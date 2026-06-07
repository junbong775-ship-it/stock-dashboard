"""급등주 스캐너 — 전체 시장에서 '급등 직전/초기' 후보를 탐색한다.

추세·정배열·패턴은 보지 않는다. 대신 전체 시장 스냅샷에서 당일 강도(변동률)로 후보를
압축한 뒤, 벌크 히스토리로 RVOL(상대 거래량)·거래대금 급증(2배↑)·갭을 계산하고
뉴스 감성과 특이 이벤트(M&A·합병완료·거래재개·FDA 등)를 가점한다.
S급 이벤트(M&A·SPAC 합병완료·거래정지 후 재개·티커변경 등)는 결과 최상단에 고정한다.

무료 데이터만 사용: 미국=스크리너 스냅샷 + yfinance 히스토리, 한국=FinanceDataReader.
프리/정규/애프터 세션 시세는 스냅샷(lastsale·Close)을 그대로 표시해 다른 탭과 일치시킨다.
"""
from __future__ import annotations

import gc
import time
import streamlit as st
import pandas as pd
import html as _html

import news_sentiment as _sent
from data_provider import _fetch_us_news, _fetch_naver_news
from bulk_data import (
    fetch_us_snapshot, fetch_kr_snapshot, fetch_us_industry_map, bulk_history,
)
from safe_exec import gather_parallel
from scan_ui import render_stage_funnel, render_session_bar
from news_view import render_news_section
# ETF/SPAC·한국 제외 판정은 메인 스캐너와 동일 규칙을 재사용한다.
from ui import _is_etf_or_spac, _is_kr_excluded

# 스냅샷 강도 상위에서 심층 분석할 종목 수 상한 (벌크 다운로드 ~10종목/초 → ~40초).
_US_CAP = 250
_KR_CAP = 150
_KRW_PER_USD = 1_350
# 뉴스 수집 상한 — 신호 후보가 많아도 메모리/시간을 묶어 Cloud에서 안정 동작.
_NEWS_CAP = 80


def _rvol_tier(rvol: float) -> tuple[int, str, str]:
    """RVOL → (점수, 라벨, 색). 1.5~3x 최선, 3~5x 양호, 5x↑ 과열, 10x↑ 추격주의."""
    if rvol >= 10:
        return 12, "추격 주의 (10x↑)", "#FF5252"
    if rvol >= 5:
        return 22, "과열 경고 (5x↑)", "#FF7043"
    if rvol >= 3:
        return 32, "양호 (3~5x)", "#69F0AE"
    if rvol >= 1.5:
        return 40, "최선 (1.5~3x)", "#00C851"
    if rvol >= 1.2:
        return 18, "보통 (1.2~1.5x)", "#FFB300"
    return 0, "약함 (<1.2x)", "#90A4AE"


def _value_surge_pts(ratio: float) -> tuple[int, str]:
    """거래대금 급증(최근 vs 직전 20일 평균) → (점수, 라벨)."""
    if ratio >= 3:
        return 20, f"폭증 ({ratio:.1f}x)"
    if ratio >= 2:
        return 15, f"급증 ({ratio:.1f}x)"
    if ratio >= 1.5:
        return 8, f"증가 ({ratio:.1f}x)"
    return 0, f"보통 ({ratio:.1f}x)"


def _score_one(snap: dict, hist: pd.DataFrame, market: str) -> dict | None:
    """스냅샷 + 히스토리로 급등 강도 점수를 산출. 추세/패턴은 보지 않는다."""
    try:
        if hist is None or len(hist) < 20:
            return None
        h = hist.dropna(subset=["Close"])
        if len(h) < 20:
            return None
        close   = float(h["Close"].iloc[-1])
        vol     = float(h["Volume"].iloc[-1])
        avg_vol = float(h["Volume"].tail(20).mean())
        if close <= 0 or avg_vol <= 0:
            return None

        rvol = vol / avg_vol
        # 거래대금 급증: 최근 5일 평균 vs 직전 20일 평균
        recent5 = float((h["Close"] * h["Volume"]).tail(5).mean())
        base20  = float((h["Close"] * h["Volume"]).tail(25).head(20).mean())
        val_ratio = recent5 / base20 if base20 > 0 else 1.0
        # 갭(직전 종가 대비 당일 시가)
        try:
            gap = (float(h["Open"].iloc[-1]) - float(h["Close"].iloc[-2])) / float(h["Close"].iloc[-2]) * 100.0
        except Exception:
            gap = 0.0

        # 표시 시세는 세션 스냅샷(프리/정규/애프터) 우선
        disp_price = snap.get("price") or close
        disp_chg   = snap.get("change_pct")
        if disp_chg is None:
            disp_chg = 0.0

        # 거래대금(표시용, KRW 기준 정규화)
        if market == "us":
            tv_krw = recent5 * _KRW_PER_USD
        else:
            tv_krw = float(snap.get("amount") or recent5)

        return {
            "code": snap.get("code") or snap.get("symbol"),
            "market": market,
            "price": disp_price,
            "intraday": float(disp_chg),
            "rvol": rvol,
            "val_ratio": val_ratio,
            "gap": gap,
            "trading_value": tv_krw,
            "mcap": snap.get("mcap") or 0.0,
        }
    except Exception:
        return None


def scan_momentum(scan_us: bool = True, scan_kr: bool = True) -> dict:
    """전체 시장 급등 후보 스캔(공개 진입점).

    실제 계산은 캐시되는 `_scan_momentum_cached`가 수행한다. 예외가 나면 그 결과를
    캐시에 남기지 않고(클리어) 안전한 빈 결과 + error 메시지를 돌려준다 — 일시적
    장애가 5분 캐시에 '고착'되어 앱이 계속 죽는 것을 막는다.
    """
    try:
        return _scan_momentum_cached(scan_us=scan_us, scan_kr=scan_kr)
    except Exception as e:  # noqa: BLE001
        try:
            _scan_momentum_cached.clear()
        except Exception:
            pass
        return {"results": [], "funnel": [], "elapsed": 0.0, "error": str(e)}


def clear_cache() -> None:
    """급등주 스캔 캐시를 비운다(새로고침 버튼용)."""
    try:
        _scan_momentum_cached.clear()
    except Exception:
        pass


@st.cache_data(ttl=300, show_spinner=False)
def _scan_momentum_cached(scan_us: bool = True, scan_kr: bool = True) -> dict:
    """전체 시장 급등 후보 스캔. 결과 + 단계별 탈락 집계를 반환한다."""
    t0 = time.time()
    c_total = c_etf = c_reduced = c_nodata = c_nosignal = 0

    us_snap = fetch_us_snapshot() if scan_us else {}
    kr_kospi = fetch_kr_snapshot("KOSPI") if scan_kr else {}
    kr_kosdaq = fetch_kr_snapshot("KOSDAQ") if scan_kr else {}
    us_ind = fetch_us_industry_map() if (scan_us and us_snap) else {}

    # ── Stage A: ETF/SPAC 제외 + 당일 강도 상위로 압축 ──
    us_rows: list[dict] = []
    for sym, row in us_snap.items():
        c_total += 1
        excl, _r = _is_etf_or_spac({
            "company_name": row.get("name", ""),
            "industry": us_ind.get(sym, ""),
            "quote_type": "EQUITY",
        })
        if excl:
            c_etf += 1
            continue
        us_rows.append(row)
    us_rows.sort(key=lambda r: (r.get("change_pct") or -999), reverse=True)
    us_keep = us_rows[:_US_CAP]
    c_reduced += max(0, len(us_rows) - len(us_keep))

    kr_rows: list[dict] = []
    for snap_map in (kr_kospi, kr_kosdaq):
        for code, row in snap_map.items():
            c_total += 1
            if _is_kr_excluded(row.get("name", "")):
                c_etf += 1
                continue
            kr_rows.append(row)
    kr_rows.sort(key=lambda r: (r.get("change_pct") or -999), reverse=True)
    kr_keep = kr_rows[:_KR_CAP]
    c_reduced += max(0, len(kr_rows) - len(kr_keep))

    # ── Stage B: 벌크 히스토리 ──
    hist_map: dict[str, pd.DataFrame] = {}
    if us_keep:
        hist_map.update(bulk_history([r["symbol"] for r in us_keep], "us", period="6mo"))
    if kr_keep:
        kr_suffix = {r["code"]: ("KS" if r.get("market") == "KOSPI" else "KQ") for r in kr_keep}
        hist_map.update(bulk_history([r["code"] for r in kr_keep], "kr",
                                     period="6mo", kr_suffix=kr_suffix))

    # ── Stage C: 점수 + 신호/뉴스/이벤트 ──
    scored: list[dict] = []
    for row, market in ([(r, "us") for r in us_keep] + [(r, "kr") for r in kr_keep]):
        key = row.get("symbol") or row.get("code")
        hist = hist_map.get(key)
        if hist is None:
            c_nodata += 1
            continue
        base = _score_one(row, hist, market)
        if base is None:
            c_nodata += 1
            continue
        base["name"] = row.get("name", key)
        scored.append(base)

    # 히스토리는 점수 계산 후 더 쓰지 않으므로 즉시 해제(Cloud 메모리 절약).
    hist_map.clear()
    gc.collect()

    # 뉴스/이벤트는 1차 신호 통과 후보에만 (속도 보존)
    def _has_signal(it: dict) -> bool:
        return (it["rvol"] >= 1.5 or it["val_ratio"] >= 2.0
                or it["intraday"] >= 5.0 or it["gap"] >= 3.0)

    signal_cands = [it for it in scored if _has_signal(it)]
    c_nosignal = len(scored) - len(signal_cands)

    # 신호 후보가 아주 많아도 뉴스 수집을 상한으로 묶는다(강도 상위 우선).
    def _strength(it: dict) -> float:
        return it["rvol"] + it["val_ratio"] + it["intraday"] / 5.0 + it["gap"] / 5.0

    news_cands = sorted(signal_cands, key=_strength, reverse=True)[:_NEWS_CAP]

    def _news_for(it: dict):
        try:
            if it["market"] == "us":
                return it["code"], _fetch_us_news(it["code"])
            return it["code"], _fetch_naver_news(it["code"])
        except Exception:
            return it["code"], []

    # 안전 병렬: 전체 마감시한 + shutdown(wait=False)로 무한 로딩/멈춤 방지.
    news_map: dict[str, list] = {}
    for _it, pair in gather_parallel(_news_for, news_cands,
                                     max_workers=8, deadline_sec=40.0):
        if pair:
            news_map[pair[0]] = pair[1]

    results: list[dict] = []
    for it in signal_cands:
        news = news_map.get(it["code"], [])
        news_score = _sent.overall_score(news) if news else 50
        ev_tier, ev_labels = _sent.classify_events(news)

        rvol_pts, rvol_label, rvol_col = _rvol_tier(it["rvol"])
        val_pts, val_label = _value_surge_pts(it["val_ratio"])
        intraday_pts = max(0.0, min(25.0, it["intraday"]))
        gap_pts = 8 if it["gap"] >= 5 else 4 if it["gap"] >= 3 else 0
        news_pts = (news_score - 50) / 50 * 10
        ev_pts = 40 if ev_tier == "S" else 15 if ev_tier == "A" else 0

        score = round(rvol_pts + val_pts + intraday_pts + gap_pts + news_pts + ev_pts)
        score = max(0, min(100, score))

        it.update({
            "rvol_label": rvol_label, "rvol_col": rvol_col,
            "val_label": val_label,
            "news": news, "news_score": news_score,
            "event_tier": ev_tier, "event_labels": ev_labels,
            "score": score,
        })
        results.append(it)

    # S급 이벤트 최상단 고정, 그다음 점수순
    results.sort(key=lambda d: (d.get("event_tier") == "S",
                                d.get("event_tier") == "A",
                                d["score"]), reverse=True)

    funnel = [
        ("총 스캔",          c_total,           "total"),
        ("ETF/제외 종목",    c_etf,             "reject"),
        ("강도 컷(상위 외)", c_reduced,         "reject"),
        ("데이터 없음",      c_nodata,          "reject"),
        ("급등 신호 없음",   c_nosignal,        "reject"),
        ("급등 후보",        len(results),      "pass"),
    ]
    return {"results": results, "funnel": funnel,
            "elapsed": time.time() - t0, "error": None}


def _fmt_value(it: dict) -> str:
    v = it.get("trading_value") or 0.0
    if v <= 0:
        return "—"
    if it["market"] == "kr":
        return f"₩{v/1e8:.0f}억"
    return f"${v/_KRW_PER_USD/1e6:.1f}M"


def _card(it: dict) -> str:
    code = _html.escape(str(it["code"]))
    name = _html.escape(str(it.get("name", it["code"])))
    is_kr = it["market"] == "kr"
    if is_kr:
        url = f"https://m.stock.naver.com/domestic/stock/{code}/total"
        px_str = f"₩{it['price']:,.0f}"
    else:
        url = f"https://m.stock.naver.com/worldstock/stock/{code}.O/total"
        px_str = f"${it['price']:,.2f}"
    chg_col = "#00C851" if it["intraday"] >= 0 else "#FF5252"
    chg_str = f"{it['intraday']:+.2f}%"
    score = it["score"]
    sc_col = "#00C851" if score >= 70 else "#FFB300" if score >= 45 else "#90A4AE"

    def _row(label, value, vcol="#D5D5EE"):
        return (
            f'<div style="display:flex;justify-content:space-between;gap:10px;'
            f'padding:4px 0;border-bottom:1px solid #23233A">'
            f'<span style="color:#8888AA;font-size:0.74rem">{label}</span>'
            f'<span style="color:{vcol};font-size:0.8rem;font-weight:600;'
            f'text-align:right">{value}</span></div>'
        )

    ev_html = ""
    if it.get("event_tier"):
        tcol = "#FF4081" if it["event_tier"] == "S" else "#FFB300"
        labels = " · ".join(_html.escape(x) for x in (it.get("event_labels") or [])[:3])
        ev_html = (
            f'<div style="margin:4px 0 8px;"><span style="background:{tcol}22;color:{tcol};'
            f'border:1px solid {tcol}66;border-radius:8px;padding:2px 8px;font-size:0.72rem;'
            f'font-weight:700">⚡ {it["event_tier"]}급 이벤트</span> '
            f'<span style="color:#B5B5D0;font-size:0.74rem">{labels}</span></div>'
        )

    news_n = len(it.get("news") or [])
    flag = "🇰🇷" if is_kr else "🇺🇸"
    return (
        '<div style="background:#1E1E2E;border:1px solid #2E2E4E;border-radius:10px;'
        'padding:14px 16px 10px;margin-bottom:4px;">'
        '<div style="display:flex;justify-content:space-between;align-items:center;">'
        f'<span style="font-size:1.0rem;font-weight:700;color:#E0E0FF">{flag} '
        f'<a href="{url}" target="_blank" rel="noopener" '
        f'style="color:#E0E0FF;text-decoration:none;border-bottom:1px dotted #6C6C9C">{name}</a>'
        f'<span style="color:#7A7A9C;font-size:0.74rem;margin-left:6px">{code}</span></span>'
        f'<span style="background:{sc_col};color:#000;font-weight:700;'
        f'padding:2px 10px;border-radius:10px;font-size:0.8rem">{score}</span></div>'
        '<div style="display:flex;justify-content:space-between;align-items:baseline;margin:6px 0 6px">'
        f'<span style="font-size:1.16rem;font-weight:700;color:#FFF">{px_str}</span>'
        f'<span style="font-size:0.88rem;font-weight:600;color:{chg_col}">{chg_str}</span></div>'
        + ev_html
        + _row("RVOL (상대거래량)", f'{it["rvol"]:.1f}x', it["rvol_col"])
        + _row("RVOL 등급", _html.escape(it["rvol_label"]), it["rvol_col"])
        + _row("거래대금 급증", _html.escape(it["val_label"]))
        + _row("갭 상승", f'{it["gap"]:+.1f}%')
        + _row("당일 변동률", chg_str, chg_col)
        + _row("거래대금", _fmt_value(it), "#FFD54F")
        + _row("뉴스 감성", f'{it["news_score"]}/100 · {news_n}건')
        + '</div>'
    )


def render_momentum_tab() -> None:
    render_session_bar()
    st.markdown("### 🔥 급등주 스캐너")
    st.caption(
        "전체 시장에서 RVOL·거래대금 급증·갭·뉴스/특이이벤트로 '급등 직전/초기'를 탐지합니다. "
        "추세·정배열·패턴은 보지 않습니다. ⚡S급 이벤트(M&A·합병완료·거래재개 등)는 최상단 고정. "
        "※ 무료 데이터 한계로 프리마켓 가격은 세션 스냅샷(lastsale)으로 대체합니다."
    )

    c1, c2 = st.columns(2)
    scan_us = c1.checkbox("🇺🇸 미국 전체", value=True, key="mo_us")
    scan_kr = c2.checkbox("🇰🇷 한국 전체", value=True, key="mo_kr")
    if st.button("🔄 급등주 스캔 새로고침", key="momentum_refresh"):
        clear_cache()

    if not (scan_us or scan_kr):
        st.info("스캔할 시장을 한 개 이상 선택하세요.")
        return

    # 스캔/렌더 전체를 보호: 어떤 예외도 앱 전체(빨간 화면)를 중단시키지 않는다.
    try:
        with st.spinner("전체 시장 급등주 스캔 중… (스냅샷 → 강도 압축 → 벌크 분석)"):
            out = scan_momentum(scan_us=scan_us, scan_kr=scan_kr)
    except Exception:
        st.error("급등주 스캔 중 일시적인 오류가 발생했습니다. 잠시 후 '새로고침'을 눌러 다시 시도해주세요.")
        return

    if out.get("error"):
        st.warning("급등주 데이터를 불러오는 중 문제가 발생했습니다. '새로고침'을 눌러 다시 시도해주세요.")
        return

    results = out.get("results") or []
    if out.get("funnel"):
        render_stage_funnel(out["funnel"], title="단계별 필터 현황", elapsed=out.get("elapsed"))

    if not results:
        st.info("현재 급등 신호(RVOL·거래대금 급증·갭·이벤트)에 해당하는 후보가 없습니다.")
        return

    st.success(f"급등 후보 {len(results)}종목 (S급 이벤트 우선 · 점수 내림차순)")
    COLS = 2
    rows = [results[i:i+COLS] for i in range(0, len(results), COLS)]
    for row in rows:
        cols = st.columns(COLS)
        for col, it in zip(cols, row):
            with col:
                try:
                    st.markdown(_card(it), unsafe_allow_html=True)
                except Exception:
                    st.warning("이 종목 카드를 표시할 수 없습니다.")
                news = it.get("news") or []
                if news:
                    try:
                        with st.expander(f"📰 뉴스 {len(news)}건 (한국어)", expanded=False):
                            render_news_section(news, key_prefix="mo", show_overall=False)
                    except Exception:
                        pass

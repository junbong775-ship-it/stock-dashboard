"""미래 10배주 스캐너 — 고성장 테마 종목을 큐레이션 유니버스에서 발굴한다.

테마(AI·우주·국방·반도체·데이터센터·에너지·전력망·로봇·자율주행·양자) 별로 큐레이션된
유니버스를 ThreadPool 로 병렬 조회한다(.info 가 느려 전체 스캔 대신 큐레이션 사용).
매출성장률(YoY)·EPS 추세·상대강도(RS)·RVOL·뉴스 키워드(정부계약/수주/가이던스/흑자전환)로
'10배 잠재력'을 종합 점수화한다.
"""
from __future__ import annotations

import streamlit as st
import yfinance as yf
import html as _html

import news_sentiment as _sent
from safe_exec import gather_parallel
from data_provider import _fetch_us_news, _session_quote, display_quote
from bulk_data import fetch_live_quotes, live_symbol
from news_view import render_news_section


# 테마별 큐레이션 유니버스 (시총 ≤ $10B 고성장 후보 중심)
THEMATIC_UNIVERSE: dict[str, list[str]] = {
    "AI 인프라":   ["SMCI", "VRT", "CRDO", "ALAB", "NBIS", "CRWV"],
    "우주":        ["RKLB", "ASTS", "LUNR", "PL", "RDW"],
    "국방":        ["KTOS", "AVAV", "DRS", "PLTR", "RCAT", "ONDS"],
    "반도체":      ["AEHR", "ACLS", "INDI", "NVTS", "WOLF", "SITM"],
    "데이터센터":  ["VRT", "CRDO", "NBIS", "APLD", "IREN", "CIFR"],
    "에너지":      ["OKLO", "SMR", "LEU", "CCJ", "UEC", "NNE"],
    "전력망":      ["GEV", "POWL", "PWR", "ASPN", "FLNC"],
    "로봇":        ["SERV", "RR", "OUST", "NNDM"],
    "자율주행":    ["INDI", "MVIS", "LIDR", "AUR"],
    "양자":        ["IONQ", "RGTI", "QBTS", "QUBT", "ARQQ", "LAES"],
}

MCAP_MAX = 10_000_000_000   # 시총 상한 $10B

# 뉴스 가산 키워드 (한/영) — 정부계약·신규수주·가이던스 상향·흑자전환
_BONUS_KEYWORDS: dict[str, tuple[str, ...]] = {
    "정부계약": ("government contract", "defense contract", "awarded", "pentagon",
                 "nasa", "정부", "계약", "수주"),
    "신규수주": ("new order", "backlog", "purchase order", "bookings", "수주", "주문"),
    "가이던스상향": ("raises guidance", "raised guidance", "upgrades outlook",
                     "beats estimates", "guidance", "가이던스", "상향"),
    "흑자전환": ("turns profitable", "profitability", "positive earnings",
                 "흑자", "흑자전환"),
}


def _theme_of(ticker: str) -> list[str]:
    return [th for th, ts in THEMATIC_UNIVERSE.items() if ticker in ts]


def _news_bonuses(news: list[dict]) -> list[str]:
    found: list[str] = []
    blob = " ".join(
        f"{(n.get('title') or '')} {(n.get('summary') or '')}" for n in news
    ).lower()
    for tag, kws in _BONUS_KEYWORDS.items():
        if any(kw.lower() in blob for kw in kws):
            found.append(tag)
    return found


def _scan_one(ticker: str) -> dict | None:
    try:
        tk   = yf.Ticker(ticker)
        hist = tk.history(period="1y")
        if hist is None or hist.empty:
            return None
        # 프리마켓에는 yfinance 가 '오늘'의 NaN 일봉을 덧붙여 마지막 종가가 NaN 이 되고,
        # 그러면 close/RS 가 NaN → min(20, NaN/5)=20 으로 상대강도 점수가 전 종목 만점으로
        # 오염된다. NaN 종가 행을 제거해 iloc[-1] 이 항상 '직전 완성 일봉'을 가리키게 한다.
        hist = hist.dropna(subset=["Close"])
        if len(hist) < 30:
            return None
        close   = float(hist["Close"].iloc[-1])
        vol     = float(hist["Volume"].iloc[-1])
        avg_vol = float(hist["Volume"].tail(20).mean())
        if close <= 0:
            return None

        info = {}
        try:
            info = tk.info or {}
        except Exception:
            info = {}
        mcap = info.get("marketCap") or 0
        if mcap and mcap > MCAP_MAX:
            return None

        rev_growth = info.get("revenueGrowth")          # YoY (소수, 예 0.45 = +45%)
        eps_fwd    = info.get("forwardEps")
        eps_trail  = info.get("trailingEps")
        industry   = info.get("industry", "") or info.get("sector", "")
        rvol       = (vol / avg_vol) if avg_vol > 0 else 0.0

        # 상대강도(RS) — 6개월(약 126봉) 전 대비 가격 상승률
        rs = 0.0
        if len(hist) >= 130:
            past = float(hist["Close"].iloc[-126])
            if past > 0:
                rs = (close - past) / past * 100.0

        try:
            news = _fetch_us_news(ticker)
        except Exception:
            news = []
        bonuses    = _news_bonuses(news)
        news_score = _sent.overall_score(news) if news else 50

        # ── 종합 점수 ──────────────────────────────────────────────────────
        # 매출성장 (최대 +35): +50% YoY 이상이면 만점 근접
        rg_pts = 0.0
        if isinstance(rev_growth, (int, float)):
            rg_pts = max(0.0, min(35.0, rev_growth * 70.0))   # 0.5 → 35
        # EPS 추세 (최대 +15): 흑자 또는 개선
        eps_pts = 0.0
        if isinstance(eps_fwd, (int, float)) and isinstance(eps_trail, (int, float)):
            if eps_fwd > eps_trail:
                eps_pts += 8
            if eps_fwd > 0:
                eps_pts += 7
        elif isinstance(eps_trail, (int, float)) and eps_trail > 0:
            eps_pts += 5
        # 상대강도 (최대 +20)
        rs_pts = max(0.0, min(20.0, rs / 5.0))                # +100% → 20
        # RVOL (최대 +10)
        rvol_pts = max(0.0, min(10.0, (rvol - 1.0) * 10.0))
        # 뉴스 키워드 보너스 (각 +5, 최대 +20)
        bonus_pts = min(20, len(bonuses) * 5)

        score = round(rg_pts + eps_pts + rs_pts + rvol_pts + bonus_pts)
        score = max(0, min(100, score))

        # 표시용 세션 시세(프리/정규/애프터) — 위에서 이미 받은 info 재사용(추가 호출 없음).
        # RS·점수는 일봉 close 기준을 유지하고, 화면 표시 price 만 세션 시세로 보정한다.
        # 시세 우선순위는 공통 엔진(display_quote)으로 일원화(US-only 유니버스).
        sess_px, _sess_chg = _session_quote(info, close)
        disp_price, _ = display_quote(
            "us", live_price=sess_px, live_change=_sess_chg,
            hist_close=close, hist_change=None,
        )

        return {
            "ticker":     ticker,
            "price":      disp_price,
            "market_session": str(info.get("marketState") or "").upper(),
            "themes":     _theme_of(ticker),
            "industry":   industry,
            "mcap":       mcap,
            "rev_growth": rev_growth,
            "eps_fwd":    eps_fwd,
            "eps_trail":  eps_trail,
            "rs":         rs,
            "rvol":       rvol,
            "bonuses":    bonuses,
            "news":       news,
            "news_score": news_score,
            "score":      score,
        }
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def scan_growth() -> list[dict]:
    """테마 유니버스를 병렬 조회해 10배주 후보(종합점수 내림차순)를 반환한다."""
    universe: list[str] = []
    seen: set[str] = set()
    for ts in THEMATIC_UNIVERSE.values():
        for t in ts:
            if t not in seen:
                seen.add(t)
                universe.append(t)

    # 안전 병렬: yfinance .info 가 hang-prone 이므로 전체 마감시한으로 보호한다.
    out: list[dict] = []
    for _t, r in gather_parallel(_scan_one, universe,
                                 max_workers=8, deadline_sec=60.0):
        if r:
            out.append(r)
    out.sort(key=lambda d: d["score"], reverse=True)
    return out


def _fmt_mcap(v) -> str:
    if not isinstance(v, (int, float)) or v <= 0:
        return "—"
    if v >= 1e9:
        return f"${v/1e9:.2f}B"
    return f"${v/1e6:.0f}M"


def _fmt_pct(v) -> str:
    return f"{v*100:+.1f}%" if isinstance(v, (int, float)) else "—"


def _card(it: dict) -> str:
    t   = _html.escape(it["ticker"])
    url = f"https://m.stock.naver.com/worldstock/stock/{t}.O/total"
    score  = it["score"]
    sc_col = "#00C851" if score >= 70 else "#FFB300" if score >= 45 else "#90A4AE"

    def _row(label, value, vcol="#D5D5EE"):
        return (
            f'<div style="display:flex;justify-content:space-between;gap:10px;'
            f'padding:4px 0;border-bottom:1px solid #23233A">'
            f'<span style="color:#8888AA;font-size:0.74rem">{label}</span>'
            f'<span style="color:{vcol};font-size:0.8rem;font-weight:600;'
            f'text-align:right">{value}</span></div>'
        )

    theme_html = ''.join(
        f'<span style="display:inline-block;background:#23233A;color:#9FA8DA;'
        f'border:1px solid #3A3A5C;border-radius:8px;padding:1px 7px;'
        f'margin:0 4px 3px 0;font-size:0.7rem;font-weight:600">{_html.escape(th)}</span>'
        for th in it.get("themes", [])
    ) or '<span style="color:#666">—</span>'

    bonus_html = ''.join(
        f'<span style="display:inline-block;background:#1B3A2A;color:#69F0AE;'
        f'border:1px solid #2E5C44;border-radius:8px;padding:1px 7px;'
        f'margin:0 4px 3px 0;font-size:0.7rem;font-weight:700">{_html.escape(b)}</span>'
        for b in it.get("bonuses", [])
    )

    eps_txt = "—"
    if isinstance(it.get("eps_fwd"), (int, float)) or isinstance(it.get("eps_trail"), (int, float)):
        ef = it.get("eps_fwd"); et = it.get("eps_trail")
        ef_s = f"{ef:.2f}" if isinstance(ef, (int, float)) else "—"
        et_s = f"{et:.2f}" if isinstance(et, (int, float)) else "—"
        eps_txt = f"{et_s} → {ef_s}"

    news_n = len(it.get("news") or [])
    return (
        '<div style="background:#1E1E2E;border:1px solid #2E2E4E;border-radius:10px;'
        'padding:14px 16px 10px;margin-bottom:4px;">'
        '<div style="display:flex;justify-content:space-between;align-items:center;">'
        f'<span style="font-size:1.02rem;font-weight:700;color:#E0E0FF">'
        f'<a href="{url}" target="_blank" rel="noopener" '
        f'style="color:#E0E0FF;text-decoration:none;border-bottom:1px dotted #6C6C9C">{t}</a></span>'
        f'<span style="background:{sc_col};color:#000;font-weight:700;'
        f'padding:2px 10px;border-radius:10px;font-size:0.8rem">{score}</span></div>'
        '<div style="margin:6px 0 8px">'
        f'<span style="font-size:1.18rem;font-weight:700;color:#FFF">${it["price"]:,.2f}</span></div>'
        f'<div style="margin-bottom:6px">{theme_html}</div>'
        + (f'<div style="margin-bottom:6px">{bonus_html}</div>' if bonus_html else '')
        + _row("매출성장(YoY)", _fmt_pct(it.get("rev_growth")), "#69F0AE")
        + _row("EPS (직전→예상)", _html.escape(eps_txt))
        + _row("상대강도(6M)", f'{it["rs"]:+.0f}%', "#64B5F6")
        + _row("RVOL", f'{it["rvol"]:.1f}x')
        + _row("시가총액", _fmt_mcap(it["mcap"]), "#FFD54F")
        + _row("업종", _html.escape(str(it.get("industry") or "—")))
        + _row("뉴스 감성", f'{it["news_score"]}/100 · {news_n}건')
        + '</div>'
    )


def render_growth_tab() -> None:
    st.markdown("### 🌱 미래 10배주 스캐너")
    st.caption(
        "AI·우주·국방·반도체·데이터센터·에너지·전력망·로봇·자율주행·양자 테마의 큐레이션 유니버스에서 "
        "매출성장률·EPS 추세·상대강도·RVOL·뉴스 키워드(정부계약/수주/가이던스/흑자전환)로 10배 잠재력을 점수화합니다."
    )
    if st.button("🔄 10배주 스캔 새로고침", key="growth_refresh"):
        scan_growth.clear()

    themes = ["전체"] + list(THEMATIC_UNIVERSE.keys())
    sel = st.selectbox("테마 필터", themes, key="growth_theme_filter")

    with st.spinner("미래 10배주 스캔 중… (테마 유니버스)"):
        results = scan_growth()

    if sel != "전체":
        results = [r for r in results if sel in r.get("themes", [])]

    if not results:
        st.info("현재 조건(시총 ≤ $10B)에 맞는 테마 후보가 없습니다.")
        return

    # 화면 출력 직전 현재가 재조회 — 최종 테마 후보만 실시간 갱신(전체 실시간 X).
    try:
        _syms = [live_symbol(it.get("ticker", ""), "us") for it in results]
        _q = fetch_live_quotes(tuple(dict.fromkeys(_syms)))
        for _it, _sym in zip(results, _syms):
            _hit = _q.get(_sym)
            if _hit:
                _it["price"] = _hit[0]
    except Exception:
        pass

    st.success(f"테마 후보 {len(results)}종목 (종합점수 내림차순)")
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
                            render_news_section(news, key_prefix="gr", show_overall=False)
                    except Exception:
                        pass

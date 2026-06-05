"""급등주 스캐너 — 소형주 급등 후보를 큐레이션 유니버스에서 탐색한다.

.info 조회는 종목당 느리므로 **전체 시장 스캔이 아니라 큐레이션된 소형주 유니버스**만
ThreadPool 로 병렬 조회한다. 급등 강도 점수는 RVOL(상대 거래량) + 당일 변동률 +
거래량 증가로 산출(프리마켓 거래량은 무료 데이터에 없음). **화면 표시 시세(price·
당일변동률)는 _session_quote 로 프리/정규/애프터 세션 시세를 반영**해 다른 탭과 일치시킨다.
"""
from __future__ import annotations

import streamlit as st
import yfinance as yf
import pandas as pd
import html as _html
from concurrent.futures import ThreadPoolExecutor, as_completed

import news_sentiment as _sent
from data_provider import _fetch_us_news, _session_quote
from news_view import render_news_section


# 큐레이션 소형주 유니버스 — 변동성 높은 저가 소형주 중심(전체 스캔 대체).
MOMENTUM_UNIVERSE: list[str] = [
    "LUNR", "ASTS", "RKLB", "PL", "IRDM", "BBAI", "SOUN", "RGTI", "QUBT", "QBTS",
    "IONQ", "ARQQ", "LAES", "OUST", "MVIS", "LIDR", "INDI", "NVTS", "WOLF", "LCID",
    "CHPT", "PLUG", "FCEL", "BLNK", "RUN", "SES", "QS", "ACHR", "JOBY", "EVTL",
    "DNA", "RXRX", "TMC", "MARA", "RIOT",
]

# RVOL(상대 거래량) 점수 구간 — 1.5~3x 최선, 3~5x 양호, 5x↑ 과열, 10x↑ 추격주의
MCAP_MAX     = 300_000_000     # 시총 상한 (소형주 정의)
PRICE_MIN    = 0.5
PRICE_MAX    = 10.0


def _rvol_tier(rvol: float) -> tuple[int, str, str]:
    """RVOL → (점수, 라벨, 색). 1.5~3x 최선, 3~5x 양호, 5x↑ 과열, 10x↑ 추격주의."""
    if rvol >= 10:
        return 10, "추격 주의 (10x↑)", "#FF5252"
    if rvol >= 5:
        return 20, "과열 경고 (5x↑)", "#FF7043"
    if rvol >= 3:
        return 30, "양호 (3~5x)", "#69F0AE"
    if rvol >= 1.5:
        return 40, "최선 (1.5~3x)", "#00C851"
    if rvol >= 1.2:
        return 15, "보통 (1.2~1.5x)", "#FFB300"
    return 0, "약함 (<1.2x)", "#90A4AE"


def _scan_one(ticker: str) -> dict | None:
    try:
        tk   = yf.Ticker(ticker)
        hist = tk.history(period="3mo")
        if hist is None or hist.empty:
            return None
        # 프리마켓에는 yfinance 가 '오늘'의 NaN 일봉을 덧붙여 마지막 종가가 NaN 이 되고,
        # 그러면 close/prev/당일변동률(intraday)이 NaN → min/max 클램프가 NaN 을 거르지
        # 못해 점수가 오염된다. NaN 종가 행을 제거해 iloc[-1] 이 '직전 완성 일봉'을 가리키게 한다.
        hist = hist.dropna(subset=["Close"])
        if len(hist) < 20:
            return None
        close   = float(hist["Close"].iloc[-1])
        prev    = float(hist["Close"].iloc[-2])
        vol     = float(hist["Volume"].iloc[-1])
        avg_vol = float(hist["Volume"].tail(20).mean())
        if avg_vol <= 0 or close <= 0:
            return None

        # 가격 필터 (저가 소형주)
        if not (PRICE_MIN <= close <= PRICE_MAX):
            return None

        # 펀더멘털 (.info — 느림, 큐레이션 유니버스라 허용)
        info = {}
        try:
            info = tk.info or {}
        except Exception:
            info = {}
        mcap  = info.get("marketCap") or 0
        flt   = info.get("floatShares") or info.get("sharesOutstanding") or 0
        # 시총 필터 (정보 있을 때만)
        if mcap and mcap > MCAP_MAX:
            return None

        rvol      = vol / avg_vol
        intraday  = (close - prev) / prev * 100.0
        # 최근 5일 vs 직전 20일 거래량 증가
        recent5   = float(hist["Volume"].tail(5).mean())
        base20    = float(hist["Volume"].tail(25).head(20).mean()) or avg_vol
        vol_inc   = recent5 / base20 if base20 > 0 else 1.0

        # 뉴스 (감성)
        try:
            news = _fetch_us_news(ticker)
        except Exception:
            news = []
        news_score = _sent.overall_score(news) if news else 50

        # ── 점수 산출 ──────────────────────────────────────────────────────
        rvol_pts, rvol_label, rvol_col = _rvol_tier(rvol)
        intraday_pts = max(0, min(20, intraday))           # 당일 상승률 (최대 +20)
        volinc_pts   = 15 if vol_inc >= 1.5 else 8 if vol_inc >= 1.2 else 0
        news_pts     = (news_score - 50) / 50 * 15         # -15 ~ +15
        score = round(rvol_pts + intraday_pts + volinc_pts + news_pts)
        score = max(0, min(100, score))

        # 표시용 세션 시세(프리/정규/애프터) — 위에서 이미 받은 info 재사용(추가 호출 없음).
        # 점수는 위에서 일봉 기준으로 이미 산출됐고, 여기서는 화면 표시용 price/당일변동률만
        # 세션 시세로 보정한다(가격과 % 의 기준이 어긋나지 않도록 둘을 함께 교체).
        sess_px, sess_chg = _session_quote(info, close)
        if sess_px and sess_px > 0:
            disp_price = sess_px
            # 세션 price 적용 시 % 도 같은 기준이어야 함: 세션 %가 없으면 일봉 종가 대비
            # 재계산(과거 일봉 % 를 새 price 옆에 그대로 두지 않는다).
            disp_chg = sess_chg if sess_chg is not None else (sess_px - close) / close * 100.0
        else:
            disp_price = close
            disp_chg   = intraday

        return {
            "ticker":     ticker,
            "price":      disp_price,
            "intraday":   disp_chg,
            "market_session": str(info.get("marketState") or "").upper(),
            "rvol":       rvol,
            "rvol_label": rvol_label,
            "rvol_col":   rvol_col,
            "vol_inc":    vol_inc,
            "mcap":       mcap,
            "float":      flt,
            "news":       news,
            "news_score": news_score,
            "score":      score,
        }
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def scan_momentum() -> list[dict]:
    """큐레이션 유니버스를 병렬 조회해 급등 후보 리스트(점수 내림차순)를 반환한다."""
    out: list[dict] = []
    seen: set[str] = set()
    universe = [t for t in MOMENTUM_UNIVERSE if not (t in seen or seen.add(t))]
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(_scan_one, t): t for t in universe}
        for f in as_completed(futs):
            r = f.result()
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


def _card(it: dict) -> str:
    t   = _html.escape(it["ticker"])
    url = f"https://m.stock.naver.com/worldstock/stock/{t}.O/total"
    chg_col = "#00C851" if it["intraday"] >= 0 else "#FF5252"
    chg_str = f"{it['intraday']:+.2f}%"
    score   = it["score"]
    sc_col  = "#00C851" if score >= 70 else "#FFB300" if score >= 45 else "#90A4AE"

    def _row(label, value, vcol="#D5D5EE"):
        return (
            f'<div style="display:flex;justify-content:space-between;gap:10px;'
            f'padding:4px 0;border-bottom:1px solid #23233A">'
            f'<span style="color:#8888AA;font-size:0.74rem">{label}</span>'
            f'<span style="color:{vcol};font-size:0.8rem;font-weight:600;'
            f'text-align:right">{value}</span></div>'
        )

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
        '<div style="display:flex;justify-content:space-between;align-items:baseline;margin:6px 0 8px">'
        f'<span style="font-size:1.18rem;font-weight:700;color:#FFF">${it["price"]:,.2f}</span>'
        f'<span style="font-size:0.88rem;font-weight:600;color:{chg_col}">{chg_str}</span></div>'
        + _row("RVOL (상대거래량)", f'{it["rvol"]:.1f}x', it["rvol_col"])
        + _row("RVOL 등급", _html.escape(it["rvol_label"]), it["rvol_col"])
        + _row("거래량 증가(5/20)", f'{it["vol_inc"]:.1f}x')
        + _row("당일 변동률", chg_str, chg_col)
        + _row("시가총액", _fmt_mcap(it["mcap"]), "#FFD54F")
        + _row("뉴스 감성", f'{it["news_score"]}/100 · {news_n}건')
        + '</div>'
    )


def render_momentum_tab() -> None:
    st.markdown("### 🔥 급등주 스캐너")
    st.caption(
        "큐레이션 소형주 유니버스에서 RVOL·당일 변동률·거래량 증가·뉴스 감성으로 급등 강도를 점수화합니다. "
        "※ 무료 데이터에 프리마켓 가격이 없어 RVOL+당일 변동률로 대체합니다."
    )
    if st.button("🔄 급등주 스캔 새로고침", key="momentum_refresh"):
        scan_momentum.clear()

    with st.spinner("급등주 스캔 중… (큐레이션 유니버스)"):
        results = scan_momentum()

    if not results:
        st.info("현재 조건(시총 ≤ $300M · 가격 $0.5~$10)에 맞는 급등 후보가 없습니다.")
        return

    st.success(f"급등 후보 {len(results)}종목 (점수 내림차순)")
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
                    with st.expander(f"📰 뉴스 {len(news)}건 (한국어)", expanded=False):
                        render_news_section(news, key_prefix="mo", show_overall=False)

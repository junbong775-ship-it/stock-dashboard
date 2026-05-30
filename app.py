import datetime
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import FinanceDataReader as fdr
from deep_translator import GoogleTranslator

# ── Constants ────────────────────────────────────────────────────────────────────
TARGET_MULT   = 1.30
STOPLOSS_MULT = 0.82

import streamlit as st
import yfinance as yf
import pandas as pd
import FinanceDataReader as fdr
import time

# --- 시장별 종목 리스트 가져오는 함수 ---
@st.cache_data
def get_tickers(market):
    if market == "S&P 500":
        df = fdr.StockListing('S&P500')
        return df['Symbol'].tolist()[:100]  # 속도를 위해 상위 100개만
    elif market == "나스닥 100":
        # NASDAQ 전체 중 시총 상위 100개 추출
        df = fdr.StockListing('NASDAQ')
        return df['Symbol'].tolist()[:100]
    elif market == "코스피 200":
        df = fdr.StockListing('KRX-KOSPI')
        return df['Code'].tolist()[:100]
    elif market == "코스닥 150":
        df = fdr.StockListing('KRX-KOSDAQ')
        return df['Code'].tolist()[:100]
    elif market == "러셀 2000":
        # NYSE/AMEX/NASDAQ 합친 목록 등에서 필터링
        df = fdr.StockListing('NYSE')
        return df['Symbol'].tolist()[:100]
    return []

# --- 메인 화면 ---
st.title("📈 자동 종목 스캐너")

# 탭 생성
tabs = ["S&P 500", "나스닥 100", "러셀 2000", "코스피 200", "코스닥 150"]
selected_tab = st.tabs(tabs)

# --- 각 탭 로직 ---
for i, tab in enumerate(selected_tab):
    with tab:
        market = tabs[i]
        if st.button(f"{market} 스캔 시작"):
            tickers = get_tickers(market)
            st.write(f"총 {len(tickers)}개 종목 스캔 중...")
            
            # 스캔 로직 (이평선 계산 등)
            for ticker in tickers:
                try:
                    df = yf.download(ticker, period="1y", progress=False)
                    if not df.empty:
                        df['MA20'] = df['Close'].rolling(window=20).mean()
                        # 여기에 원하시는 조건문 추가
                        # 예: if df['Close'].iloc[-1] > df['MA20'].iloc[-1]: ...
                except:
                    continue
            st.success("스캔 완료!")
GRADE_COLOR: dict[str, str] = {
    'SSS': '#FFD700', 'SS': '#FF8C00', 'S':  '#00C851',
    'A':   '#2196F3', 'B':  '#9E9E9E', 'C':  '#FF5722', 'D': '#F44336',
}


# ── KRX Listing (cached 1 hour) ───────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def get_krx_listing() -> pd.DataFrame:
    """Download full KRX stock list from FinanceDataReader. ~2500 rows."""
    df = fdr.StockListing('KRX')
    # Normalise column names — fdr returns 'Code' and 'Name'
    if 'Symbol' in df.columns and 'Code' not in df.columns:
        df = df.rename(columns={'Symbol': 'Code'})
    df['Code'] = df['Code'].astype(str).str.zfill(6)
    df['Name'] = df['Name'].astype(str).str.strip()
    return df[['Code', 'Name']].drop_duplicates('Code').reset_index(drop=True)


def get_ticker_by_name(name: str) -> tuple[str | None, list[str]]:
    """
    Search KRX listing for a company name.
    Returns (6-digit code | None, candidate_names_list).
    """
    try:
        listing = get_krx_listing()
    except Exception:
        return None, []

    query = name.strip()

    # 1. Exact match
    exact = listing[listing['Name'] == query]
    if not exact.empty:
        return str(exact.iloc[0]['Code']), []

    # 2. Case-insensitive exact match
    exact_ci = listing[listing['Name'].str.lower() == query.lower()]
    if not exact_ci.empty:
        return str(exact_ci.iloc[0]['Code']), []

    # 3. Contains match (partial)
    partial = listing[listing['Name'].str.contains(query, na=False, case=False)]
    if partial.empty:
        return None, []
    if len(partial) == 1:
        return str(partial.iloc[0]['Code']), []

    # Multiple — return top candidates
    return None, list(partial['Name'].head(8))


def get_name_by_code(code: str) -> str:
    """Look up display name from scanner dict or KRX listing."""
    if code in KR_NAMES:
        return KR_NAMES[code]
    try:
        listing = get_krx_listing()
        row = listing[listing['Code'] == code]
        if not row.empty:
            return str(row.iloc[0]['Name'])
    except Exception:
        pass
    return code


# ── Input Detection ───────────────────────────────────────────────────────────────
def has_korean(s: str) -> bool:
    return any('\uAC00' <= c <= '\uD7A3' for c in s)


def resolve_input(raw: str) -> tuple[str | None, str, list[str], str]:
    """
    Parse user input and return (code | None, display_name, suggestions, market).
    market is 'kr' or 'us'.
    """
    s = raw.strip()

    # Already has KS/KQ exchange suffix
    if s.upper().endswith('.KS') or s.upper().endswith('.KQ'):
        code = s.split('.')[0].zfill(6)
        return code, get_name_by_code(code), [], 'kr'

    # Pure 6-digit numeric → KR code
    if s.isdigit() and len(s) == 6:
        return s.zfill(6), get_name_by_code(s.zfill(6)), [], 'kr'

    # Contains Korean characters → search KRX listing
    if has_korean(s):
        code, candidates = get_ticker_by_name(s)
        if code:
            return code, get_name_by_code(code), [], 'kr'
        return None, s, candidates, 'kr'

    # English: check if it's a known US ticker
    upper = s.upper()
    return upper, upper, [], 'us'


# ── Technical Indicators ──────────────────────────────────────────────────────────
def _calc_rsi(close: pd.Series, period: int = 14) -> float:
    delta    = close.diff()
    avg_gain = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, float('nan'))
    return float((100 - 100 / (1 + rs)).iloc[-1])


def _calc_mfi(hist: pd.DataFrame, period: int = 14) -> float:
    tp      = (hist['High'] + hist['Low'] + hist['Close']) / 3
    mf      = tp * hist['Volume']
    pos_mf  = mf.where(tp > tp.shift(1), 0)
    neg_mf  = mf.where(tp < tp.shift(1), 0)
    pos_sum = pos_mf.rolling(period).sum()
    neg_sum = neg_mf.rolling(period).sum()
    mfi     = 100 - (100 / (1 + pos_sum / neg_sum.replace(0, float('nan'))))
    return float(mfi.iloc[-1])


def _build_result(hist: pd.DataFrame, news: list) -> dict | None:
    """Shared post-processing for both KR and US OHLCV DataFrames."""
    if hist is None or hist.empty or len(hist) < 20:
        return None
    hist = hist.copy()
    hist['MA5']  = hist['Close'].rolling(5).mean()
    hist['MA10'] = hist['Close'].rolling(10).mean()
    hist['MA20'] = hist['Close'].rolling(20).mean()

    spread = float(
        ((hist[['MA5','MA10','MA20']].max(axis=1) - hist[['MA5','MA10','MA20']].min(axis=1))
         / hist['Close'] * 100).iloc[-1]
    )
    price      = float(hist['Close'].iloc[-1])
    change_pct = float((price - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2] * 100)
    rsi        = _calc_rsi(hist['Close'])
    mfi        = _calc_mfi(hist) if 'Volume' in hist.columns and hist['Volume'].sum() > 0 else 50.0

    return dict(spread=spread, price=price, change_pct=change_pct,
                rsi=rsi, mfi=mfi, hist=hist, news=news)


# ── News helpers ─────────────────────────────────────────────────────────────────
def _extract_news_item(n: dict) -> dict | None:
    """Pull title + URL out of a yfinance news dict (handles old & new formats)."""
    content = n.get('content', {}) if isinstance(n.get('content'), dict) else {}
    title   = content.get('title', '') or n.get('title', '')
    if not title:
        return None
    # URL — try multiple locations yfinance uses across versions
    url = (
        content.get('canonicalUrl', {}).get('url', '') or
        content.get('clickThroughUrl', {}).get('url', '') or
        n.get('link', '') or
        n.get('url', '')
    )
    return {'title': title, 'url': url}


def _translate_news(raw_items: list[dict]) -> list[dict]:
    """Translate a list of {title, url} dicts; returns {original, translated, url}."""
    result = []
    for item in raw_items:
        title = item['title']
        url   = item.get('url', '')
        try:
            translated = GoogleTranslator(source='auto', target='ko').translate(title)
        except Exception:
            translated = title
        result.append({'original': title, 'translated': translated, 'url': url})
    return result


def _fetch_kr_news(code: str) -> list[dict]:
    try:
        stock    = yf.Ticker(f"{code}.KS")
        raw      = [_extract_news_item(n) for n in (stock.news or [])[:5]]
        raw      = [r for r in raw if r][:3]
        return _translate_news(raw)
    except Exception:
        return []


# ── Data Fetching ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def get_data_kr(code: str) -> dict | None:
    """Fetch Korean stock OHLCV via FinanceDataReader."""
    try:
        start = (datetime.date.today() - datetime.timedelta(days=120)).strftime('%Y-%m-%d')
        hist  = fdr.DataReader(code, start)
        if hist is None or hist.empty:
            return None
        # fdr column names are already Open/High/Low/Close/Volume
        hist = hist[['Open','High','Low','Close','Volume']].dropna()
        news = _fetch_kr_news(code)
        return _build_result(hist, news)
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def get_data_us(ticker: str) -> dict | None:
    """Fetch US stock OHLCV via yfinance."""
    try:
        stock = yf.Ticker(ticker)
        hist  = stock.history(period="3mo")
        if hist.empty:
            return None
        hist = hist[['Open','High','Low','Close','Volume']].dropna()

        raw   = [_extract_news_item(n) for n in (stock.news or [])[:5]]
        raw   = [r for r in raw if r][:3]
        news_items = _translate_news(raw)
        return _build_result(hist, news_items)
    except Exception:
        return None


def get_data(code_or_ticker: str, market: str = 'auto') -> dict | None:
    """Unified entry point. Routes to get_data_kr or get_data_us."""
    s = code_or_ticker.strip()
    # Strip .KS/.KQ suffix
    if '.' in s:
        s = s.split('.')[0]
    if market == 'kr' or (market == 'auto' and s.isdigit() and len(s) == 6):
        return get_data_kr(s.zfill(6))
    return get_data_us(s)


# ── AI Scoring ────────────────────────────────────────────────────────────────────
def calculate_ai_score(data: dict) -> tuple[int, str]:
    hist, rsi, mfi, spread = data['hist'], data['rsi'], data['mfi'], data['spread']
    ma5  = float(hist['MA5'].iloc[-1])
    ma10 = float(hist['MA10'].iloc[-1])
    ma20 = float(hist['MA20'].iloc[-1])

    ma_score = (40 if ma5 > ma10 > ma20 else
                22 if (ma5 > ma10 or ma10 > ma20) else
                0  if ma5 < ma10 < ma20 else 12)
    rsi_score = (30 if 40 <= rsi <= 60 else
                 20 if (30 <= rsi < 40) or (60 < rsi <= 70) else
                 10 if (20 <= rsi < 30) or (70 < rsi <= 80) else 0)
    mfi_score = (20 if 40 <= mfi <= 60 else
                 14 if (30 <= mfi < 40) or (60 < mfi <= 70) else
                 8  if (20 <= mfi < 30) or (70 < mfi <= 80) else 2)
    spread_score = 10 if spread <= 1.0 else (7 if spread <= 2.0 else 4)

    score = ma_score + rsi_score + mfi_score + spread_score
    grade = ('SSS' if score >= 90 else 'SS' if score >= 80 else 'S'  if score >= 70 else
             'A'   if score >= 60 else 'B'  if score >= 50 else 'C'  if score >= 40 else 'D')
    return score, grade


# ── Chart ─────────────────────────────────────────────────────────────────────────
def make_chart(hist: pd.DataFrame, label: str) -> go.Figure:
    fig = go.Figure()
    for col, color, name in [
        ('Close','#5B8DEF','종가'), ('MA5','#FF9F40','MA5'),
        ('MA10','#4BC0C0','MA10'), ('MA20','#FF6384','MA20'),
    ]:
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist[col], name=name,
            line=dict(color=color, width=1.8), opacity=0.9
        ))
    fig.update_layout(
        title=dict(text=f'{label} — 이동평균선 (3개월)', font=dict(size=13)),
        height=300, margin=dict(l=0, r=0, t=36, b=0),
        legend=dict(orientation='h', y=1.12, font=dict(size=11)),
        xaxis_title=None, yaxis_title='가격',
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


# ── Shared: Full Detail Block ─────────────────────────────────────────────────────
def render_full_detail(label: str, data: dict) -> None:
    try:
        score, grade = calculate_ai_score(data)
    except Exception:
        score, grade = 0, 'D'

    price     = data.get('price', 0)
    change    = data.get('change_pct', 0)
    spread    = data.get('spread', 0)
    rsi       = data.get('rsi', 0)
    mfi       = data.get('mfi', 0)
    hist      = data.get('hist')
    news      = data.get('news', [])
    target    = price * TARGET_MULT
    stop_loss = price * STOPLOSS_MULT
    change_str = f"+{change:.2f}%" if change >= 0 else f"{change:.2f}%"
    badge_col  = GRADE_COLOR.get(grade, '#9E9E9E')

    # ── Row 1: 핵심 지표 4개 ─────────────────────────────────────────────────
    try:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("현재가",        f"{price:,.2f}",     change_str)
        c2.metric("AI 점수",       f"{score} / 100",    f"등급 {grade}")
        c3.metric("목표가 (+30%)", f"{target:,.2f}",    f"+{target - price:,.2f}")
        c4.metric("손절가 (−18%)", f"{stop_loss:,.2f}", f"−{price - stop_loss:,.2f}",
                  delta_color="inverse")
    except Exception:
        st.warning("지표를 표시할 수 없습니다.")

    st.divider()

    # ── Row 2: 기술 지표 ──────────────────────────────────────────────────────
    try:
        t1, t2, t3, t4, t5 = st.columns(5)
        t1.metric("RSI (14)", f"{rsi:.1f}")
        t2.metric("MFI (14)", f"{mfi:.1f}")
        t3.metric("밀집도",    f"{spread:.2f}%")
        if hist is not None:
            t4.metric("MA5",  f"{hist['MA5'].iloc[-1]:,.2f}")
            t5.metric("MA10 / MA20",
                      f"{hist['MA10'].iloc[-1]:,.2f} / {hist['MA20'].iloc[-1]:,.2f}")
    except Exception:
        pass

    # ── 차트 ──────────────────────────────────────────────────────────────────
    try:
        if hist is not None:
            st.plotly_chart(make_chart(hist, label), width='stretch')
    except Exception:
        pass

    # ── 뉴스 (접힌 상태로 시작) ───────────────────────────────────────────────
    with st.expander("📰 최신 관련 뉴스 (한글)", expanded=False):
        if news:
            for item in news:
                try:
                    translated = item.get('translated') or item.get('original', '')
                    url        = item.get('url', '')
                    if url:
                        st.markdown(f"- [{translated}]({url})")
                    else:
                        st.markdown(f"- {translated}")
                except Exception:
                    pass
        else:
            st.caption("죄송합니다. 데이터를 불러올 수 없습니다.")


# ══════════════════════════════════════════════════════════════════════════════════
# Tab 1 — 종목 상세 분석
# ══════════════════════════════════════════════════════════════════════════════════
def render_analysis_tab() -> None:
    st.markdown("#### 종목 코드 또는 이름 입력")
    st.caption("예: `AAPL` · `005930` · `SK하이닉스` · `파두` · `카카오` · `LG전자`")

    raw = st.text_input(
        "종목 입력",
        placeholder="AAPL / SK하이닉스 / 파두 / 005930 …",
        label_visibility="collapsed",
        key="analysis_input",
    )

    if not raw:
        st.info("📌 종목 코드(티커) 또는 한국 회사명을 입력하면 AI 분석 결과를 보여줍니다.")
        return

    # ── Korean name search: load KRX listing (show progress on first load) ──
    if has_korean(raw):
        listing_loaded = True
        try:
            get_krx_listing()
        except Exception:
            listing_loaded = False
        if not listing_loaded:
            st.error("KRX 종목 목록을 불러올 수 없습니다. 잠시 후 다시 시도하세요.")
            return

    code, display, suggestions, market = resolve_input(raw)

    # ── No match ─────────────────────────────────────────────────────────────
    if code is None:
        st.error(f"**'{raw}'** 에 해당하는 종목을 찾을 수 없습니다.")
        if suggestions:
            st.markdown("**혹시 이 종목을 찾으셨나요?**")
            for name in suggestions:
                st.markdown(f"- `{name}`")
        else:
            st.markdown("""
**입력 형식 안내**
- 🇺🇸 미국 주식: `AAPL`, `NVDA`, `MSFT`, `TSLA`
- 🇰🇷 한국 주식 (코드): `005930` 또는 `005930.KS`
- 🇰🇷 한국 주식 (이름): 회사 이름 그대로 입력 (예: `SK하이닉스`, `파두`, `카카오`)
""")
        return

    # ── Multiple partial matches ──────────────────────────────────────────────
    if suggestions:
        st.warning(f"**'{raw}'** 에 여러 종목이 해당됩니다. 정확한 이름을 입력하세요:")
        for name in suggestions:
            st.markdown(f"- `{name}`")
        return

    # ── Fetch data ────────────────────────────────────────────────────────────
    with st.spinner(f"{display} 데이터 불러오는 중…"):
        data = get_data(code, market)

    if data is None:
        st.error("죄송합니다. 현재 이 종목의 데이터를 불러올 수 없습니다.")
        st.caption(f"{display} ({code}) — Yahoo Finance / FinanceDataReader에서 지원하지 않는 종목일 수 있습니다.")
        return

    # ── Result header card ────────────────────────────────────────────────────
    score, grade  = calculate_ai_score(data)
    price         = data['price']
    change        = data['change_pct']
    badge_col     = GRADE_COLOR.get(grade, '#9E9E9E')
    change_str    = f"▲ +{change:.2f}%" if change >= 0 else f"▼ {change:.2f}%"
    chg_color     = "#00C851" if change >= 0 else "#FF5252"

    st.markdown(f"""
<div style="background:#1E1E2E;border:1px solid #2E2E4E;border-radius:14px;
            padding:20px 24px 16px;margin-bottom:8px;">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;">
    <div>
      <div style="font-size:1.4rem;font-weight:700;color:#E0E0FF">{display}</div>
      <div style="font-size:0.85rem;color:#8888AA;margin-top:2px">{code}</div>
    </div>
    <div style="text-align:right;">
      <span style="background:{badge_col};color:#000;font-weight:700;
                   padding:4px 16px;border-radius:16px;font-size:1.1rem">{grade}</span>
      <div style="font-size:0.8rem;color:#8888AA;margin-top:4px">AI Score {score}/100</div>
    </div>
  </div>
  <div style="margin-top:14px;">
    <span style="font-size:1.6rem;font-weight:700;color:#FFFFFF">{price:,.2f}</span>
    &nbsp;&nbsp;
    <span style="font-size:1rem;font-weight:600;color:{chg_color}">{change_str}</span>
  </div>
</div>""", unsafe_allow_html=True)

    st.write("")
    render_full_detail(display, data)


# ══════════════════════════════════════════════════════════════════════════════════
# Tab 2 — 이평선 스캐너
# ══════════════════════════════════════════════════════════════════════════════════
def render_scanner_tab() -> None:
    with st.sidebar:
        st.markdown("### ⚙️ 스캐너 설정")
        market     = st.radio("마켓", ["🇺🇸 미국", "🇰🇷 한국", "전체"], horizontal=True, key="mkt")
        max_spread = st.slider("최대 밀집도 (%)", 0.5, 10.0, 3.0, 0.5, key="spread_slider")
        st.divider()
        if st.button("🔄 캐시 새로고침", use_container_width=True, key="refresh_btn"):
            st.cache_data.clear()
            st.rerun()

    if "미국" in market:
        ticker_list = [(t, t, 'us') for t in US_TICKERS]
    elif "한국" in market:
        ticker_list = [(c, KR_NAMES.get(c, c), 'kr') for c in KR_TICKERS]
    else:
        ticker_list = ([(t, t, 'us') for t in US_TICKERS] +
                       [(c, KR_NAMES.get(c, c), 'kr') for c in KR_TICKERS])

    # Scan
    results: list[dict] = []
    bar = st.progress(0, text="종목 스캔 중…")
    for i, (code, name, mkт) in enumerate(ticker_list):
        bar.progress((i + 1) / len(ticker_list),
                     text=f"스캔 중: {name}  ({i+1}/{len(ticker_list)})")
        try:
            data = get_data(code, mkт)
            if data and data['spread'] <= max_spread:
                score, grade = calculate_ai_score(data)
                results.append(dict(code=code, name=name, score=score, grade=grade, **data))
        except Exception:
            continue
    bar.empty()

    # Summary
    m1, m2, m3 = st.columns(3)
    m1.metric("스캔", f"{len(ticker_list)}종목")
    m2.metric("필터 통과", f"{len(results)}종목")
    m3.metric("기준 밀집도", f"≤ {max_spread}%")
    st.divider()

    if not results:
        st.info("조건에 맞는 종목이 없습니다. 밀집도 슬라이더를 높여보세요.")
        return

    results.sort(key=lambda x: x['score'], reverse=True)

    COLS = 2
    rows = [results[i:i+COLS] for i in range(0, len(results), COLS)]

    for row_items in rows:
        cols = st.columns(COLS)
        for col, item in zip(cols, row_items):
            with col:
                try:
                    code       = item['code']
                    name       = item['name']
                    price      = item.get('price', 0)
                    change     = item.get('change_pct', 0)
                    spread     = item.get('spread', 0)
                    score      = item.get('score', 0)
                    grade      = item.get('grade', 'D')
                    rsi        = item.get('rsi', 0)
                    mfi        = item.get('mfi', 0)
                    news       = item.get('news', [])
                    badge_col  = GRADE_COLOR.get(grade, '#9E9E9E')
                    change_str = f"+{change:.2f}%" if change >= 0 else f"{change:.2f}%"
                    chg_color  = "#00C851" if change >= 0 else "#FF5252"

                    st.markdown(f"""
<div style="background:#1E1E2E;border:1px solid #2E2E4E;border-radius:10px;
            padding:14px 18px 10px;margin-bottom:4px;">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <span style="font-size:1.05rem;font-weight:700;color:#E0E0FF">{name}</span>
    <span style="background:{badge_col};color:#000;font-weight:700;
                 padding:2px 10px;border-radius:10px;font-size:0.82rem">{grade}</span>
  </div>
  <div style="font-size:0.78rem;color:#8888AA;margin:2px 0 8px">{code} &nbsp;·&nbsp; AI Score: {score}/100</div>
  <div style="display:flex;justify-content:space-between;align-items:baseline;">
    <span style="font-size:1.2rem;font-weight:700;color:#FFF">{price:,.2f}</span>
    <span style="font-size:0.9rem;font-weight:600;color:{chg_color}">{change_str}</span>
  </div>
  <div style="display:flex;gap:16px;margin-top:8px;">
    <span style="font-size:0.78rem;color:#8888AA">밀집도 <b style="color:#C8C8E8">{spread:.2f}%</b></span>
    <span style="font-size:0.78rem;color:#8888AA">RSI <b style="color:#C8C8E8">{rsi:.1f}</b></span>
    <span style="font-size:0.78rem;color:#8888AA">MFI <b style="color:#C8C8E8">{mfi:.1f}</b></span>
  </div>
</div>""", unsafe_allow_html=True)

                    with st.expander("▼ 상세 보기 (RSI · MFI · 뉴스)"):
                        # Technical indicators
                        d1, d2, d3 = st.columns(3)
                        d1.metric("RSI (14)", f"{rsi:.1f}")
                        d2.metric("MFI (14)", f"{mfi:.1f}")
                        d3.metric("밀집도",    f"{spread:.2f}%")
                        # News links
                        st.markdown("**📰 최신 관련 뉴스 (Korean)**")
                        if news:
                            for n_item in news:
                                try:
                                    translated = n_item.get('translated', n_item.get('original', ''))
                                    url        = n_item.get('url', '')
                                    if url:
                                        st.markdown(f"- [{translated}]({url})")
                                    else:
                                        st.markdown(f"- {translated}")
                                except Exception:
                                    pass
                        else:
                            st.caption("죄송합니다. 현재 이 종목의 데이터를 불러올 수 없습니다.")

                except Exception:
                    st.warning("죄송합니다. 현재 이 종목의 데이터를 불러올 수 없습니다.")


# ══════════════════════════════════════════════════════════════════════════════════
# App Entry
# ══════════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="이평선 스크리너", page_icon="📈", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #13131F; }
[data-testid="stSidebar"]          { background-color: #1A1A2E; }
h1, h2, h3, h4 { color: #E0E0FF !important; }
</style>""", unsafe_allow_html=True)

st.title("📈 이평선 스크리너")
st.caption("5 · 10 · 20일 이동평균 밀집도 스캔 + AI 종목 분석  |  미국(yfinance) · 한국(FinanceDataReader)")

tab_analysis, tab_scanner = st.tabs(["📊 종목 상세 분석", "🔍 이평선 스캐너"])

with tab_analysis:
    render_analysis_tab()

with tab_scanner:
    render_scanner_tab()

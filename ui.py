import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import time
import math
import html as _html
from concurrent.futures import ThreadPoolExecutor, as_completed

from data_provider import (
    get_krx_listing, has_korean, resolve_input,
    get_data, KR_NAMES, get_name_by_code,
    fetch_nasdaq_tickers, fetch_sp500_tickers, fetch_russell2000_tickers,
    fetch_kospi_tickers, fetch_kosdaq_tickers,
    get_kr_meta_dict, get_kr_yf_industry,
)
from indicators import add_indicators, calc_macd
from ai_engine import (
    calculate_ai_score, build_research_view,
    classify_strategies, STRATEGY_META,
    classify_strong_patterns, STRONG_PATTERN_META,
    action_view,
    GRADE_COLOR, STAGE_META, RATING_META, ACTION_META,
)
from themes import classify_themes, theme_display, THEME_LABELS
import news_sentiment as _sent
import datetime as _dt


def _fmt_news_date(published, now=None) -> str:
    """Format a tz-aware publish datetime as 'YYYY-MM-DD HH:MM (KST) · N전'."""
    if published is None:
        return ''
    if published.tzinfo is None:
        published = published.replace(tzinfo=_dt.timezone.utc)
    kst  = published.astimezone(_dt.timezone(_dt.timedelta(hours=9)))
    now  = now or _dt.datetime.now(_dt.timezone.utc)
    secs = (now - published).total_seconds()
    if secs < 3600:
        rel = f"{max(1, int(secs // 60))}분 전"
    elif secs < 86400:
        rel = f"{int(secs // 3600)}시간 전"
    else:
        rel = f"{int(secs // 86400)}일 전"
    return f"{kst:%Y-%m-%d %H:%M} · {rel}"


def _render_news_card(n_item: dict) -> None:
    """Render one news article as a sentiment-colored card (escaped HTML)."""
    title = n_item.get('translated') or n_item.get('original', '')
    if not title:
        return
    url        = str(n_item.get('url', '') or '')
    s_emoji, s_label, color = _sent.label_badge(n_item.get('sentiment', 'neutral'))
    date_str   = _fmt_news_date(n_item.get('published'))
    safe_title = _html.escape(str(title))
    safe_date  = _html.escape(date_str)
    safe_url   = _html.escape(url, quote=True) if url.startswith(('http://', 'https://')) else ''
    title_html = (
        f'<a href="{safe_url}" target="_blank" rel="noopener noreferrer" '
        f'style="color:#DCE3F0;text-decoration:none;">{safe_title}</a>'
        if safe_url else f'<span style="color:#DCE3F0;">{safe_title}</span>'
    )
    st.markdown(
        f'<div style="border-left:3px solid {color};background:#1B1B2E;'
        f'border-radius:6px;padding:8px 11px;margin:7px 0;">'
        f'<div style="font-size:0.74rem;font-weight:700;color:{color};">'
        f'{s_emoji} {s_label}</div>'
        f'<div style="font-size:0.9rem;line-height:1.35;margin:3px 0;">{title_html}</div>'
        f'<div style="font-size:0.7rem;color:#8A90A6;">📅 {safe_date}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ── 섹터 필터 키워드 매핑 ────────────────────────────────────────────────────────
# 각 섹터 라벨 → sector/industry 필드에서 검색할 한글 키워드 목록
_SECTOR_KEYWORDS: dict[str, list[str]] = {
    '반도체':   ['반도체'],
    '가전·전자': ['소비자 가전', '전기제품', '전기 제품', '가전'],
    '전자부품':  ['전자 부품', '전자부품', '전자장비', '계측·정밀기기', '계측'],
    '광통신':   ['광통신', '통신 장비'],
    'AI':      ['소프트웨어', 'IT 서비스', '인터넷·플랫폼', '인터넷', '클라우드', '데이터'],
    # 로보틱스는 업종 텍스트가 아니라 큐레이션된 티커 집합(_ROBOTICS_TICKERS)으로만
    # 매칭한다. 키워드 fallback이 없으므로 빈 리스트로 둔다(드롭다운 라벨은 노출).
    '로보틱스':  [],
    # 양자컴퓨팅도 동일하게 큐레이션된 티커 집합(_QUANTUM_TICKERS)으로만 매칭한다.
    '양자컴퓨팅': [],
    '우주항공':  ['항공우주·방위', '항공우주', '방위', '우주', '항공'],
    '전력':    ['전력', '유틸리티', '신재생에너지', '에너지 저장', '태양광', '에너지', '복합유틸리티'],
    '바이오':   ['바이오', '제약', '의료기기', '헬스케어', '헬스IT', '진단·연구', '생명과학', '헬스'],
    '자동차':   ['자동차'],
    '금융':    ['금융', '은행', '보험', '자산운용', '자본시장', '금융지주'],
    '리츠':    ['리츠', '부동산'],
    '소비재':   ['소비재', '소매', '백화점', '여행', '호텔', '의류', '명품', '전문소매', '레저'],
    '산업재':   ['산업재', '기계', '건설', '화학', '철강', '금속', '소재', '기초소재'],
    '식품':    ['식품', '가공식품', '음료', '외식', '농화학'],
    '통신':    ['통신 서비스', '방송', '미디어', '커뮤니케이션', '무선통신', '다각화된 통신'],
}

_SECTOR_LABELS = list(_SECTOR_KEYWORDS.keys())

# ── 로보틱스 섹터 분류 ───────────────────────────────────────────────────────────
# yfinance/FinanceDataReader 업종 분류에는 '로보틱스'가 없으므로(수술용 로봇=의료기기,
# 산업용 로봇=산업기계 등으로 흩어짐), 큐레이션된 티커 집합으로 직접 매칭한다.
_ROBOTICS_TICKERS: set[str] = {
    # 미국 — 휴머노이드·서비스 로봇
    'SERV',    # Serve Robotics
    'RR',      # Richtech Robotics
    'IRBT',    # iRobot
    # 미국 — 산업용 로봇
    'ABB',     # ABB Ltd
    'ROK',     # Rockwell Automation
    'TER',     # Teradyne (Universal Robots / MiR)
    'FANUY',   # Fanuc
    # 미국 — 물류·창고 자동화
    'SYM',     # Symbotic
    'ZBRA',    # Zebra Technologies
    # 미국 — 머신비전
    'CGNX',    # Cognex
    'AMBA',    # Ambarella (비전 SoC)
    'KYCCF',   # Keyence
    # 미국 — 로봇 소프트웨어
    'PATH',    # UiPath (RPA)
    # 미국 — 로봇 부품·모션
    'NOVT',    # Novanta
    'ALNT',    # Allient (모션 컴포넌트)
    # 미국 — 수술용 로봇
    'ISRG',    # Intuitive Surgical
    'STXS',    # Stereotaxis
    'PRCT',    # PROCEPT BioRobotics
    'GMED',    # Globus Medical
    'ASXC',    # Asensus Surgical
    # 미국 — 드론·자율 로봇
    'AVAV',    # AeroVironment
    'KTOS',    # Kratos Defense
    'OUST',    # Ouster (라이다)
    'LAZR',    # Luminar (라이다)
    'RCAT',    # Red Cat (드론)
    # 한국 — 로봇 (FinanceDataReader 6자리 코드)
    '277810',  # 레인보우로보틱스
    '454910',  # 두산로보틱스
    '090360',  # 로보스타
    '056080',  # 유진로봇
    '108490',  # 로보티즈
    '348340',  # 뉴로메카
    '117730',  # 티로보틱스
    '389500',  # 에스비비테크 (로봇 감속기·부품)
    '090710',  # 휴림로봇
}

# ── 양자컴퓨팅 섹터 분류 ─────────────────────────────────────────────────────────
# 양자컴퓨팅 역시 yfinance/FinanceDataReader의 단일 업종으로 분류되지 않으므로
# 큐레이션된 티커 집합으로 직접 매칭한다(로보틱스와 동일 방식).
_QUANTUM_TICKERS: set[str] = {
    'IONQ',    # IonQ
    'RGTI',    # Rigetti Computing
    'QBTS',    # D-Wave Quantum
    'QUBT',    # Quantum Computing Inc.
    'ARQQ',    # Arqit Quantum
    'LAES',    # SEALSQ (포스트 양자 보안)
    'QMCO',    # Quantum Corp.
}


def _match_sector_filter(item: dict, selected: list[str]) -> bool:
    """선택된 섹터 중 하나라도 매칭되면 True (OR 로직)."""
    if not selected:
        return True
    text = f"{item.get('sector', '')} {item.get('industry', '')}".lower()
    code = str(item.get('code', '')).upper()
    for label in selected:
        # 로보틱스는 큐레이션된 티커 집합으로만 매칭 (키워드 fallback 없음)
        if label == '로보틱스':
            if code in _ROBOTICS_TICKERS:
                return True
            continue
        # 양자컴퓨팅도 큐레이션된 티커 집합으로만 매칭 (키워드 fallback 없음)
        if label == '양자컴퓨팅':
            if code in _QUANTUM_TICKERS:
                return True
            continue
        for kw in _SECTOR_KEYWORDS.get(label, []):
            if kw.lower() in text:
                return True
    return False


def _match_theme_filter(item: dict, selected: list[str]) -> bool:
    """선택된 테마 중 하나라도 매칭되면 True (OR 로직)."""
    if not selected:
        return True
    themes = set(classify_themes(
        item.get('code', ''), item.get('industry', ''), item.get('sector', '')
    ))
    return any(t in themes for t in selected)


def _industry_theme_html(code: str, sector: str, industry: str,
                         is_kr: bool = False) -> str:
    """결과 카드용 '업종' + '테마' 라인 HTML 생성 (모든 동적 문자열 escape)."""
    industry_disp = industry or sector or ('업종 정보 없음' if is_kr else '')
    themes = classify_themes(code, industry, sector)

    ind_txt  = _html.escape(f"업종: {industry_disp}" if industry_disp else "업종: 정보 없음")
    out = (
        f'<div style="font-size:0.8rem;color:#9090B0;margin-top:3px">{ind_txt}</div>'
    )
    if themes:
        chips = ''.join(
            f'<span style="display:inline-block;background:#23233A;color:#9FA8DA;'
            f'border:1px solid #3A3A5C;border-radius:8px;padding:1px 8px;'
            f'margin:3px 4px 0 0;font-size:0.72rem;font-weight:600">'
            f'{_html.escape(theme_display(t))}</span>'
            for t in themes
        )
        out += (
            f'<div style="margin-top:4px;font-size:0.74rem;color:#7A7A9A">'
            f'테마 {chips}</div>'
        )
    return out


def _dedup_by_ticker(items: list[dict]) -> list[dict]:
    """티커(code) 기준 중복 제거.

    NASDAQ·S&P500·Russell2000 종목 리스트가 서로 겹쳐 같은 종목이 여러 번
    스캔될 수 있으므로, 최종 표시 전에 티커당 1개만 남긴다. 입력이 점수
    내림차순으로 정렬돼 있으면 첫 등장(=최고 점수)이 유지된다.
    """
    seen: set[str] = set()
    out: list[dict] = []
    for it in items:
        key = str(it.get('code', '')).upper()
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def _match_grade_filter(item: dict, selected: list[str]) -> bool:
    """선택된 AI 등급 중 하나라도 매칭되면 True."""
    if not selected:
        return True
    return item.get('grade', '') in selected


def _match_pattern_filter(item: dict, selected: list[str]) -> bool:
    """선택된 패턴 중 감지된 것이 하나라도 있으면 True (OR 로직)."""
    if not selected:
        return True
    _label_to_key = {v: k for k, v in _PATTERN_LABELS.items()}
    patterns = item.get('patterns', {})
    for label in selected:
        key = _label_to_key.get(label)
        if key and patterns.get(key, (False,))[0]:
            return True
    return False


# ── Chart ──────────────────────────────────────────────────────────────────────────
def make_chart(hist: pd.DataFrame, label: str, height: int = 480) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.04,
        subplot_titles=[label, 'MACD'],
    )

    fig.add_trace(go.Candlestick(
        x=hist.index,
        open=hist['Open'], high=hist['High'],
        low=hist['Low'],   close=hist['Close'],
        name='캔들',
        increasing_line_color='#EF5350',
        decreasing_line_color='#26A69A',
        showlegend=False,
    ), row=1, col=1)

    for col_name, color, lbl in [
        ('MA20', '#FFD700', 'MA20'),
        ('MA60', '#FF9F40', 'MA60'),
        ('MA120', '#9B59B6', 'MA120'),
    ]:
        if col_name in hist.columns:
            fig.add_trace(go.Scatter(
                x=hist.index, y=hist[col_name], name=lbl,
                line=dict(color=color, width=1.5), opacity=0.85,
            ), row=1, col=1)

    macd_data   = calc_macd(hist['Close'])
    macd_line   = macd_data['macd']
    signal_line = macd_data['signal']
    histogram   = macd_data['hist']
    bar_colors  = ['#EF5350' if v >= 0 else '#26A69A' for v in histogram]

    fig.add_trace(go.Bar(
        x=hist.index, y=histogram, name='히스토그램',
        marker_color=bar_colors, showlegend=False,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=hist.index, y=macd_line, name='MACD',
        line=dict(color='#5B8DEF', width=1.5),
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=hist.index, y=signal_line, name='시그널',
        line=dict(color='#FF6384', width=1.5),
    ), row=2, col=1)

    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=36, b=0),
        legend=dict(orientation='h', y=1.08, font=dict(size=11)),
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_rangeslider_visible=False,
    )
    fig.update_yaxes(gridcolor='rgba(100,100,150,0.15)')
    return fig


# ── Pattern badges ─────────────────────────────────────────────────────────────────
_PATTERN_LABELS = {
    # 눌림목 패턴 (저위험 진입 구간)
    'ma10_pullback':  'MA10 눌림목',
    'ma20_pullback':  'MA20 눌림목',
    # 단일 패턴
    'new_high':       '신고가 돌파',
    'volume_surge':   '거래량 급증',
    'golden_cross':   '골든크로스',
    'cup_and_handle': '컵앤핸들',
    'double_bottom':  '더블바텀',
    'box_breakout':   '박스권 돌파',
}

_CARD_PATTERN_ICONS: dict[str, tuple[str, str]] = {
    'ma10_pullback':  ('🔥', 'MA10 눌림'),
    'ma20_pullback':  ('🔥', 'MA20 눌림'),
    'new_high':       ('🚀', '신고가 돌파'),
    'volume_surge':   ('📈', '거래량 급증'),
    'golden_cross':   ('🏆', '골든크로스'),
    'cup_and_handle': ('☕', '컵앤핸들'),
    'double_bottom':  ('🔄', '더블바텀'),
    'box_breakout':   ('📦', '박스권 돌파'),
}


def _build_card_pattern_html(patterns: dict, max_count: int = 3) -> str:
    """감지된 패턴만 뱃지 HTML로 반환 (최대 max_count개)."""
    badges: list[str] = []
    for key, (icon, label) in _CARD_PATTERN_ICONS.items():
        if len(badges) >= max_count:
            break
        val = patterns.get(key)
        if val is None:
            continue
        detected = val[0] if isinstance(val, (tuple, list)) else bool(val)
        if detected:
            badges.append(
                f"<span style='display:inline-block;background:#1A2A1A;"
                f"border:1px solid #2E4A2E;border-radius:12px;"
                f"padding:2px 9px;font-size:0.72rem;color:#A8E6A8;"
                f"margin-right:4px;margin-bottom:4px;white-space:nowrap'>"
                f"{icon} {label}</span>"
            )
    if not badges:
        return ''
    inner = ''.join(badges)
    return (
        f"<div style='margin-top:8px;display:flex;flex-wrap:wrap;gap:0'>"
        f"{inner}</div>"
    )


def _render_patterns(patterns: dict) -> None:
    st.markdown("##### 📐 패턴 분석")
    cards: list[str] = []
    for key, label in _PATTERN_LABELS.items():
        detected, confidence, _ = patterns.get(key, (False, 0.0, ''))
        conf_pct = confidence * 100
        if detected and conf_pct >= 70:
            bg      = '#0F2A0F'
            border  = '#3A7A3A'
            lbl_col = '#AAFFAA'
            icon    = '💪'
            tier    = '강한감지'
            sub_col = '#88DD88'
        elif detected and conf_pct >= 50:
            bg      = '#182818'
            border  = '#2A5A2A'
            lbl_col = '#88EE88'
            icon    = '✅'
            tier    = '감지'
            sub_col = '#77BB77'
        elif detected and conf_pct >= 30:
            bg      = '#1E1E10'
            border  = '#4A4A20'
            lbl_col = '#CCCC66'
            icon    = '🟡'
            tier    = '약함'
            sub_col = '#AAAA55'
        else:
            bg      = '#1E1E2E'
            border  = '#2E2E4E'
            lbl_col = '#8888AA'
            icon    = '⚪'
            tier    = '미감지'
            sub_col = '#555577'
        sub = f'{conf_pct:.0f}% {tier}' if detected else '미감지'
        cards.append(
            f'<div style="display:inline-flex;flex-direction:column;align-items:center;'
            f'justify-content:center;min-width:110px;height:72px;padding:6px 10px;'
            f'background:{bg};border:1px solid {border};border-radius:8px;text-align:center">'
            f'<div style="white-space:nowrap;font-size:0.82rem;font-weight:700;color:{lbl_col}">'
            f'{label}</div>'
            f'<div style="font-size:1rem;line-height:1.6">{icon}</div>'
            f'<div style="white-space:nowrap;font-size:0.72rem;color:{sub_col}">{sub}</div>'
            f'</div>'
        )
    st.markdown(
        '<div style="overflow-x:auto;-webkit-overflow-scrolling:touch;padding-bottom:4px">'
        '<div style="display:flex;flex-wrap:nowrap;gap:8px;width:max-content">'
        + ''.join(cards)
        + '</div></div>',
        unsafe_allow_html=True,
    )


# ── Full Detail Block ──────────────────────────────────────────────────────────────
def _prob_color(score: int) -> str:
    return ('#FFD700' if score >= 80 else
            '#00C851' if score >= 70 else
            '#2196F3' if score >= 50 else
            '#FF9100' if score >= 30 else '#F44336')


def _build_ai_summary(label: str, data: dict, scored: dict, research: dict) -> str:
    """감지된 신호를 종합한 상세 AI 코멘트를 마크다운 문자열로 생성한다."""
    score = scored['score']
    grade = scored['grade']
    stage = research['stage']
    rating = research['rating']
    rsi = data.get('rsi', 0.0)
    mfi = data.get('mfi', 0.0)
    price = data.get('price', 0.0)
    hist = data.get('hist')
    patterns = scored.get('patterns', {})

    stage_ko, _, stage_desc = STAGE_META.get(stage, (stage, '', ''))
    rating_ko, _, rating_cmt = RATING_META.get(rating, (rating, '', ''))
    label = _html.escape(str(label))

    parts: list[str] = []
    parts.append(
        f"**{label}** 은(는) 현재 **{stage_ko}** 국면으로 판단되며, "
        f"AI 종합 점수는 **{score}/100 ({grade}등급)**, 투자의견은 **{rating_ko}** 입니다. "
        f"{stage_desc}."
    )

    # 이동평균 구조
    if hist is not None:
        def _ma(c):
            if c in hist.columns:
                s = hist[c].dropna()
                if not s.empty:
                    return float(s.iloc[-1])
            return None
        ma20, ma60, ma120 = _ma('MA20'), _ma('MA60'), _ma('MA120')
        if None not in (ma20, ma60, ma120):
            if ma20 > ma60 > ma120:
                parts.append("20·60·120일 이동평균이 **정배열**을 이루어 중장기 상승 구조가 유효합니다.")
            elif ma20 < ma60 < ma120:
                parts.append("이동평균이 **역배열**로 하락 추세가 우세해 추세 반전 확인 전까지 신규 진입은 부담스럽습니다.")
            else:
                parts.append("이동평균이 혼조 배열로 방향성이 아직 명확하지 않습니다 — 추세 확정 신호를 확인하는 것이 안전합니다.")

    # 모멘텀 (수치 노출 없이 질적 해석만 제공)
    rsi_txt = ("과매수 구간으로 단기 조정 가능성" if rsi >= 70 else
               "양호한 상승 모멘텀" if rsi >= 50 else
               "중립 모멘텀" if rsi >= 40 else
               "약한 모멘텀")
    mfi_txt = ("자금 유입 강함" if mfi >= 60 else
               "자금 유입 양호" if mfi >= 50 else
               "자금 흐름 중립" if mfi >= 40 else "자금 이탈 우위")
    parts.append(f"모멘텀은 {rsi_txt}, 자금 흐름은 {mfi_txt} 상태입니다.")

    # 패턴
    detected = [_PATTERN_LABELS[k] for k, v in patterns.items()
                if k in _PATTERN_LABELS and isinstance(v, (tuple, list)) and v and v[0]]
    if detected:
        parts.append("감지된 핵심 패턴: **" + ", ".join(detected) + "**.")
    else:
        parts.append("현재 뚜렷하게 감지된 매수 패턴은 없습니다.")

    # 매매 플랜
    el, eh = research['entry_low'], research['entry_high']
    t1 = research['targets'][0]
    parts.append(
        f"권장 진입 구간은 **{el:,.2f} ~ {eh:,.2f}**, 손절가는 **{research['stop_loss']:,.2f}**, "
        f"1차 목표가는 **{t1:,.2f}** 로 손익비는 약 **{research['risk_reward']:.1f} : 1** 입니다. {rating_cmt}."
    )

    a_ko, a_final, _ = action_view(research)
    parts.append(f"종합 액션은 **{a_ko} → {a_final}** 으로 판단됩니다.")

    return "\n\n".join(parts)


def render_full_detail(label: str, data: dict) -> None:
    try:
        data     = add_indicators(data)
        scored   = calculate_ai_score(data)
        research = build_research_view(data, scored)
    except Exception as e:
        st.error(f"분석 중 오류가 발생했습니다: {e}")
        return

    score     = scored['score']
    grade     = scored['grade']
    patterns  = scored.get('patterns', {})

    price      = data.get('price', 0)
    change     = data.get('change_pct', 0)
    market     = data.get('market', 'us')
    hist       = data.get('hist')
    news       = data.get('news', [])

    rating  = research['rating']
    stage   = research['stage']
    action  = research.get('action', 'WATCH')
    el, eh  = research['entry_low'], research['entry_high']
    stop    = research['stop_loss']
    t1, t2, t3 = research['targets']
    rr      = research['risk_reward']

    r_ko, r_col, r_cmt = RATING_META.get(rating, (rating, '#9E9E9E', ''))
    s_ko, s_col, s_desc = STAGE_META.get(stage, (stage, '#9E9E9E', ''))
    a_ko, a_final, a_col = action_view(research)
    p_col   = _prob_color(score)

    def pct(v):
        return (v - price) / price * 100 if price else 0.0

    # ── 1) 액션 / 투자의견 / 국면 / AI 상승확률 ───────────────────────────────
    st.markdown(f"""
<div style="display:flex;flex-wrap:wrap;gap:12px;margin-bottom:14px">
  <div style="flex:1 1 180px;background:#16161F;border:1px solid {a_col}55;
              border-left:4px solid {a_col};border-radius:12px;padding:14px 18px">
    <div style="font-size:0.72rem;letter-spacing:.08em;color:#8888AA;text-transform:uppercase">액션 · Action</div>
    <div style="font-size:1.7rem;font-weight:800;color:{a_col};margin-top:2px">{a_ko}</div>
    <div style="font-size:0.85rem;font-weight:700;color:{a_col};margin-top:2px">최종판단: {a_final}</div>
  </div>
  <div style="flex:1 1 180px;background:#16161F;border:1px solid {r_col}55;
              border-left:4px solid {r_col};border-radius:12px;padding:14px 18px">
    <div style="font-size:0.72rem;letter-spacing:.08em;color:#8888AA;text-transform:uppercase">투자의견 · Rating</div>
    <div style="font-size:1.7rem;font-weight:800;color:{r_col};margin-top:2px">{r_ko}
      <span style="font-size:0.9rem;font-weight:600;color:{r_col}AA">{rating}</span></div>
    <div style="font-size:0.78rem;color:#9999BB;margin-top:2px">{r_cmt}</div>
  </div>
  <div style="flex:1 1 180px;background:#16161F;border:1px solid {s_col}55;
              border-left:4px solid {s_col};border-radius:12px;padding:14px 18px">
    <div style="font-size:0.72rem;letter-spacing:.08em;color:#8888AA;text-transform:uppercase">현재 국면 · Stage</div>
    <div style="font-size:1.7rem;font-weight:800;color:{s_col};margin-top:2px">{s_ko}</div>
    <div style="font-size:0.78rem;color:#9999BB;margin-top:2px">{s_desc}</div>
  </div>
  <div style="flex:1 1 180px;background:#16161F;border:1px solid {p_col}55;
              border-left:4px solid {p_col};border-radius:12px;padding:14px 18px">
    <div style="font-size:0.72rem;letter-spacing:.08em;color:#8888AA;text-transform:uppercase">AI 상승확률 · Probability</div>
    <div style="font-size:1.7rem;font-weight:800;color:{p_col};margin-top:2px">{score}<span style="font-size:1rem;color:#9999BB">/100 · {grade}</span></div>
    <div style="background:#2A2A3E;border-radius:6px;height:7px;margin-top:8px;overflow:hidden">
      <div style="width:{score}%;height:100%;background:{p_col}"></div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    # ── 강세추세 세부패턴 (현재국면 카드 아래) — STRONG 종목에만 표시 ──────────
    _detail_item = {**data, **scored}
    try:
        _detail_tags = classify_strategies(_detail_item, research)
    except Exception:
        _detail_tags = []
    _detail_spats = (
        classify_strong_patterns(_detail_item) if 'STRONG' in _detail_tags else []
    )
    if _detail_spats:
        _spat_chips = ''.join(
            f'<span style="display:inline-block;background:{STRONG_PATTERN_META[k][1]}22;'
            f'color:{STRONG_PATTERN_META[k][1]};border:1px solid {STRONG_PATTERN_META[k][1]}66;'
            f'border-radius:9px;padding:4px 13px;margin:0 8px 6px 0;font-size:0.92rem;'
            f'font-weight:700">{_html.escape(STRONG_PATTERN_META[k][0])}</span>'
            for k in _detail_spats if k in STRONG_PATTERN_META
        )
        st.markdown(
            '<div style="background:#16161F;border:1px solid #2E2E4E;border-radius:12px;'
            'padding:12px 18px;margin-bottom:14px">'
            '<div style="font-size:0.72rem;letter-spacing:.08em;color:#8888AA;'
            'text-transform:uppercase;margin-bottom:8px">세부패턴 · Strong Patterns</div>'
            f'<div>{_spat_chips}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown(_debug_block_html(data, research), unsafe_allow_html=True)

    # ── 2) 매매 플랜 (진입 / 손절 / 목표 / 손익비) ────────────────────────────
    def plan_card(title, value, sub, color):
        return (
            f'<div style="flex:1 1 150px;background:#1A1A28;border:1px solid #2E2E4E;'
            f'border-radius:10px;padding:12px 14px">'
            f'<div style="font-size:0.72rem;color:#8888AA">{title}</div>'
            f'<div style="font-size:1.15rem;font-weight:700;color:{color};margin-top:3px">{value}</div>'
            f'<div style="font-size:0.72rem;color:#777799;margin-top:1px">{sub}</div></div>'
        )

    st.markdown("##### 🎯 매매 플랜")
    st.markdown(
        '<div style="display:flex;flex-wrap:wrap;gap:10px;margin-bottom:6px">'
        + plan_card("진입 구간 · Entry", f"{el:,.2f} ~ {eh:,.2f}", "눌림목 분할 매수 밴드", "#2196F3")
        + plan_card("손절가 · Stop", f"{stop:,.2f}", f"{pct(stop):+.1f}% · 리스크 방어", "#FF5252")
        + plan_card("손익비 · R/R", f"{rr:.1f} : 1", "1차 목표 기준", "#B388FF")
        + '</div>'
        + '<div style="display:flex;flex-wrap:wrap;gap:10px">'
        + plan_card("목표가 ① T1", f"{t1:,.2f}", f"{pct(t1):+.1f}% · 단기", "#00C851")
        + plan_card("목표가 ② T2", f"{t2:,.2f}", f"{pct(t2):+.1f}% · 중기", "#00C851")
        + plan_card("목표가 ③ T3", f"{t3:,.2f}", f"{pct(t3):+.1f}% · 장기", "#00C851")
        + '</div>',
        unsafe_allow_html=True,
    )

    # ── 3) AI 상세 코멘트 ──────────────────────────────────────────────────────
    try:
        st.markdown("##### 🧠 AI 상세 분석")
        summary = _build_ai_summary(label, data, scored, research)
        st.markdown(
            f'<div style="background:#15151F;border:1px solid #2E2E4E;border-radius:12px;'
            f'padding:16px 20px;line-height:1.7;color:#D5D5EE;font-size:0.92rem">{summary}</div>',
            unsafe_allow_html=True,
        )
    except Exception:
        pass

    # ── 4) 패턴 분석 ───────────────────────────────────────────────────────────
    try:
        st.write("")
        _render_patterns(patterns)
    except Exception:
        pass

    # ── 5) 차트 (접이식) ───────────────────────────────────────────────────────
    if hist is not None:
        price_str  = f"{price:,.0f}원" if market == 'kr' else f"${price:,.2f}"
        change_str = f"+{change:.1f}%" if change >= 0 else f"{change:.1f}%"
        with st.expander(f"📈 차트 보기 ({price_str} / {change_str})", expanded=False):
            try:
                st.plotly_chart(make_chart(hist, label, height=360), width='stretch')
            except Exception:
                st.caption("차트를 표시할 수 없습니다.")

    # ── 6) 뉴스 ────────────────────────────────────────────────────────────────
    with st.expander("📰 최신 뉴스 (최근 3일)", expanded=False):
        if news:
            o_score          = _sent.overall_score(news)
            o_emoji, o_label = _sent.overall_label(o_score)
            o_color          = _sent.overall_color(o_score)
            st.markdown(
                f'<div style="background:{o_color}22;border:1px solid {o_color}66;'
                f'border-radius:8px;padding:9px 12px;margin-bottom:6px;'
                f'display:flex;align-items:center;justify-content:space-between;gap:8px;">'
                f'<span style="font-size:0.82rem;color:#B8C0D6;">종합 뉴스 감성</span>'
                f'<span style="font-size:0.95rem;font-weight:700;color:{o_color};">'
                f'{o_emoji} {o_label} ({o_score}/100)</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            for n_item in news:
                try:
                    _render_news_card(n_item)
                except Exception:
                    pass
        else:
            st.caption("최근 3일 내 관련 뉴스 없음")


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

    if has_korean(raw):
        try:
            get_krx_listing()
        except Exception:
            st.error("KRX 종목 목록을 불러올 수 없습니다. 잠시 후 다시 시도하세요.")
            return

    code, display, suggestions, market = resolve_input(raw)

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

    if suggestions:
        st.warning(f"**'{raw}'** 에 여러 종목이 해당됩니다. 정확한 이름을 입력하세요:")
        for name in suggestions:
            st.markdown(f"- `{name}`")
        return

    with st.spinner(f"{display} 데이터 불러오는 중…"):
        data = get_data(code, market)

    if data is None:
        st.error("죄송합니다. 현재 이 종목의 데이터를 불러올 수 없습니다.")
        st.caption(f"{display} ({code}) — Yahoo Finance / FinanceDataReader에서 지원하지 않는 종목일 수 있습니다.")
        return

    data   = add_indicators(data)
    data['market'] = market
    scored = calculate_ai_score(data)
    score  = scored['score']
    grade  = scored['grade']
    price  = data.get('price', 0.0)
    change = data.get('change_pct', 0.0)

    badge_col  = GRADE_COLOR.get(grade, '#9E9E9E')
    change_str = f"▲ +{change:.2f}%" if change >= 0 else f"▼ {change:.2f}%"
    chg_color  = "#00C851" if change >= 0 else "#FF5252"

    company_name = data.get('company_name') or display
    sector       = data.get('sector', '')
    industry     = data.get('industry', '')
    # KR stocks: supplement empty industry with Yahoo Finance data (cached 3 days)
    if market == 'kr' and not industry:
        _kr_market_en = get_kr_meta_dict().get(code, {}).get('market', '')
        industry      = get_kr_yf_industry(code, _kr_market_en)
    industry_theme_block = _industry_theme_html(code, sector, industry, market == 'kr')

    company_name = _html.escape(str(company_name))
    code_disp    = _html.escape(str(code))

    st.markdown(f"""
<div style="background:#1E1E2E;border:1px solid #2E2E4E;border-radius:14px;
            padding:20px 24px 16px;margin-bottom:8px;">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;">
    <div>
      <div style="font-size:1.4rem;font-weight:700;color:#E0E0FF">
        {company_name} <span style="color:#8888AA;font-weight:400;font-size:1.1rem">({code_disp})</span>
      </div>
      {industry_theme_block}
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
# Scanner helpers
# ══════════════════════════════════════════════════════════════════════════════════

# KR ETF 브랜드 접두사 (FDR StockListing 기준 영문 브랜드명 사용)
_KR_ETF_PREFIXES: tuple[str, ...] = (
    'KODEX ', 'TIGER ', 'KBSTAR ', 'ARIRANG ', 'KINDEX ', 'HANARO ', 'KOSEF ',
    'ACE ', 'SOL ', 'TIMEFOLIO ', 'TREX ', 'MASTER ', 'FOCUS ', 'PLUS ',
    'WOORI ', 'SMART ETF', 'HANWHA ', 'HEUNGKUK ',
)

# US SPAC 회사명 키워드 (소문자 매칭)
_US_SPAC_NAME_KW: tuple[str, ...] = (
    'acquisition corp', 'acquisition co.', 'acquisition inc',
    'blank check', 'spac ', 'merger corp', 'merger co.',
)
# Yahoo Finance industry 값 기준 SPAC 판별
_US_SPAC_INDUSTRIES: frozenset[str] = frozenset({
    'Shell Companies', 'Blank Check Companies',
})

# US ETF 이름 기반 감지 — quote_type 없는 구캐시 데이터용 폴백
# 회사명에 ' etf' 포함 여부로 우선 감지, 아래는 펀드 운용사 접두사
_US_ETF_PROVIDER_PFX: tuple[str, ...] = (
    'ishares ',       # BlackRock
    'vanguard ',      # Vanguard
    'spdr ',          # State Street
    'invesco ',       # Invesco
    'proshares ',     # ProShares
    'direxion ',      # Direxion
    'xtrackers ',     # DWS
    'wisdomtree ',    # WisdomTree
    'vaneck ',        # VanEck
    'pimco ',         # PIMCO
    'first trust ',   # First Trust
    'global x ',      # Global X
    'schwab ',        # Schwab
    'fidelity msci ', # Fidelity MSCI ETFs
    'pacer ',         # Pacer
    'amplify ',       # Amplify
    'roundhill ',     # Roundhill
    'defiance ',      # Defiance
    'innovator ',     # Innovator
)


def _is_kr_excluded(name: str) -> bool:
    """KR SPAC(스팩) 및 ETF 종목 여부."""
    n = name.strip()
    if '스팩' in n:
        return True
    nu = n.upper()
    if 'ETF' in nu:
        return True
    return any(nu.startswith(p.upper()) for p in _KR_ETF_PREFIXES)


def _is_etf_or_spac(data: dict) -> tuple[bool, str]:
    """US ETF/SPAC 여부를 (is_excluded, reason) 으로 반환.

    감지 우선순위:
    1. quote_type 필드 (신규 캐시 데이터)
    2. 회사명에 ' etf' 포함 (구캐시 폴백 — 가장 범용)
    3. 알려진 ETF 운용사 접두사 (구캐시 폴백)
    4. industry_en / industry 로 SPAC 감지
    5. SPAC 회사명 키워드
    """
    qt = data.get('quote_type', 'EQUITY')
    if qt in ('ETF', 'MUTUALFUND', 'INDEX', 'CURRENCY'):
        return True, 'ETF'

    cname = (data.get('company_name') or '').lower()

    # ── ETF 이름 기반 감지 (구캐시 폴백) ──────────────────────────────────
    if ' etf' in cname:                              # "... ETF", "... ETF Trust" 등
        return True, 'ETF'
    if any(cname.startswith(p) for p in _US_ETF_PROVIDER_PFX):
        return True, 'ETF'

    # ── SPAC 감지 ─────────────────────────────────────────────────────────
    ind_en = data.get('industry_en', '')
    if ind_en in _US_SPAC_INDUSTRIES:
        return True, 'SPAC'
    if '스팩' in data.get('industry', ''):
        return True, 'SPAC'
    if any(kw in cname for kw in _US_SPAC_NAME_KW):
        return True, 'SPAC'

    return False, ''


def _stage1_pass(data: dict | None) -> bool:
    """Stage 1: MA alignment (MA20>MA60>MA120), non-zero volume. 밀집도 필터 제거."""
    if data is None:
        return False
    hist = data.get('hist')
    if hist is None or len(hist) < 60:
        return False
    try:
        ma20  = float(hist['MA20'].iloc[-1])
        ma60  = float(hist['MA60'].iloc[-1])
        ma120 = float(hist['MA120'].iloc[-1])
        if pd.isna(ma20) or pd.isna(ma60) or pd.isna(ma120):
            return False
        if not (ma20 > ma60 > ma120):
            return False
    except Exception:
        return False
    if 'Volume' in hist.columns:
        try:
            avg_vol = float(hist['Volume'].rolling(20).mean().iloc[-1])
            if avg_vol <= 0:
                return False
        except Exception:
            pass
    return True


def _calc_momentum(hist: pd.DataFrame) -> float:
    """Momentum score 0-100: price above MA20, 5-day & 20-day returns."""
    try:
        close = hist['Close']
        price = float(close.iloc[-1])
        ma20  = float(close.rolling(20).mean().iloc[-1])
        score = 30.0 if price > ma20 else 0.0
        if len(close) >= 6:
            r5 = (price / float(close.iloc[-6]) - 1.0) * 100
            score += min(r5 / 5.0, 1.0) * 20 if r5 > 0 else 0
        if len(close) >= 21:
            r20 = (price / float(close.iloc[-21]) - 1.0) * 100
            score += min(r20 / 10.0, 1.0) * 50 if r20 > 0 else 0
        return min(float(score), 100.0)
    except Exception:
        return 0.0


# ══════════════════════════════════════════════════════════════════════════════════
# 디버그 — 액션(최종판단) 결정 사유 / 제외 사유
# ══════════════════════════════════════════════════════════════════════════════════
def _ma_last(item: dict, col: str):
    """item['hist'] 의 마지막 유효 이동평균 값을 반환한다."""
    hist = item.get('hist')
    try:
        if hist is not None and col in hist.columns:
            s = hist[col].dropna()
            if not s.empty:
                return float(s.iloc[-1])
    except Exception:
        pass
    return None


def _action_reason(research: dict) -> tuple[str, str]:
    """(action 결정 사유, 제외 사유) 를 반환한다.

    최종판단(액션)은 오직 손익비(R/R)로만 결정된다(`ai_engine._decide_action`).
    따라서 카드에 액션='제외'가 찍히는 경우는 단 하나 — R/R < 1.0 (RR_FILTER) 뿐이다.
    (ETF/SPAC/거래대금/연구뷰실패 컷은 스캐너 단계에서 결과에서 아예 제거되므로
     카드로 렌더링되지 않는다.)
    """
    rr = research.get('risk_reward', 0.0) or 0.0
    if not math.isfinite(rr):   # NaN/inf 방어 — 잘못된 입력이 매수로 둔갑하지 않게
        rr = 0.0
    action = research.get('action', 'WATCH')
    if action == 'EXCLUDE' or rr < 1.0:
        return f'R/R {rr:.2f} < 1.0 → 제외', 'RR_FILTER'
    if rr < 2.0:
        return f'1.0 ≤ R/R {rr:.2f} < 2.0 → 관망', 'NONE'
    if rr < 3.0:
        return f'2.0 ≤ R/R {rr:.2f} < 3.0 → 선매집', 'NONE'
    if rr < 5.0:
        return f'3.0 ≤ R/R {rr:.2f} < 5.0 → 매수', 'NONE'
    return f'R/R {rr:.2f} ≥ 5.0 → 적극매수', 'NONE'


def _debug_block_html(item: dict, research: dict) -> str:
    """액션 결정 디버그 정보를 카드용 HTML 블록으로 반환한다."""
    price = item.get('price', 0) or 0
    ma300 = _ma_last(item, 'MA300')
    dev = item.get('ma300_dev_pct')
    if dev is None and ma300 and price:
        dev = (price - ma300) / ma300 * 100
    rr = research.get('risk_reward', 0.0) or 0.0
    reason, excl = _action_reason(research)
    stop  = research.get('stop_loss', item.get('stop_loss'))
    basis = item.get('stop_basis') or research.get('stop_basis') or '—'
    _basis_ko = {
        'swing_low': '최근 스윙로우', 'ma20': 'MA20',
        'atr2': 'ATR(14)×2', 'fallback': '폴백(-8%)',
    }.get(basis, basis)

    price_str = f'{price:,.2f}' if isinstance(price, (int, float)) else '—'
    ma300_str = f'{ma300:,.2f}' if isinstance(ma300, (int, float)) else '—'
    dev_str   = f'{dev:+.2f}%' if isinstance(dev, (int, float)) else '—'
    stop_str  = f'{stop:,.2f}' if isinstance(stop, (int, float)) else '—'
    rr_str    = f'{rr:.2f} : 1'
    reason_e  = _html.escape(reason)
    excl_e    = _html.escape(excl)
    basis_e   = _html.escape(str(_basis_ko))
    excl_col  = '#FF8A80' if excl != 'NONE' else '#69F0AE'

    def _drow(label, value, vcol='#C8C8E8'):
        return (
            f'<div style="display:flex;justify-content:space-between;gap:10px;padding:2px 0">'
            f'<span style="color:#7A7A95;font-size:0.7rem">{label}</span>'
            f'<span style="color:{vcol};font-size:0.72rem;font-weight:600;'
            f'text-align:right">{value}</span></div>'
        )

    return (
        '<div style="margin-top:8px;padding:8px 10px;background:#15151F;'
        'border:1px dashed #3A3A55;border-radius:8px">'
        '<div style="color:#7A7A95;font-size:0.68rem;letter-spacing:.04em;'
        'margin-bottom:4px">🔍 디버그 · 액션 결정</div>'
        + _drow('현재가', price_str)
        + _drow('MA300', ma300_str)
        + _drow('MA300 이격도', dev_str)
        + _drow('손절가', stop_str, '#FF8A80')
        + _drow('손절 근거', basis_e)
        + _drow('손익비(R/R)', rr_str, '#B388FF')
        + _drow('action 결정 사유', reason_e)
        + _drow('제외 사유', excl_e, excl_col)
        + '</div>'
    )


# ══════════════════════════════════════════════════════════════════════════════════
# Tab 2 — 이평선 스캐너
# ══════════════════════════════════════════════════════════════════════════════════
def _render_scanner_card(item: dict, research: dict | None, tags: list[str]) -> None:
    """스캐너 결과 카드 한 개를 렌더링한다.

    필드 순서: 전략 → 패턴 → 업종 → 테마 → 액션 → 확률점수 →
    매수구간 → 손절가 → 목표가 → 손익비 → 최종판단.
    """
    code   = item.get('code', '')
    name   = item.get('name', '')
    price  = item.get('price', 0) or 0
    change = item.get('change_pct', 0) or 0
    spread = item.get('spread', 0) or 0
    score  = int(item.get('score', 0) or 0)
    grade  = item.get('grade', 'D')
    rsi    = item.get('rsi', 0) or 0
    mfi    = item.get('mfi', 0) or 0
    news   = item.get('news', [])
    patterns = item.get('patterns', {})
    tv_raw   = item.get('trading_value', 0.0) or 0.0
    is_kr    = item.get('market', '') == 'kr'
    s_sector   = item.get('sector', '')
    s_industry = item.get('industry', '')

    if research is None:
        try:
            research = build_research_view(item, item)
        except Exception:
            research = {}

    tv_str = (f"₩{tv_raw/1e8:.0f}억" if is_kr else f"${tv_raw/1e6:.1f}M") if tv_raw > 0 else "—"
    badge_col  = GRADE_COLOR.get(grade, '#9E9E9E')
    change_str = f"+{change:.2f}%" if change >= 0 else f"{change:.2f}%"
    chg_color  = "#00C851" if change >= 0 else "#FF5252"
    name_disp  = _html.escape(str(name))
    code_disp  = _html.escape(str(code))

    # 전략 칩 (바닥탈출은 형성/돌파 국면 라벨을 함께 표기)
    _bs = patterns.get('bottom_setup') if isinstance(patterns, dict) else None
    _bs = _bs if isinstance(_bs, dict) else {}
    _phase_ko = {'FORMING': '형성 중', 'BREAKOUT': '돌파 직후'}

    def _strat_label(t: str) -> str:
        base = STRATEGY_META[t][0]
        if t == 'BOTTOM_OUT' and _bs.get('phase') in _phase_ko:
            return f"{base} · {_phase_ko[_bs['phase']]}"
        return base

    if tags:
        strat_html = ''.join(
            f'<span style="display:inline-block;background:{STRATEGY_META[t][1]}22;'
            f'color:{STRATEGY_META[t][1]};border:1px solid {STRATEGY_META[t][1]}66;'
            f'border-radius:8px;padding:1px 8px;margin:0 4px 3px 0;font-size:0.72rem;'
            f'font-weight:700">{_html.escape(_strat_label(t))}</span>'
            for t in tags if t in STRATEGY_META
        )
    else:
        strat_html = '<span style="color:#666">—</span>'

    # 강세추세 세부패턴 칩 (터틀돌파/신고가근처/첫눌림목) — 강세추세 종목에만 표시
    strong_pats = classify_strong_patterns(item) if 'STRONG' in tags else []
    if strong_pats:
        strong_html = ''.join(
            f'<span style="display:inline-block;background:{STRONG_PATTERN_META[k][1]}22;'
            f'color:{STRONG_PATTERN_META[k][1]};border:1px solid {STRONG_PATTERN_META[k][1]}66;'
            f'border-radius:8px;padding:1px 8px;margin:0 4px 3px 0;font-size:0.72rem;'
            f'font-weight:700">{_html.escape(STRONG_PATTERN_META[k][0])}</span>'
            for k in strong_pats if k in STRONG_PATTERN_META
        )
    else:
        strong_html = ''

    # 패턴
    det_labels = [
        _PATTERN_LABELS[k] for k, v in patterns.items()
        if k in _PATTERN_LABELS and isinstance(v, (tuple, list)) and v and v[0]
    ]
    pat_html = _html.escape(', '.join(det_labels)) if det_labels else '<span style="color:#666">—</span>'

    # 업종
    industry_disp = _html.escape(s_industry or s_sector or ('업종 정보 없음' if is_kr else '정보 없음'))

    # 테마 (다중)
    themes = classify_themes(code, s_industry, s_sector)
    if themes:
        theme_html = ''.join(
            f'<span style="display:inline-block;background:#23233A;color:#9FA8DA;'
            f'border:1px solid #3A3A5C;border-radius:8px;padding:1px 7px;'
            f'margin:0 4px 3px 0;font-size:0.7rem;font-weight:600">'
            f'{_html.escape(theme_display(t))}</span>'
            for t in themes
        )
    else:
        theme_html = '<span style="color:#666">—</span>'

    # 액션 / 최종판단
    action = research.get('action', 'WATCH')
    a_ko, a_final, a_col = action_view(research)

    # 매매 플랜
    el = research.get('entry_low'); eh = research.get('entry_high')
    stop = research.get('stop_loss')
    targets = research.get('targets') or []
    t1 = targets[0] if len(targets) >= 1 else None
    rr = research.get('risk_reward', 0.0) or 0.0
    p_col = _prob_color(score)

    def _f(v):
        return f"{v:,.2f}" if isinstance(v, (int, float)) else "—"

    entry_str = f"{_f(el)} ~ {_f(eh)}" if el is not None and eh is not None else "—"

    # MA300 이격도(%) / MA300 전략 여부(YES/NO)
    dev = item.get('ma300_dev_pct')
    dev_str  = f"{dev:+.2f}%" if isinstance(dev, (int, float)) else "—"
    ma300_yn = "YES" if item.get('ma300_strategy') else "NO"
    yn_col   = "#69F0AE" if item.get('ma300_strategy') else "#FF8A80"

    def _row(label, value_html, vcolor="#D5D5EE"):
        return (
            f'<div style="display:flex;justify-content:space-between;gap:10px;'
            f'padding:4px 0;border-bottom:1px solid #23233A">'
            f'<span style="color:#8888AA;font-size:0.74rem;white-space:nowrap">{label}</span>'
            f'<span style="color:{vcolor};font-size:0.8rem;font-weight:600;'
            f'text-align:right">{value_html}</span></div>'
        )

    st.markdown(f"""
<div style="background:#1E1E2E;border:1px solid #2E2E4E;border-radius:10px;
            padding:14px 16px 10px;margin-bottom:4px;">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <span style="font-size:1.02rem;font-weight:700;color:#E0E0FF">
      {name_disp} <span style="color:#8888AA;font-weight:400;font-size:0.85rem">({code_disp})</span>
    </span>
    <span style="background:{badge_col};color:#000;font-weight:700;
                 padding:2px 10px;border-radius:10px;font-size:0.8rem">{grade}</span>
  </div>
  <div style="display:flex;justify-content:space-between;align-items:baseline;margin:6px 0 8px">
    <span style="font-size:1.18rem;font-weight:700;color:#FFF">{price:,.2f}</span>
    <span style="font-size:0.88rem;font-weight:600;color:{chg_color}">{change_str}</span>
  </div>
  {_row("전략", strat_html)}
  {_row("강세패턴", strong_html) if strong_html else ""}
  {_row("패턴", pat_html)}
  {_row("업종", industry_disp)}
  {_row("테마", theme_html)}
  {_row("MA300 이격도", dev_str, "#FFD54F")}
  {_row("MA300 전략", ma300_yn, yn_col)}
  {_row("액션", a_ko, a_col)}
  {_row("확률점수", f'{score}/100', p_col)}
  {_row("매수구간", entry_str, "#64B5F6")}
  {_row("손절가", _f(stop), "#FF8A80")}
  {_row("목표가", _f(t1), "#69F0AE")}
  {_row("손익비(R/R)", f'{rr:.1f} : 1', "#B388FF")}
  <div style="display:flex;justify-content:space-between;gap:10px;padding:6px 0 2px">
    <span style="color:#8888AA;font-size:0.74rem">최종판단</span>
    <span style="color:{a_col};font-size:0.9rem;font-weight:800">{a_final}</span>
  </div>
  <div style="display:flex;gap:14px;margin-top:8px;flex-wrap:wrap;">
    <span style="font-size:0.72rem;color:#8888AA">밀집도 <b style="color:#C8C8E8">{spread:.2f}%</b></span>
    <span style="font-size:0.72rem;color:#8888AA">RSI <b style="color:#C8C8E8">{rsi:.1f}</b></span>
    <span style="font-size:0.72rem;color:#8888AA">MFI <b style="color:#C8C8E8">{mfi:.1f}</b></span>
    <span style="font-size:0.72rem;color:#8888AA">거래대금 <b style="color:#C8C8E8">{tv_str}</b></span>
  </div>
  {_debug_block_html(item, research)}
</div>""", unsafe_allow_html=True)

    with st.expander("▼ 상세 보기 (지표 · 패턴 · 뉴스)"):
        d1, d2, d3 = st.columns(3)
        d1.metric("RSI (14)", f"{rsi:.1f}")
        d2.metric("MFI (14)", f"{mfi:.1f}")
        d3.metric("밀집도",    f"{spread:.2f}%")
        _render_patterns(patterns)
        with st.expander("🐛 패턴 원시 데이터 (디버그)"):
            debug_lines = []
            for k, v in patterns.items():
                det, conf, desc = v if isinstance(v, tuple) and len(v) == 3 else (v, 0.0, '')
                icon = "✅" if det else "⚪"
                debug_lines.append(f"{icon} {k}: detected={det}, conf={conf:.2f}, desc={desc!r}")
            st.code("\n".join(debug_lines), language=None)
        st.markdown("**📰 최신 뉴스 (최근 3일)**")
        if news:
            o_score          = _sent.overall_score(news)
            o_emoji, o_label = _sent.overall_label(o_score)
            o_color          = _sent.overall_color(o_score)
            st.markdown(
                f'<div style="font-size:0.82rem;font-weight:700;'
                f'color:{o_color};margin-bottom:4px;">'
                f'{o_emoji} 종합 감성 {o_label} ({o_score}/100)</div>',
                unsafe_allow_html=True,
            )
            for n_item in news:
                try:
                    _render_news_card(n_item)
                except Exception:
                    pass
        else:
            st.caption("최근 3일 내 관련 뉴스 없음")


def _render_results_grid(enriched: list[tuple]) -> None:
    """(item, research, tags) 튜플 리스트를 2열 그리드로 렌더링한다."""
    if not enriched:
        st.caption("이 전략에 해당하는 종목이 없습니다.")
        return
    COLS = 2
    rows = [enriched[i:i+COLS] for i in range(0, len(enriched), COLS)]
    for row_items in rows:
        cols = st.columns(COLS)
        for col, (item, research, tags) in zip(cols, row_items):
            with col:
                try:
                    _render_scanner_card(item, research, tags)
                except Exception:
                    st.warning("죄송합니다. 현재 이 종목의 데이터를 불러올 수 없습니다.")


def render_scanner_tab() -> None:
    # Fetch live listings (cached 24 h each, fallback to hardcoded)
    with st.spinner("시장 종목 목록 로딩 중…"):
        nasdaq_tickers  = fetch_nasdaq_tickers()
        sp500_tickers   = fetch_sp500_tickers()
        russell_tickers = fetch_russell2000_tickers()
        kospi_tickers   = fetch_kospi_tickers()
        kosdaq_tickers  = fetch_kosdaq_tickers()
        kr_meta         = get_kr_meta_dict()

    market_pools: list[tuple[str, str, str, list[str]]] = [
        ("chk_nasdaq",  "🇺🇸 NASDAQ",       "us", nasdaq_tickers),
        ("chk_sp500",   "🇺🇸 S&P 500",      "us", sp500_tickers),
        ("chk_russell", "🇺🇸 Russell 2000",  "us", russell_tickers),
        ("chk_kospi",   "🇰🇷 KOSPI",         "kr", kospi_tickers),
        ("chk_kosdaq",  "🇰🇷 KOSDAQ",        "kr", kosdaq_tickers),
    ]

    # ── 결과 필터 session_state 사전 초기화 ─────────────────────────────────
    # 위젯 렌더링 전에 초기화해야 default= 와 session_state 충돌이 없음
    if "sector_multisel" not in st.session_state:
        st.session_state["sector_multisel"] = []
    if "grade_multisel" not in st.session_state:
        st.session_state["grade_multisel"] = []
    if "pattern_multisel" not in st.session_state:
        st.session_state["pattern_multisel"] = []
    if "theme_multisel" not in st.session_state:
        st.session_state["theme_multisel"] = []

    with st.sidebar:
        st.markdown("## 🚀 봉봉 트레이더 스캐너")
        st.caption("Momentum Trend Scanner")
        st.divider()
        st.markdown("### 📊 스캔 대상 시장")
        checks: dict[str, bool] = {}
        for key, label, _mkt, tickers in market_pools:
            checks[key] = st.checkbox(
                f"{label}  ({len(tickers):,}종목)",
                value=(key == "chk_nasdaq"),
                key=key,
            )
        st.divider()

        # ── 결과 필터 (디스플레이 레벨) ────────────────────────────────────────
        st.markdown("### 🎯 결과 필터")
        st.caption("스캔 완료 후 결과를 즉시 좁혀볼 수 있습니다.")

        sector_sel: list[str] = st.multiselect(
            "섹터",
            options=_SECTOR_LABELS,
            key="sector_multisel",
            placeholder="전체 섹터",
        )

        theme_sel: list[str] = st.multiselect(
            "테마",
            options=THEME_LABELS,
            format_func=theme_display,
            key="theme_multisel",
            placeholder="전체 테마",
        )

        grade_sel: list[str] = st.multiselect(
            "AI 등급",
            options=["S+", "S", "A", "B", "C", "D"],
            key="grade_multisel",
            placeholder="전체 등급",
        )

        pattern_sel: list[str] = st.multiselect(
            "패턴",
            options=list(_PATTERN_LABELS.values()),
            key="pattern_multisel",
            placeholder="전체 패턴",
        )

        def _reset_filters() -> None:
            st.session_state["sector_multisel"]  = []
            st.session_state["theme_multisel"]   = []
            st.session_state["grade_multisel"]   = []
            st.session_state["pattern_multisel"] = []

        _has_active = (bool(sector_sel) or bool(theme_sel)
                       or bool(grade_sel) or bool(pattern_sel))
        st.button(
            "🔁 필터 초기화",
            use_container_width=True,
            key="reset_filter_btn",
            disabled=not _has_active,
            on_click=_reset_filters,
        )

        st.divider()
        if st.button("🔄 캐시 새로고침", use_container_width=True, key="refresh_btn"):
            st.cache_data.clear()
            st.session_state.pop("scanner_results", None)
            st.session_state.pop("scanner_ran", None)
            st.rerun()

    # Build the combined scan list from checked markets (ETF/SPAC 제외)
    scan_list: list[tuple[str, str, str]] = []
    for key, _label, mkt, tickers in market_pools:
        if checks[key]:
            if mkt == "kr":
                scan_list += [
                    (c, kr_meta.get(c, {}).get('name') or KR_NAMES.get(c, c), mkt)
                    for c in tickers
                    if not _is_kr_excluded(
                        kr_meta.get(c, {}).get('name') or KR_NAMES.get(c, c)
                    )
                ]
            else:
                scan_list += [(t, t, mkt) for t in tickers]

    total = len(scan_list)
    selected_labels = [label for key, label, _m, _t in market_pools if checks[key]]

    # 스캔 버튼은 항상 표시 — 시장 미선택 시 비활성화
    run_scan = st.button(
        "🔍 스캔 시작",
        type="primary",
        key="scan_btn",
        disabled=(total == 0),
        use_container_width=False,
    )

    if total == 0:
        st.warning("스캔할 시장을 하나 이상 선택하세요. 왼쪽 사이드바에서 시장을 선택해 주세요.")
        return

    if not run_scan and not st.session_state.get("scanner_ran"):
        markets_str = "  ·  ".join(selected_labels)
        st.info(f"📌 **스캔 시작** 버튼을 클릭하면 **{total}개 종목**을 스캔합니다.\n\n대상: {markets_str}")
        return

    cache_key = f"scanner_results_{'_'.join(k for k,v in checks.items() if v)}"
    if run_scan or st.session_state.get("scanner_ran") != cache_key:
        st.session_state["scanner_ran"] = cache_key
        st.session_state.pop("scanner_benchmark", None)

        # ── Stage 1: parallel download + fast MA/spread/volume filter ──
        stat_box   = st.empty()
        prog_bar   = st.progress(0.0)
        start_time = time.time()
        completed  = 0
        stage1_ok: list[tuple[str, str, str, dict]] = []
        etf_excluded:       int = 0
        spac_excluded:      int = 0
        null_count:         int = 0   # 데이터 없음(None) 종목
        liquidity_excluded: int = 0   # 거래대금 100억원 미만
        rr_excluded:        int = 0   # 손익비(R/R) 1 미만 제외

        _MIN_DAILY_VALUE_KRW = 10_000_000_000   # 100억원
        _KRW_PER_USD         = 1_350             # USD→KRW 환산 기준

        def _dl(item: tuple[str, str, str]) -> tuple[str, str, str, dict | None]:
            code, name, mkt = item
            return code, name, mkt, get_data(code, mkt)

        try:
            with ThreadPoolExecutor(max_workers=20) as exc:
                futures = {exc.submit(_dl, item): item for item in scan_list}
                for fut in as_completed(futures):
                    completed += 1
                    code = name = mkt = ''
                    raw: dict | None = None

                    # ① future 결과 수거 — 예외 독립 처리
                    try:
                        code, name, mkt, raw = fut.result()
                    except Exception:
                        pass   # raw stays None → counted as null below

                    # ② raw 데이터 없음 → 카운트 후 스킵
                    if raw is None:
                        null_count += 1
                    else:
                        # ③ US 종목: ETF/SPAC 판별 (예외 안전 처리)
                        _excl = False
                        if mkt == 'us':
                            try:
                                _excl_flag, _reason = _is_etf_or_spac(raw)
                            except Exception:
                                _excl_flag, _reason = False, ''
                            if _excl_flag:
                                if _reason == 'ETF':
                                    etf_excluded += 1
                                else:
                                    spac_excluded += 1
                                _excl = True

                        # ④ 거래대금 필터 (100억원 미만 제외)
                        if not _excl:
                            try:
                                hist_liq = raw.get('hist')
                                if hist_liq is not None and len(hist_liq) >= 5:
                                    tail5   = hist_liq.tail(5)
                                    avg_val = float((tail5['Close'] * tail5['Volume']).mean())
                                else:
                                    avg_val = 0.0
                                val_krw = avg_val * _KRW_PER_USD if mkt == 'us' else avg_val
                                if val_krw < _MIN_DAILY_VALUE_KRW:
                                    liquidity_excluded += 1
                                    _excl = True
                            except Exception:
                                pass   # 거래대금 계산 실패 시 필터 미적용

                        # ⑤ Stage 1 통과 여부 (ETF/SPAC/유동성 미제외 종목만)
                        if not _excl:
                            try:
                                if _stage1_pass(raw):
                                    stage1_ok.append((code, name, mkt, raw))
                            except Exception:
                                pass

                    # ⑤ 진행 상황 업데이트 (모든 종목에 대해 항상 실행)
                    elapsed = time.time() - start_time
                    rate    = completed / elapsed if elapsed > 0 else 0.0
                    eta_s   = (total - completed) / rate if rate > 0 else 0.0
                    eta_str = (f"{int(eta_s//60)}분 {int(eta_s%60)}초"
                               if eta_s >= 60 else f"{int(eta_s)}초")
                    try:
                        prog_bar.progress(completed / total)
                        stat_box.markdown(
                            f"**1단계** 데이터 로드 &nbsp;·&nbsp; "
                            f"`{completed:,} / {total:,}` &nbsp;·&nbsp; "
                            f"ETF제외 **{etf_excluded}** · SPAC제외 **{spac_excluded}** · "
                            f"거래대금제외 **{liquidity_excluded}** · "
                            f"데이터없음 **{null_count}** · 통과 **{len(stage1_ok)}** &nbsp;·&nbsp; "
                            f"**{rate:.1f} 종목/초** &nbsp;·&nbsp; 남은시간 **{eta_str}**"
                        )
                    except Exception:
                        pass
        except Exception as e:
            stat_box.error(f"1단계 오류: {e}")
            prog_bar.empty()
            return

        # ── Stage 2: full indicators + AI score on survivors only ──────
        results: list[dict] = []
        with_patterns: int  = 0   # 패턴 1개 이상 감지 종목 수
        s2_total = len(stage1_ok)
        for i, (code, name, mkt, raw) in enumerate(stage1_ok):
            try:
                pct = (i + 1) / s2_total if s2_total > 0 else 1.0
                try:
                    prog_bar.progress(pct)
                    stat_box.markdown(
                        f"**2단계** 심층 분석 &nbsp;·&nbsp; "
                        f"`{i+1} / {s2_total}` &nbsp;·&nbsp; {name} &nbsp;·&nbsp; "
                        f"패턴감지 **{with_patterns}**"
                    )
                except Exception:
                    pass
                data   = add_indicators(raw)
                scored = calculate_ai_score(data)
                # 패턴 1개 이상 감지 여부 카운트 (bottom_setup 은 dict 이므로 제외)
                if any(isinstance(v, (tuple, list)) and v and v[0]
                       for v in scored.get('patterns', {}).values()):
                    with_patterns += 1
                mom    = _calc_momentum(raw['hist'])
                _m = kr_meta.get(code, {}) if mkt == 'kr' else {}
                # KR Phase-2 stocks: fetch Yahoo Finance industry (small set, cached)
                if mkt == 'kr' and not data.get('industry'):
                    _ind = get_kr_yf_industry(code, _m.get('market', ''))
                    data = {**data, 'industry': _ind}
                # 거래대금 계산 (카드 표시용, 5일 평균)
                try:
                    _hist_tv = raw.get('hist')
                    if _hist_tv is not None and len(_hist_tv) >= 5:
                        _tv_raw = float(
                            (_hist_tv['Close'].tail(5) * _hist_tv['Volume'].tail(5)).mean()
                        )
                    else:
                        _tv_raw = 0.0
                except Exception:
                    _tv_raw = 0.0

                item = {
                    'code': code, 'name': name, 'market': mkt,
                    'momentum':       mom,
                    'kr_market':      _m.get('market', ''),
                    'kr_industry':    _m.get('industry', ''),
                    'trading_value':  _tv_raw,   # 로컬 통화 (KRW or USD)
                    **data, **scored,
                }
                # 손익비(R/R) 1 미만 → 제외 (전략 분류와 무관하게 최종판단 규칙으로 컷).
                # 연구뷰 계산이 실패하면 R/R 을 확정할 수 없으므로 fail-closed 로 제외한다.
                try:
                    _rv = build_research_view(item, item)
                except Exception:
                    rr_excluded += 1
                    continue
                _rr_val = _rv.get('risk_reward', 0.0) or 0.0
                if not (_rr_val >= 1.0):   # NaN 도 함께 제외 (NaN>=1 → False)
                    rr_excluded += 1
                    continue
                item['_research'] = _rv
                results.append(item)
            except Exception:
                continue

        # ── Benchmark ──────────────────────────────────────────────────
        total_elapsed = time.time() - start_time
        avg_rate      = total / total_elapsed if total_elapsed > 0 else 0.0
        benchmark     = {
            'total': total, 'elapsed': total_elapsed, 'rate': avg_rate,
            'etf_excluded': etf_excluded, 'spac_excluded': spac_excluded,
            'liquidity_excluded': liquidity_excluded,
            'rr_excluded': rr_excluded,
            'null_count': null_count,
            'stage1_pass': len(stage1_ok), 'stage2_pass': len(results),
            'with_patterns': with_patterns,
        }
        try:
            prog_bar.empty()
            stat_box.success(
                f"✅ 스캔 완료 &nbsp;·&nbsp; **{total:,}종목 / {total_elapsed:.1f}초** "
                f"&nbsp;·&nbsp; ETF제외 **{etf_excluded}** · SPAC제외 **{spac_excluded}** "
                f"· 거래대금제외 **{liquidity_excluded}** · 데이터없음 **{null_count}** "
                f"· R/R<1 제외 **{rr_excluded}** "
                f"&nbsp;·&nbsp; 1단계 통과 **{len(stage1_ok)}** "
                f"&nbsp;·&nbsp; 최종 결과 **{len(results)}**"
            )
        except Exception:
            pass

        results.sort(key=lambda x: (x.get('ma300_strategy', False), x.get('score', 0)), reverse=True)
        results = _dedup_by_ticker(results)
        st.session_state["scanner_results"]   = results
        st.session_state["scanner_benchmark"] = benchmark
    else:
        results = st.session_state.get("scanner_results", [])

    results   = st.session_state.get("scanner_results", [])
    benchmark = st.session_state.get("scanner_benchmark")
    # 세션에 이전(중복 제거 전) 결과가 남아있을 수 있어 방어적으로 한 번 더 적용
    results   = _dedup_by_ticker(results)

    # ── 결과 필터 적용 (디스플레이 레벨) ───────────────────────────────────────
    sector_sel  = st.session_state.get("sector_multisel", [])
    theme_sel   = st.session_state.get("theme_multisel", [])
    grade_sel   = st.session_state.get("grade_multisel", [])
    pattern_sel = st.session_state.get("pattern_multisel", [])
    filtered = [
        r for r in results
        if _match_sector_filter(r, sector_sel)
        and _match_theme_filter(r, theme_sel)
        and _match_grade_filter(r, grade_sel)
        and _match_pattern_filter(r, pattern_sel)
    ]

    if benchmark:
        b1, b2, b3, b4, b5, b6, b7, b8 = st.columns(8)
        _etf  = benchmark.get('etf_excluded', 0)
        _spac = benchmark.get('spac_excluded', 0)
        _liq  = benchmark.get('liquidity_excluded', 0)
        _null = benchmark.get('null_count', 0)
        _wpat = benchmark.get('with_patterns', 0)
        _s2   = benchmark.get('stage2_pass', 0)
        _analyzed = benchmark['total'] - _etf - _spac - _liq - _null
        b1.metric("전체 종목",     f"{benchmark['total']:,}")
        b2.metric("SPAC 제외",     f"{_spac}개")
        b3.metric("거래대금 제외", f"{_liq}개",  help="5일 평균 일 거래대금 100억원 미만 제외")
        b4.metric("분석 대상",     f"{max(_analyzed, 0):,}개")
        b5.metric("소요 시간",     f"{benchmark['elapsed']:.1f}초")
        b6.metric("최종 통과",     f"{_s2}종목")
        b7.metric("패턴 감지",     f"{_wpat}종목", delta=f"/{_s2}종목 중")
        b8.metric("ETF 제외",      f"{_etf}개", help="NASDAQ 소스는 ETF 미포함. S&P500/Russell 스캔 시 ETF가 탐지됩니다.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("스캔",        f"{total:,}종목")
        m2.metric("필터 통과",   f"{len(results)}종목")
        m3.metric("분석 대상", f"{len(results)}종목")
    st.divider()

    # ── 검색 결과 헤더 ──────────────────────────────────────────────────────────
    _active_parts: list[str] = []
    if sector_sel:
        _active_parts += sector_sel
    if theme_sel:
        _active_parts += [theme_display(t) for t in theme_sel]
    if grade_sel:
        _active_parts += [f"등급:{g}" for g in grade_sel]
    if pattern_sel:
        _active_parts += pattern_sel

    if _active_parts:
        _filter_summary = _html.escape(" + ".join(_active_parts))
        st.markdown(
            f"<div style='background:#1A1A2E;border-left:3px solid #5C6BC0;"
            f"padding:10px 14px;border-radius:6px;margin-bottom:12px;'>"
            f"<span style='color:#8888AA;font-size:0.82rem;'>적용 필터</span>&nbsp;&nbsp;"
            f"<span style='color:#C5CAE9;font-size:0.88rem;font-weight:600;'>{_filter_summary}</span>"
            f"&nbsp;&nbsp;<span style='color:#7986CB;font-size:0.88rem;'>|</span>&nbsp;&nbsp;"
            f"<span style='color:#E0E0FF;font-size:0.95rem;font-weight:700;'>검색 결과: {len(filtered)}개</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    elif results:
        st.markdown(
            f"<div style='background:#1A1A2E;border-left:3px solid #5C6BC0;"
            f"padding:10px 14px;border-radius:6px;margin-bottom:12px;'>"
            f"<span style='color:#8888AA;font-size:0.82rem;'>전체 결과</span>&nbsp;&nbsp;"
            f"<span style='color:#E0E0FF;font-size:0.95rem;font-weight:700;'>검색 결과: {len(filtered)}개</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    if not filtered:
        if not results:
            st.info("조건에 맞는 종목이 없습니다. MA20 > MA60 > MA120 정배열 종목이 없거나 거래대금 조건을 확인하세요.")
        else:
            st.warning(
                "선택한 필터에 해당하는 종목이 없습니다.\n\n"
                "다른 시장이나 섹터를 선택하거나 **필터 초기화** 버튼을 눌러보세요."
            )
        return

    # ── 전략 분류 (리서치 뷰 1회 계산 후 태그 부여) ────────────────────────────
    enriched: list[tuple] = []
    for it in filtered:
        research = it.get('_research')
        if research is None:
            try:
                research = build_research_view(it, it)
            except Exception:
                research = None
        tags = classify_strategies(it, research)
        enriched.append((it, research, tags))

    _n_pre    = sum(1 for _, _, tg in enriched if 'PREBUY' in tg)
    _n_bottom = sum(1 for _, _, tg in enriched if 'BOTTOM_OUT' in tg)
    _n_strong = sum(1 for _, _, tg in enriched if 'STRONG' in tg)

    tab_all, tab_pre, tab_bottom, tab_strong = st.tabs([
        f"전체 ({len(enriched)})",
        f"① 선매집 ({_n_pre})",
        f"② 바닥탈출 ({_n_bottom})",
        f"③ 강세추세 ({_n_strong})",
    ])
    with tab_all:
        _render_results_grid(enriched)
    with tab_pre:
        st.caption(STRATEGY_META['PREBUY'][2])
        _render_results_grid([e for e in enriched if 'PREBUY' in e[2]])
    with tab_bottom:
        st.caption(STRATEGY_META['BOTTOM_OUT'][2])

        def _bottom_key(e: tuple) -> tuple:
            bs = (e[0].get('patterns', {}) or {}).get('bottom_setup')
            bs = bs if isinstance(bs, dict) else {}
            return (bs.get('priority') or 9, abs(bs.get('breakout_pct') or 9.0))

        _bottom = sorted(
            [e for e in enriched if 'BOTTOM_OUT' in e[2]], key=_bottom_key
        )
        _render_results_grid(_bottom)
    with tab_strong:
        st.caption(STRATEGY_META['STRONG'][2])
        _render_results_grid([e for e in enriched if 'STRONG' in e[2]])

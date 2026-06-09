import streamlit as st
import pandas as pd
import time
import math
import gc
import html as _html
from safe_exec import gather_parallel

from data_provider import (
    get_krx_listing, has_korean, resolve_input,
    get_data, KR_NAMES, get_name_by_code,
    fetch_nasdaq_tickers, fetch_sp500_tickers, fetch_russell2000_tickers,
    fetch_kospi_tickers, fetch_kosdaq_tickers,
    get_kr_meta_dict, get_kr_yf_industry,
    _build_result, _fetch_us_news, _fetch_naver_news, display_quote,
)
from bulk_data import (
    fetch_us_snapshot, fetch_us_industry_map, fetch_kr_snapshot,
    bulk_history, us_snapshot_is_degraded,
    fetch_live_quotes, live_symbol,
)
from scan_ui import render_stage_funnel
from indicators import add_indicators
from ai_engine import (
    calculate_ai_score, build_research_view,
    classify_strategies, STRATEGY_META,
    classify_strong_patterns, STRONG_PATTERN_META,
    action_view, pattern_progress,
    GRADE_COLOR, STAGE_META,
)
from themes import classify_themes, theme_display, THEME_LABELS
import news_sentiment as _sent
import datetime as _dt
from news_view import render_news_section


def naver_url(code: str, market: str) -> str:
    """종목 → 네이버 금융 링크. 한국: 네이버증권 종목, 미국: 네이버 해외주식.

    미국 종목은 거래소 접미사(.O=NASDAQ 기본)를 알 수 없는 경우가 있어 .O 로 두며,
    NYSE 등 일부 종목은 페이지가 다를 수 있다(상세분석 탭은 거래소를 조회해 보정).
    """
    code = str(code or '').strip()
    if market == 'kr':
        digits = ''.join(ch for ch in code if ch.isdigit())
        return f"https://finance.naver.com/item/main.naver?code={digits or code}"
    return f"https://m.stock.naver.com/worldstock/stock/{code.upper()}.O/total"


# ── 금융 섹터 제외 ───────────────────────────────────────────────────────────────
# 은행·보험·증권·카드·금융서비스 종목은 스캐너 결과에서 자동 제외한다(섹터/업종 텍스트 매칭).
# 미국은 sector='금융'(Financial Services)으로 일괄 잡히고, 세부 업종(지방은행/생명보험/
# 자본시장 등)·한국 업종 텍스트는 키워드로 보강한다.
_FINANCIAL_KEYWORDS: tuple[str, ...] = (
    '금융', '은행', '보험', '증권', '카드', '자산운용', '자본시장',
)


def _is_financial(sector: str, industry: str) -> bool:
    """섹터/업종 텍스트가 금융권(은행·보험·증권·카드·금융서비스)이면 True."""
    text = f"{sector or ''} {industry or ''}"
    return any(kw in text for kw in _FINANCIAL_KEYWORDS)


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
    """선택된 패턴 중 감지된 것이 하나라도 있으면 True (OR 로직).

    방어적: patterns 가 dict 가 아니거나 값이 튜플/불리언/딕셔너리 등
    어떤 형태여도 예외 없이 안전하게 판정한다(필터에서 절대 죽지 않도록).
    """
    if not selected:
        return True
    patterns = item.get('patterns', {})
    if not isinstance(patterns, dict):
        return False
    _label_to_key = {v: k for k, v in _PATTERN_LABELS.items()}
    for label in selected:
        key = _label_to_key.get(label)
        if not key:
            continue
        val = patterns.get(key)
        if isinstance(val, (tuple, list)):
            detected = bool(val) and bool(val[0])
        elif isinstance(val, dict):
            detected = bool(val.get('detected'))
        else:
            detected = bool(val)
        if detected:
            return True
    return False


# ── Pattern badges ─────────────────────────────────────────────────────────────────
_PATTERN_LABELS = {
    'cup_and_handle': '컵앤핸들 형성',
    'double_bottom':  '쌍바닥 형성',
    'ma300_breakout': '300일선 돌파 시도',
    'ma_convergence': '5·10·20일선 수렴',
    'first_pullback': '첫 눌림목',
    'new_high':       '신고가 돌파',
    'golden_cross':   '골든크로스',
}

_CARD_PATTERN_ICONS: dict[str, tuple[str, str]] = {
    'cup_and_handle': ('☕', '컵앤핸들'),
    'double_bottom':  ('🔄', '쌍바닥'),
    'ma300_breakout': ('🎯', '300일선 돌파'),
    'ma_convergence': ('🧲', '5·10·20 수렴'),
    'first_pullback': ('🔥', '첫 눌림목'),
    'new_high':       ('🚀', '신고가 돌파'),
    'golden_cross':   ('🏆', '골든크로스'),
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
    rsi = data.get('rsi', 0.0)
    mfi = data.get('mfi', 0.0)
    price = data.get('price', 0.0)
    hist = data.get('hist')
    patterns = scored.get('patterns', {})

    stage_ko, _, stage_desc = STAGE_META.get(stage, (stage, '', ''))
    label = _html.escape(str(label))

    parts: list[str] = []
    parts.append(
        f"**{label}** 은(는) 현재 **{stage_ko}** 국면으로 판단되며, "
        f"AI 종합 점수는 **{score}/100 ({grade}등급)** 입니다. "
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

    a_ko, a_final, _ = action_view(research)
    parts.append(f"**최종 행동은 「{a_ko}」 — {a_final}** 입니다.")

    return "\n\n".join(parts)


def _tech_detail_html(scored: dict, research: dict) -> str:
    """상세분석용 기술 지표 카드 — 이평 기울기·데드크로스·거래량."""
    slopes = research.get('ma_slopes', {}) or {}

    def _slope(key):
        lbl, pct = slopes.get(key, ('—', 0.0))
        col = {'상승': '#00C851', '하락': '#FF5252', '횡보': '#FFB300'}.get(lbl, '#9E9E9E')
        txt = f'{lbl} ({pct:+.2f}%)' if lbl != '—' else '—'
        return col, txt

    dead     = bool(research.get('dead_cross'))
    dead_col = '#FF5252' if dead else '#69F0AE'
    dead_txt = '있음 ⚠️' if dead else '없음'

    vol       = research.get('vol_state', '—')
    vol_ratio = research.get('vol_ratio', 0.0) or 0.0
    vol_col   = {'급증': '#FF7043', '증가': '#69F0AE',
                 '감소': '#90A4AE', '보통': '#C8C8E8'}.get(vol, '#C8C8E8')
    vol_txt   = f'{vol} ({vol_ratio:.1f}x)' if vol and vol != '—' else '—'

    def _row(label, value, vcol='#D5D5EE'):
        return (
            f'<div style="display:flex;justify-content:space-between;gap:10px;'
            f'padding:5px 0;border-bottom:1px solid #23233A">'
            f'<span style="color:#8888AA;font-size:0.8rem">{_html.escape(label)}</span>'
            f'<span style="color:{vcol};font-size:0.86rem;font-weight:600;'
            f'text-align:right">{_html.escape(str(value))}</span></div>'
        )

    c5, t5   = _slope('ma5')
    c10, t10 = _slope('ma10')
    c20, t20 = _slope('ma20')

    return (
        '<div style="background:#15151F;border:1px solid #2E2E4E;border-radius:12px;'
        'padding:14px 18px;margin-bottom:14px">'
        '<div style="font-size:0.72rem;letter-spacing:.08em;color:#8888AA;'
        'text-transform:uppercase;margin-bottom:6px">기술 지표 상세 · Technicals</div>'
        + _row('MA5 기울기', t5, c5)
        + _row('MA10 기울기', t10, c10)
        + _row('MA20 기울기', t20, c20)
        + _row('데드크로스 여부', dead_txt, dead_col)
        + _row('거래량 상태', vol_txt, vol_col)
        + '</div>'
    )


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
    news       = data.get('news', [])

    stage   = research['stage']
    el, eh  = research['entry_low'], research['entry_high']
    stop    = research['stop_loss']
    t1, t2, t3 = research['targets']
    rr      = research['risk_reward']

    s_ko, s_col, s_desc = STAGE_META.get(stage, (stage, '#9E9E9E', ''))
    a_ko, a_final, a_col = action_view(research)
    p_col   = _prob_color(score)

    def pct(v):
        return (v - price) / price * 100 if price else 0.0

    # ── 1) 최종 행동(단일 결정) / 국면 / AI 상승확률 ──────────────────────────
    st.markdown(f"""
<div style="display:flex;flex-wrap:wrap;gap:12px;margin-bottom:14px">
  <div style="flex:1 1 240px;background:#16161F;border:1px solid {a_col}55;
              border-left:4px solid {a_col};border-radius:12px;padding:14px 18px">
    <div style="font-size:0.72rem;letter-spacing:.08em;color:#8888AA;text-transform:uppercase">최종 행동 · Final Action</div>
    <div style="font-size:1.7rem;font-weight:800;color:{a_col};margin-top:2px">{a_ko}</div>
    <div style="font-size:0.85rem;font-weight:700;color:{a_col};margin-top:2px">{a_final}</div>
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

    # ── 2) 기술 지표 상세 (이평 기울기 · 데드크로스 · 거래량) ────────────────────
    try:
        st.markdown("##### 🔬 기술 지표 상세")
        st.markdown(_tech_detail_html(scored, research), unsafe_allow_html=True)
    except Exception:
        pass

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

    # ── 5) 뉴스 ────────────────────────────────────────────────────────────────
    # 내장 차트 섹션은 제거됨: 종목명이 네이버 차트로 바로 연결되므로 중복이며,
    # 화면을 많이 차지하고 렌더링을 느리게 했다.
    with st.expander("📰 최신 뉴스 (최근 3일)", expanded=False):
        render_news_section(news, key_prefix="an")


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

    # 화면 출력 직전 현재가 재조회 — **표시(헤더)용 가격/등락률만** 최신화한다.
    # data 는 건드리지 않는다: render_full_detail 이 data 로 점수/연구뷰를 재계산하므로
    # data['price'] 를 바꾸면 점수/액션이 흔들려 'scoring 불변' 제약을 위반한다. 따라서
    # 표시용 지역변수 price/change 만 덮어써 카드에 최신가를 보인다(get_data 15분 캐시 무관).
    try:
        _kr_board = get_kr_meta_dict().get(code, {}).get('market', '') if market == 'kr' else ''
        _sym = live_symbol(code, market, _kr_board)
        _q = fetch_live_quotes((_sym,))
        if _q.get(_sym):
            price, change = _q[_sym]
    except Exception:
        pass

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
    n_url        = _html.escape(naver_url(code, market))

    st.markdown(f"""
<div style="background:#1E1E2E;border:1px solid #2E2E4E;border-radius:14px;
            padding:20px 24px 16px;margin-bottom:8px;">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;">
    <div>
      <div style="font-size:1.4rem;font-weight:700;color:#E0E0FF">
        <a href="{n_url}" target="_blank" rel="noopener"
           style="color:#E0E0FF;text-decoration:none;border-bottom:1px dotted #6C6C9C">{company_name}</a>
        <span style="color:#8888AA;font-weight:400;font-size:1.1rem">({code_disp})</span>
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

    최종판단(액션)은 단 하나의 결정 — 투자의견(매수 자격) + 진입구간 위치로 산출된다
    (`ai_engine._decide_action`).
      매수 + 진입구간 도달 → 바로 진입 · 근처 → 분할 진입 · 위 → 관찰리스트 ·
      조건 미충족(매수의견 아님/매수 차단) → 제외
    """
    action = research.get('action', 'EXCLUDE')
    rating = research.get('rating', '')
    price  = research.get('price') or 0.0
    eh     = research.get('entry_high') or 0.0
    gap    = ((price - eh) / eh * 100) if eh else 0.0

    if research.get('buy_blocked'):
        return '실적 급락 + 이평 하락 + 데드크로스 → 매수 금지', 'EARNINGS_GUARD'
    if action == 'EXCLUDE':
        return f'투자의견 {rating} (매수 아님) → 제외', 'NOT_BUY'
    if action == 'ENTER':
        return f'매수 + 진입구간 도달 (상단 대비 {gap:+.1f}%) → 바로 진입', 'NONE'
    if action == 'SPLIT':
        return f'매수 + 진입구간 근처 (상단 대비 {gap:+.1f}%) → 분할 진입', 'NONE'
    return f'매수 + 진입구간 위 (상단 대비 {gap:+.1f}%) → 관찰리스트', 'NONE'


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

    필드 순서: 전략 → 패턴 → 업종 → 테마 → 현재 국면 → 거래량 상태 →
    확률점수 → 매수구간 → 손절가 → 목표가 → 손익비 → 최종 행동(단일 결정).
    """
    code   = item.get('code', '')
    name   = item.get('name', '')
    # 세션(프리/정규/애프터) 스냅샷 시세 우선, 없으면 일봉 종가로 폴백
    price  = item.get('session_price') or item.get('price', 0) or 0
    change = item.get('session_change')
    if change is None:
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
    n_url      = _html.escape(naver_url(code, item.get('market', '')))

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

    # 패턴
    det_labels = [
        _PATTERN_LABELS[k] for k, v in patterns.items()
        if k in _PATTERN_LABELS and isinstance(v, (tuple, list)) and v and v[0]
    ]
    pat_html = _html.escape(', '.join(det_labels)) if det_labels else '<span style="color:#666">—</span>'

    # 액션 / 최종판단
    a_ko, a_final, a_col = action_view(research)

    # 패턴 진행률 / 거래량 상태
    pp_name, pp_pct, pp_phase = pattern_progress(patterns)
    pp_html = (
        f'{_html.escape(pp_name)} · {pp_pct:.0f}% '
        f'<span style="color:#8888AA;font-weight:400">({_html.escape(pp_phase)})</span>'
        if pp_name != '—' else '<span style="color:#666">—</span>'
    )
    vol_state = research.get('vol_state', '—')
    vol_ratio = research.get('vol_ratio', 0.0) or 0.0
    vol_str = (f'{_html.escape(str(vol_state))} ({vol_ratio:.1f}x)'
               if vol_state and vol_state != '—' else '—')
    vol_col = {'급증': '#FF7043', '증가': '#69F0AE',
               '감소': '#90A4AE', '보통': '#C8C8E8'}.get(vol_state, '#C8C8E8')

    p_col = _prob_color(score)

    # 현재가 표기: KR ₩(정수) / US $(소수 2자리), 시장 태그 동반
    price_str = f"₩{price:,.0f}" if is_kr else f"${price:,.2f}"
    mkt_tag   = "🇰🇷 KR" if is_kr else "🇺🇸 US"

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
      <a href="{n_url}" target="_blank" rel="noopener"
         style="color:#E0E0FF;text-decoration:none;border-bottom:1px dotted #6C6C9C">{name_disp}</a>
      <span style="color:#8888AA;font-weight:400;font-size:0.85rem">({code_disp})</span>
    </span>
    <span style="background:{badge_col};color:#000;font-weight:700;
                 padding:2px 10px;border-radius:10px;font-size:0.8rem">{grade}</span>
  </div>
  <div style="display:flex;justify-content:space-between;align-items:baseline;margin:6px 0 8px">
    <span style="font-size:1.18rem;font-weight:700;color:#FFF">{price_str}
      <span style="font-size:0.7rem;font-weight:600;color:#8888AA;margin-left:6px">{mkt_tag}</span>
    </span>
    <span style="font-size:0.88rem;font-weight:600;color:{chg_color}">{change_str}</span>
  </div>
  {_row("전략", strat_html)}
  {_row("패턴", pat_html)}
  {_row("패턴 진행률", pp_html)}
  {_row("거래량 상태", vol_str, vol_col)}
  {_row("확률점수", f'{score}/100', p_col)}
  <div style="display:flex;justify-content:space-between;align-items:center;gap:10px;
              margin-top:8px;padding:8px 10px;background:{a_col}1A;
              border:1px solid {a_col}55;border-radius:8px">
    <span style="color:#A0A0C0;font-size:0.74rem;font-weight:600">최종 행동</span>
    <span style="color:{a_col};font-size:1.0rem;font-weight:800">{a_ko}</span>
  </div>
</div>""", unsafe_allow_html=True)

    with st.expander("▼ 상세 보기 (패턴 · 뉴스)"):
        _render_patterns(patterns)
        st.markdown("**📰 최신 뉴스 (최근 3일)**")
        render_news_section(news, key_prefix="sc")


def _refresh_display_quotes(items: list[dict]) -> None:
    """최종 결과 종목만 화면 출력 직전 현재가를 재조회해 표시 시세를 갱신한다.

    7천 종목 전체 실시간이 아니라 *최종 통과 종목*(보통 ≤100)만 단일 벌크 일봉으로
    다시 받아 카드가 항상 최신가를 보이게 한다. 점수·전략·패턴 등 분석 결과는 그대로
    두고, 표시용 `session_price`/`session_change` 만 덮어쓴다. 실패한 종목은 기존
    표시가를 유지(폴백)한다."""
    if not items:
        return
    try:
        syms = [live_symbol(it.get('code', ''), it.get('market', ''),
                            it.get('kr_market', '')) for it in items]
        quotes = fetch_live_quotes(tuple(dict.fromkeys(syms)))
        for it, sym in zip(items, syms):
            q = quotes.get(sym)
            if q:
                it['session_price'], it['session_change'] = q[0], q[1]
    except Exception:
        pass


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
    # 전체 시장 스냅샷 로딩 (캐시 15분) — 종목당 호출 없이 일괄 수집
    _t_uni0 = time.time()
    with st.spinner("전체 시장 스냅샷 로딩 중…"):
        us_snap     = fetch_us_snapshot()
        kospi_snap  = fetch_kr_snapshot("KOSPI")
        kosdaq_snap = fetch_kr_snapshot("KOSDAQ")
        kr_meta     = get_kr_meta_dict()
    _universe_load_sec = time.time() - _t_uni0

    # 시장별 통합 풀: US = NASDAQ+NYSE+AMEX, KR = KOSPI+KOSDAQ.
    # 거래소 선택(체크박스)은 제거 — 상단 시장 탭(미국/한국)으로 단순화(모바일 대응).
    us_rows = list(us_snap.values())
    kr_rows = list(kospi_snap.values()) + list(kosdaq_snap.values())

    # ── 결과 필터 session_state 사전 초기화 ─────────────────────────────────
    # 위젯 렌더링 전에 초기화해야 default= 와 session_state 충돌이 없음
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

        # ── 결과 필터 (디스플레이 레벨) ────────────────────────────────────────
        st.markdown("### 🎯 결과 필터")
        st.caption("스캔 완료 후 결과를 즉시 좁혀볼 수 있습니다.")

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
            st.session_state["theme_multisel"]   = []
            st.session_state["grade_multisel"]   = []
            st.session_state["pattern_multisel"] = []

        _has_active = (bool(theme_sel)
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
            st.session_state.pop("scanner_cache", None)
            st.rerun()

    # ── 시장 선택 (단일 선택기: 🇺🇸 미국 / 🇰🇷 한국) ──────────────────────────────
    # st.radio 는 항상 유효값을 반환하고 해제(None)가 없어 선택이 확실히 반영된다.
    # (이전 st.segmented_control + `or "us"` 는 해제 시 None→US 로 되돌아가는 버그가 있었음.)
    market_choice = st.radio(
        "시장 선택",
        options=["us", "kr"],
        format_func=lambda m: ("🇺🇸 미국시장 (NASDAQ·NYSE·AMEX)" if m == "us"
                               else "🇰🇷 한국시장 (KOSPI·KOSDAQ)"),
        horizontal=True,
        key="scanner_market",
        label_visibility="collapsed",
    )

    if market_choice == "us":
        sel_rows, sel_label = us_rows, "🇺🇸 미국시장 (NASDAQ · NYSE · AMEX)"
        if us_snapshot_is_degraded(us_snap):
            st.warning(
                "⚠️ NASDAQ 스냅샷 소스가 차단(HTTP 401)되어 **FDR 심볼 폴백**으로 동작 중입니다. "
                "미국 종목 목록은 정상 표시되지만 **시총 사전필터가 비활성화**되고 "
                "**가격은 다운로드 후 일봉 종가 기준**이라 스캔이 느려지고 프리마켓 시세가 반영되지 않을 수 있습니다.",
                icon="⚠️",
            )
    else:
        sel_rows, sel_label = kr_rows, "🇰🇷 한국시장 (KOSPI · KOSDAQ)"

    # 선택 시장의 전체 후보 수집 (ETF/거래대금/시총 필터는 깔때기에서 — 탈락 수 집계)
    candidates: list[tuple[str, str, str, dict]] = []
    for row in sel_rows:
        code = row["code"] if market_choice == "kr" else row["symbol"]
        candidates.append((code, row.get("name") or code, market_choice, row))

    total = len(candidates)
    selected_labels = [sel_label]

    st.caption(f"대상: {sel_label}  ·  {total:,}종목")

    # 스캔 버튼은 항상 표시 — 스냅샷 미수신 시 비활성화
    run_scan = st.button(
        "🔍 스캔 시작",
        type="primary",
        key="scan_btn",
        disabled=(total == 0),
        use_container_width=False,
    )

    if total == 0:
        st.warning("시장 스냅샷을 불러오지 못했습니다. 잠시 후 사이드바의 '🔄 캐시 새로고침'을 눌러 다시 시도해 주세요.")
        return

    # ── 결과 캐시(10분) ─────────────────────────────────────────────────────────
    # 같은 시장을 10분 내 다시 보면 재스캔 없이 즉시 표시(시장 전환 후 복귀 포함).
    _SCAN_CACHE_TTL = 600
    _scan_cache: dict = st.session_state.setdefault("scanner_cache", {})
    _entry = _scan_cache.get(market_choice)
    _fresh = bool(_entry) and (time.time() - _entry.get("ts", 0) < _SCAN_CACHE_TTL)

    if not run_scan and not _fresh:
        st.info(f"📌 **스캔 시작** 버튼을 클릭하면 **{total:,}개 종목**을 스캔합니다.\n\n대상: {sel_label}")
        return

    if run_scan:
        # ── 깔때기: 스냅샷 사전필터(Stage 1) → 벌크 히스토리 → 정배열/AI 심층 ──
        prog_bar   = st.progress(0.0)
        funnel_box = st.empty()
        start_time = time.time()

        _MIN_DAILY_VALUE_KRW = 5_000_000_000    # 50억원 — 평균 거래대금 최소 기준 (미만이면 제외)
        _KRW_PER_USD         = 1_350            # USD→KRW 환산 기준
        # Stage 1 사전필터 기준 (패턴 분석 전 컷)
        _MIN_PRICE_USD  = 2.0                   # 주가 $2 이하 제외
        _MIN_AVG_VOLUME = 100_000               # 평균 거래량 10만주 이하 제외
        _MIN_MCAP_USD   = 100_000_000           # 시총 $1억 미만 제외
        _MIN_MCAP_KRW   = _MIN_MCAP_USD * _KRW_PER_USD
        # 심층분석(다운로드+점수) 후보 상한 — 사전필터 통과가 많아도 상위 N개만 분석.
        _DEEP_CANDIDATE_CAP  = 300    # 다운로드/심층분석 대상 최대 수 (상한 300, 속도 우선)
        _DEEP_CANDIDATE_WARN = 500    # 사전필터 통과가 이 수를 넘으면 원인 로그
        _MAX_OUTPUT          = 50     # 최종 출력 상한 (점수순 자동 컷오프)

        c_total  = len(candidates)
        c_etf = c_spac = c_liq = c_nodata = c_trend = c_fin = c_fail = 0
        c_cheap = c_mcap = c_vol = c_capcut = 0

        # 단계별 소요 시간 측정용 (속도 진단).
        stage_t: dict[str, float] = {"유니버스 로딩": _universe_load_sec}
        _t_a0 = time.time()

        # US 업종 맵 (금융 필터/테마용, 24h 캐시, 비치명적). US 미선택 시 생략.
        _need_us = any(m == 'us' for (_c, _n, m, _s) in candidates)
        us_ind   = fetch_us_industry_map() if _need_us else {}

        def _interim(stage_label):
            try:
                render_stage_funnel(
                    [
                        ("총 스캔",            c_total,  "total"),
                        ("ETF/SPAC 제외",      c_etf + c_spac, "reject"),
                        ("저가($2↓) 제외",     c_cheap, "reject"),
                        ("시총($100M↓) 제외",  c_mcap,  "reject"),
                        (f"심층분석 상한({_DEEP_CANDIDATE_CAP}) 초과", c_capcut, "reject"),
                        ("데이터 없음",         c_nodata,"reject"),
                        ("거래량(10만↓) 제외", c_vol,   "reject"),
                        ("거래대금 제외",       c_liq,   "reject"),
                        ("추세(정배열) 제외",   c_trend, "reject"),
                        ("금융 제외",           c_fin,   "reject"),
                        (stage_label,          len(prelim), "info"),
                    ],
                    title="단계별 필터 현황 (진행 중)",
                    container=funnel_box,
                )
            except Exception:
                pass

        # ── Stage A: 스냅샷 사전필터 (ETF/SPAC · KR 거래대금 · US 시총 프록시) ──
        prelim: list[tuple[str, str, str, dict]] = []
        for code, name, mkt, snap in candidates:
            if mkt == 'us':
                try:
                    excl, reason = _is_etf_or_spac({
                        'company_name': name,
                        'industry':     us_ind.get(code, ''),
                        'quote_type':   'EQUITY',
                    })
                except Exception:
                    excl, reason = False, ''
                if excl:
                    if reason == 'ETF':
                        c_etf += 1
                    else:
                        c_spac += 1
                    continue
                # Stage 1: 저가·소형주 사전 제외 (스냅샷에 price·mcap 존재 → 다운로드 전 컷).
                _px = snap.get('price')
                if _px is not None and _px <= _MIN_PRICE_USD:
                    c_cheap += 1
                    continue
                _mc = snap.get('mcap')
                if _mc is not None and _mc < _MIN_MCAP_USD:
                    c_mcap += 1
                    continue
                # 평균 거래량은 스냅샷에 없어(스크리너 무거래량) Stage C 히스토리에서 계산.
            else:  # kr
                if _is_kr_excluded(name):
                    c_etf += 1
                    continue
                if (snap.get('amount') or 0.0) < _MIN_DAILY_VALUE_KRW:
                    c_liq += 1
                    continue
                # Stage 1: 소형주·저거래량 사전 제외 (KR 스냅샷엔 mcap·volume 존재).
                _mc = snap.get('mcap') or 0.0
                if _mc and _mc < _MIN_MCAP_KRW:
                    c_mcap += 1
                    continue
                if (snap.get('volume') or 0.0) < _MIN_AVG_VOLUME:
                    c_vol += 1
                    continue
            prelim.append((code, name, mkt, snap))

        n_prefiltered = len(prelim)   # Stage 1(기본 조건) 통과 수 — 심층분석 후보 원수

        # ── 심층분석 후보 상한 적용 (다운로드 前 컷 — 과부하·속도 방지) ──
        # US 스냅샷엔 거래량이 없어 거래량/RVOL 사전필터가 불가능 → 시총·주가만으로
        # 거른 뒤에도 후보가 수천 개 남는다. 따라서 유동성 프록시(US=시총, KR=거래대금)
        # 내림차순으로 정렬해 상위 _DEEP_CANDIDATE_CAP 개만 다운로드/심층분석한다.
        if n_prefiltered > _DEEP_CANDIDATE_WARN:
            _cause = ("US 스냅샷에 거래량이 없어 거래량/RVOL 사전필터를 적용하지 못함"
                      " (시총·주가 범위만 적용 가능)"
                      if market_choice == 'us'
                      else "KR 거래대금·시총·거래량 필터 후에도 후보가 과다함")
            print(f"[스캐너 경고] 심층분석 후보 {n_prefiltered:,}개 > "
                  f"{_DEEP_CANDIDATE_WARN}개 (시장={market_choice}). 원인: {_cause}"
                  f" → 상위 {_DEEP_CANDIDATE_CAP}개로 자동 컷오프.", flush=True)

        def _cand_rank(t):
            # 1순위: 유동성 프록시(KR=거래대금, US=시총). US 스냅샷이 열화(FDR 폴백)되어
            # 시총이 비면 2순위 |등락률|로 의미 있는 순위를 유지한다.
            _s = t[3]
            _primary = (_s.get('amount') or 0.0) if t[2] == 'kr' else (_s.get('mcap') or 0.0)
            _secondary = abs(_s.get('change_pct') or 0.0)
            return (_primary, _secondary)

        if n_prefiltered > _DEEP_CANDIDATE_CAP:
            prelim.sort(key=_cand_rank, reverse=True)
            c_capcut = n_prefiltered - _DEEP_CANDIDATE_CAP
            prelim   = prelim[:_DEEP_CANDIDATE_CAP]

        n_deep = len(prelim)   # 실제 심층분석(다운로드) 대상 수
        print(f"[스캐너] 심층분석 후보: {n_deep:,}개 "
              f"(기본필터 통과 {n_prefiltered:,}개"
              + (f", 상한 초과 {c_capcut:,}개 컷오프)" if c_capcut else ")"),
              flush=True)

        stage_t['스냅샷·사전필터'] = time.time() - _t_a0
        _t_b0 = time.time()
        _interim("심층 후보")

        # ── Stage B: 벌크 히스토리 다운로드 (시장별, ~10종목/초) ──
        us_codes = [c for c, _n, m, _s in prelim if m == 'us']
        kr_codes = [c for c, _n, m, _s in prelim if m == 'kr']
        kr_suffix = {c: ('KS' if (s.get('market') == 'KOSPI') else 'KQ')
                     for c, _n, m, s in prelim if m == 'kr'}

        hist_map: dict[str, pd.DataFrame] = {}

        def _prog_dl(done, tot, base, span):
            try:
                prog_bar.progress(min(base + span * done / max(tot, 1), 1.0))
            except Exception:
                pass

        try:
            if us_codes:
                hist_map.update(bulk_history(
                    us_codes, 'us',
                    progress=lambda d, t: _prog_dl(d, t, 0.0, 0.55 if kr_codes else 1.0),
                ))
            if kr_codes:
                hist_map.update(bulk_history(
                    kr_codes, 'kr', kr_suffix=kr_suffix,
                    progress=lambda d, t: _prog_dl(d, t, 0.55 if us_codes else 0.0,
                                                   0.45 if us_codes else 1.0),
                ))
        except Exception as e:
            funnel_box.error(f"히스토리 다운로드 오류: {e}")

        survivors = [(c, n, m, s) for c, n, m, s in prelim if c in hist_map]
        c_nodata  = len(prelim) - len(survivors)

        stage_t['데이터 다운로드'] = time.time() - _t_b0
        _t_c0 = time.time()

        # ── Stage C: _build_result → 거래대금(US) → 정배열 → AI 심층 분석 ──
        results: list[dict] = []
        with_patterns = 0
        s2_total = len(survivors)
        _ind_sec = 0.0   # 지표 계산 누적 시간
        _pat_sec = 0.0   # 패턴 탐지(AI 점수) 누적 시간
        # ── 심층분석 시작 직전 로그 (요구사항: 통과/대상 수 명시) ──
        print(f"[스캐너] 필터 통과: {n_prefiltered:,}개", flush=True)
        print(f"[스캐너] 심층분석 대상: {s2_total:,}개 "
              f"(상한 {_DEEP_CANDIDATE_CAP} · 다운로드 성공분)", flush=True)
        for i, (code, name, mkt, snap) in enumerate(survivors):
            try:
                if i % 50 == 0:
                    _interim("심층 분석 중")
                hist = hist_map.get(code)
                raw  = _build_result(hist, [])   # 뉴스는 최종 통과 종목만 나중에 수집
                if raw is None:
                    c_nodata += 1
                    continue
                # US 거래대금 필터 (스냅샷에 거래량이 없어 히스토리로 계산)
                if mkt == 'us':
                    if len(hist) >= 5:
                        t5 = hist.tail(5)
                        avg_vol = float(t5['Volume'].mean())
                        avg_val = float((t5['Close'] * t5['Volume']).mean())
                    else:
                        avg_vol = avg_val = 0.0
                    # Stage 1: 평균 거래량 10만주 미만 제외
                    if avg_vol < _MIN_AVG_VOLUME:
                        c_vol += 1
                        continue
                    if avg_val * _KRW_PER_USD < _MIN_DAILY_VALUE_KRW:
                        c_liq += 1
                        continue
                # 추세(정배열 MA20>MA60>MA120) 필터
                if not _stage1_pass(raw):
                    c_trend += 1
                    continue
                # 메타 부여
                _m = kr_meta.get(code, {}) if mkt == 'kr' else {}
                raw['company_name'] = name
                if mkt == 'us':
                    raw['sector']   = ''
                    raw['industry'] = us_ind.get(code, '')
                else:
                    raw['sector']   = _m.get('market', '')
                    raw['industry'] = _m.get('industry', '')

                _ti0 = time.time()
                data   = add_indicators(raw)
                _ind_sec += time.time() - _ti0
                _tp0 = time.time()
                scored = calculate_ai_score(data)
                _pat_sec += time.time() - _tp0
                if any(isinstance(v, (tuple, list)) and v and v[0]
                       for v in scored.get('patterns', {}).values()):
                    with_patterns += 1
                # 금융권 제외
                if _is_financial(data.get('sector', ''), data.get('industry', '')):
                    c_fin += 1
                    continue
                mom = _calc_momentum(hist)
                if len(hist) >= 5:
                    t5 = hist.tail(5)
                    tv = float((t5['Close'] * t5['Volume']).mean())
                else:
                    tv = 0.0
                _disp_px, _disp_chg = display_quote(
                    mkt,
                    live_price=snap.get('price'),
                    live_change=snap.get('change_pct'),
                    hist_close=data.get('price'),
                    hist_change=data.get('change_pct'),
                )
                item = {
                    'code': code, 'name': name, 'market': mkt,
                    'momentum':        mom,
                    'kr_market':       _m.get('market', ''),
                    'kr_industry':     _m.get('industry', ''),
                    'trading_value':   tv,
                    # 표시 시세는 공통 엔진(display_quote)으로 일원화한다 — US는 라이브
                    # 스냅샷(lastsale) 우선, KR은 항상 히스토리(yfinance) 종가로 폴백.
                    'session_price':   _disp_px,
                    'session_change':  _disp_chg,
                    **data, **scored,
                }
                try:
                    item['_research'] = build_research_view(item, item)
                except Exception:
                    c_fail += 1
                    continue
                results.append(item)
            except Exception:
                c_fail += 1
                continue

        stage_t['지표 계산'] = _ind_sec
        stage_t['패턴 탐지'] = _pat_sec
        _t_n0 = time.time()

        # 메모리 정리: 히스토리 맵은 분석이 끝나면 더 필요 없다 → 즉시 해제.
        hist_map.clear()
        gc.collect()

        # ── 뉴스 수집: 최종 결과 상위 ~100종목만 (속도 보존) ──
        results.sort(key=lambda x: (x.get('ma300_strategy', False),
                                    x.get('score', 0)), reverse=True)

        # ── 최종 출력 상한: 점수순 상위 _MAX_OUTPUT 개만 (자동 컷오프) ──
        _n_scored = len(results)
        if _n_scored > _MAX_OUTPUT:
            results = results[:_MAX_OUTPUT]
        print(f"[스캐너] 최종 출력: {len(results):,}개 "
              f"(점수통과 {_n_scored:,}개"
              + (f" → 상위 {_MAX_OUTPUT}개 컷오프)" if _n_scored > _MAX_OUTPUT else ")"),
              flush=True)

        def _fetch_news_for(it):
            try:
                if it['market'] == 'us':
                    return it['code'], _fetch_us_news(it['code'])
                return it['code'], _fetch_naver_news(it['code'])
            except Exception:
                return it['code'], []

        news_targets = results   # 이미 ≤50개 (최종 출력 상한)
        if news_targets:
            # 안전 병렬: 전체 마감시한 + shutdown(wait=False)로 무한 로딩/멈춤 방지.
            # 속도 우선 — 워커↑·마감시한↓(10~20초 예산 보호). 시한 내 미수신 종목은 뉴스 없음.
            news_map = {}
            for _it, pair in gather_parallel(_fetch_news_for, news_targets,
                                             max_workers=12, deadline_sec=12.0):
                if pair:
                    news_map[pair[0]] = pair[1]
            for it in results:
                it['news'] = news_map.get(it['code'], [])
        else:
            for it in results:
                it.setdefault('news', [])

        stage_t['뉴스 수집'] = time.time() - _t_n0
        total_elapsed = time.time() - start_time
        # ── 스캔 요약 로그 (단계별 종목 수 + 소요 시간) ──
        _stage_log = " · ".join(f"{k} {v:.1f}s" for k, v in stage_t.items())
        print(
            f"[스캐너 요약] 전체 {c_total:,} → 기본필터 통과 {n_prefiltered:,} → "
            f"심층분석 {n_deep:,} → 최종 출력 {len(results):,}개 | "
            f"총 {total_elapsed:.1f}s ({_stage_log})", flush=True)
        gc.collect()
        try:
            prog_bar.empty()
        except Exception:
            pass

        funnel_stages = [
            ("총 스캔",            c_total,        "total"),
            ("ETF/SPAC 제외",      c_etf + c_spac, "reject"),
            ("저가($2↓) 제외",     c_cheap,        "reject"),
            ("시총($100M↓) 제외",  c_mcap,         "reject"),
            (f"심층분석 상한({_DEEP_CANDIDATE_CAP}) 초과", c_capcut, "reject"),
            ("데이터 없음",         c_nodata,       "reject"),
            ("거래량(10만↓) 제외", c_vol,          "reject"),
            ("거래대금 제외",       c_liq,          "reject"),
            ("추세(정배열) 제외",   c_trend,        "reject"),
            ("금융 제외",           c_fin,          "reject"),
            ("분석 실패 제외",      c_fail,         "reject"),
            ("최종 통과",           len(results),   "pass"),
        ]
        funnel_box.empty()

        # 데이터 소스 표기 (실제 사용 중인 시세 출처)
        if market_choice == 'us':
            _data_source = ("FinanceDataReader 폴백 (NASDAQ 401 차단)"
                            if us_snapshot_is_degraded(us_snap)
                            else "NASDAQ 스크리너 (세션 스냅샷)")
        else:
            _data_source = "FinanceDataReader (KRX 스냅샷)"

        benchmark = {
            'total': c_total, 'elapsed': total_elapsed,
            'universe': c_total, 'filtered': n_deep, 'final': len(results),
            'prefilter_pass': n_prefiltered, 'deep_candidates': n_deep,
            'capcut_excluded': c_capcut,
            'etf_excluded': c_etf, 'spac_excluded': c_spac,
            'price_excluded': c_cheap, 'mcap_excluded': c_mcap,
            'volume_excluded': c_vol, 'liquidity_excluded': c_liq,
            'null_count': c_nodata, 'trend_excluded': c_trend,
            'financial_excluded': c_fin, 'fail_excluded': c_fail,
            'stage2_pass': len(results), 'with_patterns': with_patterns,
            'stage_times': stage_t,
            'universe_load': _universe_load_sec,
            'download_sec': stage_t.get('데이터 다운로드', 0.0),
            'indicator_sec': _ind_sec, 'pattern_sec': _pat_sec,
            'data_source': _data_source, 'updated_at': time.time(),
        }

        results = _dedup_by_ticker(results)
        _entry = {
            "results":   results,
            "funnel":    funnel_stages,
            "benchmark": benchmark,
            "ts":        time.time(),
        }
        _scan_cache[market_choice] = _entry

    # (여기 도달 = 방금 스캔했거나 10분 내 캐시가 존재)
    results        = _entry["results"]
    funnel_stages  = _entry["funnel"]
    benchmark      = _entry["benchmark"]
    st.session_state["scanner_results"]        = results
    st.session_state["scanner_funnel"]         = funnel_stages
    st.session_state["scanner_funnel_elapsed"] = benchmark.get("elapsed")
    st.session_state["scanner_benchmark"]      = benchmark
    # 세션에 이전(중복 제거 전) 결과가 남아있을 수 있어 방어적으로 한 번 더 적용
    results = _dedup_by_ticker(results)

    # ── 화면 출력 직전 현재가 재조회 (최종 결과 종목만 실시간) ──────────────────
    # 7천 종목 전체 실시간은 포기하되, 최종 통과 종목만 단일 벌크 일봉으로 다시 받아
    # 표시 시세를 최신화한다(스캔 캐시가 10분 지나도 카드 가격은 항상 최신).
    _refresh_display_quotes(results)

    # ── 결과 필터 적용 (디스플레이 레벨) ───────────────────────────────────────
    theme_sel   = st.session_state.get("theme_multisel", [])
    grade_sel   = st.session_state.get("grade_multisel", [])
    pattern_sel = st.session_state.get("pattern_multisel", [])
    filtered = [
        r for r in results
        if _match_theme_filter(r, theme_sel)
        and _match_grade_filter(r, grade_sel)
        and _match_pattern_filter(r, pattern_sel)
    ]

    # ── 필터 전/후 종목 수 + 진단 로그 ─────────────────────────────────────────
    # results = 스캔을 통과한 *최종* 종목(보통 수십~수백). 필터는 이 최종 집합에만
    # 적용되며, 유니버스(수천 종목)를 재렌더링하지 않는다. (요구 4·6)
    _n_before = len(results)
    _n_after  = len(filtered)
    _filter_names: list[str] = []
    if theme_sel:
        _filter_names += [theme_display(t) for t in theme_sel]
    if grade_sel:
        _filter_names += [f"등급:{g}" for g in grade_sel]
    if pattern_sel:
        _filter_names += list(pattern_sel)
    # 워크플로 콘솔 로그(요구 1): "[스캐너 결과 필터] 컵앤핸들 형성 → 23/210개"
    if _filter_names:
        print(f"[스캐너 결과 필터] {' + '.join(_filter_names)} "
              f"→ {_n_after}/{_n_before}개", flush=True)

    _funnel = st.session_state.get("scanner_funnel")
    if _funnel:
        render_stage_funnel(
            _funnel,
            title="단계별 필터 현황",
            elapsed=st.session_state.get("scanner_funnel_elapsed"),
        )
    _bench = st.session_state.get("scanner_benchmark") or {}
    if _bench.get("universe") is not None:
        _mc1, _mc2, _mc3 = st.columns(3)
        _mc1.metric("유니버스", f"{_bench.get('universe', 0):,}")
        _mc2.metric("심층분석 대상", f"{_bench.get('deep_candidates', _bench.get('filtered', 0)):,}")
        _mc3.metric("최종 출력", f"{_bench.get('final', 0):,}")
        # 성능 분석: 단계별 소요 시간 (요구 B)
        _p1, _p2, _p3, _p4, _p5 = st.columns(5)
        _p1.metric("유니버스 로딩", f"{_bench.get('universe_load', 0):.1f}s")
        _p2.metric("데이터 다운로드", f"{_bench.get('download_sec', 0):.1f}s")
        _p3.metric("지표 계산", f"{_bench.get('indicator_sec', 0):.1f}s")
        _p4.metric("패턴 탐지", f"{_bench.get('pattern_sec', 0):.1f}s")
        _p5.metric("전체 스캔", f"{_bench.get('elapsed', 0):.1f}s")
    # 데이터 소스 + 최종 업데이트 시각 (요구 E)
    _src = _bench.get("data_source")
    _upd = _bench.get("updated_at")
    if _src or _upd:
        import datetime as _dtmod
        _upd_str = (_dtmod.datetime.fromtimestamp(_upd).strftime("%Y-%m-%d %H:%M:%S")
                    if _upd else "—")
        st.caption(
            f"🛰️ 데이터 소스: **{_html.escape(str(_src or '—'))}**  ·  "
            f"🕒 최종 업데이트: {_upd_str}  ·  "
            f"📈 시세 기준: 세션 스냅샷(lastsale)/일봉 종가 — "
            f"브로커 프리마켓 호가와 다를 수 있습니다(무료 데이터 한계)."
        )
    st.divider()

    # ── 검색 결과 헤더 (필터 전/후 종목 수 항상 표시 — 요구 6) ───────────────────
    if _filter_names:
        _filter_summary = _html.escape(" + ".join(_filter_names))
        st.markdown(
            f"<div style='background:#1A1A2E;border-left:3px solid #5C6BC0;"
            f"padding:10px 14px;border-radius:6px;margin-bottom:12px;'>"
            f"<span style='color:#8888AA;font-size:0.82rem;'>적용 필터</span>&nbsp;&nbsp;"
            f"<span style='color:#C5CAE9;font-size:0.88rem;font-weight:600;'>{_filter_summary}</span>"
            f"&nbsp;&nbsp;<span style='color:#7986CB;font-size:0.88rem;'>|</span>&nbsp;&nbsp;"
            f"<span style='color:#E0E0FF;font-size:0.95rem;font-weight:700;'>"
            f"필터 전 {_n_before:,}개 → 필터 후 {_n_after:,}개</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    elif results:
        st.markdown(
            f"<div style='background:#1A1A2E;border-left:3px solid #5C6BC0;"
            f"padding:10px 14px;border-radius:6px;margin-bottom:12px;'>"
            f"<span style='color:#8888AA;font-size:0.82rem;'>전체 결과 (필터 없음)</span>&nbsp;&nbsp;"
            f"<span style='color:#E0E0FF;font-size:0.95rem;font-weight:700;'>"
            f"필터 전 {_n_before:,}개 → 필터 후 {_n_after:,}개</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    if not filtered:
        if not results:
            st.info("조건에 맞는 종목이 없습니다. MA20 > MA60 > MA120 정배열 종목이 없거나 거래대금 조건을 확인하세요.")
        else:
            _names = " + ".join(_filter_names) if _filter_names else "선택한 필터"
            st.warning(
                f"**조건에 맞는 종목 없음 (0개)**\n\n"
                f"'{_names}' 에 해당하는 종목이 최종 결과 {_n_before:,}개 중 없습니다.\n\n"
                "다른 패턴·시장을 선택하거나 사이드바의 **🔁 필터 초기화** 버튼을 눌러보세요."
            )
        return

    # 렌더링 투명성(요구 4): 유니버스가 아니라 최종 통과 종목만 카드로 그린다.
    st.caption(f"🃏 카드 렌더링: {_n_after:,}개 (최종 결과 {_n_before:,}개 기준 · 유니버스 재렌더링 아님)")

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

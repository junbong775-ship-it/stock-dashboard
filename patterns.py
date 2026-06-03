import pandas as pd
import numpy as np


def detect_cup_and_handle(df: pd.DataFrame) -> tuple[bool, float, str]:
    """Rounded bottom (cup) followed by short consolidation (handle)."""
    close = df['Close'].values
    if len(close) < 60:
        return False, 0.0, "데이터 부족"

    window = close[-60:]
    left_peak  = window[:10].max()
    cup_bottom = window[10:50].min()
    right_peak = window[50:].max()
    handle_low = window[55:].min() if len(window) >= 60 else window[-5:].min()

    depth          = (left_peak - cup_bottom) / left_peak if left_peak > 0 else 0
    symmetry       = abs(right_peak - left_peak) / left_peak if left_peak > 0 else 1.0
    handle_retrace = (right_peak - handle_low) / right_peak if right_peak > 0 else 1.0

    detected = (
        depth >= 0.08 and
        symmetry <= 0.20 and
        handle_retrace <= 0.12 and
        right_peak >= left_peak * 0.90
    )
    confidence  = min(1.0, depth * 3) * (1 - symmetry) if detected else 0.0
    description = "컵앤핸들 패턴 감지 — 저점 반등 후 짧은 횡보 구간 확인" if detected else "패턴 없음"
    return detected, round(confidence, 2), description


def detect_double_bottom(df: pd.DataFrame) -> tuple[bool, float, str]:
    """Two price troughs at similar levels with a middle peak."""
    close = df['Close'].values
    if len(close) < 40:
        return False, 0.0, "데이터 부족"

    window = close[-40:]
    left   = window[:15]
    mid    = window[15:25]
    right  = window[25:]

    trough1    = left.min()
    trough2    = right.min()
    midpeak    = mid.max()
    avg_trough = (trough1 + trough2) / 2
    similarity = abs(trough1 - trough2) / avg_trough if avg_trough > 0 else 1.0
    peak_rise  = (midpeak - avg_trough) / avg_trough if avg_trough > 0 else 0

    detected = (
        similarity <= 0.10 and
        peak_rise >= 0.03 and
        close[-1] >= midpeak * 0.93
    )
    confidence  = (1 - similarity * 10) * min(1.0, peak_rise * 10) if detected else 0.0
    description = "더블바텀 패턴 감지 — 두 저점 수렴 후 저항선 돌파" if detected else "패턴 없음"
    return detected, round(max(0.0, confidence), 2), description


def detect_box_breakout(df: pd.DataFrame) -> tuple[bool, float, str]:
    """Price breaking above a recent resistance range with volume confirmation."""
    if len(df) < 30:
        return False, 0.0, "데이터 부족"

    box    = df.iloc[-30:-5]
    recent = df.iloc[-5:]

    resistance  = box['High'].max()
    box_floor   = box['Low'].min()
    box_range   = resistance - box_floor
    current     = float(df['Close'].iloc[-1])
    avg_vol_box = box['Volume'].mean()
    avg_vol_now = recent['Volume'].mean()

    if box_range <= 0 or avg_vol_box <= 0:
        return False, 0.0, "패턴 없음"

    breakout     = current > resistance
    vol_confirm  = avg_vol_now > avg_vol_box * 1.3
    breakout_pct = (current - resistance) / resistance if resistance > 0 else 0

    detected    = breakout and vol_confirm
    confidence  = min(1.0, breakout_pct * 20) if detected else 0.0
    description = (
        f"박스권 돌파 감지 — 저항선 {resistance:,.2f} 상향 돌파, "
        f"거래량 {avg_vol_now/avg_vol_box:.1f}배"
    ) if detected else "패턴 없음"
    return detected, round(confidence, 2), description


def detect_golden_cross(df: pd.DataFrame) -> tuple[bool, float, str]:
    """MA20 crossing above MA60 within the last 20 bars."""
    if 'MA20' not in df.columns or 'MA60' not in df.columns:
        return False, 0.0, "이동평균선 데이터 없음"

    window = df.dropna(subset=['MA20', 'MA60']).tail(25)
    if len(window) < 2:
        return False, 0.0, "데이터 부족"

    ma20 = window['MA20'].values
    ma60 = window['MA60'].values

    crossed  = False
    bars_ago = None
    for i in range(1, min(20, len(ma20))):
        idx = len(ma20) - 1 - i
        if idx < 1:
            break
        if ma20[idx] > ma60[idx] and ma20[idx - 1] <= ma60[idx - 1]:
            crossed  = True
            bars_ago = i
            break

    if not crossed:
        return False, 0.0, "패턴 없음"

    recency_score = (20 - bars_ago) / 20
    gap           = (ma20[-1] - ma60[-1]) / ma60[-1] if ma60[-1] > 0 else 0
    confidence    = recency_score * min(1.0, 1 + gap * 10)
    description   = f"골든크로스 감지 — MA20이 MA60을 {bars_ago}봉 전에 상향 돌파"
    return True, round(min(1.0, confidence), 2), description


def detect_new_high(df: pd.DataFrame) -> tuple[bool, float, str]:
    """최근 52주(최대 250봉) 최고가 돌파."""
    close = df['Close'].values
    if len(close) < 60:
        return False, 0.0, "데이터 부족"

    window   = min(250, len(close) - 1)
    high_52w = close[-window - 1:-1].max()
    current  = close[-1]

    if not (current > high_52w):
        return False, 0.0, "패턴 없음"

    pct        = (current - high_52w) / high_52w
    confidence = min(1.0, pct * 20)
    desc       = f"52주 신고가 돌파 — {high_52w:,.2f} 상향 돌파 ({pct*100:.1f}%↑)"
    return True, round(confidence, 2), desc


TURTLE_WINDOW   = 20      # 터틀 돌파 — 돈치언 상단 기간(직전 20봉 고가)
NEAR_HIGH_BAND  = 0.05    # 신고가 근처 — 52주 고점 대비 허용 폭(5% 이내, 신고가 포함)


def detect_turtle_breakout(df: pd.DataFrame) -> tuple[bool, float, str]:
    """터틀 트레이딩 — 직전 20봉 고가(돈치언 상단) 돌파.

    강세추세 세부패턴(표시 전용). 점수/액션 로직과 무관 — get_all_patterns 에
    포함하지 않고 classify_strong_patterns 에서 직접 평가한다.
    """
    if len(df) < TURTLE_WINDOW + 5:
        return False, 0.0, "데이터 부족"
    high  = (df['High'] if 'High' in df.columns else df['Close']).values
    close = df['Close'].values
    prior_high = float(high[-TURTLE_WINDOW - 1:-1].max())
    current    = float(close[-1])
    if prior_high <= 0 or current < prior_high:
        return False, 0.0, "패턴 없음"
    pct        = (current - prior_high) / prior_high
    confidence = min(1.0, pct * 25 + 0.45)
    desc       = f"터틀 돌파 — 20일 고가 {prior_high:,.2f} 상향 ({pct*100:.1f}%↑)"
    return True, round(confidence, 2), desc


def detect_near_high(df: pd.DataFrame) -> tuple[bool, float, str]:
    """52주(최대 250봉) 최고가 5% 이내 근접(신고가 포함).

    강세추세 세부패턴(표시 전용). 점수/액션 로직과 무관.
    """
    close = df['Close'].values
    if len(close) < 60:
        return False, 0.0, "데이터 부족"
    window   = min(250, len(close))
    high_52w = float(close[-window:].max())
    current  = float(close[-1])
    if high_52w <= 0:
        return False, 0.0, "데이터 없음"
    gap = (high_52w - current) / high_52w   # 0 이면 신고가
    if gap > NEAR_HIGH_BAND:
        return False, 0.0, "패턴 없음"
    confidence = min(1.0, (1.0 - gap / NEAR_HIGH_BAND) * 0.6 + 0.4)
    if current >= high_52w:
        desc = f"52주 신고가 도달 — {high_52w:,.2f}"
    else:
        desc = f"52주 신고가 근처 — 고점 대비 -{gap*100:.1f}%"
    return True, round(confidence, 2), desc


def detect_volume_surge(df: pd.DataFrame) -> tuple[bool, float, str]:
    """최근 20일 평균 거래량 대비 150% 이상 급증."""
    if 'Volume' not in df.columns or len(df) < 22:
        return False, 0.0, "데이터 부족"

    vol        = df['Volume'].values
    avg_vol20  = vol[-21:-1].mean()
    today_vol  = float(vol[-1])

    if avg_vol20 <= 0:
        return False, 0.0, "거래량 데이터 없음"

    ratio    = today_vol / avg_vol20
    detected = ratio >= 1.5

    if not detected:
        return False, 0.0, "패턴 없음"

    confidence = min(1.0, (ratio - 1.5) / 2.5 + 0.4)
    desc       = f"거래량 급증 — 20일 평균 대비 {ratio:.1f}배 ({ratio*100:.0f}%)"
    return True, round(confidence, 2), desc


def detect_ma10_pullback(df: pd.DataFrame) -> tuple[bool, float, str]:
    """MA10 > MA20 > MA60 정배열 + 현재가 MA10 ±3% 눌림목 진입."""
    for col in ('MA10', 'MA20', 'MA60'):
        if col not in df.columns:
            return False, 0.0, "이동평균선 데이터 없음"

    window = df.dropna(subset=['MA10', 'MA20', 'MA60']).tail(15)
    if len(window) < 5:
        return False, 0.0, "데이터 부족"

    ma10    = float(window['MA10'].iloc[-1])
    ma20    = float(window['MA20'].iloc[-1])
    ma60    = float(window['MA60'].iloc[-1])
    current = float(window['Close'].iloc[-1])

    if not (ma10 > ma20 > ma60):
        return False, 0.0, "패턴 없음"

    deviation = abs(current - ma10) / ma10 if ma10 > 0 else 1.0
    if deviation > 0.03:
        return False, 0.0, "패턴 없음"

    # 최근 3~5일 내 고점 대비 하락 (눌림목)
    recent_closes = window['Close'].values
    peak_5d = float(recent_closes[:-1][-5:].max()) if len(recent_closes) >= 6 else float(recent_closes[:-1].max())
    if current >= peak_5d:
        return False, 0.0, "패턴 없음"

    pullback_pct = (peak_5d - current) / peak_5d if peak_5d > 0 else 0
    if pullback_pct < 0.005:
        return False, 0.0, "패턴 없음"

    # 거래량 감소 확인 (최근 3일 < 10일 평균)
    vol_declining = False
    if 'Volume' in window.columns and len(window) >= 10:
        avg_recent = float(window['Volume'].tail(3).mean())
        avg_10d    = float(window['Volume'].tail(10).mean())
        vol_declining = avg_10d > 0 and avg_recent < avg_10d * 0.90

    proximity_score = (0.03 - deviation) / 0.03
    confidence = min(1.0,
        proximity_score * 0.55
        + (0.25 if vol_declining else 0.0)
        + min(pullback_pct * 8, 0.20)
    )
    desc = (
        f"MA10 눌림목 — 정배열(MA10>MA20>MA60), "
        f"MA10 대비 {deviation*100:.1f}% 이내"
        + (", 거래량 감소 확인" if vol_declining else "")
    )
    return True, round(confidence, 2), desc


def detect_ma20_pullback(df: pd.DataFrame) -> tuple[bool, float, str]:
    """MA20 > MA60 > MA120 정배열 + 현재가 MA20 ±5% 눌림목 진입."""
    for col in ('MA20', 'MA60', 'MA120'):
        if col not in df.columns:
            return False, 0.0, "이동평균선 데이터 없음"

    window = df.dropna(subset=['MA20', 'MA60', 'MA120']).tail(20)
    if len(window) < 10:
        return False, 0.0, "데이터 부족"

    ma20    = float(window['MA20'].iloc[-1])
    ma60    = float(window['MA60'].iloc[-1])
    ma120   = float(window['MA120'].iloc[-1])
    current = float(window['Close'].iloc[-1])

    if not (ma20 > ma60 > ma120):
        return False, 0.0, "패턴 없음"

    deviation = abs(current - ma20) / ma20 if ma20 > 0 else 1.0
    if deviation > 0.05:
        return False, 0.0, "패턴 없음"

    # 최근 5~10일 내 고점 대비 하락
    recent_closes = window['Close'].values
    peak_10d = float(recent_closes[:-1][-10:].max()) if len(recent_closes) >= 11 else float(recent_closes[:-1].max())
    if current >= peak_10d:
        return False, 0.0, "패턴 없음"

    pullback_pct = (peak_10d - current) / peak_10d if peak_10d > 0 else 0
    if pullback_pct < 0.01:
        return False, 0.0, "패턴 없음"

    # 거래량 감소 확인 (최근 5일 < 20일 평균)
    vol_declining = False
    if 'Volume' in window.columns and len(window) >= 20:
        avg_recent = float(window['Volume'].tail(5).mean())
        avg_20d    = float(window['Volume'].tail(20).mean())
        vol_declining = avg_20d > 0 and avg_recent < avg_20d * 0.90

    proximity_score = (0.05 - deviation) / 0.05
    confidence = min(1.0,
        proximity_score * 0.55
        + (0.25 if vol_declining else 0.0)
        + min(pullback_pct * 4, 0.20)
    )
    desc = (
        f"MA20 눌림목 — 정배열(MA20>MA60>MA120), "
        f"MA20 대비 {deviation*100:.1f}% 이내"
        + (", 거래량 감소 확인" if vol_declining else "")
    )
    return True, round(confidence, 2), desc


# ──────────────────────────────────────────────────────────────────────────────
# 바닥탈출 셋업 — '패턴 형성 단계'와 '돌파 직전/직후(5% 이내)' 구간 분류
# ──────────────────────────────────────────────────────────────────────────────
# 핵심 원칙: 패턴의 "존재 여부"가 아니라 "형성 단계 + 돌파 직후 구간"을 찾는다.
#   1순위 FORMING   — 더블바텀/컵앤핸들 형성 중, 박스권 상단 5% 이내
#   2순위 BREAKOUT  — 목선/컵림/박스상단 돌파 후 5% 이내
#   제외   EXTENDED  — 돌파 후 5% 초과(이미 상승 진행) → 바닥탈출 아님(강세추세로)
BOTTOM_BREAKOUT_BAND = 0.05     # 돌파 직후/상단 근접 허용 폭 (±5%)

# ── 바닥탈출 유효 조건(이미 크게 상승한 종목 제외) ────────────────────────────
#   ① MA300 이격도 +25% 초과         → 바닥탈출 제외(강세추세로)
#   ② 패턴 돌파 후 상승폭 +20% 초과   → 바닥탈출 아닌 강세추세로 재분류
#   ③ 돌파선 대비 +15% 이상           → 바닥탈출 점수 제거(강세추세 자격은 별도 판정)
#   ④ 유지 상태: 형성중(FORMING)·돌파직전(FORMING)·돌파완료 초기(BREAKOUT ≤+5%)
BOTTOM_MA300_DEV_MAX   = 0.25   # ① MA300 이격도 상한
BOTTOM_STRONG_PCT      = 0.20   # ② 돌파 후 상승폭 — 초과 시 강세추세 재분류
BOTTOM_SCORE_DROP_PCT  = 0.15   # ③ 돌파선 대비 상승폭 — 이상이면 바닥탈출 점수 제거


def _ma300_dev(df: pd.DataFrame, close: float) -> float | None:
    """현재가의 MA300 이격도(소수)를 반환한다. MA300 부재/무효 시 None."""
    if 'MA300' not in df.columns:
        return None
    try:
        ma300 = float(df['MA300'].iloc[-1])
    except Exception:
        return None
    if not np.isfinite(ma300) or ma300 <= 0:
        return None
    return (close - ma300) / ma300


def _phase_from_level(close: float, level: float) -> tuple[str | None, float]:
    """현재가와 핵심 돌파 레벨(목선/컵림/박스상단)의 관계로 국면을 분류한다.

    반환: (phase, breakout_pct) — phase ∈ {FORMING, BREAKOUT, EXTENDED, None}
      FORMING  : 레벨 아래(아직 미돌파)
      BREAKOUT : 레벨 돌파 후 +5% 이내
      EXTENDED : 레벨 돌파 후 +5% 초과(상승 상당 진행)
    """
    if not (np.isfinite(level) and np.isfinite(close)) or level <= 0:
        return None, 0.0
    pct = (close - level) / level
    if pct < 0:
        return 'FORMING', pct
    if pct <= BOTTOM_BREAKOUT_BAND:
        return 'BREAKOUT', pct
    return 'EXTENDED', pct


def detect_bottom_setup(df: pd.DataFrame) -> dict:
    """바닥탈출 후보를 형성 단계/돌파 직후(5% 이내) 중심으로 분류한다.

    더블바텀·컵앤핸들·박스권 세 구조의 '핵심 돌파 레벨'(목선/컵림/박스상단) 대비
    현재가 위치로 국면을 판정한다. 형성 중(1순위) 또는 돌파 직후 5% 이내(2순위)만
    바닥탈출로 인정하고, 돌파 후 5% 초과는 extended=True 로 표시(→ 강세추세).
    """
    none = {
        'qualifies': False, 'priority': None, 'phase': None,
        'pattern': None, 'breakout_pct': 0.0, 'extended': False, 'desc': '패턴 없음',
    }
    close_arr = df['Close'].values
    if len(close_arr) < 40:
        return none
    close = float(close_arr[-1])

    candidates: list[dict] = []   # FORMING/BREAKOUT 후보
    extended = False              # 돌파 5% 초과 베이스 패턴 존재 여부

    # ── 더블바텀: 핵심 레벨 = 목선(중앙 고점) ────────────────────────────
    w = close_arr[-40:]
    trough1 = float(w[:15].min())
    trough2 = float(w[25:].min())
    neckline = float(w[15:25].max())
    avg_trough = (trough1 + trough2) / 2
    if avg_trough > 0 and neckline > 0:
        similarity = abs(trough1 - trough2) / avg_trough
        peak_rise  = (neckline - avg_trough) / avg_trough
        # 두 저점 수렴 + 중앙 반등 + 둘째 저점에서 회복 중
        if similarity <= 0.10 and peak_rise >= 0.03 and close > trough2:
            phase, pct = _phase_from_level(close, neckline)
            if phase == 'EXTENDED':
                extended = True
            elif phase in ('FORMING', 'BREAKOUT'):
                candidates.append({
                    'pattern': 'double_bottom', 'phase': phase, 'breakout_pct': pct,
                    'priority': 1 if phase == 'FORMING' else 2,
                    'desc': (f"더블바텀 형성 중 — 목선까지 {abs(pct)*100:.1f}%"
                             if phase == 'FORMING'
                             else f"더블바텀 목선 돌파 직후 (+{pct*100:.1f}%)"),
                })

    # ── 컵앤핸들: 핵심 레벨 = 컵 림(좌/우 고점) ──────────────────────────
    if len(close_arr) >= 60:
        cw = close_arr[-60:]
        left_peak  = float(cw[:10].max())
        cup_bottom = float(cw[10:50].min())
        right_peak = float(cw[50:].max())
        handle_low = float(cw[55:].min())
        rim = max(left_peak, right_peak)
        if left_peak > 0 and rim > 0:
            depth          = (left_peak - cup_bottom) / left_peak
            symmetry       = abs(right_peak - left_peak) / left_peak
            handle_retrace = (right_peak - handle_low) / right_peak if right_peak > 0 else 1.0
            if depth >= 0.08 and symmetry <= 0.20 and handle_retrace <= 0.12:
                phase, pct = _phase_from_level(close, rim)
                if phase == 'EXTENDED':
                    extended = True
                elif phase in ('FORMING', 'BREAKOUT'):
                    candidates.append({
                        'pattern': 'cup_and_handle', 'phase': phase, 'breakout_pct': pct,
                        'priority': 1 if phase == 'FORMING' else 2,
                        'desc': (f"컵앤핸들 형성 중 — 컵 림까지 {abs(pct)*100:.1f}%"
                                 if phase == 'FORMING'
                                 else f"컵앤핸들 돌파 직후 (+{pct*100:.1f}%)"),
                    })

    # ── 박스권: 핵심 레벨 = 박스 상단(저항) ──────────────────────────────
    if len(df) >= 30:
        box = df.iloc[-30:-5]
        resistance = float(box['High'].max())
        box_floor  = float(box['Low'].min())
        box_range  = resistance - box_floor
        if resistance > 0 and box_range > 0:
            range_pct = box_range / resistance
            # 진짜 횡보 박스(폭 3~35%) + 박스 하단 위
            if 0.03 <= range_pct <= 0.35 and close >= box_floor:
                phase, pct = _phase_from_level(close, resistance)
                if phase == 'EXTENDED':
                    extended = True
                elif phase == 'BREAKOUT':
                    candidates.append({
                        'pattern': 'box', 'phase': 'BREAKOUT', 'breakout_pct': pct,
                        'priority': 2,
                        'desc': f"박스권 상단 돌파 직후 (+{pct*100:.1f}%)",
                    })
                elif phase == 'FORMING' and pct >= -BOTTOM_BREAKOUT_BAND:
                    # 박스권은 '상단 5% 이내'만 1순위 (하단 횡보는 제외)
                    candidates.append({
                        'pattern': 'box', 'phase': 'FORMING', 'breakout_pct': pct,
                        'priority': 1,
                        'desc': f"박스권 상단 근접 — 저항까지 {abs(pct)*100:.1f}%",
                    })

    if not candidates:
        return {
            **none, 'extended': extended,
            'desc': '돌파 후 5% 초과 (강세추세)' if extended else '패턴 없음',
        }

    # 1순위(형성 중) 우선, 동순위는 돌파 레벨에 가장 근접한 후보
    candidates.sort(key=lambda c: (c['priority'], abs(c['breakout_pct'])))
    best = candidates[0]

    # ── 바닥탈출 유효 조건 — 이미 크게 상승한 종목은 강세추세로 보냄 ──────────
    ma300_dev = _ma300_dev(df, close)
    bpct      = best['breakout_pct']
    # ①·② MA300 이격도 +25% 초과 또는 돌파 후 +20% 초과 → 강세추세로 재분류
    if (ma300_dev is not None and ma300_dev > BOTTOM_MA300_DEV_MAX) or bpct > BOTTOM_STRONG_PCT:
        if ma300_dev is not None and ma300_dev > BOTTOM_MA300_DEV_MAX:
            reason = f'MA300 이격 +{ma300_dev*100:.0f}% (강세추세)'
        else:
            reason = f'돌파 후 +{bpct*100:.0f}% (강세추세)'
        return {**none, 'extended': True, 'desc': reason}
    # ③ 돌파선 대비 +15% 이상 → 바닥탈출 점수 제거(강세추세 자격은 classify에서 별도 판정)
    if bpct >= BOTTOM_SCORE_DROP_PCT:
        return {**none, 'extended': extended, 'desc': f'돌파 후 +{bpct*100:.0f}% (바닥탈출 제외)'}

    return {
        'qualifies': True,
        'priority': best['priority'],
        'phase': best['phase'],
        'pattern': best['pattern'],
        'breakout_pct': best['breakout_pct'],
        'extended': extended,
        'desc': best['desc'],
    }


PATTERN_CONF_THRESHOLD = 0.30   # 이 값 미만 신뢰도 → 미감지 처리


def _apply_conf_threshold(
    detected: bool, conf: float, desc: str
) -> tuple[bool, float, str]:
    """신뢰도 30% 미만이면 detected=False 로 다운그레이드."""
    if detected and conf < PATTERN_CONF_THRESHOLD:
        return False, conf, f"신뢰도 부족 ({conf*100:.0f}%)"
    return detected, conf, desc


def get_all_patterns(df: pd.DataFrame) -> dict:
    # ① 개별 패턴 독립 평가
    cup_and_handle  = detect_cup_and_handle(df)
    double_bottom   = detect_double_bottom(df)
    box_breakout    = detect_box_breakout(df)
    golden_cross    = detect_golden_cross(df)
    new_high        = detect_new_high(df)
    volume_surge    = detect_volume_surge(df)
    ma10_pullback   = detect_ma10_pullback(df)
    ma20_pullback   = detect_ma20_pullback(df)

    # ② 신뢰도 30% 임계값 일괄 적용
    raw = {
        'new_high':       new_high,
        'volume_surge':   volume_surge,
        'golden_cross':   golden_cross,
        'ma10_pullback':  ma10_pullback,
        'ma20_pullback':  ma20_pullback,
        'cup_and_handle': cup_and_handle,
        'double_bottom':  double_bottom,
        'box_breakout':   box_breakout,
    }
    out = {k: _apply_conf_threshold(*v) for k, v in raw.items()}
    # 바닥탈출 셋업(형성/돌파 국면)은 임계값 처리 대상이 아니라 dict 로 별도 첨부.
    out['bottom_setup'] = detect_bottom_setup(df)
    return out

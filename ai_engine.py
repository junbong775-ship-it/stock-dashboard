import math

from indicators import calc_rsi, calc_macd, calc_mfi
from patterns import get_all_patterns, detect_turtle_breakout, detect_near_high

TARGET_MULT   = 1.30
# 손절가는 구조 기반(스윙로우/MA20/ATR×2)으로 산출한다. STOPLOSS_MULT 는 구조
# 데이터가 전혀 없을 때만 쓰는 최후 폴백(-8%) 이며, 과거의 고정 -18% 손절은 폐기됨.
STOPLOSS_FALLBACK_MULT = 0.92

GRADE_COLOR: dict[str, str] = {
    'S+': '#FFD700',  # 골드 — 최고등급
    'S':  '#00C851',  # 그린
    'A':  '#2196F3',  # 블루
    'B':  '#9E9E9E',  # 그레이
    'C':  '#FF5722',  # 오렌지
    'D':  '#F44336',  # 레드
}

# ── 배점 설계 원칙 ─────────────────────────────────────────────────────────────
# 기본점 최대 (패턴 0개): MA(35) + RSI(20) + MFI(15) + 밀집도(4) = 74점
# S등급 기준 75점 → 패턴 1개 이상 필수 (최소 가중치 8점 → 82점)
# MACD는 독립 항목 제거, 패턴 점수에 흡수 (MACD 골든크로스 = 패턴으로 표현)
# 패턴 가중치: 눌림목(12/10) > 추세돌파(9) > 기술적패턴(8/8/7) > 거래량(6)
# ──────────────────────────────────────────────────────────────────────────────

_PATTERN_WEIGHTS: dict[str, int] = {
    'first_pullback': 12,  # 첫 눌림목 — 저위험 진입, 가장 높은 신뢰
    'new_high':        9,  # 52주 신고가 돌파
    'ma300_breakout':  9,  # 300일선 돌파 시도
    'golden_cross':    8,  # MA20 > MA60 골든크로스
    'ma_convergence':  8,  # 5·10·20일선 수렴
    'cup_and_handle':  8,  # 컵앤핸들 — 중장기 반등
    'double_bottom':   8,  # 더블바텀 — 바닥 반전
}


def _atr(hist, period: int = 14):
    """ATR(평균진폭) — True Range 의 단순이동평균. 데이터 부족 시 None."""
    if not all(c in hist.columns for c in ('High', 'Low', 'Close')):
        return None
    high  = hist['High']
    low   = hist['Low']
    close = hist['Close']
    prev  = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev).abs()
    tr3 = (low - prev).abs()
    tr  = tr1.where(tr1 >= tr2, tr2)
    tr  = tr.where(tr >= tr3, tr3)
    val = tr.rolling(period).mean().dropna()
    return float(val.iloc[-1]) if not val.empty else None


def _recent_swing_low(hist, lookback: int = 30, k: int = 2):
    """최근 스윙로우 — 좌우 k봉보다 낮은 피벗 저점 중 가장 최근 값.
    피벗이 없으면 최근 lookback 구간 최저가로 폴백. 데이터 부족 시 None."""
    if 'Low' not in hist.columns:
        return None
    low = hist['Low'].dropna()
    arr = low.values
    n   = len(arr)
    if n < (2 * k + 1):
        return None
    start = max(k, n - lookback)
    pivot = None
    for i in range(start, n - k):
        left  = arr[i - k:i]
        right = arr[i + 1:i + k + 1]
        if arr[i] < left.min() and arr[i] <= right.min():
            pivot = float(arr[i])   # 가장 최근 피벗으로 갱신
    if pivot is None:
        win = arr[max(0, n - lookback):]
        pivot = float(win.min()) if len(win) else None
    return pivot


def _structure_stop(hist, price: float, ma20):
    """구조 기반 손절가를 산출한다.

    후보(우선순위): ① 최근 스윙로우 · ② MA20 · ③ ATR(14)×2 (price-2·ATR).
    유효 조건: 0 < 값 < 현재가. **우선순위가 높은 순서대로 첫 유효 후보**를 손절가로
    쓴다(스윙로우 → MA20 → ATR×2). 고정 % 손절은 사용하지 않으며, 구조 데이터가
    전혀 없을 때만 -8% 폴백을 쓴다.

    설계 메모: 'MA20' 는 진입 밴드(min(MA10,MA20)~price)의 하단과 사실상 겹쳐
    위험폭이 1% 바닥으로 눌리며 R/R 이 비현실적으로 폭발(10~14:1)한다. 따라서
    스윙로우(진입 밴드보다 충분히 아래)를 1순위로 두어 현실적인 R/R(보통 2~4:1)을
    얻는다 — '현재가에 가장 가까운 값'을 무조건 고르면 항상 MA20 이 선택되어 R/R
    신호가 무력화되므로 명시된 우선순위(1>2>3)를 선택 규칙으로 채택한다.

    반환: (stop_loss, basis)  basis ∈ {swing_low, ma20, atr2, fallback}
    """
    atr      = _atr(hist, 14)
    atr_stop = price - 2 * atr if (atr is not None and math.isfinite(atr)) else None
    candidates = [
        ('swing_low', _recent_swing_low(hist)),
        ('ma20',      ma20),
        ('atr2',      atr_stop),
    ]
    for name, v in candidates:   # 우선순위 순서대로 첫 유효 후보 채택
        if v is not None and math.isfinite(v) and 0 < v < price:
            return float(v), name
    # 폴백 — 유효한 구조 손절이 하나도 없을 때만
    if atr_stop is not None and atr_stop > 0:
        return atr_stop, 'atr2'
    return price * STOPLOSS_FALLBACK_MULT, 'fallback'


def calculate_ai_score(data: dict) -> dict:
    hist   = data['hist']
    rsi    = data.get('rsi', calc_rsi(hist['Close']))
    mfi    = data.get('mfi', 50.0)
    spread = data.get('spread', 0.0)

    # ── MA 정배열 (35점) ──────────────────────────────────────────────────────
    ma20  = float(hist['MA20'].iloc[-1])  if 'MA20'  in hist.columns and not hist['MA20'].isna().all()  else None
    ma60  = float(hist['MA60'].iloc[-1])  if 'MA60'  in hist.columns and not hist['MA60'].isna().all()  else None
    ma120 = float(hist['MA120'].iloc[-1]) if 'MA120' in hist.columns and not hist['MA120'].isna().all() else None
    ma300 = float(hist['MA300'].iloc[-1]) if 'MA300' in hist.columns and not hist['MA300'].isna().all() else None

    if ma20 is not None and ma60 is not None and ma120 is not None:
        ma_score = (35 if ma20 > ma60 > ma120 else
                    20 if (ma20 > ma60 or ma60 > ma120) else
                    0  if ma20 < ma60 < ma120 else 10)
    elif ma20 is not None and ma60 is not None:
        ma_score = (25 if ma20 > ma60 else 10)
    else:
        ma_score = 10

    # ── RSI (최대 20점) ───────────────────────────────────────────────────────
    # 40-70: 추세+눌림목 모멘텀 구간 → 20점 (상향 확대)
    # 35-40 / 70-75: 경계 구간 → 14점
    # 30-35 / 75-80: 침체/과열 경계 → 9점
    # 20-30 / 80-90: 침체/과열 → 4점
    # 그 외: 0점
    rsi_score = (20 if 40   <= rsi <= 70 else
                 14 if (35  <= rsi <  40) or (70  < rsi <= 75) else
                 9  if (30  <= rsi <  35) or (75  < rsi <= 80) else
                 4  if (20  <= rsi <  30) or (80  < rsi <= 90) else 0)

    # ── MFI (최대 15점) ───────────────────────────────────────────────────────
    mfi_score = (15 if 40   <= mfi <= 60 else
                 10 if (30  <= mfi <  40) or (60  < mfi <= 70) else
                 5  if (20  <= mfi <  30) or (70  < mfi <= 80) else 1)

    # ── 밀집도 (최대 4점) — 정보 참고용으로 소폭만 반영 ──────────────────────
    spread_score = (4 if spread <=  3.0 else
                    3 if spread <=  8.0 else
                    2 if spread <= 15.0 else
                    1 if spread <= 30.0 else 0)

    # ── MACD — 독립 항목 제거, 패턴에 흡수 ───────────────────────────────────
    macd_data  = data.get('macd') or calc_macd(hist['Close'])

    # ── 패턴 점수 (패턴 1개당 가중치 합산, 상한 없음 → 100점 cap 적용) ───────
    pattern_data  = get_all_patterns(hist)
    pattern_score = sum(
        _PATTERN_WEIGHTS.get(k, 6)
        for k, v in pattern_data.items()
        if isinstance(v, (tuple, list)) and len(v) == 3 and v[0]
    )

    # ── 300일선 전략 가산점 (최우선 전략) ────────────────────────────────────
    #   300일선 이격도(절대값) 중심 평가 — 돌파 후 경과 기간은 보지 않는다.
    #   이격도 0~3%   : +20
    #   이격도 3~5%   : +15
    #   이격도 5~7%   : +5
    #   이격도 7% 초과 : 0
    #   첫 눌림목(정배열 눌림목)    : +15
    #   거래량 증가                 : +10
    #   정배열 20>60>120>300        : +10
    price_now = data.get('price', float(hist['Close'].iloc[-1]))
    # 300일선 이격도(%) = (현재가 - MA300) / MA300 × 100  (양수: 위, 음수: 아래)
    ma300_dev_pct = (
        (price_now - ma300) / ma300 * 100.0
        if ma300 is not None and ma300 > 0 else None
    )
    abs_dev = abs(ma300_dev_pct) if ma300_dev_pct is not None else None
    if   abs_dev is None:   ma300_dev_score = 0
    elif abs_dev <= 3.0:    ma300_dev_score = 20
    elif abs_dev <= 5.0:    ma300_dev_score = 15
    elif abs_dev <= 7.0:    ma300_dev_score = 5
    else:                   ma300_dev_score = 0
    # 선매집 조건: 300일선 이격도 ±5% 이내
    ma300_near = abs_dev is not None and abs_dev <= 5.0
    ma300_aligned = (ma20 is not None and ma60 is not None and ma120 is not None
                     and ma300 is not None and ma20 > ma60 > ma120 > ma300)
    # 첫 눌림목 — 정배열 눌림목 패턴(MA10/MA20 통합)
    first_pullback = bool(pattern_data.get('first_pullback', (False,))[0])
    # 거래량 증가 — 최근 5일 평균 > 직전 20일 평균 ×1.15
    vol_increase = False
    if 'Volume' in hist.columns:
        v = hist['Volume'].dropna()
        if len(v) >= 25:
            recent = float(v.tail(5).mean())
            base   = float(v.tail(25).head(20).mean())
            vol_increase = base > 0 and recent > base * 1.15
    ma300_bonus = (ma300_dev_score
                   + (15 if first_pullback else 0) + (10 if vol_increase else 0)
                   + (10 if ma300_aligned else 0))
    ma300_strategy = bool(ma300_near or ma300_aligned)
    ma_aligned = bool(ma20 is not None and ma60 is not None and ma120 is not None
                      and ma20 > ma60 > ma120)

    # ── 최종 합산 ─────────────────────────────────────────────────────────────
    # 기본점 최대: 35+20+15+4 = 74  →  패턴 0개이면 최대 74점
    # 300일선 전략 가산점 최대 +55 (이격도 20 + 첫눌림목 15 + 거래량 10 + 정배열 10, 100점 cap)
    # S+: 85점 이상 + 패턴 2개 이상
    # S : 80점 이상 + 패턴 1개 이상
    # A : 75점 이상 (패턴 조건 없음, 현실적으로 1개 이상 필요)
    # B : 45점 이상
    # C : 30점 이상
    # D : 30점 미만
    raw_score  = ma_score + rsi_score + mfi_score + spread_score + pattern_score + ma300_bonus
    score      = min(raw_score, 100)
    n_patterns = sum(
        1 for v in pattern_data.values()
        if isinstance(v, (tuple, list)) and len(v) == 3 and v[0]
    )

    if   score >= 85 and n_patterns >= 2: grade = 'S+'
    elif score >= 80 and n_patterns >= 1: grade = 'S'
    elif score >= 75:                     grade = 'A'
    elif score >= 45:                     grade = 'B'
    elif score >= 30:                     grade = 'C'
    else:                                 grade = 'D'

    price     = data.get('price', float(hist['Close'].iloc[-1]))
    target    = price * TARGET_MULT
    stop_loss, stop_basis = _structure_stop(hist, price, ma20)

    return {
        'score':        score,
        'grade':        grade,
        'target_price': target,
        'stop_loss':    stop_loss,
        'stop_basis':   stop_basis,
        'patterns':     pattern_data,
        'macd':         macd_data,
        # 정배열 / 300일선 전략 플래그 (전략 분류·우선 정렬에 사용)
        'ma_aligned':            ma_aligned,
        'ma300_bonus':           ma300_bonus,
        'ma300_dev_pct':         ma300_dev_pct,
        'ma300_near':            ma300_near,
        'ma300_aligned':         ma300_aligned,
        'ma300_first_pullback':  first_pullback,
        'ma300_vol_increase':    vol_increase,
        'first_pullback':        first_pullback,
        'vol_increase':          vol_increase,
        'ma300_strategy':        ma300_strategy,
    }


# ══════════════════════════════════════════════════════════════════════════════════
# 리서치 뷰 — 매매 판단을 위한 상위 레벨 해석 (스테이지 / 레이팅 / 진입·청산 플랜)
# ══════════════════════════════════════════════════════════════════════════════════

# 국면(스테이지): (한글 라벨, 색상, 한 줄 설명)
STAGE_META: dict[str, tuple[str, str, str]] = {
    'Bottom':     ('바닥권',   '#9C6BFF', '저점권에서 반등 시도 — 분할 관찰 구간'),
    'Breakout':   ('돌파',     '#00B8D4', '저항 돌파·신고가 — 추세 전환 초입'),
    'Uptrend':    ('상승추세', '#00C851', '정배열 상승 진행 — 눌림 매수 유효'),
    'Overheated': ('과열',     '#FF9100', '단기 과열 — 신규 추격 매수 부담'),
    'Downtrend':  ('하락추세', '#F44336', '역배열 하락 — 매수 보류 권장'),
}

# 투자의견(레이팅): (한글 라벨, 색상, 한 줄 코멘트)
RATING_META: dict[str, tuple[str, str, str]] = {
    'BUY':   ('매수', '#00C851', '진입 매력 구간 — 분할 매수 고려'),
    'HOLD':  ('관망', '#FFB300', '조건 충족 대기 — 신규 진입은 신중'),
    'AVOID': ('회피', '#F44336', '추세·점수 약함 — 신규 진입 비권장'),
}

# 액션(최종판단) — 단 하나의 최종 결정. 투자의견(매수 자격) + 진입구간 위치를 결합한다.
#   매수 + 진입구간 도달  → 바로 진입(ENTER)
#   매수 + 진입구간 근처  → 분할 진입(SPLIT)
#   매수 + 진입구간 위    → 관찰리스트(WATCH)
#   조건 미충족(매수의견 아님·매수 차단) → 제외(EXCLUDE)
ACTION_META: dict[str, tuple[str, str, str]] = {
    'ENTER':   ('바로 진입',  '진입구간 내부 — 즉시 매수 가능',  '#00C851'),
    'SPLIT':   ('분할 진입',  '진입구간 상단 +10% 이내 — 분할 매수', '#2196F3'),
    'WATCH':   ('관찰리스트', '진입구간 +10% 이상 — 눌림 대기',  '#FFB300'),
    'EXCLUDE': ('제외',       '조건 미충족 — 신규 매수 보류',    '#F44336'),
}

# 진입구간(매수 밴드 상단) 대비 현재가 위치 밴드.
#   현재가 ≤ 상단×(1+1%)         → 진입구간 내부/도달 → 바로 진입
#   상단×(1+1%) < 현재가 < ×(1+10%) → 진입구간 상단 근처 → 분할 진입
#   현재가 ≥ 상단×(1+10%)        → 진입구간 위(상당 이격) → 관찰리스트
# 강세 돌파주가 진입구간보다 약간 위라는 이유만으로 관찰리스트로 밀리지 않도록
# 분할 진입 밴드를 +10% 직전까지 넓게 잡는다(과거 +3%는 지나치게 보수적이었음).
ENTRY_READY_BAND = 0.01
ENTRY_CHASE_BAND = 0.10


def action_view(research: dict) -> tuple[str, str, str]:
    """액션 카드 표시값(액션 라벨, 부가설명, 색)을 반환한다 — 단 하나의 최종 행동.

    최종판단(액션)은 투자의견(매수 자격) + 진입구간 위치로 결정된다(ACTION_META 1:1 매핑).
    """
    action = research.get('action', 'EXCLUDE')
    return ACTION_META.get(action, ACTION_META['EXCLUDE'])

# 목표가 멀티플 (T1 단기 / T2 중기 / T3 장기)
TARGET_MULTS: tuple[float, float, float] = (1.10, 1.20, 1.30)


def _last_ma(hist, col: str):
    if col in hist.columns:
        s = hist[col].dropna()
        if not s.empty:
            return float(s.iloc[-1])
    return None


def _determine_stage(hist, price: float, rsi: float) -> str:
    """이동평균 배열·RSI·52주 고저 위치로 현재 국면을 분류한다."""
    ma10  = _last_ma(hist, 'MA10')
    ma20  = _last_ma(hist, 'MA20')
    ma60  = _last_ma(hist, 'MA60')
    ma120 = _last_ma(hist, 'MA120')

    closes = hist['Close'].dropna()
    window = closes.tail(252)
    hi = float(window.max()) if not window.empty else price
    lo = float(window.min()) if not window.empty else price
    near_high = hi > 0 and price >= hi * 0.97
    from_low  = (price - lo) / lo * 100 if lo > 0 else 0.0

    aligned = ma20 is not None and ma60 is not None and ma120 is not None and ma20 > ma60 > ma120
    inverse = ma20 is not None and ma60 is not None and ma120 is not None and ma20 < ma60 < ma120
    above20 = ma20 is not None and price > ma20

    if rsi >= 78 or (ma20 is not None and price > ma20 * 1.25):
        return 'Overheated'
    if inverse or (ma60 is not None and price < ma60 and rsi < 45 and not above20):
        return 'Bottom' if (rsi < 40 and from_low < 18) else 'Downtrend'
    if near_high and ma20 is not None and ma60 is not None and ma20 > ma60:
        return 'Breakout'
    if (aligned and 40 <= rsi < 78) or (ma20 is not None and ma60 is not None and ma20 > ma60 and above20):
        return 'Uptrend'
    if rsi < 45:
        return 'Bottom'
    return 'Uptrend' if above20 else 'Downtrend'


def _decide_rating(score: int, stage: str) -> str:
    if stage == 'Downtrend' or score < 45:
        return 'AVOID'
    if stage == 'Overheated':
        return 'HOLD'
    if score >= 72 and stage in ('Bottom', 'Breakout', 'Uptrend'):
        return 'BUY'
    return 'HOLD'


def _decide_action(rating: str, price: float, entry_high: float | None,
                   buy_blocked: bool = False) -> str:
    """단 하나의 최종 행동(액션)을 결정한다 — 투자의견 + 진입구간 위치.

    이 값이 화면에 표시되는 유일한 최종 결정이며, 투자의견(rating)과 충돌하지 않도록
    rating 을 액션 안으로 흡수한다.
      · 조건 미충족(매수의견 아님·매수 차단)            → 제외(EXCLUDE)
      · 매수 + 진입구간 내부/도달(현재가 ≤ 상단×1.01)     → 바로 진입(ENTER)
      · 매수 + 상단 근처(상단×1.01 < 현재가 < ×1.10)     → 분할 진입(SPLIT)
      · 매수 + 상단 +10% 이상(현재가 ≥ 상단×1.10)        → 관찰리스트(WATCH)
    """
    if buy_blocked or rating != 'BUY':
        return 'EXCLUDE'
    if not entry_high or entry_high <= 0 or not price or price <= 0:
        return 'WATCH'
    gap = (price - entry_high) / entry_high       # 진입 밴드 상단 대비 현재가 이격
    if gap <= ENTRY_READY_BAND:
        return 'ENTER'
    if gap < ENTRY_CHASE_BAND:
        return 'SPLIT'
    return 'WATCH'


def _entry_zone(hist, price: float) -> tuple[float, float]:
    """눌림목 매수 밴드 — 현재가 이하의 MA10/MA20 지지 구간."""
    ma10 = _last_ma(hist, 'MA10')
    ma20 = _last_ma(hist, 'MA20')
    supports = [v for v in (ma10, ma20) if v is not None and v <= price]
    if supports:
        entry_low  = min(supports)
        entry_high = min(max(supports), price)
    else:
        entry_low  = price * 0.96
        entry_high = price * 0.99
    if entry_low > entry_high:
        entry_low, entry_high = entry_high * 0.97, entry_high
    return entry_low, entry_high


# ══════════════════════════════════════════════════════════════════════════════════
# 추세 품질 진단 — 이평선 기울기 · 데드크로스 · 거래량 상태 · 실적 급락 가드
# (단순 정배열만 보지 않고 추세의 '방향성·건전성'을 함께 평가하기 위한 헬퍼)
# ══════════════════════════════════════════════════════════════════════════════════

def _ma_slope(hist, col: str, lookback: int = 5) -> tuple[str, float]:
    """이동평균선 기울기 — 최근 lookback봉 변화율(%)과 방향 라벨.

    반환: (label, pct)  label ∈ {상승, 하락, 횡보, '—'(데이터 부족)}
    """
    if col not in hist.columns:
        return '—', 0.0
    s = hist[col].dropna()
    if len(s) < lookback + 1:
        return '—', 0.0
    last = float(s.iloc[-1])
    prev = float(s.iloc[-1 - lookback])
    if prev == 0:
        return '횡보', 0.0
    pct = (last - prev) / abs(prev) * 100.0
    label = '상승' if pct > 0.5 else '하락' if pct < -0.5 else '횡보'
    return label, round(pct, 2)


def compute_ma_slopes(hist) -> dict:
    """MA5/10/20 기울기를 한 번에 계산한다. (MA20은 10봉 기준으로 평활)"""
    return {
        'ma5':  _ma_slope(hist, 'MA5', 5),
        'ma10': _ma_slope(hist, 'MA10', 5),
        'ma20': _ma_slope(hist, 'MA20', 10),
    }


def detect_dead_cross(hist, lookback: int = 20) -> bool:
    """데드크로스 — MA20이 MA60을 최근 lookback봉 내 하향 돌파(골든크로스의 반대)."""
    if 'MA20' not in hist.columns or 'MA60' not in hist.columns:
        return False
    w = hist.dropna(subset=['MA20', 'MA60']).tail(lookback + 2)
    if len(w) < 2:
        return False
    a = w['MA20'].values
    b = w['MA60'].values
    for i in range(1, min(lookback, len(a))):
        idx = len(a) - 1 - i
        if idx < 1:
            break
        if a[idx] < b[idx] and a[idx - 1] >= b[idx - 1]:
            return True
    return False


def volume_state(hist) -> tuple[str, float]:
    """거래량 상태 — 최근 5일 평균 / 직전 20일 평균 비율과 라벨.

    반환: (label, ratio)  label ∈ {급증, 증가, 보통, 감소, '—'}
    """
    if 'Volume' not in hist.columns:
        return '—', 0.0
    v = hist['Volume'].dropna()
    if len(v) < 25:
        return '—', 0.0
    recent = float(v.tail(5).mean())
    base   = float(v.tail(25).head(20).mean())
    if base <= 0:
        return '—', 0.0
    ratio = recent / base
    label = ('급증' if ratio >= 2.0 else
             '증가' if ratio >= 1.2 else
             '감소' if ratio < 0.8 else '보통')
    return label, round(ratio, 2)


def detect_earnings_shock(hist, lookback: int = 12, drop_pct: float = 7.0) -> bool:
    """실적 발표 후 급락(추정) — 최근 lookback봉 중 하루 종가 낙폭이 drop_pct% 이상.

    무료 데이터에 실적 캘린더가 없어, 단일봉 급락(갭다운 포함)으로 근사한다.
    """
    c = hist['Close'].dropna()
    if len(c) < lookback + 1:
        return False
    rets = c.pct_change().tail(lookback) * 100.0
    return bool((rets <= -drop_pct).any())


def earnings_reaction_guard(hist) -> bool:
    """매수 금지 신호 — 실적 급락 + 이평선 하락 + 데드크로스 동시 충족.

    이 조합이면 단순 정배열이어도 신규 매수를 막고 관망/제외로 강등한다.
    (스펙 5: '실적 발표 후 급락 + 이평선 하락 + 데드크로스 → 매수 금지'.)
    """
    slopes = compute_ma_slopes(hist)
    valid  = [slopes[k] for k in ('ma5', 'ma10', 'ma20') if slopes[k][0] != '—']
    if len(valid) < 2:
        return False
    ma_down = all(lbl == '하락' for lbl, _ in valid)
    return ma_down and detect_dead_cross(hist) and detect_earnings_shock(hist)


# ── 메인 스캐너 표시용 — 대표 패턴명 · 패턴 진행률(0~100) · 국면 라벨 ──────────────
_PATTERN_KO: dict[str, str] = {
    'cup_and_handle': '컵앤핸들',
    'double_bottom':  '쌍바닥',
    'box':            '박스권 돌파',
    'ma300_breakout': '300일선 돌파시도',
    'ma_convergence': '5·10·20 수렴',
    'first_pullback': '첫 눌림목',
    'new_high':       '전고점 돌파',
    'golden_cross':   '골든크로스',
}


def _bottom_progress(bs: dict) -> float:
    pct = bs.get('breakout_pct', 0.0) or 0.0
    if bs.get('phase') == 'FORMING':          # 레벨 아래 — 가까울수록 진행률↑ (20~70%)
        gap = min(abs(pct), 0.10)
        return round(20 + (1 - gap / 0.10) * 50)
    if bs.get('phase') == 'BREAKOUT':         # 돌파 직후 — 70~100%
        return round(70 + min(max(pct, 0.0) / 0.05, 1.0) * 30)
    return 0.0


def pattern_progress(patterns: dict) -> tuple[str, float, str]:
    """대표 패턴명·진행률(0~100)·국면 라벨을 반환한다(메인 스캐너 카드 표시용).

    바닥탈출 셋업(형성/돌파)이 우선, 없으면 가중치 높은 정규 패턴의 신뢰도를 진행률로.
    반환: (pattern_ko, progress_pct, phase_ko)
    """
    bs = patterns.get('bottom_setup')
    if isinstance(bs, dict) and bs.get('qualifies'):
        pat   = _PATTERN_KO.get(bs.get('pattern'), '바닥 패턴')
        prog  = _bottom_progress(bs)
        phase = '형성 중' if bs.get('phase') == 'FORMING' else '돌파 직후'
        return pat, prog, phase
    order = ['ma300_breakout', 'first_pullback', 'ma_convergence',
             'new_high', 'golden_cross', 'cup_and_handle', 'double_bottom']
    for k in order:
        v = patterns.get(k)
        if isinstance(v, (tuple, list)) and len(v) == 3 and v[0]:
            return _PATTERN_KO.get(k, k), round(min(float(v[1]), 1.0) * 100), '진행 중'
    return '—', 0.0, '관망'


def build_research_view(data: dict, scored: dict) -> dict:
    """calculate_ai_score 결과를 매매 판단용 리서치 뷰로 확장한다."""
    hist  = data['hist']
    price = data.get('price', float(hist['Close'].iloc[-1]))
    rsi   = data.get('rsi', 50.0)
    score = scored['score']

    stage  = _determine_stage(hist, price, rsi)
    rating = _decide_rating(score, stage)

    entry_low, entry_high = _entry_zone(hist, price)
    stop_loss = scored['stop_loss']
    targets   = [price * m for m in TARGET_MULTS]

    # 손익비는 실제 진입 기준(진입 밴드 중앙값)으로 계산한다.
    # 리스크 하한을 현재가의 1%로 두어, 진입 밴드가 손절가에 근접한 종목에서
    # R/R 이 비현실적으로 폭발(수십억:1)하는 것을 막는다.
    entry_mid = (entry_low + entry_high) / 2
    risk      = max(entry_mid - stop_loss, price * 0.01)
    reward    = max(targets[0] - entry_mid, 0.0)
    rr        = reward / risk if risk > 0 else 0.0
    if not math.isfinite(rr):   # NaN/inf 방어 — 잘못된 입력이 적극매수로 둔갑하지 않게
        rr = 0.0

    # ── 추세 품질 진단 (스펙 4·5) ───────────────────────────────────────────────
    slopes               = compute_ma_slopes(hist)
    dead_cross           = detect_dead_cross(hist)
    vol_label, vol_ratio = volume_state(hist)

    # 실적 급락 + 이평선 하락 + 데드크로스 → 매수 금지(투자의견 회피로 강등).
    buy_blocked = earnings_reaction_guard(hist)
    if buy_blocked:
        rating = 'AVOID'

    # 최종판단(액션)은 단 하나의 결정 — 투자의견(매수 자격) + 진입구간 위치를 결합.
    action = _decide_action(rating, price, entry_high, buy_blocked)

    return {
        'stage':        stage,
        'rating':       rating,
        'action':       action,
        'probability':  score,
        'price':        price,
        'entry_low':    entry_low,
        'entry_high':   entry_high,
        'stop_loss':   stop_loss,
        'stop_basis':  scored.get('stop_basis'),
        'targets':     targets,
        'risk_reward': rr,
        # 추세 품질 진단 필드 (상세분석/스캐너 카드 표시 + 매수 가드)
        'ma_slopes':    slopes,
        'dead_cross':   dead_cross,
        'vol_state':    vol_label,
        'vol_ratio':    vol_ratio,
        'buy_blocked':  buy_blocked,
    }


# ══════════════════════════════════════════════════════════════════════════════════
# 전략 분류 — ① 선매집 / ② 바닥탈출 / ③ 강세추세
# ══════════════════════════════════════════════════════════════════════════════════

# 전략 키 → (라벨, 색상, 한 줄 설명)
STRATEGY_META: dict[str, tuple[str, str, str]] = {
    'PREBUY':     ('① 선매집',   '#2196F3', '300일선 ±5% · 거래량 증가 · 추세 전환 초기'),
    'BOTTOM_OUT': ('② 바닥탈출', '#00BFA5', '패턴 형성 중 · 돌파 직후 5% 이내 (쌍바닥·컵앤핸들·박스권)'),
    'STRONG':     ('③ 강세추세', '#00C851', '정배열 · 신고가 · 터틀 돌파 · 첫 눌림목'),
}


# 강세추세 세부패턴 (표시 전용 — 점수/액션 로직과 완전 분리): (라벨, 색상, 설명)
STRONG_PATTERN_META: dict[str, tuple[str, str, str]] = {
    'turtle':         ('🏹 터틀돌파',   '#FFB300', '직전 20일 고가(돈치언 상단) 돌파'),
    'near_high':      ('🔥 신고가근처', '#FF7043', '52주 신고가 5% 이내 (신고가 포함)'),
    'first_pullback': ('🎯 첫눌림목',   '#42A5F5', '정배열 이평선(10/20일) 첫 눌림목'),
}


def _pat_det(patterns: dict, key: str) -> bool:
    v = patterns.get(key)
    return bool(v[0]) if isinstance(v, (tuple, list)) and v else False


def classify_strong_patterns(item: dict) -> list[str]:
    """강세추세 종목의 세부 패턴(터틀돌파/신고가근처/첫눌림목)을 분류한다.

    표시 전용 분류 — 기존 패턴 점수·등급·액션 로직에 영향을 주지 않도록
    get_all_patterns 에 포함하지 않고 여기서 직접 평가한다. (반환 키는
    STRONG_PATTERN_META 의 키와 일치하며, 0개 이상 동시 해당 가능.)
    """
    out: list[str] = []
    hist = item.get('hist')
    if hist is not None and len(hist) > 0:
        try:
            if detect_turtle_breakout(hist)[0]:
                out.append('turtle')
        except Exception:
            pass
        try:
            if detect_near_high(hist)[0]:
                out.append('near_high')
        except Exception:
            pass
    patterns = item.get('patterns', {})
    if _pat_det(patterns, 'first_pullback'):
        out.append('first_pullback')
    return out


def classify_strategies(item: dict, research: dict | None = None) -> list[str]:
    """종목의 '현재 지배적(dominant) 단계' 하나만 반환한다(없으면 빈 리스트).

    라이프사이클은 선매집 → 바닥탈출 → 강세추세 순으로 진행되며, 한 종목은 자신이
    충족하는 가장 앞선(가장 상승한) 단계 **하나**로만 분류된다.
    우선순위: ③ 강세추세 > ② 바닥탈출 > ① 선매집.

    따라서 이미 상승해 강세추세 조건(정배열·신고가·첫 눌림목·골든크로스·돌파 진행)을
    충족한 종목은 바닥탈출(형성/돌파)을 동시에 달지 않는다 — 강세추세가 지배 단계가 된다.
    반환 타입은 호환을 위해 list 이지만 원소는 최대 1개다.
    """
    patterns = item.get('patterns', {})
    bs = patterns.get('bottom_setup')
    bs = bs if isinstance(bs, dict) else {}

    # ③ 강세추세 — 정배열 / 신고가 / 첫 눌림목 / 골든크로스 / 바닥 패턴 돌파 진행(5%↑)
    is_strong = (item.get('ma_aligned', False) or item.get('ma300_aligned', False)
                 or _pat_det(patterns, 'new_high')
                 or _pat_det(patterns, 'first_pullback')
                 or _pat_det(patterns, 'golden_cross')
                 or bs.get('extended', False))
    # ② 바닥탈출 — 패턴 형성 중(1순위) 또는 돌파 직후 5% 이내(2순위)만.
    is_bottom = bs.get('qualifies', False)
    # ① 선매집 — 300일선 ±5% 근접(거래량 증가·추세 전환 초기 동반 시 강화)
    is_prebuy = item.get('ma300_near', False)

    # 가장 앞선 단계 하나만 — 강세추세가 바닥탈출/선매집을 덮어쓴다.
    if is_strong:
        return ['STRONG']
    if is_bottom:
        return ['BOTTOM_OUT']
    if is_prebuy:
        return ['PREBUY']
    return []

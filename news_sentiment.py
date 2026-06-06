"""Lightweight lexicon-based sentiment analysis for stock-news headlines.

No external LLM dependency — deterministic keyword scoring tuned for Korean +
English financial vocabulary. Provides a per-article label and a 0-100 overall
score for a set of articles. Korean keywords are matched as substrings (the
language is agglutinative, so particles attach to stems); English keywords are
matched with word boundaries to avoid substring false positives.
"""
from __future__ import annotations

import re

LABEL_POSITIVE = 'positive'
LABEL_NEGATIVE = 'negative'
LABEL_NEUTRAL = 'neutral'

_POSITIVE = {
    # Korean
    '상승', '급등', '강세', '호재', '신고가', '돌파', '개선', '흑자', '성장',
    '수주', '계약', '호실적', '최대', '사상최대', '증가', '확대', '매수', '상향',
    '기대', '반등', '회복', '인수', '협력', '파트너십', '제휴', '투자', '승인',
    '출시', '성공', '호조', '수혜', '신제품', '훈풍', '낙관', '플러스', '흥행',
    # English
    'surge', 'soar', 'soars', 'jump', 'jumps', 'rally', 'gain', 'gains', 'rise',
    'rises', 'beat', 'beats', 'record', 'high', 'highs', 'upgrade', 'upgraded',
    'growth', 'profit', 'profits', 'strong', 'boost', 'partnership', 'deal',
    'win', 'wins', 'approval', 'approved', 'launch', 'launches', 'outperform',
    'bullish', 'rebound', 'breakthrough', 'expansion', 'optimistic', 'buy',
}

_NEGATIVE = {
    # Korean
    '하락', '급락', '폭락', '약세', '악재', '우려', '적자', '손실', '감소',
    '축소', '매도', '하향', '부진', '경고', '리스크', '위기', '소송', '조사',
    '제재', '파산', '결함', '리콜', '부정', '실패', '충격', '둔화', '감원',
    '구조조정', '디폴트', '청산', '급감', '쇼크', '악화', '비관', '논란',
    # English
    'fall', 'falls', 'drop', 'drops', 'plunge', 'plunges', 'slump', 'loss',
    'losses', 'miss', 'misses', 'downgrade', 'downgraded', 'weak', 'cut',
    'cuts', 'lawsuit', 'probe', 'recall', 'warning', 'warn', 'risk', 'bearish',
    'decline', 'declines', 'concern', 'concerns', 'layoff', 'layoffs',
    'bankruptcy', 'default', 'crash', 'tumble', 'sink', 'sinks', 'sell',
}


def _word_in(word: str, text: str) -> bool:
    if word.isascii():
        return re.search(r'\b' + re.escape(word) + r'\b', text) is not None
    return word in text


def _count_hits(text: str) -> tuple[int, int]:
    if not text:
        return 0, 0
    lower = text.lower()
    pos = sum(1 for w in _POSITIVE if _word_in(w, lower))
    neg = sum(1 for w in _NEGATIVE if _word_in(w, lower))
    return pos, neg


def analyze(text: str) -> tuple[str, float]:
    """Return (label, score) for one headline. score is in [-1.0, 1.0]."""
    pos, neg = _count_hits(text)
    total = pos + neg
    if total == 0:
        return LABEL_NEUTRAL, 0.0
    score = (pos - neg) / total
    if score > 0.15:
        return LABEL_POSITIVE, score
    if score < -0.15:
        return LABEL_NEGATIVE, score
    return LABEL_NEUTRAL, score


def overall_score(articles: list[dict]) -> int:
    """0-100 overall sentiment from articles carrying a 'sent_score' in [-1, 1]."""
    scores = [float(a.get('sent_score', 0.0)) for a in articles]
    if not scores:
        return 50
    mean = sum(scores) / len(scores)
    return max(0, min(100, int(round(50 + mean * 50))))


def overall_label(score: int) -> tuple[str, str]:
    """Return (emoji, korean_label) for a 0-100 overall score."""
    if score >= 80:
        return '🟢', '매우 긍정적'
    if score >= 60:
        return '🟢', '긍정적'
    if score >= 40:
        return '⚪', '중립적'
    if score >= 20:
        return '🔴', '부정적'
    return '🔴', '매우 부정적'


def overall_color(score: int) -> str:
    if score >= 60:
        return '#2ecc71'
    if score >= 40:
        return '#9aa0b5'
    return '#e74c3c'


def label_badge(label: str) -> tuple[str, str, str]:
    """Return (emoji, korean_label, color) for a per-article sentiment label."""
    if label == LABEL_POSITIVE:
        return '🟢', '긍정적', '#2ecc71'
    if label == LABEL_NEGATIVE:
        return '🔴', '부정적', '#e74c3c'
    return '⚪', '중립적', '#9aa0b5'


# ══════════════════════════════════════════════════════════════════════════════
# 특수 이벤트 분류기 (급등주 스캐너용) — 뉴스 제목/요약에서 촉매 이벤트를 탐지한다.
#   S급(최우선): 인수합병 · SPAC 합병 완료 · 거래정지 후 재개 · 티커 변경 ·
#                Reverse Merger · OTC→Nasdaq 상장.
#   A급: FDA 승인/임상 · 계약 수주 · 정부/국방 계약 · 실적 서프라이즈 ·
#        가이던스 상향 · 파트너십.
# 키워드는 (라벨, 키워드 튜플) 형태이며 정렬 우선순위는 정의 순서를 따른다.
# 한국어는 부분일치, 영어는 소문자 부분일치(제목이 짧아 단어경계는 생략).
# ══════════════════════════════════════════════════════════════════════════════
_S_EVENTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ('인수합병(M&A)', ('merger', 'acquisition', 'acquire', 'acquires', 'acquired',
                       'to be acquired', 'buyout', 'takeover', '인수', '합병', '피인수')),
    ('SPAC 합병완료', ('spac', 'de-spac', 'business combination', 'completes merger',
                       'completes business combination', '스팩', '스팩합병')),
    ('거래재개', ('resume trading', 'resumes trading', 'trading resumes',
                 'trading halt lifted', 'reinstated', '거래재개', '거래 재개', '매매재개')),
    ('티커변경', ('ticker change', 'change its ticker', 'new ticker', 'symbol change',
                 'will trade under', '티커변경', '종목코드 변경')),
    ('Reverse Merger', ('reverse merger', 'reverse takeover', '우회상장')),
    ('나스닥 상장', ('uplist', 'uplisting', 'uplists', 'begins trading on nasdaq',
                    'approved for listing on nasdaq', 'to list on nasdaq',
                    'nasdaq listing', '나스닥 상장', '나스닥 이전상장')),
)

_A_EVENTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ('FDA/임상', ('fda approval', 'fda approves', 'fda clearance', 'phase 3',
                 'phase iii', 'phase 2', 'clinical trial', 'topline', 'breakthrough therapy',
                 'fast track', '임상', 'fda 승인', '승인')),
    ('정부/국방 계약', ('defense contract', 'pentagon', 'department of defense',
                       'government contract', 'nasa', 'air force', 'army', 'navy',
                       'awarded contract', '국방', '방산', '정부계약', '방위')),
    ('계약수주', ('new order', 'contract win', 'wins contract', 'awarded',
                 'purchase order', 'backlog', 'bookings', 'secures order', '수주', '계약체결', '공급계약')),
    ('실적서프라이즈', ('beats estimates', 'beats expectations', 'tops estimates',
                       'earnings beat', 'record revenue', 'record earnings', 'surprise',
                       '서프라이즈', '어닝서프라이즈', '실적 호조', '사상최대')),
    ('가이던스상향', ('raises guidance', 'raised guidance', 'lifts guidance',
                     'upgrades outlook', 'boosts forecast', '가이던스 상향', '실적 전망 상향', '상향')),
    ('파트너십', ('partnership', 'partners with', 'collaboration', 'strategic alliance',
                 'joint venture', '파트너십', '협력', '제휴', '공동개발')),
)


def classify_event(text: str) -> tuple[str, str, str]:
    """뉴스 텍스트 → (tier, 라벨, 매칭키워드). tier ∈ {'S','A',''}.

    S급을 먼저 검사하고, 없으면 A급, 둘 다 없으면 ('', '', '') 반환."""
    if not text:
        return '', '', ''
    low = text.lower()
    for label, kws in _S_EVENTS:
        for kw in kws:
            if (kw in low) if kw.isascii() else (kw in text):
                return 'S', label, kw
    for label, kws in _A_EVENTS:
        for kw in kws:
            if (kw in low) if kw.isascii() else (kw in text):
                return 'A', label, kw
    return '', '', ''


def classify_events(articles: list[dict]) -> tuple[str, list[str]]:
    """기사 묶음 → (최고 tier, 발견된 이벤트 라벨 리스트). 최고 tier 는 'S'>'A'>''."""
    tier = ''
    labels: list[str] = []
    seen: set[str] = set()
    for a in articles or []:
        blob = f"{a.get('title') or ''} {a.get('headline_ko') or ''} {a.get('summary') or ''}"
        t, label, _kw = classify_event(blob)
        if label and label not in seen:
            seen.add(label)
            labels.append(label)
        if t == 'S':
            tier = 'S'
        elif t == 'A' and tier != 'S':
            tier = 'A'
    return tier, labels

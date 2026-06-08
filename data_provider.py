import datetime
import re
import time
import random
import streamlit as st
import yfinance as yf
import pandas as pd
import FinanceDataReader as fdr
from deep_translator import GoogleTranslator
from email.utils import parsedate_to_datetime
from urllib.parse import quote

import news_sentiment

NASDAQ_TICKERS = [
    'MSFT', 'AAPL', 'NVDA', 'TSLA', 'AMZN', 'META', 'GOOGL', 'AVGO', 'AMD', 'NFLX',
    'ADBE', 'CRM',  'ORCL', 'COST', 'TXN',  'QCOM', 'MU',    'LRCX', 'KLAC','SNPS',
]

SP500_TICKERS = [
    'PEP',  'KO',   'MCD',  'WMT',  'PG',   'CVX',  'XOM',  'JPM',  'BAC',  'DIS',
    'PFE',  'ABBV', 'TMO',  'DHR',  'HON',  'GE',   'V',    'MA',   'UNH',  'JNJ',
]

RUSSELL_TICKERS = [
    'SAIA', 'TREX', 'AAON', 'RLI',  'COOP', 'ITRI', 'AEIS', 'CALX', 'FORM', 'HALO',
    'PRGS', 'CHDN', 'CSWI', 'IRTC', 'NEOG', 'EVTC', 'CRVL', 'ACLS', 'MGLN', 'LBRT',
]

KOSPI_TICKERS = [
    '005930', '000660', '373220', '207940', '005380',
    '005490', '035720', '035420', '051910', '068270',
    '034730', '105560', '012330', '006400', '066570',
    '055550', '017670', '000270', '012450', '011200',
    '096770', '028260', '003550', '032830', '086790',
    '018260', '009150', '010130', '138040', '030200',
]

KOSDAQ_TICKERS = [
    '086520', '247540', '196170', '112040', '263750',
    '039030', '058470', '145020', '214150', '041510',
    '122870', '035900', '293490', '036930', '091990',
]

US_TICKERS = NASDAQ_TICKERS + SP500_TICKERS + RUSSELL_TICKERS
KR_TICKERS = KOSPI_TICKERS + KOSDAQ_TICKERS


# ── Live-fetch market listings (cached 24 h, fallback to hardcoded lists) ──────

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_nasdaq_tickers() -> list[str]:
    try:
        import requests
        r = requests.get(
            'https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nasdaq/nasdaq_tickers.txt',
            timeout=12,
        )
        tickers = [t.strip() for t in r.text.split('\n')
                   if t.strip() and '/' not in t and len(t.strip()) <= 5]
        if len(tickers) > 100:
            return sorted(tickers)
    except Exception:
        pass
    return NASDAQ_TICKERS


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_sp500_tickers() -> list[str]:
    try:
        import requests
        r = requests.get(
            'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv',
            timeout=12,
        )
        lines = r.text.strip().split('\n')
        tickers = [ln.split(',')[0].strip() for ln in lines[1:] if ln.strip()]
        tickers = [t for t in tickers if t and '/' not in t and len(t) <= 5]
        if len(tickers) > 400:
            return sorted(tickers)
    except Exception:
        pass
    return SP500_TICKERS


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_russell2000_tickers() -> list[str]:
    """NYSE + AMEX stocks, S&P500 removed — approximates the Russell 2000 universe."""
    try:
        import requests
        base = 'https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main'
        r_nyse = requests.get(f'{base}/nyse/nyse_tickers.txt', timeout=12)
        r_amex = requests.get(f'{base}/amex/amex_tickers.txt', timeout=12)
        nyse = {t.strip() for t in r_nyse.text.split('\n')
                if t.strip() and '/' not in t and len(t.strip()) <= 5}
        amex = {t.strip() for t in r_amex.text.split('\n')
                if t.strip() and '/' not in t and len(t.strip()) <= 5}
        sp500 = set(fetch_sp500_tickers())
        combined = sorted((nyse | amex) - sp500)
        if len(combined) > 100:
            return combined
    except Exception:
        pass
    return RUSSELL_TICKERS


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_kospi_tickers() -> list[str]:
    try:
        df = fdr.StockListing('KOSPI')
        col = 'Code' if 'Code' in df.columns else df.columns[0]
        codes = list(df[col].dropna().astype(str).str.zfill(6))
        if len(codes) > 100:
            return codes
    except Exception:
        pass
    return KOSPI_TICKERS


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_kosdaq_tickers() -> list[str]:
    try:
        df = fdr.StockListing('KOSDAQ')
        col = 'Code' if 'Code' in df.columns else df.columns[0]
        codes = list(df[col].dropna().astype(str).str.zfill(6))
        if len(codes) > 100:
            return codes
    except Exception:
        pass
    return KOSDAQ_TICKERS

KR_NAMES: dict[str, str] = {
    '005930': '삼성전자',           '000660': 'SK하이닉스',
    '373220': 'LG에너지솔루션',     '207940': '삼성바이오로직스',
    '005380': '현대차',             '005490': 'POSCO홀딩스',
    '035720': '카카오',             '035420': '네이버',
    '051910': 'LG화학',             '068270': '셀트리온',
    '034730': 'SK',                 '105560': 'KB금융',
    '012330': '현대모비스',         '006400': '삼성SDI',
    '066570': 'LG전자',             '055550': '신한지주',
    '017670': 'SK텔레콤',           '000270': '기아',
    '012450': '한화에어로스페이스',  '011200': 'HMM',
    '096770': 'SK이노베이션',       '028260': '삼성물산',
    '003550': 'LG',                 '032830': '삼성생명',
    '086790': '하나금융지주',       '018260': '삼성에스디에스',
    '009150': '삼성전기',           '010130': '고려아연',
    '138040': '메리츠금융지주',     '030200': 'KT',
    '086520': '에코프로',           '247540': '에코프로비엠',
    '196170': '알테오젠',           '112040': '위메이드',
    '263750': '펄어비스',           '039030': '이오테크닉스',
    '058470': '리노공업',           '145020': '휴젤',
    '214150': '클래시스',           '041510': 'SM엔터테인먼트',
    '122870': '와이지엔터테인먼트',  '035900': 'JYP Ent.',
    '293490': '카카오게임즈',       '036930': '주성엔지니어링',
    '091990': '셀트리온헬스케어',
}


@st.cache_data(ttl=3600, show_spinner=False)
def get_krx_listing() -> pd.DataFrame:
    df = fdr.StockListing('KRX')
    if 'Symbol' in df.columns and 'Code' not in df.columns:
        df = df.rename(columns={'Symbol': 'Code'})
    df['Code'] = df['Code'].astype(str).str.zfill(6)
    df['Name'] = df['Name'].astype(str).str.strip()
    return df[['Code', 'Name']].drop_duplicates('Code').reset_index(drop=True)


@st.cache_data(ttl=86400, show_spinner=False)
def get_kr_meta_dict() -> dict[str, dict]:
    """Returns {code: {name, market, industry}} for KOSPI + KOSDAQ (cached 24 h)."""
    result: dict[str, dict] = {}
    for market_name in ('KOSPI', 'KOSDAQ'):
        try:
            df       = fdr.StockListing(market_name)
            code_col = next(
                (c for c in df.columns if c.lower() in ('code', 'symbol', '티커')),
                df.columns[0],
            )
            name_col = next(
                (c for c in df.columns if c.lower() in ('name', '종목명', '이름')), None
            )
            ind_col  = next(
                (c for c in df.columns if c.lower() in ('industry', 'sector', '업종', '섹터')), None
            )
            df        = df.copy()
            df[code_col] = df[code_col].astype(str).str.zfill(6)
            codes = df[code_col].tolist()
            names = df[name_col].astype(str).str.strip().tolist() if name_col else codes
            inds  = df[ind_col].astype(str).str.strip().tolist()  if ind_col  else [''] * len(codes)
            for code, name, ind in zip(codes, names, inds):
                result[code] = {
                    'name':     name,
                    'market':   market_name,
                    'industry': '' if ind in ('nan', 'None', 'NaN', '') else ind,
                }
        except Exception:
            pass
    return result


def get_ticker_by_name(name: str) -> tuple[str | None, list[str]]:
    try:
        listing = get_krx_listing()
    except Exception:
        return None, []

    query = name.strip()

    exact = listing[listing['Name'] == query]
    if not exact.empty:
        return str(exact.iloc[0]['Code']), []

    exact_ci = listing[listing['Name'].str.lower() == query.lower()]
    if not exact_ci.empty:
        return str(exact_ci.iloc[0]['Code']), []

    partial = listing[listing['Name'].str.contains(query, na=False, case=False)]
    if partial.empty:
        return None, []
    if len(partial) == 1:
        return str(partial.iloc[0]['Code']), []

    return None, list(partial['Name'].head(8))


def get_name_by_code(code: str) -> str:
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


def has_korean(s: str) -> bool:
    return any('\uAC00' <= c <= '\uD7A3' for c in s)


def resolve_input(raw: str) -> tuple[str | None, str, list[str], str]:
    s = raw.strip()

    if s.upper().endswith('.KS') or s.upper().endswith('.KQ'):
        code = s.split('.')[0].zfill(6)
        return code, get_name_by_code(code), [], 'kr'

    if s.isdigit() and len(s) == 6:
        return s.zfill(6), get_name_by_code(s.zfill(6)), [], 'kr'

    if has_korean(s):
        code, candidates = get_ticker_by_name(s)
        if code:
            return code, get_name_by_code(code), [], 'kr'
        return None, s, candidates, 'kr'

    upper = s.upper()
    return upper, upper, [], 'us'


def _extract_news_item(n: dict) -> dict | None:
    content = n.get('content', {}) if isinstance(n.get('content'), dict) else {}
    title   = content.get('title', '') or n.get('title', '')
    if not title:
        return None
    url = (
        content.get('canonicalUrl', {}).get('url', '') or
        content.get('clickThroughUrl', {}).get('url', '') or
        n.get('link', '') or
        n.get('url', '')
    )
    return {'title': title, 'url': url}


def _parse_rss_items(
    xml: str,
) -> list[tuple[str, str, datetime.datetime | None, str]]:
    """Return [(title, link, published, source), ...] from an RSS XML string.

    `published` is a timezone-aware UTC datetime when the RSS <pubDate> can be
    parsed, otherwise None. `source` is the publisher name (from the <source>
    tag or the trailing " - 출처" suffix Google News appends to titles).
    """
    items: list[tuple[str, str, datetime.datetime | None, str]] = []
    for raw in re.findall(r'<item>(.*?)</item>', xml, re.DOTALL):
        title_m = re.search(r'<title>(.*?)</title>', raw, re.DOTALL)
        link_m  = re.search(r'<link>(.*?)</link>', raw, re.DOTALL)
        date_m  = re.search(r'<pubDate>(.*?)</pubDate>', raw, re.DOTALL)
        src_m   = re.search(r'<source[^>]*>(.*?)</source>', raw, re.DOTALL)
        if not title_m:
            continue
        title = re.sub(r'<!\[CDATA\[|\]\]>', '', title_m.group(1)).strip()
        # publisher name: prefer the <source> tag, else the " - 출처" suffix
        source = ''
        if src_m:
            source = re.sub(r'<!\[CDATA\[|\]\]>', '', src_m.group(1)).strip()
        if not source:
            suffix_m = re.search(r'\s+-\s+([^\-]{2,40})\s*$', title)
            if suffix_m:
                source = suffix_m.group(1).strip()
        # strip trailing " - 출처이름" suffix common in Google News
        title = re.sub(r'\s+-\s+[^\-]{2,40}\s*$', '', title).strip()
        link  = link_m.group(1).strip() if link_m else ''
        published: datetime.datetime | None = None
        if date_m:
            try:
                published = parsedate_to_datetime(date_m.group(1).strip())
                if published is not None and published.tzinfo is None:
                    published = published.replace(tzinfo=datetime.timezone.utc)
            except Exception:
                published = None
        if title:
            items.append((title, link, published, source))
    return items


def _recent_sorted(
    items: list[tuple[str, str, datetime.datetime | None, str]],
    days: int = 3,
    limit: int = 5,
) -> list[tuple[str, str, datetime.datetime, str]]:
    """Keep items published within `days`, sorted newest-first, capped at `limit`.

    Items without a parseable publish date are dropped — recency can't be
    verified for them.
    """
    now    = datetime.datetime.now(datetime.timezone.utc)
    cutoff = now - datetime.timedelta(days=days)
    dated  = [
        (title, link, pub.astimezone(datetime.timezone.utc), source)
        for title, link, pub, source in items
        if pub is not None and pub.astimezone(datetime.timezone.utc) >= cutoff
    ]
    dated.sort(key=lambda t: t[2], reverse=True)
    return dated[:limit]


def _make_news_item(
    original: str,
    url: str,
    published: datetime.datetime,
    source: str = '',
    subject: str = '',
) -> dict:
    """Build a news item with a deterministic Korean headline (no translation).

    `subject` is the ticker (US) or company name (KR) used as the headline prefix.
    The Korean headline is synthesized from the original title's finance topics +
    sentiment, so it is ALWAYS Korean and English-only headlines never appear.
    """
    label, score = news_sentiment.analyze(original)
    headline_ko  = _compose_headline_ko(original, subject, label)
    return {
        'original':    original,
        'headline_ko': headline_ko,
        # 'translated' kept for backward compatibility (now = the Korean headline)
        'translated':  headline_ko,
        'url':         url,
        'published':   published,
        'source':      source,
        'sentiment':   label,
        'sent_score':  score,
        'summary_ko':  _compose_summary_ko(headline_ko, label, score),
    }


def _is_mostly_korean(text: str) -> bool:
    han     = sum(1 for c in text if '\uac00' <= c <= '\ud7a3')
    letters = sum(1 for c in text if c.isalpha())
    return letters > 0 and han / letters > 0.3


# ── 한글 헤드라인 생성 (번역 서비스 미사용) ──────────────────────────────────────
# 무료 번역 서비스(Google/MyMemory)가 이 서버 IP를 차단/쿼터 소진시켜 영어 제목이
# 그대로 노출되던 문제가 있었다. 그래서 외부 번역에 의존하지 않고, RSS 제목에서
# 금융 토픽 키워드를 탐지해 결정론적으로 한글 헤드라인을 합성한다. 키워드가 하나도
# 안 잡히면 감성(긍정/부정/중립) 기반 한글 문구로 폴백 → 영어 단독 헤드라인은 절대
# 표시되지 않는다. (한국어 RSS 제목은 이미 한글이라 그대로 사용.)
#
# 정의 순서 = 우선순위. 각 항목은 (영문 키워드들, 한글 토픽). 더 구체적인 토픽을
# 위쪽에 둔다. 매칭은 소문자/단어경계 기준(_topic_hit).
_TOPIC_KO: list[tuple[tuple[str, ...], str]] = [
    (('upgrade', 'upgraded', 'outperform', 'overweight'),                 '투자의견 상향'),
    (('downgrade', 'downgraded', 'underperform', 'underweight'),          '투자의견 하향'),
    (('price target', 'analyst', 'analysts', 'initiates', 'rating'),      '목표주가·투자의견'),
    (('beat', 'beats', 'tops estimates', 'tops'),                         '실적 호조'),
    (('miss', 'misses', 'falls short'),                                   '실적 부진'),
    (('earnings', 'revenue', 'eps', 'quarter', 'quarterly', 'results',
      'q1', 'q2', 'q3', 'q4'),                                            '실적 발표'),
    (('guidance', 'forecast', 'outlook', 'raises guidance',
      'cuts guidance'),                                                   '가이던스·전망'),
    (('dividend',),                                                       '배당'),
    (('buyback', 'repurchase', 'repurchases'),                            '자사주 매입'),
    (('acquire', 'acquires', 'acquisition', 'merger', 'buyout',
      'takeover'),                                                        '인수·합병'),
    (('partnership', 'partners', 'deal', 'agreement', 'contract',
      'collaboration', 'teams up'),                                       '계약·파트너십'),
    (('lawsuit', 'sues', 'sued', 'settlement', 'litigation'),             '소송'),
    (('sec', 'probe', 'investigation', 'antitrust', 'regulator',
      'fine', 'fined'),                                                   '조사·규제'),
    (('fda', 'approval', 'approved', 'cleared'),                          '승인'),
    (('launch', 'launches', 'unveils', 'unveil', 'release', 'releases',
      'introduces', 'debut'),                                             '신제품·서비스 출시'),
    (('recall', 'recalls'),                                               '리콜'),
    (('layoff', 'layoffs', 'job cuts', 'restructuring'),                  '감원·구조조정'),
    (('bankruptcy', 'default', 'insolvency', 'chapter 11'),               '재무 위기'),
    (('stock split', 'split'),                                            '주식 분할'),
    (('ipo', 'public offering', 'goes public'),                           '상장·공모'),
    (('ceo', 'cfo', 'executive', 'resigns', 'resign', 'steps down',
      'appoints', 'appointed'),                                          '경영진 변동'),
    (('insider', 'sells shares', 'sold shares', 'buys shares',
      'bought shares', 'stake'),                                          '내부자·지분 거래'),
    (('short seller', 'short interest', 'short report'),                  '공매도'),
    (('artificial intelligence', ' ai ', 'chip', 'chips', 'semiconductor',
      'gpu'),                                                             'AI·반도체'),
    (('electric vehicle', ' ev ', 'battery'),                             '전기차·배터리'),
    (('surge', 'surges', 'soar', 'soars', 'jump', 'jumps', 'rally',
      'rallies', 'rockets', 'spike', 'spikes', 'record high',
      'all-time high', 'hits high'),                                      '주가 급등·강세'),
    (('plunge', 'plunges', 'tumble', 'tumbles', 'sink', 'sinks', 'crash',
      'slump', 'slumps', 'drop', 'drops', 'fall', 'falls', 'slide',
      'slides'),                                                          '주가 하락·약세'),
]

_SENT_FALLBACK_KO = {
    news_sentiment.LABEL_POSITIVE: '긍정적 소식',
    news_sentiment.LABEL_NEGATIVE: '부정적 소식',
    news_sentiment.LABEL_NEUTRAL:  '주요 동향',
}


def _topic_hit(keyword: str, text_lower: str) -> bool:
    """Whitespace-padded substring match (text is space-padded by caller)."""
    if keyword.isascii():
        return f' {keyword.strip()} ' in text_lower or keyword in text_lower
    return keyword in text_lower


def _detect_topics_ko(title: str, max_topics: int = 2) -> list[str]:
    """Detect up to `max_topics` Korean finance topics from an English title."""
    if not title:
        return []
    lower = f' {title.lower()} '
    found: list[str] = []
    for keywords, ko in _TOPIC_KO:
        if any(_topic_hit(k, lower) for k in keywords):
            if ko not in found:
                found.append(ko)
        if len(found) >= max_topics:
            break
    return found


def _compose_headline_ko(title: str, subject: str, label: str) -> str:
    """Deterministically build a Korean headline (no translation service).

    Korean RSS titles are returned as-is. For English titles we synthesize a
    headline from detected finance topics + the ticker/company subject, falling
    back to a sentiment phrase so the result is ALWAYS Korean (never English-only).
    """
    if _is_mostly_korean(title):
        return title.strip()
    subj   = (subject or '').strip()
    topics = _detect_topics_ko(title)
    body   = '·'.join(topics) if topics else _SENT_FALLBACK_KO.get(
        label, _SENT_FALLBACK_KO[news_sentiment.LABEL_NEUTRAL])
    # Invariant: the result is ALWAYS Korean (never the English title). `body` is
    # always a Korean phrase (topic or sentiment fallback), so even an empty
    # subject yields a Korean headline.
    return f"{subj} · {body}" if subj else body


_SENT_SUMMARY_KO = {
    news_sentiment.LABEL_POSITIVE: '긍정적 신호로 주가에 우호적으로 작용할 수 있습니다.',
    news_sentiment.LABEL_NEGATIVE: '부정적 요인으로 주가에 부담을 줄 수 있습니다.',
    news_sentiment.LABEL_NEUTRAL:  '시장에 미치는 영향은 제한적일 것으로 보입니다.',
}


def _impact_word(impact: int) -> str:
    if impact >= 70:
        return '높음'
    if impact >= 55:
        return '다소 높음'
    if impact <= 30:
        return '낮음'
    if impact <= 45:
        return '다소 낮음'
    return '보통'


def _compose_summary_ko(headline_ko: str, label: str, score: float) -> str:
    """Build a short 2~3 line Korean summary from the data we have for free.

    No article body is fetched, no LLM and no translation service is used, so we
    cannot invent article facts. The Korean headline carries the topic; we add a
    sentiment + impact reading so the user can grasp the takeaway from the list.
    """
    impact = max(0, min(100, int(round(50 + float(score or 0.0) * 50))))
    head   = (headline_ko or '').strip().rstrip('.')
    lines  = []
    if head:
        lines.append(f"📰 {head}.")
    lines.append(f"📊 시장 반응: {_SENT_SUMMARY_KO.get(label, _SENT_SUMMARY_KO[news_sentiment.LABEL_NEUTRAL])}")
    lines.append(f"⚡ 영향도 {impact}/100 ({_impact_word(impact)})")
    return '\n'.join(lines)


@st.cache_data(ttl=900, show_spinner=False)
def _fetch_naver_news(code: str, count: int = 5) -> list[dict]:
    """Fetch stock-specific Korean news via Google News RSS (company name search).

    15분 캐시 — 순차 뉴스 수집(safe_exec)이어도 반복 스캔 시 네트워크 호출을 생략한다."""
    try:
        import requests
        meta         = get_kr_meta_dict()
        company_name = meta.get(code, {}).get('name', '')
        if not company_name:
            return []
        url = (f"https://news.google.com/rss/search?"
               f"q={quote(company_name + ' when:3d')}&hl=ko&gl=KR&ceid=KR:ko")
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=8)
        r.raise_for_status()
        recent = _recent_sorted(_parse_rss_items(r.text), days=3, limit=count)
        return [
            _make_news_item(original=title, url=link, published=pub,
                            source=source, subject=company_name)
            for title, link, pub, source in recent
        ]
    except Exception as exc:
        print(f"[KR news ERROR] {type(exc).__name__}: {exc}")
        return []


@st.cache_data(ttl=900, show_spinner=False)
def _fetch_us_news(ticker: str, count: int = 5) -> list[dict]:
    """Fetch US stock-specific news via Google News RSS (Korean headline synthesized,
    no translation service used).

    15분 캐시 — 순차 뉴스 수집(safe_exec)이어도 반복 스캔 시 네트워크 호출을 생략한다."""
    try:
        import requests
        url = (f"https://news.google.com/rss/search?"
               f"q={quote(ticker + ' stock when:3d')}&hl=en-US&gl=US&ceid=US:en")
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=8)
        r.raise_for_status()
        recent = _recent_sorted(_parse_rss_items(r.text), days=3, limit=count)
        return [
            _make_news_item(original=title, url=link, published=pub,
                            source=source, subject=ticker.upper())
            for title, link, pub, source in recent
        ]
    except Exception as exc:
        print(f"[US news ERROR] {type(exc).__name__}: {exc}")
        return []


# ── 뉴스 원문 상세 ───────────────────────────────────────────────────────────────
# 기사 본문 크롤링/번역 기능은 제거됨. 무료(키 없는) 번역 서비스가 이 서버 IP를
# 차단하고 본문 추출 성공률도 낮아 비용 대비 신뢰성이 떨어졌다. 대신 뉴스 카드에
# 한글 제목 + 짧은 한글 요약(_compose_summary_ko)을 바로 보여주고, 제목을 누르면
# 원문 기사로 바로 이동한다(news_view.render_news_section).


def _build_result(hist: pd.DataFrame, news: list) -> dict | None:
    if hist is None or hist.empty or len(hist) < 20:
        return None
    hist = hist.copy()
    hist['MA5']   = hist['Close'].rolling(5).mean()
    hist['MA10']  = hist['Close'].rolling(10).mean()
    hist['MA20']  = hist['Close'].rolling(20).mean()
    hist['MA60']  = hist['Close'].rolling(60).mean()
    hist['MA120'] = hist['Close'].rolling(120).mean()
    hist['MA300'] = hist['Close'].rolling(300).mean()

    ma_cols = ['MA20', 'MA60', 'MA120']
    last_valid = hist[ma_cols].iloc[-1].dropna()
    if len(last_valid) >= 2:
        spread = float((last_valid.max() - last_valid.min()) / hist['Close'].iloc[-1] * 100)
    else:
        spread = 0.0

    price = float(hist['Close'].iloc[-1])
    try:
        prev        = float(hist['Close'].iloc[-2])
        change_pct  = (price - prev) / prev * 100 if prev != 0 else 0.0
    except (IndexError, ZeroDivisionError):
        change_pct  = 0.0

    return dict(spread=spread, price=price, change_pct=change_pct, hist=hist, news=news)


_KR_MARKET_KO = {'KOSPI': '코스피', 'KOSDAQ': '코스닥'}


_YF_SECTOR_KO: dict[str, str] = {
    'Basic Materials':        '기초소재',
    'Communication Services': '커뮤니케이션',
    'Consumer Cyclical':      '경기소비재',
    'Consumer Defensive':     '필수소비재',
    'Energy':                 '에너지',
    'Financial Services':     '금융',
    'Healthcare':             '헬스케어',
    'Industrials':            '산업재',
    'Real Estate':            '부동산',
    'Technology':             '기술',
    'Utilities':              '유틸리티',
}

_YF_INDUSTRY_KO: dict[str, str] = {
    'Semiconductors':                    '반도체',
    'Consumer Electronics':              '가전·전자',
    'Communication Equipment':           '통신 장비',
    'Electronic Components':             '전자 부품',
    'Scientific & Technical Instruments':'계측·정밀기기',
    'Internet Content & Information':    '인터넷·플랫폼',
    'Software - Application':            '소프트웨어',
    'Software - Infrastructure':         '소프트웨어 인프라',
    'Information Technology Services':   'IT 서비스',
    'Computer Hardware':                 '컴퓨터 하드웨어',
    'Auto Manufacturers':                '자동차',
    'Auto Parts':                        '자동차 부품',
    'Steel':                             '철강',
    'Specialty Chemicals':               '특수화학',
    'Basic Materials':                   '기초소재',
    'Oil & Gas E&P':                     '석유·가스 탐사',
    'Oil & Gas Integrated':              '석유·가스',
    'Drug Manufacturers - General':      '제약',
    'Drug Manufacturers - Specialty & Generic': '제약·바이오',
    'Biotechnology':                     '바이오기술',
    'Medical Devices':                   '의료기기',
    'Diagnostics & Research':            '진단·연구',
    'Banks - Regional':                  '지방은행',
    'Banks - Diversified':               '종합은행',
    'Insurance - Life':                  '생명보험',
    'Insurance - Property & Casualty':   '손해보험',
    'Asset Management':                  '자산운용',
    'Capital Markets':                   '자본시장',
    'Real Estate Services':              '부동산 서비스',
    'REIT - Specialty':                  '특수 리츠',
    'REIT - Office':                     '오피스 리츠',
    'REIT - Residential':                '주거 리츠',
    'REIT - Retail':                     '리테일 리츠',
    'Travel Services':                   '여행 서비스',
    'Airlines':                          '항공',
    'Hotels & Motels':                   '호텔·숙박',
    'Restaurants':                       '외식',
    'Department Stores':                 '백화점',
    'Specialty Retail':                  '전문소매',
    'Apparel Retail':                    '의류소매',
    'Luxury Goods':                      '명품·럭셔리',
    'Retail - Apparel & Specialty':      '의류·전문소매',
    'Retail - Defensive':                '필수소매',
    'Grocery Stores':                    '식료품',
    'Telecom Services':                  '통신 서비스',
    'Fiber Optic Communication':         '광통신',
    'Broadcasting':                      '방송',
    'Entertainment':                     '엔터테인먼트',
    'Electronic Gaming & Multimedia':    '게임·멀티미디어',
    'Shell Companies':                   '스팩·페이퍼컴퍼니',
    'Financial Conglomerates':           '금융지주',
    'Staffing & Employment Services':    '인력파견',
    'Leisure':                           '여가·레저',
    'Resorts & Casinos':                 '리조트·카지노',
    'Travel & Leisure':                  '여행·레저',
    'Personal Services':                 '개인서비스',
    'Waste Management':                  '환경·폐기물',
    'Specialty Chemicals':               '특수화학',
    'Agricultural Chemicals':            '농화학',
    'Paper & Paper Products':            '종이·제지',
    'Metal Fabrication':                 '금속가공',
    'Aluminum':                          '알루미늄',
    'Copper':                            '구리',
    'Other Industrial Metals & Mining':  '기타금속·광업',
    'Coal':                              '석탄',
    'Building Materials':                '건자재',
    'Furnishings, Fixtures & Appliances':'가구·가전',
    'Footwear & Accessories':            '신발·액세서리',
    'Medical Care Facilities':           '의료시설',
    'Health Information Services':       '헬스IT서비스',
    'Education & Training Services':     '교육·연수',
    'Publishing':                        '출판',
    'Security & Protection Services':    '보안서비스',
    'Specialty Business Services':       '전문사업서비스',
    'Rental & Leasing Services':         '임대·리스',
    'Marine Shipping':                   '해운',
    'Trucking':                          '트럭운송',
    'Railroads':                         '철도',
    'Ports & Harbors':                   '항만',
    'Engineering & Construction':        '건설·엔지니어링',
    'Aerospace & Defense':               '항공우주·방위',
    'Industrial Machinery':              '산업기계',
    'Specialty Industrial Machinery':    '특수산업기계',
    'Electrical Equipment & Parts':      '전기장비·부품',
    'Utilities - Regulated Electric':    '전력',
    'Utilities - Renewable':             '신재생에너지',
    'Agricultural Inputs':               '농업',
    'Farm Products':                     '농산물',
    'Packaged Foods':                    '가공식품',
    'Beverages - Non-Alcoholic':         '음료(비주류)',
    'Beverages - Brewers':               '주류',
}


@st.cache_data(ttl=86400 * 3, show_spinner=False)
def get_kr_yf_industry(code: str, market_en: str) -> str:
    """Fetch industry for a Korean stock from Yahoo Finance and translate to Korean (3-day cache).
    Called only from the single-stock analysis tab — NOT from the scanner."""
    try:
        suffix      = 'KS' if market_en == 'KOSPI' else 'KQ'
        info        = yf.Ticker(f'{code}.{suffix}').info
        industry_en = info.get('industry', '') or ''
        if not industry_en:
            return ''
        # Use lookup table first; fall back to machine translation
        if industry_en in _YF_INDUSTRY_KO:
            return _YF_INDUSTRY_KO[industry_en]
        try:
            return GoogleTranslator(source='en', target='ko').translate(industry_en)
        except Exception:
            return industry_en
    except Exception:
        return ''


# 시세 신선도 유지(15분) — get_data_us 주석 참고. 24h 캐시는 하루 지난 가격을 보여준다.
@st.cache_data(ttl=900, show_spinner=False)
def get_data_kr(code: str) -> dict | None:
    try:
        start  = (datetime.date.today() - datetime.timedelta(days=620)).strftime('%Y-%m-%d')
        hist   = fdr.DataReader(code, start)
        if hist is None or hist.empty:
            return None
        hist   = hist[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        news   = _fetch_naver_news(code)
        result = _build_result(hist, news)
        if result is None:
            return None
        meta               = get_kr_meta_dict().get(code, {})
        result['company_name'] = meta.get('name', code)
        result['sector']       = _KR_MARKET_KO.get(meta.get('market', ''), meta.get('market', ''))
        result['industry']     = meta.get('industry', '')
        return result
    except Exception:
        return None


def _yf_history_with_retry(stock: "yf.Ticker", attempts: int = 3):
    """Fetch yfinance OHLCV history, retrying on Yahoo rate-limit errors.

    Yahoo Finance frequently returns `YFRateLimitError` ("Too Many Requests")
    from datacenter IPs even for perfectly valid tickers. A single attempt would
    surface as "unsupported ticker" to the user, so we retry with jittered
    backoff before giving up and letting the caller fall back to FinanceDataReader.
    Returns a non-empty DataFrame on success, else None.
    """
    for i in range(attempts):
        try:
            hist = stock.history(period="2y")
            if hist is not None and not hist.empty:
                return hist
        except Exception:
            pass
        if i < attempts - 1:
            time.sleep(0.6 * (i + 1) + random.uniform(0, 0.4))
    return None


def _fdr_history_us(ticker: str):
    """US OHLCV history via FinanceDataReader — fallback when yfinance is rate-limited.

    FinanceDataReader pulls from a different source (Stooq) and is not subject to
    Yahoo's rate limiting, so it reliably serves valid US tickers (SOFI, RDW, …)
    when yfinance returns nothing. Returns the standard OHLCV frame or None.
    """
    try:
        start = (datetime.date.today() - datetime.timedelta(days=820)).strftime('%Y-%m-%d')
        hist  = fdr.DataReader(ticker, start)
        if hist is None or hist.empty:
            return None
        return hist[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    except Exception:
        return None


# 가격/시세 데이터는 신선해야 한다 — 기존 24h 캐시가 하루 지난 가격을 그대로
# 보여줘 현재가·등락률이 실제 시장과 어긋났다(RDW 데이터 부정확 버그). 15분 캐시로
# 낮춰 장중 움직임을 반영하면서도 한 번의 스캔 안에서는 재사용해 야후 레이트리밋
# 폭주를 피한다. (가격/전일종가/등락률은 모두 동일한 hist['Close']에서 계산.)
def _session_quote(info: dict, daily_close: float | None) -> tuple[float | None, float | None]:
    """현재 거래 세션(프리마켓·정규장·애프터마켓)에 맞는 '표시용' (가격, 등락률) 반환.

    yfinance .info 의 세션별 필드를 marketState 로 선택한다 — 표시용 시세만 다루며
    패턴/추세 계산용 hist 와는 무관하다.
      · PRE(프리)   → preMarketPrice  / preMarketChangePercent
      · POST/CLOSED → postMarketPrice / postMarketChangePercent
      · 그 외(정규장) → regularMarketPrice / regularMarketChangePercent
    세션 시세가 없으면 정규장 → currentPrice → 일봉 종가(daily_close) 순으로 폴백.
    등락률 필드가 없으면 적정 기준가(프리/애프터=직전 정규장가, 정규장=전일종가)로 재계산.
    """
    def f(x):
        try:
            return float(x) if x is not None else None
        except (TypeError, ValueError):
            return None

    state    = str(info.get('marketState') or '').upper()
    reg_px   = f(info.get('regularMarketPrice')) or f(info.get('currentPrice'))
    reg_prev = f(info.get('regularMarketPreviousClose')) or f(info.get('previousClose'))
    reg_chg  = f(info.get('regularMarketChangePercent'))

    def chg(px, base, given):
        if given is not None:
            return given
        if px is not None and base:
            return (px - base) / base * 100.0
        return None

    # 프리마켓: 등락률 기준은 직전 정규장 종가(reg_px)
    if state.startswith('PRE'):
        px = f(info.get('preMarketPrice'))
        if px:
            return px, chg(px, reg_px, f(info.get('preMarketChangePercent')))
    # 애프터마켓/마감: 등락률 기준은 정규장 종가(reg_px)
    if state.startswith('POST') or state == 'CLOSED':
        px = f(info.get('postMarketPrice'))
        if px:
            return px, chg(px, reg_px, f(info.get('postMarketChangePercent')))
    # 정규장 또는 세션 시세 부재
    if reg_px:
        return reg_px, chg(reg_px, reg_prev, reg_chg)
    if daily_close:
        return daily_close, None
    return None, None


def display_quote(market, live_price, live_change, hist_close, hist_change):
    """모든 탭(스캐너·급등주·미래10배주·상세분석) 공통의 '표시용 시세' 선택기.

    한 곳에서만 시세 우선순위를 정의해 탭별 가격이 어긋나지 않게 한다.
      · US: 라이브 세션가(NASDAQ 스냅샷 lastsale / yfinance .info 세션가)가 있으면
            우선 사용, 없으면 일봉 종가로 폴백.
      · KR: 무료 실시간 벌크 소스가 없고(KRX/pykrx 차단, 스냅샷 캐시는 수일 지연)
            잘못된 시세보다 정확한 직전 일봉이 낫기 때문에 항상 yfinance 일봉 종가 사용.
    반환: (표시가격, 표시등락률). 등락률은 매칭되는 소스의 값을 따른다.
    """
    def _f(x):
        try:
            return float(x) if x is not None else None
        except (TypeError, ValueError):
            return None

    lp, lc = _f(live_price), _f(live_change)
    hp, hc = _f(hist_close), _f(hist_change)
    if str(market).lower() == 'us' and lp and lp > 0:
        return lp, (lc if lc is not None else hc)
    return (hp or 0.0), (hc if hc is not None else 0.0)


@st.cache_data(ttl=900, show_spinner=False)
def get_data_us(ticker: str) -> dict | None:
    stock = None
    hist  = None
    # 1) Primary: yfinance with retry on rate-limit.
    try:
        stock = yf.Ticker(ticker)
        raw   = _yf_history_with_retry(stock)
        if raw is not None:
            hist = raw[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    except Exception:
        hist = None

    # 2) Fallback: FinanceDataReader (different source, no Yahoo rate limit).
    if hist is None or hist.empty:
        hist = _fdr_history_us(ticker)

    # Both sources failed → the ticker is genuinely unavailable.
    if hist is None or hist.empty:
        return None

    news_items = _fetch_us_news(ticker)
    result     = _build_result(hist, news_items)
    if result is None:
        return None

    # Company metadata via yfinance .info — best-effort; never fatal (the price
    # series above already succeeded, possibly from the fdr fallback).
    info: dict = {}
    try:
        info        = stock.info if stock is not None else {}
        sector_en   = info.get('sector', '')
        industry_en = info.get('industry', '')
        result['company_name'] = info.get('longName') or info.get('shortName') or ticker
        result['quote_type']   = info.get('quoteType', 'EQUITY')
        result['industry_en']  = industry_en
        result['sector']       = _YF_SECTOR_KO.get(sector_en, sector_en)
        # Industry: lookup table → Google Translate → English fallback
        result['industry'] = _YF_INDUSTRY_KO.get(industry_en, '')
        if not result['industry'] and industry_en:
            try:
                result['industry'] = GoogleTranslator(source='en', target='ko').translate(industry_en)
            except Exception:
                result['industry'] = industry_en
    except Exception:
        result['company_name'] = ticker
        result['sector']       = ''
        result['industry']     = ''

    # 시세 정확도(quote accuracy) — 현재 거래 세션 반영: 일봉 히스토리의 마지막 행은
    # '마지막 *완료* 세션'이라 새 세션 동안 실시간 시세와 어긋난다(RDW: 히스토리 18.62
    # vs 실시간). get_data_us 는 위에서 *이미* info 를 받았으므로(추가 네트워크 호출 없음)
    # marketState 에 따라 프리마켓/정규장/애프터마켓 시세로 표시 price·change_pct 를
    # 보정한다(_session_quote). 패턴·추세용 hist 는 그대로 둔다. yfinance 가 막혀 FDR
    # 폴백을 쓴 경우엔 info 가 비어 보정이 적용되지 않고 일봉 종가가 그대로 쓰인다.
    try:
        result['market_session'] = str(info.get('marketState') or '').upper()
        daily_close = result.get('price')   # _build_result 가 넣어둔 일봉 마지막 종가
        daily_chg   = result.get('change_pct')
        sess_px, sess_chg = _session_quote(info, daily_close)
        # 표시 시세는 다른 탭과 동일한 공통 엔진(display_quote)으로 일원화 — US는
        # 세션 시세(라이브) 우선, 없으면 일봉 종가/등락률로 폴백. 세션가는 있는데
        # 세션 등락률이 없으면 일봉 종가 기준으로 재계산해 가격/등락률 불일치를 막는다.
        live_chg = sess_chg
        if live_chg is None and sess_px and daily_close and daily_close > 0:
            live_chg = (sess_px - daily_close) / daily_close * 100.0
        result['price'], result['change_pct'] = display_quote(
            'us', live_price=sess_px, live_change=live_chg,
            hist_close=daily_close, hist_change=daily_chg,
        )
    except Exception:
        pass
    return result


def get_data(code_or_ticker: str, market: str = 'auto') -> dict | None:
    s = code_or_ticker.strip()
    up = s.upper()

    # 한국 종목 — .KS/.KQ 접미사 또는 6자리 숫자 코드.
    if up.endswith('.KS') or up.endswith('.KQ'):
        return get_data_kr(s.split('.')[0].zfill(6))
    if market == 'kr':
        return get_data_kr(s.split('.')[0].zfill(6))
    if market == 'auto' and s.isdigit() and len(s) == 6:
        return get_data_kr(s.zfill(6))

    # 미국 종목 — yfinance 는 클래스주에 '.' 대신 '-' 를 쓴다 (예: BRK.B → BRK-B).
    us_ticker = up.replace('.', '-')
    return get_data_us(us_ticker)

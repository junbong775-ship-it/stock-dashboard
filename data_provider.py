import datetime
import re
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


def _parse_rss_items(xml: str) -> list[tuple[str, str, datetime.datetime | None]]:
    """Return [(title, link, published), ...] from an RSS XML string.

    `published` is a timezone-aware UTC datetime when the RSS <pubDate> can be
    parsed, otherwise None.
    """
    items: list[tuple[str, str, datetime.datetime | None]] = []
    for raw in re.findall(r'<item>(.*?)</item>', xml, re.DOTALL):
        title_m = re.search(r'<title>(.*?)</title>', raw, re.DOTALL)
        link_m  = re.search(r'<link>(.*?)</link>', raw, re.DOTALL)
        date_m  = re.search(r'<pubDate>(.*?)</pubDate>', raw, re.DOTALL)
        if not title_m:
            continue
        title = re.sub(r'<!\[CDATA\[|\]\]>', '', title_m.group(1)).strip()
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
            items.append((title, link, published))
    return items


def _recent_sorted(
    items: list[tuple[str, str, datetime.datetime | None]],
    days: int = 3,
    limit: int = 5,
) -> list[tuple[str, str, datetime.datetime]]:
    """Keep items published within `days`, sorted newest-first, capped at `limit`.

    Items without a parseable publish date are dropped — recency can't be
    verified for them.
    """
    now    = datetime.datetime.now(datetime.timezone.utc)
    cutoff = now - datetime.timedelta(days=days)
    dated  = [
        (title, link, pub.astimezone(datetime.timezone.utc))
        for title, link, pub in items
        if pub is not None and pub.astimezone(datetime.timezone.utc) >= cutoff
    ]
    dated.sort(key=lambda t: t[2], reverse=True)
    return dated[:limit]


def _make_news_item(
    original: str,
    translated: str,
    url: str,
    published: datetime.datetime,
) -> dict:
    label, score = news_sentiment.analyze(f"{original} {translated}")
    return {
        'original':   original,
        'translated': translated,
        'url':        url,
        'published':  published,
        'sentiment':  label,
        'sent_score': score,
    }


def _fetch_naver_news(code: str, count: int = 5) -> list[dict]:
    """Fetch stock-specific Korean news via Google News RSS (company name search)."""
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
            _make_news_item(original=title, translated=title, url=link, published=pub)
            for title, link, pub in recent
        ]
    except Exception as exc:
        print(f"[KR news ERROR] {type(exc).__name__}: {exc}")
        return []


def _fetch_us_news(ticker: str, count: int = 5) -> list[dict]:
    """Fetch US stock-specific news via Google News RSS, translated to Korean."""
    try:
        import requests
        url = (f"https://news.google.com/rss/search?"
               f"q={quote(ticker + ' stock when:3d')}&hl=en-US&gl=US&ceid=US:en")
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=8)
        r.raise_for_status()
        recent = _recent_sorted(_parse_rss_items(r.text), days=3, limit=count)
        items: list[dict] = []
        for title, link, pub in recent:
            try:
                translated = GoogleTranslator(source='auto', target='ko').translate(title)
            except Exception:
                translated = title
            items.append(_make_news_item(original=title, translated=translated,
                                         url=link, published=pub))
        return items
    except Exception as exc:
        print(f"[US news ERROR] {type(exc).__name__}: {exc}")
        return []


def _build_result(hist: pd.DataFrame, news: list) -> dict | None:
    if hist is None or hist.empty or len(hist) < 20:
        return None
    hist = hist.copy()
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


@st.cache_data(ttl=86400, show_spinner=False)
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


@st.cache_data(ttl=86400, show_spinner=False)
def get_data_us(ticker: str) -> dict | None:
    try:
        stock = yf.Ticker(ticker)
        hist  = stock.history(period="2y")
        if hist.empty:
            return None
        hist       = hist[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        news_items = _fetch_us_news(ticker)
        result     = _build_result(hist, news_items)
        if result is None:
            return None
        try:
            info        = stock.info
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
        return result
    except Exception:
        return None


def get_data(code_or_ticker: str, market: str = 'auto') -> dict | None:
    s = code_or_ticker.strip()
    if '.' in s:
        s = s.split('.')[0]
    if market == 'kr' or (market == 'auto' and s.isdigit() and len(s) == 6):
        return get_data_kr(s.zfill(6))
    return get_data_us(s)

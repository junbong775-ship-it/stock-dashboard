"""테마 분류 — 종목을 핵심 투자 테마(다중 분류)로 분류한다.

yfinance / FinanceDataReader 의 업종(industry) 분류는 단일 산업 기준이라
'AI 인프라', '실리콘포토닉스' 같은 교차산업 테마를 잡아내지 못한다. 따라서
대표 종목을 큐레이션한 티커 집합(THEME_TICKERS)을 1차 기준으로 삼고,
업종 텍스트가 신뢰성 있게 매핑되는 일부 테마(반도체·방산)만 키워드 보조
매칭(THEME_KEYWORDS)을 추가한다. (scanner-sectors 메모리 규칙과 동일)

티커 비교는 대문자/문자열 기준. 미국은 알파벳 심볼, 한국은 6자리 숫자 코드.
"""

# 표시·필터 순서 (예시: AI 인프라, 광통신, 실리콘포토닉스 …)
THEME_LABELS: list[str] = [
    'AI 인프라',
    'AI 서버',
    '데이터센터',
    '광통신',
    '실리콘포토닉스',
    '반도체',
    '전력반도체',
    'HBM 테스트',
    'AI 전력인프라',
    '원전',
    'SMR',
    '전력망',
    'ESS',
    '사이버보안',
    '양자컴퓨팅',
    '우주',
    '방산',
]

# 화면 표시용 라벨 (ESS 는 풀네임으로 노출)
THEME_DISPLAY: dict[str, str] = {
    'ESS': '에너지저장장치(ESS)',
}

# ── 테마별 큐레이션 티커 집합 (1차 기준) ──────────────────────────────────────
THEME_TICKERS: dict[str, set[str]] = {
    'AI 인프라': {
        # 미국 — AI 가속기·서버·네트워킹·인프라
        'NVDA', 'AMD', 'AVGO', 'MRVL', 'SMCI', 'DELL', 'ANET', 'VRT',
        'ARM', 'TSM', 'ASML', 'MU', 'CRDO', 'ALAB', 'PSTG', 'AEHR',
        'NBIS', 'CRWV', 'ORCL', 'CIEN',
        # 한국 — HBM·AI 메모리·후공정
        '000660',  # SK하이닉스
        '005930',  # 삼성전자
        '042700',  # 한미반도체
        '058470',  # 리노공업
    },
    '광통신': {
        # 미국 — 광트랜시버·광부품·광통신 장비
        'COHR', 'LITE', 'FN', 'AAOI', 'INFN', 'CIEN', 'POET', 'ANET',
        # 한국 — 광통신 부품
        '138080',  # 오이솔루션
        '073490',  # 이노와이어리스
        '046120',  # 오르비텍(참고)
    },
    '실리콘포토닉스': {
        # 미국 — 실리콘 포토닉스 / 광집적
        'COHR', 'LITE', 'FN', 'AAOI', 'POET', 'MRVL', 'GFS', 'AVGO',
        'CRDO', 'ALAB',
    },
    '반도체': {
        # 미국 — 설계·파운드리·장비·소재
        'NVDA', 'AMD', 'INTC', 'AVGO', 'QCOM', 'TXN', 'MU', 'TSM', 'AEHR',
        'ASML', 'AMAT', 'LRCX', 'KLAC', 'MRVL', 'ARM', 'GFS', 'ON',
        'ADI', 'NXPI', 'MCHP', 'TER', 'ENTG', 'SWKS', 'QRVO',
        # 한국 — 반도체·장비·소재
        '000660', '005930', '042700', '058470', '240810',  # 원익IPS
        '357780',  # 솔브레인
        '036930',  # 주성엔지니어링
        '095340',  # ISC
    },
    '원전': {
        # 미국 — 원자력 발전·SMR·우라늄
        'CEG', 'VST', 'SMR', 'OKLO', 'LEU', 'CCJ', 'BWXT', 'NNE',
        'UEC', 'UUUU', 'DNN', 'NLR',
        # 한국 — 원전 EPC·기자재
        '034020',  # 두산에너빌리티
        '052690',  # 한전기술
        '051600',  # 한전KPS
        '105840',  # 우진
        '100090',  # 삼강엠앤티(참고)
    },
    '전력망': {
        # 미국 — 전력기기·송배전·전력 인프라
        'GEV', 'ETN', 'PWR', 'HUBB', 'NVT', 'AGX', 'PRIM', 'MYRG',
        'POWL', 'VRT',
        # 한국 — 전력기기·변압기·전선
        '015760',  # 한국전력
        '010120',  # LS ELECTRIC
        '298040',  # 효성중공업
        '033100',  # 제룡전기
        '006260',  # LS
        '009830',  # 한화솔루션(참고)
    },
    'ESS': {
        # 미국 — 배터리·ESS 시스템
        'TSLA', 'ENPH', 'SEDG', 'FLNC', 'STEM', 'EOSE', 'NPWR',
        # 한국 — 2차전지·소재
        '373220',  # LG에너지솔루션
        '006400',  # 삼성SDI
        '247540',  # 에코프로비엠
        '003670',  # 포스코퓨처엠
        '066970',  # 엘앤에프
        '137400',  # 피엔티
    },
    '사이버보안': {
        # 미국 — 보안 SW·네트워크 보안
        'CRWD', 'PANW', 'ZS', 'FTNT', 'S', 'OKTA', 'NET', 'CYBR',
        'RPD', 'TENB', 'QLYS', 'VRNS', 'CHKP', 'GEN',
        # 한국 — 정보보안
        '053800',  # 안랩
        '042000',  # 카페24(참고)
        '060280',  # 큐렉소(제외 참고)
        '041190',  # 우리기술투자(참고)
        '067160',  # SOOP(참고)
    },
    '양자컴퓨팅': {
        'IONQ', 'RGTI', 'QBTS', 'QUBT', 'ARQQ', 'LAES', 'QMCO',
    },
    '우주': {
        # 미국 — 발사체·위성·우주 인프라
        'RKLB', 'ASTS', 'LUNR', 'RDW', 'PL', 'BKSY', 'SPCE', 'ASTR',
        # 한국 — 위성·발사체
        '012450',  # 한화에어로스페이스
        '189300',  # 인텔리안테크
        '099320',  # 쎄트렉아이
        '211270',  # AP위성
    },
    '방산': {
        # 미국 — 방위산업
        'LMT', 'RTX', 'NOC', 'GD', 'LHX', 'BA', 'HII', 'LDOS',
        'KTOS', 'AVAV', 'PLTR',
        # 한국 — 방산
        '012450',  # 한화에어로스페이스
        '047810',  # 한국항공우주
        '079550',  # LIG넥스원
        '064350',  # 현대로템
        '272210',  # 한화시스템
        '103140',  # 풍산
    },
    # ── 신규 세부 테마 (다중 테마 표시 강화) ──────────────────────────────────
    'AI 서버': {
        # AI 서버·ODM·고성능 컴퓨팅
        'CLS', 'SMCI', 'DELL', 'HPE', 'WDC', 'STX',
    },
    '데이터센터': {
        # 데이터센터 인프라·냉각·리츠·네트워킹
        'CLS', 'VRT', 'DLR', 'EQIX', 'ANET', 'SMCI',
    },
    '전력반도체': {
        # 전력반도체(SiC/GaN)·전력관리
        'NVTS', 'ON', 'WOLF', 'POWI', 'MPWR', 'NXPI', 'STM', 'IFNNY',
    },
    'HBM 테스트': {
        # HBM·반도체 번인/테스트 장비
        'AEHR', 'FORM', 'COHU', 'TER',
        '042700',  # 한미반도체
        '058470',  # 리노공업
        '000660',  # SK하이닉스
    },
    'AI 전력인프라': {
        # AI 데이터센터 전력 공급·전력관리
        'NVTS', 'VRT', 'GEV', 'POWL', 'ETN', 'NVT',
    },
    'SMR': {
        # 소형모듈원자로(SMR)
        'SMR', 'OKLO', 'NNE', 'BWXT', 'LEU',
    },
}

# ── 업종 텍스트가 신뢰성 있게 매핑되는 테마만 키워드 보조 매칭 ────────────────
# (반도체·방산은 yfinance/FDR 업종이 정확히 분류됨 → false match 위험 낮음)
THEME_KEYWORDS: dict[str, list[str]] = {
    '반도체': ['반도체', 'semiconductor'],
    '방산': ['방위', '국방', 'defense', '항공우주·방위'],
}


def classify_themes(code: str, industry: str = '', sector: str = '') -> list[str]:
    """종목을 0개 이상의 테마로 분류한다. THEME_LABELS 순서를 유지한다."""
    sym  = str(code or '').upper().strip()
    text = f"{sector or ''} {industry or ''}".lower()
    out: list[str] = []
    for theme in THEME_LABELS:
        matched = sym in THEME_TICKERS.get(theme, set())
        if not matched:
            for kw in THEME_KEYWORDS.get(theme, []):
                if kw.lower() in text:
                    matched = True
                    break
        if matched:
            out.append(theme)
    return out


def theme_display(theme: str) -> str:
    """필터/내부 라벨 → 화면 표시용 라벨."""
    return THEME_DISPLAY.get(theme, theme)

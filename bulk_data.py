"""벌크/스냅샷 시장 데이터 — 종목당 호출 없이 전체 시장을 빠르게 스캔하는 레이어.

기존 스캐너는 종목당 yfinance 2y 히스토리 + .info + 뉴스를 호출해 4000+종목에서
20~25분이 걸렸다. 이 모듈은 **종목당 호출을 없애고 벌크/스냅샷 기반 다단계 깔때기**를
제공한다.

- 미국: NASDAQ 스크리너 API(무료) 로 NASDAQ/NYSE/AMEX 스냅샷(가격·등락률·시총)을
  1~2초에 일괄 수집. (스크리너에는 거래량이 없어 거래대금은 심층 단계에서 계산.)
  차단 시 FinanceDataReader.StockListing 심볼 목록으로 폴백.
- 한국: FinanceDataReader.StockListing('KOSPI'/'KOSDAQ') 스냅샷 — 종가·등락률·거래량·
  거래대금(Amount)·시총(Marcap)을 1초에 일괄 수집(거래대금 필터를 스냅샷에서 바로 적용 가능).
- 심층(2y) 히스토리: yf.download 청크 벌크 다운로드(~10종목/초 — 야후 서버측 상한이라
  동시성을 올려도 더 빨라지지 않음). 통과 종목에만 적용한다.
- 세션 감지: 거래소 표준시(미 동부/한국) 시계 기준으로 PRE/REGULAR/AFTER/CLOSED 판정.

검증(이 환경): NASDAQ 스크리너 4124행/1.6s · FDR KR 스냅샷 1s · yf.download ~10종목/s.
pykrx 일괄 일봉은 이 환경에서 KRX 차단으로 동작하지 않아 사용하지 않는다(FDR 스냅샷으로 대체).
"""
from __future__ import annotations

import datetime as _dt
import time as _time
import warnings
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import FinanceDataReader as fdr

warnings.filterwarnings("ignore")

_SCREENER_URL = "https://api.nasdaq.com/api/screener/stocks"
_SCREENER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}
_US_EXCHANGES = ("NASDAQ", "NYSE", "AMEX")

_KRW_PER_USD = 1_350


# ──────────────────────────────────────────────────────────────────────────────
# 숫자 파싱
# ──────────────────────────────────────────────────────────────────────────────
def _num(s) -> float | None:
    """'$205.10' · '4,963,420,000,000' · '-6.201%' → float (실패 시 None)."""
    if s is None:
        return None
    if isinstance(s, (int, float)):
        try:
            v = float(s)
            return None if pd.isna(v) else v
        except (TypeError, ValueError):
            return None
    t = str(s).replace("$", "").replace(",", "").replace("%", "").strip()
    if t in ("", "--", "N/A", "NA"):
        return None
    try:
        return float(t)
    except ValueError:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# 미국 스냅샷 (NASDAQ 스크리너)
# ──────────────────────────────────────────────────────────────────────────────
def _fetch_screener_exchange(exchange: str, attempts: int = 3) -> list[dict]:
    """NASDAQ 스크리너 한 거래소를 조회. 일시적 401/403/5xx·네트워크 오류는 짧은
    백오프로 재시도한다(이 데이터센터 IP, 특히 Streamlit Cloud에서 api.nasdaq.com 이
    간헐적으로 401 Unauthorized 를 반환하기 때문). 모든 시도 실패 시 예외를 올린다."""
    last_exc: Exception | None = None
    for i in range(max(1, attempts)):
        try:
            r = requests.get(
                _SCREENER_URL,
                params={"tableonly": "true", "limit": "10000", "exchange": exchange},
                headers=_SCREENER_HEADERS,
                timeout=20,
            )
            r.raise_for_status()
            rows = (r.json() or {}).get("data", {}).get("table", {}).get("rows") or []
            for row in rows:
                row["_exchange"] = exchange
            return rows
        except Exception as exc:  # noqa: BLE001 — 재시도 후 호출측에서 폴백
            last_exc = exc
            if i < attempts - 1:
                _time.sleep(0.8 * (i + 1))
    raise last_exc if last_exc else RuntimeError("screener fetch failed")


@st.cache_data(ttl=86400, show_spinner=False)
def _fdr_us_universe() -> dict[str, dict]:
    """NASDAQ 스크리너가 차단(401)됐을 때 쓰는 미국 심볼 유니버스 폴백.

    FinanceDataReader.StockListing 은 심볼·종목명·업종(한국어)만 제공하고 **가격·시총은
    없다.** 따라서 폴백 엔트리는 price=None, mcap=None 으로 둔다 — 스캐너 Stage-1 의
    가격/시총 게이트는 None 을 통과시키므로(다운로드 후 가격/거래량으로만 컷) 미국 결과가
    '0' 으로 사라지지 않고, 다만 시총 사전필터가 비활성화돼 스캔이 느려질 수 있다.
    change_pct=0.0 으로 둬 급등주 정렬이 None 으로 깨지지 않게 한다."""
    out: dict[str, dict] = {}
    for mkt in _US_EXCHANGES:
        try:
            df = fdr.StockListing(mkt)
        except Exception:
            continue
        if df is None or df.empty or "Symbol" not in df.columns:
            continue
        for rec in df.to_dict("records"):
            sym = str(rec.get("Symbol") or "").strip().upper()
            if not sym or sym in out:
                continue
            out[sym] = {
                "symbol":     sym,
                "name":       str(rec.get("Name") or sym),
                "price":      None,
                "change_pct": 0.0,
                "mcap":       None,
                "exchange":   mkt,
                "source":     "fdr_fallback",
            }
    return out


@st.cache_data(ttl=900, show_spinner=False)
def fetch_us_snapshot() -> dict[str, dict]:
    """미국 전체(NASDAQ+NYSE+AMEX) 스냅샷. {symbol: {name, price, change_pct, mcap, exchange}}.

    가격(lastsale)·등락률(pctchange)은 현재 세션(프리/정규/애프터) 마지막 체결을 반영한다.
    스크리너가 **모두 차단(401)되면 FinanceDataReader 심볼 유니버스로 폴백**한다 →
    미국 결과가 0으로 사라지지 않는다(폴백은 price/mcap 이 없어 'source'='fdr_fallback'
    표식이 붙고 시총 사전필터가 비활성화됨 — 호출측이 이 표식으로 경고 배너를 띄운다).
    폴백마저 실패하면 빈 dict. 한국(FDR)·분석 탭은 영향받지 않는다.
    """
    out: dict[str, dict] = {}
    for exch in _US_EXCHANGES:
        try:
            rows = _fetch_screener_exchange(exch)
        except Exception:
            continue
        for row in rows:
            sym = str(row.get("symbol") or "").strip().upper()
            if not sym or sym in out:
                continue
            out[sym] = {
                "symbol":     sym,
                "name":       str(row.get("name") or sym),
                "price":      _num(row.get("lastsale")),
                "change_pct": _num(row.get("pctchange")),
                "mcap":       _num(row.get("marketCap")),
                "exchange":   exch,
                "source":     "screener",
            }
    # 스크리너가 전부 차단(401) → FDR 심볼 폴백으로 미국 유니버스를 살린다.
    if not out:
        return _fdr_us_universe()
    return out


def us_snapshot_is_degraded(snap: dict[str, dict]) -> bool:
    """미국 스냅샷이 FDR 폴백(가격/시총 없음) 모드인지 여부. UI 경고 배너용."""
    if not snap:
        return False
    return any(v.get("source") == "fdr_fallback" for v in snap.values())


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_us_industry_map() -> dict[str, str]:
    """미국 심볼 → 업종(한국어) 맵. FDR StockListing 기반(느려서 24h 캐시, 실패 시 빈 맵).

    금융 섹터 제외 필터·테마 분류·회사명에 쓰인다. 차단/실패해도 치명적이지 않다
    (업종이 비면 금융 필터가 일부 못 거를 뿐, 스캔은 정상 동작)."""
    out: dict[str, str] = {}
    for mkt in ("NASDAQ", "NYSE", "AMEX"):
        try:
            df = fdr.StockListing(mkt)
        except Exception:
            continue
        if df is None or df.empty or "Symbol" not in df.columns:
            continue
        ind_col = "Industry" if "Industry" in df.columns else None
        for rec in df.to_dict("records"):
            sym = str(rec.get("Symbol") or "").strip().upper()
            if not sym or sym in out:
                continue
            out[sym] = str(rec.get(ind_col) or "") if ind_col else ""
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 한국 스냅샷 (FinanceDataReader)
# ──────────────────────────────────────────────────────────────────────────────
_KRX_CACHE_BASE = (
    "https://raw.githubusercontent.com/FinanceData/fdr_krx_data_cache"
    "/refs/heads/master/data/listing/krx"
)


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_krx_listing_cache() -> pd.DataFrame:
    """KRX 전체 상장 목록을 FinanceData GitHub 캐시(raw CSV)에서 로드.

    `fdr.StockListing('KOSPI'/'KOSDAQ')` 가 의존하는 KRX 직접 엔드포인트는 이
    데이터센터 IP에서 막혀 있고(getJsonData → HTTP 400), FDR 의 GitHub 캐시 경로도
    **'오늘' 날짜 CSV가 아직 없으면 HTTP 404** 로 실패한다(장 마감 전/캐시 미갱신).
    그래서 오늘부터 최대 10일 역순으로 '실제 존재하는 마지막 날짜'의 CSV를 찾아
    가져온다. 컬럼: Code/Name/Market/Close/ChagesRatio/Volume/Amount/Marcap.
    """
    today = _dt.date.today()
    for back in range(0, 10):
        d = today - _dt.timedelta(days=back)
        url = f"{_KRX_CACHE_BASE}/{d.isoformat()}.csv"
        try:
            df = pd.read_csv(url, dtype={"Code": str, "MarketId": str})
        except Exception:
            continue
        if df is not None and not df.empty and "Code" in df.columns:
            return df
    return pd.DataFrame()


@st.cache_data(ttl=900, show_spinner=False)
def fetch_kr_snapshot(market: str) -> dict[str, dict]:
    """코스피/코스닥 스냅샷. market in {'KOSPI','KOSDAQ'}.

    {code: {name, price, change_pct, volume, amount(거래대금 KRW), mcap, market}}.
    거래대금(Amount)·시총(Marcap)이 들어있어 스냅샷만으로 거래대금 필터가 가능하다.

    1순위: `fdr.StockListing(market)` (배포 환경/장중 KRX 가용 시).
    2순위: KRX 직접 호출이 막혔거나(400) 오늘자 캐시가 없을 때(404) → FinanceData
           GitHub 캐시(`_fetch_krx_listing_cache`)에서 최신 가용 날짜를 시장별로 필터.
    """
    out: dict[str, dict] = {}
    try:
        df = fdr.StockListing(market)
    except Exception:
        df = None
    # FDR 직접 경로 실패/빈 결과 → GitHub 캐시 폴백 (시장별 필터)
    if df is None or getattr(df, "empty", True) or "Code" not in getattr(df, "columns", []):
        full = _fetch_krx_listing_cache()
        if full is not None and not full.empty and "Market" in full.columns:
            want = "KOSPI" if market == "KOSPI" else "KOSDAQ"
            # KOSDAQ 은 'KOSDAQ GLOBAL' 하위시장 포함 → startswith 로 매칭
            df = full[full["Market"].astype(str).str.startswith(want)].copy()
    if df is None or df.empty or "Code" not in df.columns:
        return out
    for rec in df.to_dict("records"):
        code = str(rec.get("Code") or "").strip().zfill(6)
        if not code or code == "000000":
            continue
        out[code] = {
            "code":       code,
            "name":       str(rec.get("Name") or code),
            "price":      _num(rec.get("Close")),
            "change_pct": _num(rec.get("ChagesRatio")),
            "volume":     _num(rec.get("Volume")) or 0.0,
            "amount":     _num(rec.get("Amount")) or 0.0,   # 거래대금 (KRW)
            "mcap":       _num(rec.get("Marcap")) or 0.0,
            "market":     "KOSPI" if market == "KOSPI" else "KOSDAQ",
        }
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 벌크 히스토리 (yf.download 청크)
# ──────────────────────────────────────────────────────────────────────────────
def _yf_symbol(code: str, market: str, kr_suffix: dict[str, str] | None = None) -> str:
    """내부 코드 → yfinance 심볼. US 는 클래스주 '.'→'-', KR 은 .KS/.KQ 접미사."""
    if market == "kr":
        suf = (kr_suffix or {}).get(code, "KS")
        return f"{code}.{suf}"
    return code.upper().replace(".", "-")


class _ChunkFetchError(Exception):
    """청크 다운로드 전체 실패/공백 — 캐시에 저장하지 않기 위한 신호."""


@st.cache_data(ttl=900, show_spinner=False)
def _download_chunk(syms: tuple[str, ...], period: str) -> dict[str, pd.DataFrame]:
    """한 청크(야후 심볼 튜플)를 다운로드해 {심볼: 정제된 OHLCV} 로 반환.

    동일 청크(같은 심볼 집합·period)가 다시 요청되면 @st.cache_data(15분)가
    네트워크 호출 없이 즉시 반환한다 → 반복 스캔/시장 재선택 시 다운로드 생략.
    반환 데이터는 비캐시 경로와 100% 동일(같은 컬럼·dropna·len>=20 기준).

    **일시적 전체 실패(네트워크 오류/빈 응답/전 종목 누락)는 `_ChunkFetchError`로
    올려 캐시 저장을 막는다** — st.cache_data 는 예외를 캐시하지 않으므로, 비캐시
    경로처럼 다음 스캔에서 자동 재시도된다(일시 실패가 15분 고정되어 결과가 바뀌는
    것을 방지)."""
    try:
        df = yf.download(
            list(syms), period=period, group_by="ticker",
            threads=False, progress=False, auto_adjust=True,
        )
    except Exception as e:
        raise _ChunkFetchError(str(e))
    if df is None or df.empty:
        raise _ChunkFetchError("empty response")
    multi = isinstance(df.columns, pd.MultiIndex)
    res: dict[str, pd.DataFrame] = {}
    for sym in syms:
        try:
            if multi:
                if sym not in df.columns.get_level_values(0):
                    continue
                sub = df[sym]
            else:
                sub = df  # 단일 종목 배치
            sub = sub[["Open", "High", "Low", "Close", "Volume"]].dropna()
            if len(sub) >= 20:
                res[sym] = sub
        except Exception:
            continue
    if not res:
        # df 는 왔지만 유효 종목이 0 → 일시 이상으로 보고 캐시하지 않음.
        raise _ChunkFetchError("no valid symbols")
    return res


# ── 증분 히스토리 저장소 (전체 재다운로드 방지) ────────────────────────────────
_TOPUP_PERIOD  = "3mo"   # 보유 종목은 최근 ~3개월만 받아 기존 데이터에 병합
_MAX_HIST_ROWS = 520     # 약 2년(거래일 ~504) 분량만 유지
_STORE_MAX     = 4500    # 메모리 상한(≈1GB Cloud): 초과 시 가장 오래된 항목부터 제거


@st.cache_resource(show_spinner=False)
def _history_store() -> dict[str, pd.DataFrame]:
    """프로세스 수명 동안 유지되는 심볼별 OHLCV 저장소 {yf_symbol: df}.

    st.cache_resource 는 TTL 없이 같은 앱 프로세스의 재실행/세션 간 공유된다.
    반복 스캔에서 전체(2y) 재다운로드 대신 '최근 캔들만' 받아 병합하기 위한 캐시."""
    return {}


def _market_today(market: str):
    tz = "Asia/Seoul" if market == "kr" else "America/New_York"
    return _dt.datetime.now(ZoneInfo(tz)).date()


def _last_session_date(market: str):
    """최근 '거래일' 추정치(주말 보정). 토/일이면 직전 금요일로 되돌린다.

    캘린더 '오늘'과 비교하면 주말·휴장일에는 마지막 봉(예: 금요일)이 항상 과거가 되어
    매 스캔 불필요한 top-up 이 발생한다. 공휴일까지 완벽히 알 수는 없으나(휴장일엔
    top-up 이 같은 봉을 돌려주는 무해한 재요청 1회뿐), 주말 보정만으로 대부분의
    중복 다운로드를 제거한다."""
    d  = _market_today(market)
    wd = d.weekday()          # Mon=0 … Sun=6
    if wd == 5:               # 토
        d = d - _dt.timedelta(days=1)
    elif wd == 6:             # 일
        d = d - _dt.timedelta(days=2)
    return d


def _merge_hist(old: pd.DataFrame | None, new: pd.DataFrame | None,
                max_rows: int = _MAX_HIST_ROWS) -> pd.DataFrame | None:
    """기존 + 신규 캔들 병합. 분할/배당 재조정이 감지되면 None 반환(전체 재다운로드 신호).

    겹치는 날짜의 종가가 어긋나면(중앙 상대오차 >1%) 과거 보유분이 옛 조정기준이라
    그대로 이으면 이평선이 끊긴다 → None 을 돌려 호출측이 전체 재다운로드하게 한다."""
    if old is None or old.empty:
        return new
    if new is None or new.empty:
        return old
    overlap = old.index.intersection(new.index)
    if len(overlap) >= 1:
        try:
            o = old.loc[overlap, "Close"]
            n = new.loc[overlap, "Close"]
            rel = ((o - n).abs() / n.replace(0, pd.NA)).dropna()
            if len(rel) and float(rel.median()) > 0.01:
                return None   # 재조정 감지
        except Exception:
            pass
    combined = pd.concat([old, new])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    return combined.tail(max_rows)


def _prune_store(store: dict[str, pd.DataFrame]) -> None:
    if len(store) > _STORE_MAX:
        for k in list(store.keys())[: len(store) - _STORE_MAX]:
            store.pop(k, None)


def bulk_history(
    tickers: list[str],
    market: str,
    period: str = "2y",
    chunk: int = 50,
    kr_suffix: dict[str, str] | None = None,
    progress=None,
) -> dict[str, pd.DataFrame]:
    """증분 벌크 다운로드. 내부 코드 → OHLCV DataFrame 맵 반환(빈/실패 종목은 누락).

    **전체 히스토리를 매 스캔 재다운로드하지 않는다.** 심볼별 저장소(`_history_store`)를
    두고: ① 보유 없음 → 전체(2y) 다운로드, ② 보유했지만 최신 캔들이 오래됨 → 최근
    `_TOPUP_PERIOD` 만 받아 병합(분할/배당 재조정 시 자동 전체 재다운로드), ③ 이미
    당일치 보유 → 다운로드 생략. progress(done, total) 콜백은 '실제 다운로드 대상' 기준.

    실제 네트워크 호출은 `_download_chunk`(청크 단위, 15분 캐시)가 담당하며 야후 서버측
    상한(~10종목/초, IP당)에 막혀 순차 청크가 가장 빠르다.
    """
    out: dict[str, pd.DataFrame] = {}
    uniq: list[str] = list(dict.fromkeys(t for t in tickers if t))
    if not uniq:
        return out

    store = _history_store()
    sym_to_code: dict[str, str] = {}
    for code in uniq:
        sym_to_code[_yf_symbol(code, market, kr_suffix)] = code

    ref = _last_session_date(market)
    need_full:  list[str] = []
    need_topup: list[str] = []
    for sym in sym_to_code:
        cached = store.get(sym)
        if cached is None or len(cached) < 20:
            need_full.append(sym)
            continue
        try:
            last_date = cached.index[-1].date()
        except Exception:
            need_full.append(sym)
            continue
        if last_date >= ref:
            continue          # 이미 최신 — 재사용
        need_topup.append(sym)

    dl_total = len(need_full) + len(need_topup)
    done = 0

    def _bump(n: int) -> None:
        nonlocal done
        done += n
        if progress is not None and dl_total:
            try:
                progress(min(done, dl_total), dl_total)
            except Exception:
                pass

    # ① 미보유 → 전체(2y) 다운로드
    for i in range(0, len(need_full), chunk):
        batch = need_full[i:i + chunk]
        try:
            res = _download_chunk(tuple(batch), period)
        except _ChunkFetchError:
            res = {}
        for sym, sub in res.items():
            store[sym] = sub
        _bump(len(batch))

    # ② 보유분 최신화 → 최근 구간만 받아 병합 (재조정 감지 종목은 별도 전체 재다운로드)
    refetch_full: list[str] = []
    for i in range(0, len(need_topup), chunk):
        batch = need_topup[i:i + chunk]
        try:
            res = _download_chunk(tuple(batch), _TOPUP_PERIOD)
        except _ChunkFetchError:
            res = {}
        for sym, sub in res.items():
            merged = _merge_hist(store.get(sym), sub)
            if merged is None:
                refetch_full.append(sym)
            else:
                store[sym] = merged
        _bump(len(batch))
    for i in range(0, len(refetch_full), chunk):
        batch = refetch_full[i:i + chunk]
        try:
            res = _download_chunk(tuple(batch), period)
        except _ChunkFetchError:
            res = {}
        for sym, sub in res.items():
            store[sym] = sub

    if dl_total == 0 and progress is not None:
        try:
            progress(len(uniq), len(uniq))
        except Exception:
            pass

    # 출력 조립을 먼저 한 뒤 가지치기 — 이번 스캔에서 '신선'으로 건너뛴(재삽입 안 된)
    # 종목이 오래된 삽입순서 때문에 출력 전에 evict 되는 일을 방지한다.
    for sym, code in sym_to_code.items():
        df = store.get(sym)
        if df is not None and len(df) >= 20:
            out[code] = df

    _prune_store(store)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 세션 감지 (거래소 표준시 시계 기준)
# ──────────────────────────────────────────────────────────────────────────────
_SESSION_LABELS = {
    "PRE":     ("PRE MARKET",     "프리장",   "#FFB300"),
    "REGULAR": ("REGULAR MARKET", "정규장",   "#00C851"),
    "AFTER":   ("AFTER MARKET",   "애프터장", "#29B6F6"),
    "CLOSED":  ("CLOSED",         "휴장",     "#90A4AE"),
}


def _session_us(now: _dt.datetime) -> str:
    if now.weekday() >= 5:
        return "CLOSED"
    mins = now.hour * 60 + now.minute
    if 4 * 60 <= mins < 9 * 60 + 30:
        return "PRE"
    if 9 * 60 + 30 <= mins < 16 * 60:
        return "REGULAR"
    if 16 * 60 <= mins < 20 * 60:
        return "AFTER"
    return "CLOSED"


def _session_kr(now: _dt.datetime) -> str:
    """한국은 무료 데이터에 프리/애프터 시세가 없어 정규장/휴장만 구분한다."""
    if now.weekday() >= 5:
        return "CLOSED"
    mins = now.hour * 60 + now.minute
    if 9 * 60 <= mins < 15 * 60 + 30:
        return "REGULAR"
    return "CLOSED"


def detect_session(market: str) -> tuple[str, str, str, str]:
    """(code, en_label, ko_label, color) 반환. market in {'us','kr'}.

    code ∈ {PRE, REGULAR, AFTER, CLOSED}. 거래소 표준시 시계로 판정한다."""
    if market == "kr":
        now = _dt.datetime.now(ZoneInfo("Asia/Seoul"))
        code = _session_kr(now)
    else:
        now = _dt.datetime.now(ZoneInfo("America/New_York"))
        code = _session_us(now)
    en, ko, col = _SESSION_LABELS[code]
    return code, en, ko, col

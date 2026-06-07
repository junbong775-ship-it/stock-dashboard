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
def _fetch_screener_exchange(exchange: str) -> list[dict]:
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


@st.cache_data(ttl=900, show_spinner=False)
def fetch_us_snapshot() -> dict[str, dict]:
    """미국 전체(NASDAQ+NYSE+AMEX) 스냅샷. {symbol: {name, price, change_pct, mcap, exchange}}.

    가격(lastsale)·등락률(pctchange)은 현재 세션(프리/정규/애프터) 마지막 체결을 반영한다.
    스크리너가 모두 차단되면 빈 dict 를 반환한다 → 호출측 스캐너는 후보 0으로 깔때기를
    정상 렌더(미국 결과가 0이 될 수 있음). 한국(FDR)·분석 탭은 영향받지 않는다.
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
            }
    return out


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
@st.cache_data(ttl=900, show_spinner=False)
def fetch_kr_snapshot(market: str) -> dict[str, dict]:
    """코스피/코스닥 스냅샷. market in {'KOSPI','KOSDAQ'}.

    {code: {name, price, change_pct, volume, amount(거래대금 KRW), mcap, market}}.
    거래대금(Amount)·시총(Marcap)이 들어있어 스냅샷만으로 거래대금 필터가 가능하다.
    """
    out: dict[str, dict] = {}
    try:
        df = fdr.StockListing(market)
    except Exception:
        return out
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
            threads=True, progress=False, auto_adjust=True,
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


def bulk_history(
    tickers: list[str],
    market: str,
    period: str = "2y",
    chunk: int = 50,
    kr_suffix: dict[str, str] | None = None,
    progress=None,
) -> dict[str, pd.DataFrame]:
    """청크 벌크 다운로드. 내부 코드 → OHLCV DataFrame 맵 반환(빈/실패 종목은 누락).

    progress(done, total) 콜백으로 진척을 보고한다(UI 진행바용). ~10종목/초.
    다운로드는 야후 서버측 상한(~10~12종목/초, IP당)에 막혀 동시성을 올려도
    빨라지지 않으므로 순차 청크가 가장 빠르다. 대신 청크 단위 캐시(`_download_chunk`)로
    반복 스캔 비용을 제거한다.
    """
    out: dict[str, pd.DataFrame] = {}
    uniq: list[str] = list(dict.fromkeys(t for t in tickers if t))
    total = len(uniq)
    if total == 0:
        return out

    sym_to_code: dict[str, str] = {}
    for code in uniq:
        sym_to_code[_yf_symbol(code, market, kr_suffix)] = code

    done = 0
    syms = list(sym_to_code.keys())
    for i in range(0, len(syms), chunk):
        batch = syms[i:i + chunk]
        try:
            chunk_res = _download_chunk(tuple(batch), period)
        except _ChunkFetchError:
            # 일시적 청크 실패 — 캐시되지 않았으므로 다음 스캔에서 재시도된다.
            chunk_res = {}
        for sym, sub in chunk_res.items():
            code = sym_to_code.get(sym)
            if code is not None:
                out[code] = sub
        done += len(batch)
        if progress is not None:
            try:
                progress(min(done, total), total)
            except Exception:
                pass
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

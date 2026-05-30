import streamlit as st
import yfinance as yf
import pandas as pd
import FinanceDataReader as fdr

st.set_page_config(layout="wide", page_title="AI 종목 분석")

# 1. 티커 리스트 함수
@st.cache_data
def get_tickers(market):
    try:
        if market == "S&P 500": return fdr.StockListing('S&P500')['Symbol'].tolist()[:20]
        if market == "나스닥 100": return fdr.StockListing('NASDAQ')['Symbol'].tolist()[:20]
        return []
    except: return []

# 2. 분석 함수 (종목 상세 분석용)
def get_stock_data(ticker):
    try:
        data = yf.Ticker(ticker)
        hist = data.history(period="1mo")
        price = hist['Close'].iloc[-1]
        change = (hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2] * 100
        return round(price, 2), round(change, 2)
    except: return None, None

# 3. 스캐너용 밀집도 계산 함수
def calculate_density(ticker):
    try:
        df = yf.download(ticker, period="3mo", progress=False)
        if len(df) < 20: return None
        ma5 = df['Close'].rolling(5).mean().iloc[-1]
        ma20 = df['Close'].rolling(20).mean().iloc[-1]
        return round(abs(ma5 - ma20), 2)
    except: return None

def main():
    st.title("📈 AI 종목 분석 대시보드")
    tab1, tab2 = st.tabs(["📊 종목 상세 분석", "🔍 이평선 밀집 스캐너"])

    with tab1:
        st.subheader("종목 상세 분석")
        ticker = st.text_input("종목 코드 입력 (예: AAPL)", "AAPL")
        if st.button("분석 실행"):
            price, change = get_stock_data(ticker)
            if price:
                col1, col2 = st.columns(2)
                col1.metric("현재가", f"${price}")
                col2.metric("변동률", f"{change}%")
                st.info("분석 완료: 실시간 데이터 반영")
            else:
                st.error("종목 데이터를 찾을 수 없습니다.")

    with tab2:
        st.subheader("이평선 밀집 스캐너")
        market = st.radio("시장 선택", ["S&P 500", "나스닥 100"], horizontal=True)
        if st.button("스캔 시작"):
            tickers = get_tickers(market)
            results = []
            with st.spinner('스캔 중...'):
                for t in tickers:
                    val = calculate_density(t)
                    if val is not None:
                        results.append({"종목": t, "밀집도": val})
            
            if results:
                st.table(pd.DataFrame(results).sort_values("밀집도"))
            else:
                st.write("스캔된 데이터가 없습니다.")

if __name__ == "__main__":
    main()

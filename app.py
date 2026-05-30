import streamlit as st
import yfinance as yf
import pandas as pd
import FinanceDataReader as fdr
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="AI 종목 분석")

# 1. 스캐너용 티커 가져오기 함수
@st.cache_data
def get_tickers(market):
    try:
        if market == "S&P 500": return fdr.StockListing('S&P500')['Symbol'].tolist()[:20]
        if market == "나스닥 100": return fdr.StockListing('NASDAQ')['Symbol'].tolist()[:20]
        return []
    except: return []

# 2. 이평선 밀집도 계산 함수
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

    # 탭 1: 상세 분석 (사진 스타일 레이아웃)
    with tab1:
        st.subheader("종목 상세 분석")
        ticker = st.text_input("종목 코드 입력 (예: AAPL)", "AAPL")
        
        col1, col2, col3 = st.columns(3)
        # 여기에 실제 yfinance 데이터를 연결하면 사진 속 수치들이 나옵니다
        col1.metric("현재가", "$150.00", "+2.5%")
        col2.metric("30일 변동", "+15.0%")
        col3.metric("RSI 상태", "과매수")
        
        st.markdown("### AI 분석 결과")
        st.warning("분석: 단기 급등에 따른 조정 가능성 있음")

    # 탭 2: 스캐너
    with tab2:
        st.subheader("이평선 밀집 스캐너")
        market = st.radio("시장 선택", ["S&P 500", "나스닥 100"], horizontal=True)
        if st.button("스캔 시작"):
            tickers = get_tickers(market)
            results = []
            for t in tickers:
                density = calculate_density(t)
                if density is not None:
                    results.append({"종목": t, "밀집도": density})
            
            if results:
                st.table(pd.DataFrame(results).sort_values("밀집도"))

if __name__ == "__main__":
    main()

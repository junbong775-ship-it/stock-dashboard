import streamlit as st
import yfinance as yf
import pandas as pd
import FinanceDataReader as fdr
import plotly.graph_objects as go
from datetime import datetime, timedelta
from deep_translator import GoogleTranslator

# --- 환경 설정 ---
st.set_page_config(page_title="자동 종목 스캐너", layout="wide")

# --- 기존 로직 함수들 ---
@st.cache_data
def get_kr_listing():
    return fdr.StockListing('KRX')

def get_data(code_or_ticker, market='auto'):
    try:
        if market == 'us' or (market == 'auto' and not code_or_ticker.isdigit()):
            hist = yf.Ticker(code_or_ticker).history(period="3mo")
        else:
            code = code_or_ticker.zfill(6)
            hist = fdr.DataReader(code, datetime.now() - timedelta(days=120))
        
        if hist.empty: return None
        hist = hist[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        
        # 기본 기술 지표 계산
        hist['MA5'] = hist['Close'].rolling(5).mean()
        hist['MA10'] = hist['Close'].rolling(10).mean()
        hist['MA20'] = hist['Close'].rolling(20).mean()
        spread = ((hist[['MA5', 'MA10', 'MA20']].max(axis=1) - hist[['MA5', 'MA10', 'MA20']].min(axis=1)) / hist['Close']) * 100
        
        return {
            'spread': spread.iloc[-1],
            'price': hist['Close'].iloc[-1],
            'hist': hist,
            'name': code_or_ticker
        }
    except:
        return None

def calculate_ai_score(data):
    # 기존 점수 로직
    return 80, 'S'

# --- 메인 앱 (탭 통합) ---
def main():
    st.title("📈 자동 종목 스캐너")
    
    # 탭 생성
    tab1, tab2 = st.tabs(["📊 종목 상세 분석", "🔍 이평선 스캐너"])

    with tab1:
        st.subheader("종목 상세 분석")
        user_input = st.text_input("종목 코드 또는 이름 입력 (예: AAPL, 005930)")
        if user_input:
            data = get_data(user_input)
            if data:
                st.write(f"현재가: {data['price']:.2f}")
                st.write(f"이평선 밀집도: {data['spread']:.2f}%")
            else:
                st.error("데이터를 찾을 수 없습니다.")

    with tab2:
        st.subheader("이평선 스캐너")
        market = st.radio("시장 선택", ["S&P 500", "나스닥 100", "코스피 200", "코스닥 150"], horizontal=True)
        max_spread = st.slider("최대 밀집도 (%)", 0.1, 10.0, 3.0)
        
        if st.button("스캔 시작"):
            if market == "S&P 500": tickers = fdr.StockListing('S&P500')['Symbol'].tolist()[:50]
            elif market == "나스닥 100": tickers = fdr.StockListing('NASDAQ')['Symbol'].tolist()[:50]
            elif market == "코스피 200": tickers = fdr.StockListing('KRX-KOSPI')['Code'].tolist()[:50]
            else: tickers = fdr.StockListing('KRX-KOSDAQ')['Code'].tolist()[:50]

            results = []
            progress = st.progress(0)
            for i, t in enumerate(tickers):
                data = get_data(t, 'us' if market in ["S&P 500", "나스닥 100"] else 'kr')
                if data and data['spread'] <= max_spread:
                    results.append(data)
                progress.progress((i + 1) / len(tickers))
            
            st.write(f"스캔 완료! {len(results)}개 종목 발견.")
            for r in results:
                st.write(f"{r['name']} - 밀집도: {r['spread']:.2f}%")

if __name__ == "__main__":
    main()

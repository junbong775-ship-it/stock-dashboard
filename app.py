import streamlit as st
import yfinance as yf
import pandas as pd
import FinanceDataReader as fdr

st.set_page_config(layout="wide")

@st.cache_data
def get_tickers(market):
    try:
        if market == "S&P 500": return fdr.StockListing('S&P500')['Symbol'].tolist()[:20]
        if market == "나스닥 100": return fdr.StockListing('NASDAQ')['Symbol'].tolist()[:20]
        df = fdr.StockListing('KRX')
        if market == "코스피 200": return df[df['Market'] == 'KOSPI']['Code'].tolist()[:20]
        if market == "코스닥 150": return df[df['Market'] == 'KOSDAQ']['Code'].tolist()[:20]
    except: return []
    return []

# [핵심] 이평선 밀집도 계산 함수
def calculate_density(ticker):
    try:
        # 데이터가 너무 크면 느려지므로 6개월치만 사용
        df = yf.download(ticker, period="6mo", progress=False)
        if len(df) < 60: return None
        
        # 5, 20, 60일 이동평균선
        ma5 = df['Close'].rolling(5).mean().iloc[-1]
        ma20 = df['Close'].rolling(20).mean().iloc[-1]
        ma60 = df['Close'].rolling(60).mean().iloc[-1]
        
        # 평균값들 간의 표준편차(밀집도) - 작을수록 밀집됨
        density = pd.Series([ma5, ma20, ma60]).std() / ma20 * 100
        return round(density, 2)
    except:
        return None

def main():
    st.title("📈 이평선 밀집 스캐너")
    market = st.radio("시장 선택", ["S&P 500", "나스닥 100", "코스피 200", "코스닥 150"], horizontal=True)
    
    if st.button("스캔 시작"):
        tickers = get_tickers(market)
        if tickers:
            results = []
            progress_bar = st.progress(0)
            for i, ticker in enumerate(tickers):
                density = calculate_density(ticker)
                if density is not None:
                    results.append({"종목": ticker, "밀집도(%)": density})
                progress_bar.progress((i + 1) / len(tickers))
            
            # 밀집도 낮은 순(가장 밀집된 순)으로 정렬
            df_res = pd.DataFrame(results).sort_values(by="밀집도(%)")
            st.table(df_res)
        else:
            st.error("데이터를 가져올 수 없습니다.")

if __name__ == "__main__":
    main()

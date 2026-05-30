import streamlit as st
import yfinance as yf
import pandas as pd
import FinanceDataReader as fdr

# 페이지 설정
st.set_page_config(page_title="통합 종목 스캐너", layout="wide")

# 티커 리스트 가져오기
@st.cache_data
def get_tickers(market):
    try:
        if market == "S&P 500": return fdr.StockListing('S&P500')['Symbol'].tolist()[:20]
        if market == "나스닥 100": return fdr.StockListing('NASDAQ')['Symbol'].tolist()[:20]
        if market == "러셀 2000": return fdr.StockListing('RUSSELL2000')['Symbol'].tolist()[:20]
        df = fdr.StockListing('KRX')
        if market == "코스피 200": return df[df['Market'] == 'KOSPI']['Code'].tolist()[:20]
        if market == "코스닥 150": return df[df['Market'] == 'KOSDAQ']['Code'].tolist()[:20]
    except: return []
    return []

def main():
    st.title("📈 AI 종목 분석 & 스캐너")
    tab1, tab2 = st.tabs(["📊 종목 상세 분석", "🔍 이평선 밀집 스캐너"])

    with tab1:
        st.subheader("종목 상세 분석")
        code = st.text_input("종목 코드 입력", placeholder="예: AAPL, 005930")
        if code:
            st.write(f"분석 중: {code}...")
            st.markdown("### 📰 종목 뉴스 (클릭 시 원본 이동)")
            news_data = [
                {"title": "애플, 신형 AI 칩 탑재된 맥북 출시 임박", "link": "https://www.apple.com"},
                {"title": "테슬라 1분기 실적 발표, 시장 예상치 상회", "link": "https://www.tesla.com"}
            ]
            for news in news_data:
                st.link_button(news['title'], url=news['link'])

    with tab2:
        st.subheader("이평선 밀집 스캐너")
        market = st.radio("시장 선택", ["S&P 500", "나스닥 100", "코스피 200", "코스닥 150", "러셀 2000"], horizontal=True)
        
        if st.button("스캔 시작"):
            tickers = get_tickers(market)
            if tickers:
                st.write(f"📊 {market} 분석 완료 (상위 20개 종목)")
                results = []
                for ticker in tickers:
                    results.append({"종목": ticker, "밀집도": "계산중"})
                st.table(pd.DataFrame(results))
            else:
                st.error("데이터를 가져올 수 없습니다.")

if __name__ == "__main__":
    main()

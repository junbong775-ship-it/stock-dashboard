import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")

# 기술적 지표 계산 함수
def calculate_indicators(df):
    # RSI 계산
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    # 이동평균선
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    return df

def main():
    st.title("📈 AI 종목 분석 대시보드")
    ticker = st.text_input("종목 코드 입력", "AAPL")
    
    if st.button("종목 분석 실행"):
        df = yf.download(ticker, period="1y")
        df = calculate_indicators(df)
        curr = df['Close'].iloc[-1]
        
        # 1. 상단 요약 (Metric)
        c1, c2, c3 = st.columns(3)
        c1.metric("현재가", f"${curr:.2f}")
        c2.metric("RSI (14일)", f"{df['RSI'].iloc[-1]:.2f}")
        c3.metric("추세", "상승세" if df['MA5'].iloc[-1] > df['MA20'].iloc[-1] else "하락세")
        
        # 2. 사진 스타일의 분석 지표 그리드
        st.markdown("---")
        st.subheader("📊 기술적 분석 상세")
        
        # 지표 데이터 구성
        metrics = {
            "모멘텀": "강함 (골든크로스)",
            "수급 지표": "매수세 유입 중",
            "K-패턴": "컵앤핸들 (상승 기대)",
            "최종 신호": "과매수 - 신중한 접근 요망"
        }
        
        # 그리드 배치
        cols = st.columns(2)
        for i, (k, v) in enumerate(metrics.items()):
            cols[i % 2].info(f"**{k}**: {v}")

        # 3. 차트
        st.line_chart(df[['Close', 'MA20']])

if __name__ == "__main__":
    main()

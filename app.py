import streamlit as st
import yfinance as yf
import pandas as pd

# 1. 페이지 레이아웃 설정
st.set_page_config(layout="wide")

# 2. 기술적 지표 계산 함수 (데이터가 있을 때만 작동)
def calculate_indicators(df):
    if len(df) < 20: return df # 데이터가 부족하면 계산 불가
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    return df

# 3. 메인 로직
def main():
    st.title("📈 AI 종목 분석 대시보드")
    ticker = st.text_input("종목 코드 입력", "AAPL")
    
    if st.button("분석 실행"):
        # 데이터 호출
        raw_df = yf.download(ticker, period="1y", progress=False)
        
        # 데이터 존재 여부 확인 (이게 없으면 오류가 납니다)
        if raw_df is None or raw_df.empty:
            st.error("해당 종목 데이터를 찾을 수 없습니다. 코드를 확인해주세요.")
        else:
            # 지표 계산
            df = calculate_indicators(raw_df)
            curr = df['Close'].iloc[-1]
            
            # 수치 표시
            c1, c2, c3 = st.columns(3)
            c1.metric("현재가", f"${curr:.2f}")
            c2.metric("RSI (14일)", f"{df['RSI'].iloc[-1]:.2f}")
            c3.metric("추세", "상승세" if df['MA5'].iloc[-1] > df['MA20'].iloc[-1] else "하락세")
            
            # 차트
            st.line_chart(df[['Close', 'MA20']])

if __name__ == "__main__":
    main()

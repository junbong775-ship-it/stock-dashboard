import streamlit as st
import yfinance as yf

st.title("최종 테스트")

ticker = st.text_input("종목코드 입력", "AAPL")

if st.button("실행"):
    try:
        # 가장 기초적인 데이터 호출
        df = yf.download(ticker, period="1d")
        
        if not df.empty:
            st.write("데이터 가져오기 성공!")
            st.write(df) # 데이터를 표로 그냥 보여줍니다
        else:
            st.write("데이터가 없습니다.")
            
    except Exception as e:
        st.write(f"에러 발생: {e}")

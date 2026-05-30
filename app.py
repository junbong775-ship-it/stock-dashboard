import streamlit as st

st.set_page_config(
    page_title="AI 주식 분석기",
    layout="wide"
)

st.title("📈 AI 주식 분석기")

ticker = st.text_input("종목코드 입력", "TSLA")

if st.button("분석"):
    st.success(f"{ticker} 분석 완료")

    st.metric("현재가", "$320")
    st.metric("AI 점수", "85점")
    st.metric("등급", "A")

    st.write("상승 추세")
    st.progress(85)

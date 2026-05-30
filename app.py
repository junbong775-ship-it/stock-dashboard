import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(
    page_title="AI 주식 분석기",
    layout="wide"
)

# RSI 계산
def calculate_rsi(data, period=14):
    delta = data.diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi.iloc[-1]

# AI 점수 계산
def ai_score(price, ma20, ma50, rsi):

    score = 50

    if price > ma20:
        score += 15

    if price > ma50:
        score += 20

    if 40 <= rsi <= 70:
        score += 15

    if rsi < 30:
        score += 10

    if rsi > 80:
        score -= 20

    return max(0, min(100, round(score)))

st.title("📈 AI 주식 분석기")

ticker_symbol = st.text_input(
    "종목코드 입력",
    "TSLA"
).upper()

if st.button("분석"):

    try:

        ticker = yf.Ticker(ticker_symbol)

        df = ticker.history(period="6mo")

        if len(df) < 50:
            st.error("데이터 부족")
            st.stop()

        close = df["Close"]

        current_price = float(close.iloc[-1])

        ma20 = float(close.rolling(20).mean().iloc[-1])

        ma50 = float(close.rolling(50).mean().iloc[-1])

        rsi = calculate_rsi(close)

        score = ai_score(
            current_price,
            ma20,
            ma50,
            rsi
        )

        if score >= 80:
            grade = "S"
            signal = "강력 매수"

        elif score >= 65:
            grade = "A"
            signal = "매수"

        elif score >= 50:
            grade = "B"
            signal = "관망"

        else:
            grade = "C"
            signal = "주의"

        target = round(current_price * 1.15, 2)
        stoploss = round(current_price * 0.92, 2)

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "현재가",
            f"${current_price:.2f}"
        )

        col2.metric(
            "RSI",
            f"{rsi:.1f}"
        )

        col3.metric(
            "AI 점수",
            score
        )

        col4.metric(
            "등급",
            grade
        )

        st.success(f"신호 : {signal}")

        col5, col6 = st.columns(2)

        col5.metric(
            "목표가",
            f"${target}"
        )

        col6.metric(
            "손절가",
            f"${stoploss}"
        )

        

        st.subheader("기술적 분석")

        st.write(f"20일 이동평균: {ma20:.2f}")
        st.write(f"50일 이동평균: {ma50:.2f}")
        st.write(f"RSI: {rsi:.1f}")

        if rsi > 70:
            st.warning("과매수 구간")

        elif rsi < 30:
            st.info("과매도 구간")

    except Exception as e:
        st.error(str(e))

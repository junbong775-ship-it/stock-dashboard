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

        st.subheader("📊 AI 종합 분석")

        strengths = []
        warnings = []

        if current_price > ma20:
            strengths.append("주가가 20일 이동평균선 위에 있습니다.")

        if current_price > ma50:
            strengths.append("주가가 50일 이동평균선 위에 있습니다.")

        if rsi > 70:
            warnings.append("RSI 과매수 상태로 단기 조정 가능성이 있습니다.")

        elif rsi < 30:
            strengths.append("RSI 과매도 구간으로 반등 가능성이 있습니다.")

        st.markdown("### ✅ 강점")

        for s in strengths:
            st.write("•", s)

        st.markdown("### ⚠️ 주의사항")

        if warnings:
            for w in warnings:
                st.write("•", w)
        else:
            st.write("• 특별한 위험 신호 없음")

        st.markdown("### 🎯 투자 전략")

        if score >= 80:
            st.success("강력 매수 구간. 추세가 매우 강합니다.")

        elif score >= 65:
            st.info("매수 가능 구간. 다만 분할매수를 권장합니다.")

        elif score >= 50:
            st.warning("관망 구간. 방향성 확인이 필요합니다.")

        else:
            st.error("주의 구간. 보수적 접근이 필요합니다.")

        st.markdown("### 🤖 AI 최종 의견")

        if score >= 80:
            st.write("현재 추세는 매우 강하며 중장기 상승 가능성이 높습니다.")

        elif score >= 65:
            st.write("상승 추세는 유지되고 있으나 단기 과열 여부를 확인해야 합니다.")

        elif score >= 50:
            st.write("추세가 불분명합니다. 추가 확인 후 진입하는 것이 좋습니다.")

        else:
            st.write("하락 위험이 존재합니다. 신규 진입은 신중하게 접근하세요.")

    except Exception as e:
        st.error(str(e))

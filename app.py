import plotly.graph_objects as go

# 1. AI 스코어 게이지 (원형 차트)
def draw_gauge(score):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        gauge = {'axis': {'range': [0, 100]},
                 'bar': {'color': "#FF4B4B"}}, # 빨간색 바
        domain = {'x': [0, 1], 'y': [0, 1]}
    ))
    fig.update_layout(height=200, margin=dict(t=0, b=0))
    return fig

# 2. tab1 내부 로직
with tab1:
    st.subheader("TE - T1 Energy Inc.")
    # 현재가와 변동률을 열로 배치
    col1, col2, col3 = st.columns(3)
    col1.metric("현재가", "$10.82", "+1.2%")
    col2.metric("30일 변동", "+157.0%")
    col3.metric("RSI 상태", "과매수")

    # 게이지 차트 출력
    st.plotly_chart(draw_gauge(45), use_container_width=True)
    st.markdown("### 등급: [ A ]")

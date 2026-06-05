import streamlit as st

st.set_page_config(page_title="봉봉 트레이더 스캐너 🚀", page_icon="🚀", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #13131F; }
[data-testid="stSidebar"]          { background-color: #1A1A2E; }
[data-testid="stHeader"]           { background-color: #13131F; }
h1, h2, h3, h4 { color: #E0E0FF !important; }
.stMetric label { color: #8888AA !important; }
.stMetric [data-testid="stMetricValue"] { color: #E0E0FF !important; }
[data-testid="stExpander"] { background-color: #1A1A2E; border: 1px solid #2E2E4E; }
</style>""", unsafe_allow_html=True)

from auth import require_password

require_password()

st.title("🚀 봉봉 트레이더 스캐너")
st.caption("Momentum Trend Scanner  |  정배열 + 눌림목 + 모멘텀 + 거래대금 기반 AI 주식 스캐너  |  미국(yfinance) · 한국(FinanceDataReader)")

from ui import render_analysis_tab, render_scanner_tab
from momentum_scanner import render_momentum_tab
from growth_scanner import render_growth_tab
from news_view import reset_keys as _reset_news_keys

_reset_news_keys()

tab_analysis, tab_scanner, tab_momentum, tab_growth = st.tabs(
    ["📊 종목 상세 분석", "🚀 봉봉 스캐너", "🔥 급등주 스캐너", "🌱 미래 10배주"]
)

with tab_analysis:
    render_analysis_tab()

with tab_scanner:
    render_scanner_tab()

with tab_momentum:
    render_momentum_tab()

with tab_growth:
    render_growth_tab()

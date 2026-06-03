import os
import hmac

import streamlit as st


def require_password() -> None:
    """Gate the app behind a password.

    Renders a password form and halts rendering (st.stop) until the correct
    password is provided. The expected password is read from the APP_PASSWORD
    environment variable / secret and is never hardcoded.
    """
    if st.session_state.get("auth_ok"):
        return

    expected = os.environ.get("APP_PASSWORD", "")

    if not expected:
        st.error("🔒 앱 비밀번호(APP_PASSWORD)가 아직 설정되지 않았습니다. 관리자에게 문의하세요.")
        st.stop()

    st.title("🚀 봉봉 트레이더 스캐너")
    st.markdown("### 🔒 접속하려면 비밀번호를 입력하세요")
    with st.form("login_form", clear_on_submit=False):
        pw = st.text_input(
            "비밀번호",
            type="password",
            label_visibility="collapsed",
            placeholder="비밀번호 입력",
        )
        submitted = st.form_submit_button("입장하기", use_container_width=True)

    if submitted:
        if hmac.compare_digest(pw, expected):
            st.session_state["auth_ok"] = True
            st.rerun()
        else:
            st.error("❌ 비밀번호가 올바르지 않습니다.")

    st.stop()

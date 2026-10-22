import streamlit as st

import app_ui

st.set_page_config(page_title="Spam Detector", page_icon="📧", layout="centered")
st.markdown(app_ui.CSS, unsafe_allow_html=True)

st.navigation(
    [
        st.Page("views/check_message.py", title="Check a message", icon=":material/mail:", default=True),
        st.Page("views/fool_it.py", title="Try to fool it", icon=":material/sports_esports:"),
        st.Page("views/model_report.py", title="How well it works", icon=":material/insights:"),
    ],
    position="top",
).run()

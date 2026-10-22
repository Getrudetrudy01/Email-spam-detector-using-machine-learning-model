import streamlit as st

import app_ui

st.set_page_config(page_title="Spam Detector", page_icon="📧", layout="centered")
st.markdown(app_ui.CSS, unsafe_allow_html=True)

CHECK_PAGE = st.Page("views/check_message.py", title="Check a message", icon=":material/mail:", default=True)
REPORT_PAGE = st.Page("views/model_report.py", title="How well it works", icon=":material/insights:")

# A top bar instead of the sidebar menu, which phones hide behind a button.
page = st.navigation([CHECK_PAGE, REPORT_PAGE], position="hidden")
nav1, nav2, _ = st.columns([1, 1, 1])
nav1.page_link(CHECK_PAGE)
nav2.page_link(REPORT_PAGE)
page.run()

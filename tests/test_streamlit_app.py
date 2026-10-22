import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import spam_core

pytestmark = pytest.mark.skipif(not spam_core.MODEL_PATH.exists(), reason="run train_model.py first")

# AppTest in Streamlit 1.39 does not run pages behind st.navigation, so each page is tested directly.
APP = str(spam_core.ROOT / "views" / "check_message.py")
REPORT = str(spam_core.ROOT / "views" / "model_report.py")


@pytest.fixture
def feedback_path(tmp_path, monkeypatch):
    path = tmp_path / "feedback.csv"
    monkeypatch.setattr(spam_core, "FEEDBACK_PATH", path)
    return path


def click(at, label):
    next(b for b in at.button if b.label == label).click().run()


def html_text(at):
    return " ".join(m.value for m in at.markdown)


def test_example_then_feedback_is_saved(feedback_path):
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    click(at, "Prize winner")
    assert "verdict-spam" in html_text(at)
    assert "w-spam" in html_text(at), "the message should come back with spam words highlighted"
    assert not feedback_path.exists(), "nothing should be stored before the user gives feedback"

    click(at, "Yes, it's right")
    assert "saved" in at.success[0].value

    saved = pd.read_csv(feedback_path)
    assert saved[["predicted_label", "feedback"]].values.tolist() == [["spam", "correct"]]


def test_threshold_changes_the_verdict():
    at = AppTest.from_file(APP, default_timeout=60).run()
    click(at, "Bank warning")  # scores around 60%
    assert "verdict-spam" in html_text(at)
    at.slider(key="threshold_check").set_value(80).run()
    assert "verdict-ham" in html_text(at)
    assert at.session_state["threshold"] == 80


def test_empty_message_warns():
    at = AppTest.from_file(APP, default_timeout=60).run()
    click(at, "Check message")
    assert "Paste or type a message" in at.warning[0].value


def test_clear_resets_result():
    at = AppTest.from_file(APP, default_timeout=60).run()
    click(at, "Lunch plans")
    assert "verdict-ham" in html_text(at)
    click(at, "Clear")
    assert at.text_area[0].value == ""
    assert "verdict" not in html_text(at)


def test_report_page_renders():
    at = AppTest.from_file(REPORT, default_timeout=120).run()
    assert not at.exception
    assert [m.label for m in at.metric][:2] == ["Spam caught", "Spam missed"]
    assert at.tabs[0].label.startswith("Missed spam")

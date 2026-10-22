import streamlit as st

import app_ui
import spam_core

MAX_CHARS = 5000

EXAMPLES = {
    "Prize winner": "WINNER!! You have been selected to receive a £900 prize reward. To claim call 09061701461. Valid 12 hours only.",
    "Lunch plans": "Hey, are we still meeting for lunch today at 1pm? I'll be at the usual place.",
    "Bank warning": "Your bank account has been suspended. Verify your details at secure-login-check.com to restore access.",
    "Parcel scam": "Hi, your package could not be delivered. Reschedule here: bit.ly/3xYz",
}

# ---------------------------
# Session state callbacks
# ---------------------------
# Streamlit reruns the whole script on every click, so the last result is kept
# in session state. Otherwise the feedback buttons would vanish (and never
# save) on the rerun their own click triggers. Only the probability is stored:
# the spam / not spam call is made at render time, so moving the threshold
# slider updates it.
def run_detection():
    text = st.session_state["text_input"].strip()
    if not text:
        st.session_state["result"] = None
        st.session_state["empty_warning"] = True
        return
    model, _, _ = app_ui.load_artifacts()
    st.session_state["result"] = {
        "message": text,
        "spam_prob": spam_core.spam_probability(model, [text])[0],
        "char_prob": spam_core.spam_probability(app_ui.load_char_model(), [text])[0],
        "feedback": None,
    }
    st.session_state["empty_warning"] = False

def use_example(text):
    st.session_state["text_input"] = text
    run_detection()

def give_feedback(feedback, pred):
    result = st.session_state["result"]
    spam_core.append_feedback(result["message"], pred, result["spam_prob"], feedback)
    result["feedback"] = feedback

def clear_text():
    st.session_state["text_input"] = ""
    st.session_state["result"] = None
    st.session_state["empty_warning"] = False

# First visit: show a worked example straight away, so the highlighting is the
# first thing people see rather than an empty box.
if "result" not in st.session_state:
    st.session_state["empty_warning"] = False
    use_example(EXAMPLES["Prize winner"])
# Streamlit forgets a widget's value while another page is shown, so refill the
# box from the last result when coming back.
if "text_input" not in st.session_state:
    st.session_state["text_input"] = st.session_state["result"]["message"] if st.session_state["result"] else ""

model, metrics, weights = app_ui.load_artifacts()

# ---------------------------
# Input
# ---------------------------
st.title("Spam detector")
st.write("Paste a text message or short email. You'll see whether it looks like spam, "
         "and which words made the model decide.")

st.text_area("Message", height=150, key="text_input", max_chars=MAX_CHARS,
             placeholder="Type or paste a message here...")

col1, col2 = st.columns([3, 1])
with col1:
    st.button("Check message", type="primary", width="stretch", on_click=run_detection)
with col2:
    st.button("Clear", width="stretch", on_click=clear_text)

st.caption("Or try an example:")
example_cols = st.columns(len(EXAMPLES))
for col, (label, text) in zip(example_cols, EXAMPLES.items()):
    with col:
        st.button(label, width="stretch", on_click=use_example, args=(text,))

if st.session_state["empty_warning"]:
    st.warning("Paste or type a message first, then select **Check message**.")

# ---------------------------
# Result
# ---------------------------
result = st.session_state["result"]
threshold = st.session_state.get("threshold", round(spam_core.THRESHOLD * 100)) / 100

if result:
    st.divider()
    spam_prob = result["spam_prob"]
    pred = int(spam_prob >= threshold)
    kind = "spam" if pred else "ham"
    label = "⚠ Spam" if pred else "✓ Not spam"
    st.markdown(
        f'<div class="verdict"><span class="verdict-label verdict-{kind}">{label}</span>'
        f'<span class="verdict-prob">{spam_prob:.0%} spam probability</span></div>'
        + app_ui.meter(spam_prob, threshold),
        unsafe_allow_html=True,
    )
    char_prob = result["char_prob"]
    agrees = (char_prob >= threshold) == bool(pred)
    st.markdown(
        f'<div class="second-opinion">Second opinion from a model that reads letter patterns: '
        f'<b>{char_prob:.0%} spam</b>{"" if agrees else " (it disagrees)"}.</div>',
        unsafe_allow_html=True,
    )

    segments = spam_core.explain(model, result["message"], weights)
    st.markdown("**Why**")
    st.markdown(app_ui.highlighted_message(segments) + app_ui.highlight_key(), unsafe_allow_html=True)

    spam_words, ham_words = app_ui.top_words(segments)
    lines = []
    if spam_words:
        lines.append(f'<div class="chips">Signs of spam: {app_ui.chips(spam_words, "spam")}</div>')
    if ham_words:
        lines.append(f'<div class="chips">Signs it is not spam: {app_ui.chips(ham_words, "ham")}</div>')
    if not lines:
        lines.append('<div class="chips">None of these words appeared in the training data, '
                     'so the model fell back on how common spam is overall.</div>')
    st.markdown("".join(lines), unsafe_allow_html=True)

    st.write("")
    if result["feedback"] is None:
        st.markdown("**Was this right?**")
        st.caption("Sending an answer saves the message text so it can be used to retrain the model.")
        col1, col2 = st.columns(2)
        with col1:
            st.button("Yes, it's right", width="stretch", on_click=give_feedback, args=("correct", pred))
        with col2:
            st.button("No, it's wrong", width="stretch", on_click=give_feedback, args=("incorrect", pred))
    else:
        st.success("Thanks. Your answer has been saved.")

# ---------------------------
# Strictness
# ---------------------------
st.divider()
st.markdown("**How strict should it be?**")
threshold = app_ui.threshold_slider("threshold_check")
st.markdown(app_ui.tradeoff_sentence(threshold))
st.caption("**How well it works** lists every test message the model gets wrong at this setting. "
           "**Try to fool it** lets you write spam that gets past it.")

import streamlit as st

import app_ui
import spam_core

# Each mission is a real-looking spam message plus the parts that make the scam
# work. A rewrite only counts if it keeps those parts, so deleting the phone
# number is not a way to win.
MISSIONS = {
    "Prize scam": {
        "message": "WINNER!! You have been selected to receive a £900 prize reward. To claim call 09061701461. Valid 12 hours only.",
        "goal": "Get people to call **09061701461** to claim **£900**.",
        "must_include": ["09061701461", "900"],
    },
    "Text to win": {
        "message": "Free entry into our weekly competition, just text the word WIN to 80086 NOW",
        "goal": "Get people to text **WIN** to **80086**.",
        "must_include": ["WIN", "80086"],
    },
    "Urgent award": {
        "message": "URGENT! Your mobile number has won a £2000 cash award. Call 09066362206 now to claim.",
        "goal": "Get people to call **09066362206** about a **£2000** award.",
        "must_include": ["09066362206", "2000"],
    },
}
MIN_WORDS = 5

st.session_state.setdefault("challenge_results", {})
st.session_state.setdefault("challenge_tally", {"attempts": 0, "fooled_main": 0, "fooled_both": 0})

model, _, weights = app_ui.load_artifacts()
char_model = app_ui.load_char_model()


def problems_with(text: str, mission: dict) -> list[str]:
    problems = [f"Keep **{part}** in the message, or the scam no longer works."
                for part in mission["must_include"] if part.lower() not in text.lower()]
    if len(text.split()) < MIN_WORDS:
        problems.append(f"Write at least {MIN_WORDS} words, so it still reads as a message.")
    return problems


def test_attempt(name: str):
    mission = MISSIONS[name]
    text = st.session_state[f"attempt_{name}"].strip()
    problems = problems_with(text, mission)
    if problems:
        st.session_state["challenge_results"][name] = {"problems": problems}
        return
    word_prob = spam_core.spam_probability(model, [text])[0]
    char_prob = spam_core.spam_probability(char_model, [text])[0]
    fooled_main = word_prob < spam_core.THRESHOLD
    fooled_both = fooled_main and char_prob < spam_core.THRESHOLD
    tally = st.session_state["challenge_tally"]
    tally["attempts"] += 1
    tally["fooled_main"] += fooled_main
    tally["fooled_both"] += fooled_both
    st.session_state["challenge_results"][name] = {
        "text": text, "word_prob": word_prob, "char_prob": char_prob,
    }


def reset_attempt(name: str):
    st.session_state[f"attempt_{name}"] = MISSIONS[name]["message"]
    st.session_state["challenge_results"].pop(name, None)


def model_card(title: str, subtitle: str, prob: float) -> str:
    caught = prob >= spam_core.THRESHOLD
    kind, label = ("spam", "⚠ Caught it") if caught else ("ham", "✓ Fooled")
    return (
        f'<div class="card"><div class="card-title">{title}</div>'
        f'<div class="card-sub">{subtitle}</div>'
        f'<div class="verdict"><span class="verdict-label verdict-{kind}" style="font-size: 1.5rem">{label}</span>'
        f'<span class="verdict-prob">{prob:.0%} spam</span></div>'
        + app_ui.meter(prob, spam_core.THRESHOLD) + "</div>"
    )


# ---------------------------
# Page
# ---------------------------
st.title("Try to fool it")
st.write(
    "Rewrite a spam message so the detector lets it through. Keep the part that makes the scam "
    "work, like the phone number, or it doesn't count."
)

name = st.segmented_control("Mission", list(MISSIONS), default=list(MISSIONS)[0], key="mission") or list(MISSIONS)[0]
mission = MISSIONS[name]
st.markdown(f"**Goal:** {mission['goal']}")

st.session_state.setdefault(f"attempt_{name}", mission["message"])
st.text_area("Your version", key=f"attempt_{name}", height=110, max_chars=500)

col1, col2 = st.columns([3, 1])
with col1:
    st.button("Test my version", type="primary", width="stretch", on_click=test_attempt, args=(name,))
with col2:
    st.button("Start over", width="stretch", on_click=reset_attempt, args=(name,))

with st.expander("Stuck? Three tricks spammers use"):
    st.markdown(
        "- Swap letters for look-alike numbers: *fr33 pr1ze*\n"
        "- Break words up: *c.l.a.i.m* or *n o w*\n"
        "- Make it sound like a friend, not an advert"
    )

result = st.session_state["challenge_results"].get(name)
if result and "problems" in result:
    st.warning("\n\n".join(result["problems"]))
elif result:
    word_prob, char_prob = result["word_prob"], result["char_prob"]
    st.divider()
    col1, col2 = st.columns(2)
    col1.markdown(model_card("Main detector", "Reads whole words", word_prob), unsafe_allow_html=True)
    col2.markdown(model_card("Second opinion", "Reads letter patterns", char_prob), unsafe_allow_html=True)

    fooled_main = word_prob < spam_core.THRESHOLD
    fooled_char = char_prob < spam_core.THRESHOLD
    if fooled_main and fooled_char:
        st.success("**You fooled both.** Rewording a scam to sound normal is exactly how real ones get "
                   "through. A model that only sees words has no way to know what a message is *for*.")
    elif fooled_main:
        st.info("**You got past the main detector, but the second opinion caught it.** It reads fragments "
                "of letters, so disguised spelling still looks like the spam it learned from.")
    elif fooled_char:
        st.info("**The main detector caught it, but you fooled the second opinion.** Try it on the "
                "main detector too: the highlighted words below show what gave you away.")
    else:
        st.error("**Both caught it.** The highlighted words below are what gave you away.")

    st.markdown("**What the main detector saw**")
    st.markdown(app_ui.highlighted_message(spam_core.explain(model, result["text"], weights)) + app_ui.highlight_key(),
                unsafe_allow_html=True)

tally = st.session_state["challenge_tally"]
if tally["attempts"]:
    st.divider()
    st.caption(
        f"Your attempts: {tally['attempts']} · got past the main detector: {tally['fooled_main']} · "
        f"fooled both: {tally['fooled_both']}"
    )

"""
app_ui.py
Pieces shared by the app's pages: cached model loading, the shared spam
threshold, and the HTML for the highlighted message and the score meter.
"""

import html

import streamlit as st

import spam_core

# Blue = not spam, orange = spam, everywhere in the app (charts included).
# A colour-blind-safe pair: validated for CVD separation and contrast.
HAM_COLOR = "#2a78d6"
SPAM_COLOR = "#eb6834"
HAM_RGB = "42, 120, 214"
SPAM_RGB = "235, 104, 52"

CSS = f"""
<style>
/* Streamlit's accent is red, which reads as "spam" next to the orange that
   means spam here. Its colour is baked into generated classes and changes
   with widget state, so shift the hue of the accented widgets instead: reds
   become violet, greys and text are untouched, and both themes keep working. */
[data-testid="stBaseButton-primary"],
[data-testid="stBaseButton-secondary"]:hover,
[data-testid="stSlider"] [data-baseweb="slider"],
[data-baseweb="tab"][aria-selected="true"],
[data-baseweb="tab-highlight"] {{
  filter: hue-rotate(255deg);
}}

.specimen {{
  font-family: 'Source Code Pro', ui-monospace, monospace;
  font-size: 0.95rem;
  line-height: 2.1;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  padding: 0.9rem 1.1rem;
  border: 1px solid rgba(128, 128, 128, 0.35);
  border-radius: 6px;
  margin: 0.25rem 0 0.5rem;
}}
.specimen .w {{ padding: 0.1em 0.15em; border-radius: 3px; }}
.specimen .w-spam {{ border-bottom: 2px solid {SPAM_COLOR}; }}
.specimen .w-ham {{ border-bottom: 2px dotted {HAM_COLOR}; }}

.key {{ font-size: 0.85rem; opacity: 0.85; display: flex; gap: 1.25rem; flex-wrap: wrap; }}
.key span span {{ padding: 0 0.3em; border-radius: 3px; }}

.verdict {{ display: flex; align-items: baseline; gap: 0.75rem; flex-wrap: wrap; margin: 0.5rem 0 0.25rem; }}
.verdict-label {{ font-size: 2rem; font-weight: 700; letter-spacing: -0.01em; }}
.verdict-spam {{ color: {SPAM_COLOR}; }}
.verdict-ham {{ color: {HAM_COLOR}; }}
.verdict-prob {{ font-size: 1rem; opacity: 0.8; }}

.meter {{ position: relative; height: 2.6rem; margin: 0.25rem 0 1rem; }}
.meter-track {{ position: absolute; top: 0.9rem; left: 0; right: 0; height: 6px; border-radius: 3px;
  background: rgba(128, 128, 128, 0.25); overflow: hidden; }}
.meter-zone {{ position: absolute; top: 0; bottom: 0; right: 0; background: rgba({SPAM_RGB}, 0.35); }}
.meter-dot {{ position: absolute; top: 0.6rem; width: 14px; height: 14px; margin-left: -7px; border-radius: 50%;
  border: 2px solid var(--background-color, #fff); box-sizing: border-box; }}
.meter-tick {{ position: absolute; top: 0.45rem; width: 2px; height: 1.1rem; margin-left: -1px; background: currentColor; opacity: 0.7; }}
.meter-label {{ position: absolute; top: 1.65rem; font-size: 0.75rem; opacity: 0.75; transform: translateX(-50%); white-space: nowrap; }}
.meter-end {{ position: absolute; top: 1.65rem; font-size: 0.75rem; opacity: 0.5; }}

.chips {{ display: flex; flex-wrap: wrap; gap: 0.4rem; align-items: center; font-size: 0.9rem; margin: 0.2rem 0; }}
.chip {{ font-family: 'Source Code Pro', ui-monospace, monospace; font-size: 0.85rem; padding: 0.05rem 0.45rem; border-radius: 3px; }}
.chip-spam {{ background: rgba({SPAM_RGB}, 0.25); }}
.chip-ham {{ background: rgba({HAM_RGB}, 0.22); }}
</style>
"""


# ---------------------------
# Cached loading
# ---------------------------
@st.cache_resource
def load_artifacts():
    model = spam_core.load_model()
    return model, spam_core.load_metrics(), spam_core.feature_weights(model)


@st.cache_resource
def load_test_predictions():
    return spam_core.test_predictions(load_artifacts()[0])


# ---------------------------
# Shared threshold
# ---------------------------
def threshold_slider(key: str) -> float:
    """A spam-threshold slider that keeps its value across pages.

    Streamlit drops a widget's state when another page is shown, so the value
    lives under a separate "threshold" key that every page's slider syncs to.
    """
    st.session_state.setdefault("threshold", round(spam_core.THRESHOLD * 100))
    st.session_state[key] = st.session_state["threshold"]

    def sync():
        st.session_state["threshold"] = st.session_state[key]

    st.slider(
        "Spam threshold", min_value=5, max_value=95, step=5, format="%d%%", key=key, on_change=sync,
        help="Messages scoring at or above this are marked as spam. "
             "Higher catches less spam but wrongly flags fewer real messages.",
    )
    return st.session_state["threshold"] / 100


def tradeoff_sentence(threshold: float) -> str:
    r = spam_core.rates_at(load_test_predictions(), threshold)
    return (
        f"At {threshold:.0%}, on {r['spam_total'] + r['ham_total']:,} test messages the model never saw, "
        f"it caught **{r['caught']} of {r['spam_total']} spam** ({r['recall']:.0%}) and wrongly flagged "
        f"**{r['false_alarms']} of {r['ham_total']} normal messages**."
    )


# ---------------------------
# HTML pieces
# ---------------------------
def _strength(score: float, scale: float) -> float:
    return min(abs(score) / scale, 1.0)


def highlighted_message(segments, scale: float = 2.0) -> str:
    """The message with each word tinted by how much it pushed towards spam or ham.

    Colour is backed by a second cue (solid underline = spam, dotted = not spam)
    and a hover tooltip, so the meaning never rests on colour alone.
    """
    parts = []
    for text, score in segments:
        text = html.escape(text)
        if score is None or abs(score) < 0.05:
            parts.append(text)
            continue
        strength = _strength(score, scale)
        alpha = 0.12 + 0.48 * strength
        if score > 0:
            css, rgb, tip = "w-spam", SPAM_RGB, "pushes towards spam"
        else:
            css, rgb, tip = "w-ham", HAM_RGB, "pushes towards not spam"
        underline = css if strength > 0.35 else ""
        parts.append(
            f'<span class="w {underline}" style="background: rgba({rgb}, {alpha:.2f})" '
            f'title="{tip} ({score:+.2f})">{text}</span>'
        )
    return f'<div class="specimen">{"".join(parts)}</div>'


def highlight_key() -> str:
    return (
        '<div class="key">'
        f'<span><span style="background: rgba({SPAM_RGB}, 0.45); border-bottom: 2px solid {SPAM_COLOR}">word</span> pushes towards spam</span>'
        f'<span><span style="background: rgba({HAM_RGB}, 0.4); border-bottom: 2px dotted {HAM_COLOR}">word</span> pushes towards not spam</span>'
        '<span>Stronger colour = bigger effect. Plain words were ignored.</span>'
        '</div>'
    )


def meter(spam_prob: float, threshold: float) -> str:
    color = SPAM_COLOR if spam_prob >= threshold else HAM_COLOR
    return (
        '<div class="meter" role="img" '
        f'aria-label="Spam probability {spam_prob:.0%}, threshold {threshold:.0%}">'
        f'<div class="meter-track"><div class="meter-zone" style="left: {threshold:.1%}"></div></div>'
        f'<div class="meter-tick" style="left: {threshold:.1%}"></div>'
        f'<div class="meter-dot" style="left: {spam_prob:.1%}; background: {color}"></div>'
        '<div class="meter-end" style="left: 0">0%</div>'
        f'<div class="meter-label" style="left: {min(max(threshold, 0.12), 0.88):.1%}">threshold {threshold:.0%}</div>'
        '<div class="meter-end" style="right: 0">100%</div>'
        '</div>'
    )


def top_words(segments, n: int = 4) -> tuple[list[str], list[str]]:
    """The words that pushed hardest each way, deduplicated, strongest first."""
    seen, scored = set(), []
    for text, score in segments:
        if score is not None and text.lower() not in seen:
            seen.add(text.lower())
            scored.append((score, text))
    spam = [t for s, t in sorted(scored, reverse=True) if s > 0.05][:n]
    ham = [t for s, t in sorted(scored) if s < -0.05][:n]
    return spam, ham


def chips(words: list[str], kind: str) -> str:
    return " ".join(f'<span class="chip chip-{kind}">{html.escape(w)}</span>' for w in words)

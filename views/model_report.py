import altair as alt
import pandas as pd
import streamlit as st

import app_ui
import spam_core

model, metrics, weights = app_ui.load_artifacts()
predictions = app_ui.load_test_predictions()

st.title("How well it works")
st.write(
    f"Every number here comes from {len(predictions):,} messages the model never saw while training. "
    "Move the threshold to see the trade-off between catching spam and flagging real messages."
)

threshold = app_ui.threshold_slider("threshold_report")
r = spam_core.rates_at(predictions, threshold)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Spam caught", f"{r['caught']} of {r['spam_total']}", help="Recall: share of spam marked as spam")
c2.metric("Spam missed", r["missed"])
c3.metric("Real messages flagged", f"{r['false_alarms']} of {r['ham_total']}", help="False alarms")
c4.metric("Right when it says spam", f"{r['precision']:.0%}", help="Precision")

# ---------------------------
# Score distribution
# ---------------------------
st.subheader("How sure it is")
st.write("Each bar counts test messages by the spam probability the model gave them. "
         "Everything right of the line is marked as spam.")


def score_chart(df: pd.DataFrame, title: str, color: str) -> alt.LayerChart:
    bars = alt.Chart(df).mark_bar(color=color, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X("spam_prob:Q", bin=alt.Bin(extent=[0, 1], step=0.05),
                scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", title="Spam probability")),
        y=alt.Y("count():Q", title="Messages"),
        tooltip=[alt.Tooltip("spam_prob:Q", bin=alt.Bin(extent=[0, 1], step=0.05), title="Spam probability", format=".0%"),
                 alt.Tooltip("count():Q", title="Messages")],
    )
    rule = alt.Chart(pd.DataFrame({"t": [threshold]})).mark_rule(strokeWidth=2, strokeDash=[4, 3]).encode(
        x="t:Q", tooltip=[alt.Tooltip("t:Q", title="Threshold", format=".0%")]
    )
    return (bars + rule).properties(title=title, height=140)


ham = predictions[predictions["label"] == 0]
spam = predictions[predictions["label"] == 1]
st.altair_chart(
    alt.vconcat(
        score_chart(spam, f"Spam ({len(spam)})", app_ui.SPAM_COLOR),
        score_chart(ham, f"Normal messages ({len(ham)})", app_ui.HAM_COLOR),
    ),
    use_container_width=True,
)
st.caption("Most messages sit at the far ends, so the threshold only matters for the few in the middle.")

# ---------------------------
# Words
# ---------------------------
st.subheader("Words it relies on most")
st.write("Words are shown after stemming, so *rington* covers ringtone and ringtones.")

ranked = sorted(((w, f) for f, w in weights.items() if " " not in f), reverse=True)
top = [(f, w, "Spam") for w, f in ranked[:10]] + [(f, w, "Not spam") for w, f in ranked[-10:]]
words = pd.DataFrame(top, columns=["word", "weight", "direction"])
st.altair_chart(
    alt.Chart(words).mark_bar(cornerRadius=3).encode(
        x=alt.X("weight:Q", title="← pushes towards not spam · pushes towards spam →"),
        y=alt.Y("word:N", sort=alt.EncodingSortField("weight", order="descending"), title=None,
                axis=alt.Axis(labelOverlap=False)),
        color=alt.Color("direction:N", scale=alt.Scale(domain=["Spam", "Not spam"],
                        range=[app_ui.SPAM_COLOR, app_ui.HAM_COLOR]), legend=alt.Legend(title=None, orient="top")),
        tooltip=["word", alt.Tooltip("weight:Q", format="+.2f", title="Weight"), "direction"],
    ).properties(height=520),
    use_container_width=True,
)
st.info(
    "**A shortcut it learned:** *lt* and *gt* are among the strongest signs of a normal message. "
    "They come from `&lt;#&gt;`, a placeholder the dataset's authors used to hide numbers. It appears in "
    "214 normal messages and no spam, so the model treats it as proof a message is genuine. "
    "Real messages never contain it, so this signal is useless outside this dataset.",
    icon=":material/lightbulb:",
)

# ---------------------------
# Mistakes
# ---------------------------
st.subheader("What it gets wrong")
missed = spam[spam["spam_prob"] < threshold].sort_values("spam_prob")
false_alarms = ham[ham["spam_prob"] >= threshold].sort_values("spam_prob", ascending=False)


def show_messages(df: pd.DataFrame, empty: str) -> None:
    if df.empty:
        st.write(empty)
        return
    st.markdown(app_ui.highlight_key(), unsafe_allow_html=True)
    for message, prob in zip(df["message"].head(20), df["spam_prob"].head(20)):
        st.markdown(f"**{prob:.0%}** spam probability" +
                    app_ui.highlighted_message(spam_core.explain(model, message, weights)),
                    unsafe_allow_html=True)
    if len(df) > 20:
        st.caption(f"Showing 20 of {len(df)}.")


tab_missed, tab_flagged = st.tabs([f"Missed spam ({len(missed)})", f"Real messages flagged ({len(false_alarms)})"])
with tab_missed:
    st.caption("Spam the model let through, most confidently wrong first.")
    show_messages(missed, "No spam was missed at this threshold.")
with tab_flagged:
    st.caption("Normal messages the model marked as spam, most confidently wrong first.")
    show_messages(false_alarms, "No real messages were flagged at this threshold.")

# ---------------------------
# Limits
# ---------------------------
st.subheader("Known limits")
st.markdown(f"""
- **It learned from UK text messages collected around 2011.** Modern scams look different. The
  *Parcel scam* example on the first page scores under 1% because delivery scams with short links
  barely existed in the training data.
- **It reads words, not meaning.** Spam written with ordinary, friendly words gets through, and a real
  message full of words like *free*, *win* or *claim* can be flagged.
- **Disguised spelling is a weak spot.** "FR33 PR1ZE" is not in the vocabulary, so the model only
  catches it if other words give it away.
- **The test set has only {len(spam)} spam messages,** so each missed one moves recall by almost a percentage point.
""")

st.caption(
    f"Model: {metrics['model'].replace('_', ' ')} on TF-IDF word and word-pair features, chosen by "
    f"5-fold cross-validation and trained on {metrics['n_train']:,} messages."
)

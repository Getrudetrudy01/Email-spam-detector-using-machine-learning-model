import streamlit as st
import joblib
from pathlib import Path
import pandas as pd
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from datetime import datetime

# ---------------------------
# File paths
# ---------------------------
MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "spam_classifier.joblib"
VECT_PATH = MODELS_DIR / "vectorizer.joblib"
FEEDBACK_PATH = Path("feedback_data.csv")

# ---------------------------
# Load model + vectorizer
# ---------------------------
@st.cache_data
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECT_PATH)
    return model, vectorizer

# ---------------------------
# Preprocess text
# ---------------------------
def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
    try:
        from nltk.stem import PorterStemmer
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]
    except Exception:
        tokens = [re.sub(r'(ing|ly|ed|s)$', '', t) for t in tokens]
    return " ".join(tokens)

# ---------------------------
# Predict message
# ---------------------------
def predict_message(text: str, model, vectorizer):
    pre = preprocess_text(text)
    X = vectorizer.transform([pre])
    try:
        prob = model.predict_proba(X)[0]
        spam_prob = float(prob[1])
    except Exception:
        spam_prob = None
    pred = model.predict(X)[0]
    return int(pred), spam_prob

# ---------------------------
# Save feedback
# ---------------------------
def save_feedback(message, predicted_label, confidence, feedback=None):
    # Create file with header if not exists
    if not FEEDBACK_PATH.exists():
        df = pd.DataFrame(columns=["timestamp", "message", "predicted_label", "confidence", "feedback"])
        df.to_csv(FEEDBACK_PATH, index=False)

    new_entry = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "message": message,
        "predicted_label": "spam" if predicted_label == 1 else "ham",
        "confidence": confidence,
        "feedback": feedback
    }])

    new_entry.to_csv(FEEDBACK_PATH, mode="a", header=False, index=False)

# ---------------------------
# Streamlit Page Config
# ---------------------------
st.set_page_config(page_title="Spam Detector", page_icon="📧", layout="centered")

# ---------------------------
# Sidebar "About" Section
# ---------------------------
st.sidebar.title(" ℹAbout")
st.sidebar.info("""
**Email/SMS Spam Detector**  
Detects whether a message is **Spam** or **Not Spam** using Machine Learning (TF-IDF + Naive Bayes).

 This app learns over time through user feedback.
Feedback is stored safely and used for future model improvements.

""")

# ---------------------------
# Main Interface
# ---------------------------
st.title(" Email / SMS Spam Detector")
st.write("Paste your email or SMS text below and click **Detect**.")

model, vectorizer = load_artifacts()

# Store text input in session state for Clear button
if "text_input" not in st.session_state:
    st.session_state["text_input"] = ""

def clear_text():
    st.session_state["text_input"] = ""

text = st.text_area("Message", height=200, key="text_input", placeholder="Type or paste an email/SMS here...")

col1, col2 = st.columns(2)
with col1:
    detect_btn = st.button(" Detect", use_container_width=True)
with col2:
    clear_btn = st.button(" Clear", use_container_width=True, on_click=clear_text)

# ---------------------------
# Run detection
# ---------------------------
if detect_btn:
    if not text.strip():
        st.warning(" Please paste a message first.")
    else:
        pred, spam_prob = predict_message(text, model, vectorizer)
        label = "SPAM" if pred == 1 else "NOT SPAM"

        if pred == 1:
            st.error(f" Result: **{label}**")
        else:
            st.success(f" Result: **{label}**")

        if spam_prob is not None:
            st.write(f"**Spam probability:** {spam_prob:.2%}")

        # Save initial prediction (no feedback yet)
        save_feedback(text, pred, spam_prob)

        st.info("Please provide feedback to help improve the model:")

        col1, col2 = st.columns(2)
        with col1:
            if st.button(" Prediction Correct"):
                save_feedback(text, pred, spam_prob, feedback="correct")
                st.success(" Thanks! Your feedback has been saved.")

        with col2:
            if st.button(" Prediction Incorrect"):
                save_feedback(text, pred, spam_prob, feedback="incorrect")
                st.warning(" Feedback noted — thank you!")

        st.caption("Your feedback helps this app learn and improve future spam detection accuracy.")

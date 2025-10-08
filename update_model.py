"""
update_model.py
Retrains the spam detection model by merging original dataset (spam.csv)
with user feedback collected in feedback_data.csv.
"""

import pandas as pd
import joblib
import re
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from datetime import datetime

# ---------------------------
# Paths
# ---------------------------
DATA_PATH = Path("spam.csv")
FEEDBACK_PATH = Path("feedback_data.csv")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# ---------------------------
# Text preprocessing
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
# Load base dataset
# ---------------------------
def load_spam_csv():
    try:
        df = pd.read_csv(DATA_PATH, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(DATA_PATH, encoding="latin1")

    # detect columns
    cols = [c.lower() for c in df.columns]
    if "label" in cols and "message" in cols:
        df = df.rename(columns={df.columns[cols.index("label")]: "label",
                                df.columns[cols.index("message")]: "message"})
    elif "v1" in cols and "v2" in cols:
        df = df.rename(columns={df.columns[cols.index("v1")]: "label",
                                df.columns[cols.index("v2")]: "message"})
    else:
        df = df.rename(columns={df.columns[0]: "label", df.columns[1]: "message"})

    df = df[["label", "message"]].dropna()
    df['label'] = df['label'].astype(str).str.strip().str.lower()
    label_map = {'spam': 1, 'ham': 0}
    df['label_num'] = df['label'].map(label_map)
    return df

# ---------------------------
# Merge feedback dataset
# ---------------------------
def load_feedback_data():
    if not FEEDBACK_PATH.exists():
        print(" No feedback data found. Using only the original dataset.")
        return pd.DataFrame()

    feedback = pd.read_csv(FEEDBACK_PATH)
    feedback = feedback.dropna(subset=['message', 'predicted_label'])
    feedback['label_num'] = feedback['predicted_label'].map({'spam': 1, 'ham': 0})
    
    # Only keep confirmed feedback if available
    confirmed = feedback[feedback['feedback'].isin(['correct', 'incorrect'])]
    if not confirmed.empty:
        confirmed['label_num'] = confirmed.apply(
            lambda x: x['label_num'] if x['feedback'] == 'correct' else (1 - x['label_num']),
            axis=1
        )
        return confirmed[['message', 'label_num']]
    else:
        print(" No confirmed feedback found. Using only the original dataset.")
        return pd.DataFrame()

# ---------------------------
# Main retraining logic
# ---------------------------
print(" Loading data...")
base_df = load_spam_csv()
feedback_df = load_feedback_data()

# Combine datasets
if not feedback_df.empty:
    combined_df = pd.concat([
        base_df[['message', 'label_num']],
        feedback_df[['message', 'label_num']]
    ], ignore_index=True)
else:
    combined_df = base_df[['message', 'label_num']]

print(f" Total training samples: {len(combined_df)}")

# Preprocess
combined_df['message_clean'] = combined_df['message'].apply(preprocess_text)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    combined_df['message_clean'], combined_df['label_num'], 
    test_size=0.2, random_state=42, stratify=combined_df['label_num']
)

# Vectorize
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train model
print(" Training model...")
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = model.predict(X_test_tfidf)
acc = accuracy_score(y_test, y_pred)
print(f" Model retrained successfully. Accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred, digits=4))

# Save new model and vectorizer
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = MODELS_DIR / f"spam_classifier_{timestamp}.joblib"
vect_path = MODELS_DIR / f"vectorizer_{timestamp}.joblib"

joblib.dump(model, model_path, compress=3)
joblib.dump(vectorizer, vect_path, compress=3)

# Optionally overwrite the "active" model for Streamlit
joblib.dump(model, MODELS_DIR / "spam_classifier.joblib", compress=3)
joblib.dump(vectorizer, MODELS_DIR / "vectorizer.joblib", compress=3)

print(f" Saved new model: {model_path.name}")
print(f" Saved new vectorizer: {vect_path.name}")
print(" Retraining complete.")

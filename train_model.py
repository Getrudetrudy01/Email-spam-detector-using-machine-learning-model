# train_model.py

# Train a spam classifier from spam.csv and save the model + vectorizer.
# Python 3.10+ compatible. Run: python train_model.py


from pathlib import Path
import re
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ---------------------------
# Config
# ---------------------------
DATA_PATH = Path("spam.csv") 
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42
TEST_SIZE = 0.20

# ---------------------------
# Helpers - robust column detection
# ---------------------------
def load_spam_csv(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin1", low_memory=False)
    # common variants: 'label'/'message' (user said), or 'v1'/'v2' (sms dataset)
    cols = [c.lower() for c in df.columns]
    if "label" in cols and "message" in cols:
        df = df.rename(columns={df.columns[cols.index("label")]: "label",
                                df.columns[cols.index("message")]: "message"})
    elif "v1" in cols and "v2" in cols:
        df = df.rename(columns={df.columns[cols.index("v1")]: "label",
                                df.columns[cols.index("v2")]: "message"})
    else:
        # guess: first two columns -> label, message
        df = df.rename(columns={df.columns[0]: "label", df.columns[1]: "message"})
    df = df[["label", "message"]].dropna()
    return df

# ---------------------------
# Text preprocessing
# ---------------------------
def preprocess_text(text: str) -> str:
    # 1) normalize and remove non-alphanumeric chars
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # 2) tokenize and remove stopwords
    tokens = text.split()
    tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
    # 3) stemming (try nltk PorterStemmer; fallback to crude rule)
    try:
        from nltk.stem import PorterStemmer
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]
    except Exception:
        # fallback simple heuristic: drop common suffixes
        tokens = [re.sub(r'(ing|ly|ed|s)$', '', t) for t in tokens]
    return " ".join(tokens)

# ---------------------------
# Load data
# ---------------------------
print("Loading data...")
df = load_spam_csv(DATA_PATH)
print(f"Loaded {len(df)} rows. Sample:")
print(df.head(5).to_string(index=False))

# Map common label names to 0/1
df['label'] = df['label'].astype(str).str.strip().str.lower()
label_map = {}
# heuristics
if df['label'].isin(['spam','ham']).all():
    label_map = {'spam':1, 'ham':0}
elif df['label'].isin(['1','0','spam','ham']).all():
    # convert 'spam'/'ham' or '1'/'0'
    if df['label'].isin(['spam']).any():
        label_map = {'spam':1, 'ham':0, '1':1, '0':0}
    else:
        label_map = {'1':1, '0':0}
else:
    # try mapping any value that contains 'spam'
    df['label_detect'] = df['label'].apply(lambda x: 1 if 'spam' in x else 0)
    label_map = None

if label_map:
    df['label_num'] = df['label'].map(label_map).astype(int)
else:
    df['label_num'] = df['label_detect'].astype(int)
    df = df.drop(columns=['label_detect'])

# ---------------------------
# Preprocess messages
# ---------------------------
print("Preprocessing messages (this may take a moment)...")
df['message_clean'] = df['message'].apply(preprocess_text)

# ---------------------------
# Train / Test split
# ---------------------------
X = df['message_clean'].values
y = df['label_num'].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# ---------------------------
# Vectorize
# ---------------------------
vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.9)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ---------------------------
# Train: MultinomialNB
# ---------------------------
print("Training MultinomialNB...")
nb = MultinomialNB(alpha=1.0)
nb.fit(X_train_tfidf, y_train)
y_pred = nb.predict(X_test_tfidf)

print("MultinomialNB results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, digits=4))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# ---------------------------
# Optional: Compare LogisticRegression
# ---------------------------
print("\n(OPTIONAL) Training LogisticRegression for comparison...")
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_tfidf, y_train)
y_pred_lr = lr.predict(X_test_tfidf)
print("Logistic Regression accuracy:", accuracy_score(y_test, y_pred_lr))

# ---------------------------
# Save best model + vectorizer
# ---------------------------
# choose nb for speed; you can swap to lr if preferred
MODEL_PATH = MODELS_DIR / "spam_classifier.joblib"
VECT_PATH = MODELS_DIR / "vectorizer.joblib"

joblib.dump(nb, MODEL_PATH, compress=3)
joblib.dump(vectorizer, VECT_PATH, compress=3)

print(f"Saved model to {MODEL_PATH}")
print(f"Saved vectorizer to {VECT_PATH}")

# Extra: save a tiny metadata file
metadata = {"model": "MultinomialNB", "vectorizer": "TfidfVectorizer", "notes": "preprocessed lowercase, punctuation removed, stemming"}
joblib.dump(metadata, MODELS_DIR / "metadata.joblib")
print("Training complete.")

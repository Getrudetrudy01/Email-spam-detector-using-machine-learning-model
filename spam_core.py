"""
spam_core.py
Shared code for training, retraining and serving the spam classifier.

Training and the app must treat text identically, so everything that touches
the dataset, preprocessing, model files or feedback lives here.
"""

import csv
import hashlib
import json
import random
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from nltk.stem import PorterStemmer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# ---------------------------
# Paths and settings
# ---------------------------
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "spam.csv"
FEEDBACK_PATH = ROOT / "feedback_data.csv"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "spam_pipeline.joblib"
CHAR_MODEL_PATH = MODELS_DIR / "char_pipeline.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
PREVIOUS_DIR = MODELS_DIR / "previous"

RANDOM_STATE = 42
TEST_SIZE = 0.20
THRESHOLD = 0.5
LABELS = {"ham": 0, "spam": 1}
FEEDBACK_COLUMNS = ["timestamp", "message", "predicted_label", "spam_probability", "feedback"]

# ---------------------------
# Text preprocessing
# ---------------------------
_stemmer = PorterStemmer()
_non_alnum = re.compile(r"[^a-z0-9\s]")


def preprocess_text(text) -> str:
    """Lowercase, strip punctuation, drop stopwords and stem.

    Used as the TfidfVectorizer preprocessor, so the saved pipeline applies it
    to raw text by itself and the app never has to call it directly.
    """
    text = _non_alnum.sub(" ", str(text).lower())
    return " ".join(_stemmer.stem(t) for t in text.split() if t not in ENGLISH_STOP_WORDS)


# ---------------------------
# Dataset
# ---------------------------
def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the SMS Spam Collection as columns message, label (0 = ham, 1 = spam).

    Duplicate messages are dropped so the same text cannot land in both the
    train and test split.
    """
    raw = pd.read_csv(path, encoding="latin1", dtype=str)
    label_col, text_col, *overflow = raw.columns
    # About 50 messages contain unquoted commas, so their text spills into the
    # trailing "Unnamed" columns. Join it back instead of truncating the message.
    message = raw[[text_col, *overflow]].apply(lambda row: ",".join(row.dropna()), axis=1)
    labels = raw[label_col].str.strip().str.lower()

    unknown = set(labels.dropna()) - set(LABELS)
    if unknown:
        raise ValueError(f"Unexpected labels in {path.name}: {sorted(unknown)}")

    df = pd.DataFrame({"message": message.str.strip(), "label": labels.map(LABELS)})
    df = df.dropna()
    df = df[df["message"] != ""]
    df = df.drop_duplicates(subset="message", keep="first").reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    return df


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic stratified split, so retraining is scored on the same test set."""
    return train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["label"])


def dataset_fingerprint(df: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for message, label in zip(df["message"], df["label"]):
        digest.update(f"{label}\t{message}\n".encode("utf-8"))
    return digest.hexdigest()[:16]


# ---------------------------
# Model
# ---------------------------
CANDIDATES = {
    "multinomial_nb": lambda: MultinomialNB(alpha=0.1),
    "logistic_regression": lambda: LogisticRegression(C=10, class_weight="balanced", max_iter=3000),
}

# Two ways of turning text into features:
# - word: stemmed words and word pairs. Easy to explain, so it gives the main verdict.
# - char: 2-5 letter fragments within words. Harder to explain, but it still
#   recognises "fr33 pr1ze" as close to "free prize".
VECTORIZERS = {
    "word": lambda: TfidfVectorizer(preprocessor=preprocess_text, ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True),
    "char": lambda: TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
}


def build_pipeline(model_name: str, features: str = "word") -> Pipeline:
    """TF-IDF + classifier, with sigmoid calibration so predict_proba is meaningful."""
    return Pipeline([
        ("tfidf", VECTORIZERS[features]()),
        ("clf", CalibratedClassifierCV(CANDIDATES[model_name](), method="sigmoid", cv=5)),
    ])


def spam_probability(pipeline: Pipeline, messages) -> list[float]:
    return pipeline.predict_proba(list(messages))[:, 1].tolist()


def evaluate(pipeline: Pipeline, test: pd.DataFrame) -> dict:
    """Metrics on the spam class. Accuracy is reported but is not the headline:
    the data is ~87% ham, so always answering "ham" already scores ~87%."""
    y_true = test["label"].to_numpy()
    proba = pipeline.predict_proba(test["message"].tolist())[:, 1]
    y_pred = (proba >= THRESHOLD).astype(int)
    return {
        "spam_precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "spam_recall": round(float(recall_score(y_true, y_pred)), 4),
        "spam_f1": round(float(f1_score(y_true, y_pred)), 4),
        "average_precision": round(float(average_precision_score(y_true, proba)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "threshold": THRESHOLD,
    }


def test_predictions(pipeline: Pipeline) -> pd.DataFrame:
    """Spam probabilities for the held-out test set (columns message, label, spam_prob)."""
    test = split_dataset(load_dataset())[1].reset_index(drop=True)
    return test.assign(spam_prob=spam_probability(pipeline, test["message"]))


def rates_at(predictions: pd.DataFrame, threshold: float) -> dict:
    """What a given threshold means on the test set, in counts and rates."""
    flagged = predictions["spam_prob"] >= threshold
    is_spam = predictions["label"] == 1
    caught = int((flagged & is_spam).sum())
    false_alarms = int((flagged & ~is_spam).sum())
    return {
        "caught": caught,
        "missed": int((~flagged & is_spam).sum()),
        "false_alarms": false_alarms,
        "spam_total": int(is_spam.sum()),
        "ham_total": int((~is_spam).sum()),
        "recall": caught / max(int(is_spam.sum()), 1),
        "precision": caught / max(caught + false_alarms, 1),
        "false_alarm_rate": false_alarms / max(int((~is_spam).sum()), 1),
    }


# ---------------------------
# Explanations
# ---------------------------
_word = re.compile(r"[A-Za-z0-9]+")


def feature_weights(pipeline: Pipeline) -> dict[str, float]:
    """How strongly each TF-IDF feature points towards spam (+) or ham (-).

    For Naive Bayes this is log P(word | spam) - log P(word | ham); for logistic
    regression it is the coefficient. Averaged over the calibration folds.
    """
    weights = []
    for calibrated in pipeline.named_steps["clf"].calibrated_classifiers_:
        est = calibrated.estimator
        if hasattr(est, "feature_log_prob_"):
            weights.append(est.feature_log_prob_[1] - est.feature_log_prob_[0])
        else:
            weights.append(est.coef_[0])
    mean = sum(weights) / len(weights)
    return dict(zip(pipeline.named_steps["tfidf"].get_feature_names_out(), mean.tolist()))


def explain(pipeline: Pipeline, text: str, weights: dict[str, float] | None = None) -> list[tuple[str, float | None]]:
    """Split text into (segment, score) pairs covering the whole string.

    Words get a score: positive pushed the message towards spam, negative
    towards ham, and None means the model ignored it (a stopword, or not in the
    vocabulary). Everything between words gets None. A word's score is its
    TF-IDF value times its weight, with each word pair's score shared by both words.
    """
    weights = weights if weights is not None else feature_weights(pipeline)
    tfidf = pipeline.named_steps["tfidf"]
    row = tfidf.transform([text])
    names = tfidf.get_feature_names_out()

    stem_scores: dict[str, float] = {}
    for idx, value in zip(row.indices, row.data):
        feature = names[idx]
        parts = feature.split()
        for stem in parts:
            stem_scores[stem] = stem_scores.get(stem, 0.0) + float(value) * weights[feature] / len(parts)

    segments, pos = [], 0
    for match in _word.finditer(text):
        if match.start() > pos:
            segments.append((text[pos:match.start()], None))
        stem = preprocess_text(match.group())
        segments.append((match.group(), stem_scores.get(stem) if stem else None))
        pos = match.end()
    if pos < len(text):
        segments.append((text[pos:], None))
    return segments


# ---------------------------
# Disguised spam
# ---------------------------
_LEET = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"}


def disguise(text: str, seed: int = 0) -> str:
    """Rewrite text the way spammers dodge filters: digits for letters
    ("fr33"), dotted words ("c.l.a.i.m") and spaced-out words ("n o w").
    Deterministic for a given seed, so results are repeatable."""
    rng = random.Random(seed)

    def rewrite(match):
        word = match.group()
        roll = rng.random()
        if len(word) < 3 or roll >= 0.8:
            return word
        if roll < 0.45:
            return "".join(_LEET.get(c.lower(), c) if rng.random() < 0.6 else c for c in word)
        return (".", " ")[roll >= 0.65].join(word)

    return re.sub(r"[A-Za-z]+", rewrite, text)


def disguised_spam_recall(pipeline: Pipeline, test: pd.DataFrame) -> float:
    """Share of the test spam still caught after every message is disguised."""
    spam = test.loc[test["label"] == 1, "message"]
    disguised = [disguise(message, seed) for seed, message in enumerate(spam)]
    return round(float((pipeline.predict_proba(disguised)[:, 1] >= THRESHOLD).mean()), 4)


def save_model(pipeline: Pipeline, metrics: dict, char_pipeline: Pipeline | None = None) -> None:
    """Save the pipeline(s) and metrics, keeping the current files in models/previous/."""
    MODELS_DIR.mkdir(exist_ok=True)
    for path in (MODEL_PATH, CHAR_MODEL_PATH, METRICS_PATH):
        if path.exists():
            PREVIOUS_DIR.mkdir(exist_ok=True)
            shutil.copy2(path, PREVIOUS_DIR / path.name)

    metrics = {
        **metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
    }
    joblib.dump(pipeline, MODEL_PATH, compress=3)
    if char_pipeline is not None:
        joblib.dump(char_pipeline, CHAR_MODEL_PATH, compress=3)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def load_model() -> Pipeline:
    return joblib.load(MODEL_PATH)


def load_char_model() -> Pipeline:
    return joblib.load(CHAR_MODEL_PATH)


def load_metrics() -> dict | None:
    if not METRICS_PATH.exists():
        return None
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


# ---------------------------
# Feedback
# ---------------------------
def append_feedback(message: str, predicted_label: int, spam_prob: float, feedback: str, path: Path | None = None) -> None:
    path = path or FEEDBACK_PATH
    if feedback not in ("correct", "incorrect"):
        raise ValueError(f"feedback must be 'correct' or 'incorrect', got {feedback!r}")
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(FEEDBACK_COLUMNS)
        writer.writerow([
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            message,
            "spam" if predicted_label == 1 else "ham",
            round(spam_prob, 4),
            feedback,
        ])


def load_feedback(path: Path | None = None) -> pd.DataFrame:
    """Turn feedback rows into labelled examples (columns message, label).

    "correct" keeps the predicted label and "incorrect" flips it. Messages that
    received conflicting labels are dropped rather than guessed.
    """
    path = path or FEEDBACK_PATH
    if not path.exists():
        return pd.DataFrame(columns=["message", "label"])

    fb = pd.read_csv(path, dtype=str)
    fb = fb[fb["feedback"].isin(["correct", "incorrect"]) & fb["predicted_label"].isin(list(LABELS))]
    fb = fb.dropna(subset=["message"])
    fb["message"] = fb["message"].str.strip()
    fb = fb[fb["message"] != ""]

    predicted = fb["predicted_label"].map(LABELS)
    label = predicted.where(fb["feedback"] == "correct", 1 - predicted)
    labelled = pd.DataFrame({"message": fb["message"], "label": label.astype(int)})

    n_labels = labelled.groupby("message")["label"].nunique()
    consistent = n_labels[n_labels == 1].index
    labelled = labelled[labelled["message"].isin(consistent)]
    return labelled.drop_duplicates(subset="message").reset_index(drop=True)

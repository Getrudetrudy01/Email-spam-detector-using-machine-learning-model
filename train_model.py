# train_model.py

# Train the spam classifiers from spam.csv and save them to models/.
# Python 3.10+ compatible. Run: python train_model.py
#
# Two models are trained:
# - word model: gives the app's main verdict, and its words can be highlighted
# - char model: reads letter fragments, so it copes better with disguised
#   spelling like "fr33 pr1ze". The app shows it as a second opinion.

from sklearn.model_selection import StratifiedKFold, cross_val_score

import spam_core


def select_model(train, features: str) -> tuple[str, dict]:
    """Pick the classifier with the best cross-validated spam F1 on the training split.

    The test split is not used here, so the test score stays an honest estimate.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=spam_core.RANDOM_STATE)
    scores = {}
    for name in spam_core.CANDIDATES:
        f1 = cross_val_score(
            spam_core.build_pipeline(name, features), train["message"].tolist(), train["label"], cv=cv, scoring="f1"
        )
        scores[name] = round(float(f1.mean()), 4)
        print(f"  {name}: CV spam F1 = {f1.mean():.4f} (+/- {f1.std():.4f})")
    return max(scores, key=scores.get), scores


def train(train_df, test_df, features: str):
    print(f"Selecting {features} model with 5-fold cross-validation...")
    model_name, cv_scores = select_model(train_df, features)
    print(f"Selected: {model_name}")

    pipeline = spam_core.build_pipeline(model_name, features)
    pipeline.fit(train_df["message"].tolist(), train_df["label"])
    metrics = {
        "model": model_name,
        "cv_spam_f1": cv_scores,
        **spam_core.evaluate(pipeline, test_df),
        "disguised_spam_recall": spam_core.disguised_spam_recall(pipeline, test_df),
    }
    print(f"Held-out test metrics ({features} model):")
    for key in ("spam_precision", "spam_recall", "spam_f1", "accuracy", "confusion_matrix", "disguised_spam_recall"):
        print(f"  {key}: {metrics[key]}")
    return pipeline, metrics


def main() -> None:
    print("Loading data...")
    df = spam_core.load_dataset()
    train_df, test_df = spam_core.split_dataset(df)
    print(f"{len(df)} unique messages ({df['label'].sum()} spam). Train {len(train_df)}, test {len(test_df)}.")

    word_pipeline, word_metrics = train(train_df, test_df, "word")
    char_pipeline, char_metrics = train(train_df, test_df, "char")

    spam_core.save_model(word_pipeline, {
        **word_metrics,
        "char": char_metrics,
        "dataset_fingerprint": spam_core.dataset_fingerprint(df),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "n_feedback": 0,
    }, char_pipeline=char_pipeline)
    for path in (spam_core.MODEL_PATH, spam_core.CHAR_MODEL_PATH, spam_core.METRICS_PATH):
        print(f"Saved {path.relative_to(spam_core.ROOT)}")


if __name__ == "__main__":
    main()

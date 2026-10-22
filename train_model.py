# train_model.py

# Train the spam classifier from spam.csv and save it to models/.
# Python 3.10+ compatible. Run: python train_model.py

from sklearn.model_selection import StratifiedKFold, cross_val_score

import spam_core


def select_model(train) -> tuple[str, dict]:
    """Pick the candidate with the best cross-validated spam F1 on the training split.

    The test split is not used here, so the test score stays an honest estimate.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=spam_core.RANDOM_STATE)
    scores = {}
    for name in spam_core.CANDIDATES:
        f1 = cross_val_score(
            spam_core.build_pipeline(name), train["message"].tolist(), train["label"], cv=cv, scoring="f1"
        )
        scores[name] = round(float(f1.mean()), 4)
        print(f"  {name}: CV spam F1 = {f1.mean():.4f} (+/- {f1.std():.4f})")
    return max(scores, key=scores.get), scores


def main() -> None:
    print("Loading data...")
    df = spam_core.load_dataset()
    train, test = spam_core.split_dataset(df)
    print(f"{len(df)} unique messages ({df['label'].sum()} spam). Train {len(train)}, test {len(test)}.")

    print("Selecting model with 5-fold cross-validation...")
    model_name, cv_scores = select_model(train)
    print(f"Selected: {model_name}")

    pipeline = spam_core.build_pipeline(model_name)
    pipeline.fit(train["message"].tolist(), train["label"])
    metrics = spam_core.evaluate(pipeline, test)

    print("Held-out test metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    spam_core.save_model(pipeline, {
        "model": model_name,
        "cv_spam_f1": cv_scores,
        "dataset_fingerprint": spam_core.dataset_fingerprint(df),
        "n_train": len(train),
        "n_test": len(test),
        "n_feedback": 0,
        **metrics,
    })
    print(f"Saved model to {spam_core.MODEL_PATH.relative_to(spam_core.ROOT)}")
    print(f"Saved metrics to {spam_core.METRICS_PATH.relative_to(spam_core.ROOT)}")


if __name__ == "__main__":
    main()

"""
update_model.py
Retrains the spam classifier on spam.csv plus labelled user feedback from
feedback_data.csv, and replaces the live model only if it is at least as good.

Feedback is never used for evaluation: both the current and the new model are
scored on the same held-out split of spam.csv, so the numbers are comparable.

Feedback labels come from anonymous users and are trusted as-is, so read
feedback_data.csv before running this. A batch of deliberately wrong labels
can still degrade the model in ways the test set does not catch.

Run: python update_model.py            (promote only if spam F1 does not drop)
     python update_model.py --force    (promote regardless)
"""

import argparse
import sys

import pandas as pd

import spam_core


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="replace the live model even if it scores worse")
    args = parser.parse_args()

    current_metrics = spam_core.load_metrics()
    if current_metrics is None or not spam_core.MODEL_PATH.exists():
        print("No trained model found. Run: python train_model.py")
        return 1

    print("Loading data...")
    df = spam_core.load_dataset()
    fingerprint = spam_core.dataset_fingerprint(df)
    if fingerprint != current_metrics.get("dataset_fingerprint"):
        print("spam.csv has changed since the live model was trained, so the test sets differ.")
        print("Run python train_model.py to retrain from scratch instead.")
        return 1

    train, test = spam_core.split_dataset(df)
    feedback = spam_core.load_feedback()
    # Drop feedback that duplicates a test message (it would leak into evaluation)
    # or a training message (the dataset label wins).
    feedback = feedback[~feedback["message"].isin(df["message"])]
    if feedback.empty:
        print("No new labelled feedback to learn from. Nothing to do.")
        return 0
    print(f"Adding {len(feedback)} feedback examples ({feedback['label'].sum()} spam) to {len(train)} training messages.")

    combined = pd.concat([train, feedback], ignore_index=True)
    model_name = current_metrics["model"]
    pipeline = spam_core.build_pipeline(model_name)
    pipeline.fit(combined["message"].tolist(), combined["label"])
    new_metrics = spam_core.evaluate(pipeline, test)

    # Score the live model now rather than trusting metrics.json, in case it was edited.
    old_metrics = spam_core.evaluate(spam_core.load_model(), test)
    print(f"{'metric':<20}{'current':>10}{'new':>10}")
    for key in ("spam_precision", "spam_recall", "spam_f1", "average_precision", "accuracy"):
        print(f"{key:<20}{old_metrics[key]:>10.4f}{new_metrics[key]:>10.4f}")

    if new_metrics["spam_f1"] < old_metrics["spam_f1"] and not args.force:
        print("New model has lower spam F1. Keeping the current model (use --force to override).")
        return 1

    # Only the word model learns from feedback; the char model and its metrics stay as they are.
    spam_core.save_model(pipeline, {
        "model": model_name,
        "cv_spam_f1": current_metrics.get("cv_spam_f1"),
        "disguised_spam_recall": spam_core.disguised_spam_recall(pipeline, test),
        "char": current_metrics.get("char"),
        "dataset_fingerprint": fingerprint,
        "n_train": len(combined),
        "n_test": len(test),
        "n_feedback": len(feedback),
        **new_metrics,
    })
    print("Promoted the new model. The previous one is in models/previous/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

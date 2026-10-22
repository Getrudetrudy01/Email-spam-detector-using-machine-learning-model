import pandas as pd
import pytest

import spam_core


def test_preprocess_text():
    assert spam_core.preprocess_text("WINNER!! You have WON a £1000 prize") == "winner won 1000 prize"
    assert spam_core.preprocess_text("") == ""
    assert spam_core.preprocess_text("Call 08000930705 NOW") == "08000930705"


def test_load_dataset_has_no_duplicates_and_valid_labels():
    df = spam_core.load_dataset()
    assert list(df.columns) == ["message", "label"]
    assert df["message"].is_unique
    assert set(df["label"]) == {0, 1}


def test_load_dataset_rejoins_messages_split_on_commas():
    df = spam_core.load_dataset()
    rejoined = df[df["message"].str.startswith("\\Wen u miss someone")]
    assert rejoined["message"].iloc[0].endswith('just Keep-in-touch\\" gdeve.."')


def test_load_dataset_rejects_unknown_labels(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("v1,v2\nham,hello\nmaybe,hi\n", encoding="latin1")
    with pytest.raises(ValueError, match="maybe"):
        spam_core.load_dataset(path)


def test_split_is_deterministic_and_disjoint():
    df = spam_core.load_dataset()
    train_a, test_a = spam_core.split_dataset(df)
    train_b, test_b = spam_core.split_dataset(df)
    assert test_a.index.equals(test_b.index)
    assert set(train_a["message"]).isdisjoint(test_a["message"])


def test_feedback_round_trip(tmp_path):
    path = tmp_path / "feedback.csv"
    spam_core.append_feedback("free prize, call now", 1, 0.9, "correct", path)
    spam_core.append_feedback("see you at lunch", 1, 0.6, "incorrect", path)
    # Same message labelled both ways is ambiguous and must be dropped.
    spam_core.append_feedback("hmm", 0, 0.4, "correct", path)
    spam_core.append_feedback("hmm", 0, 0.4, "incorrect", path)

    labelled = spam_core.load_feedback(path).set_index("message")["label"].to_dict()
    assert labelled == {"free prize, call now": 1, "see you at lunch": 0}


def test_append_feedback_rejects_missing_feedback(tmp_path):
    with pytest.raises(ValueError):
        spam_core.append_feedback("hi", 0, 0.1, None, tmp_path / "feedback.csv")


def test_load_feedback_without_file(tmp_path):
    assert spam_core.load_feedback(tmp_path / "missing.csv").empty


@pytest.mark.skipif(not spam_core.MODEL_PATH.exists(), reason="run train_model.py first")
def test_saved_model_classifies_obvious_examples():
    model = spam_core.load_model()
    spam, ham = spam_core.spam_probability(model, [
        "WINNER!! You have won a £1000 cash prize! Call now to claim, txt STOP to end",
        "Hey, are we still meeting for lunch today at 1pm?",
    ])
    assert spam > spam_core.THRESHOLD > ham


@pytest.mark.skipif(not spam_core.MODEL_PATH.exists(), reason="run train_model.py first")
def test_explain_covers_the_whole_message_and_points_the_right_way():
    text = "WINNER!! Claim your prize, see you at lunch"
    segments = spam_core.explain(spam_core.load_model(), text)
    assert "".join(t for t, _ in segments) == text
    scores = {t: s for t, s in segments if s is not None}
    assert scores["prize"] > 0 and scores["Claim"] > 0
    assert scores["lunch"] < 0
    assert "your" not in scores, "stopwords are ignored by the model, so they get no score"


def test_rates_at():
    preds = pd.DataFrame({"label": [1, 1, 0, 0], "spam_prob": [0.9, 0.3, 0.6, 0.1]})
    r = spam_core.rates_at(preds, 0.5)
    assert (r["caught"], r["missed"], r["false_alarms"]) == (1, 1, 1)
    assert r["recall"] == 0.5 and r["precision"] == 0.5 and r["false_alarm_rate"] == 0.5


def test_disguise_is_repeatable_and_keeps_numbers():
    text = "Claim your free prize now, call 09061701461"
    assert spam_core.disguise(text, 3) == spam_core.disguise(text, 3)
    assert spam_core.disguise(text, 3) != text
    assert "09061701461" in spam_core.disguise(text, 3)


@pytest.mark.skipif(not spam_core.CHAR_MODEL_PATH.exists(), reason="run train_model.py first")
def test_char_model_sees_through_disguised_spelling():
    disguised = "W1NN3R!! You have been s3lected to rec3ive a £900 pr1ze r3ward. To cl41m c4ll 09061701461."
    word, char = spam_core.load_model(), spam_core.load_char_model()
    assert spam_core.spam_probability(char, [disguised])[0] > spam_core.THRESHOLD
    assert spam_core.spam_probability(word, [disguised])[0] < spam_core.THRESHOLD


@pytest.mark.skipif(not spam_core.MODEL_PATH.exists(), reason="run train_model.py first")
def test_saved_metrics_match_saved_model():
    test = spam_core.split_dataset(spam_core.load_dataset())[1]
    metrics = spam_core.evaluate(spam_core.load_model(), test)
    saved = spam_core.load_metrics()
    assert metrics["spam_f1"] == saved["spam_f1"]
    assert saved["dataset_fingerprint"] == spam_core.dataset_fingerprint(spam_core.load_dataset())

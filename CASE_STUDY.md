# Case study: what a "98% accurate" spam filter was hiding

I built the first version of this project during a data science internship in 2024 and turned it into a Streamlit app in 2025. It reported about 98% accuracy. When I came back to review it as an engineer, I found that the headline number was roughly right, but almost everything around it was wrong. This is what I found, what I changed, and what the model still can't do.

**Live app:** https://email-spam-detector-ml-project.streamlit.app/

## What the audit found

**1. The saved model had never been trained.** The original app loaded `model1.pkl`, a 153-byte file. A trained Naive Bayes model stores learned word statistics and is far larger. The notebook re-created the model object (`mnb = MultinomialNB()`) for a voting ensemble just before pickling it, and the ensemble trains its own copies, so what got saved was an empty shell. That app would have crashed on its first prediction.

**2. The test set leaked into training.** The dataset has 403 duplicate messages out of 5,572 (7%). The training script never removed them, so the same message could appear in both the training and the test set, and the model was partly graded on messages it had memorised. The notebook also fitted its TF-IDF vocabulary on all the data before splitting it.

**3. Accuracy was the wrong number to report.** 87% of the messages are not spam, so a model that never flags anything already scores 87%. What matters is how much spam gets caught (recall) and how often a flagged message really is spam (precision).

**4. The feedback feature never saved feedback.** The "correct" and "incorrect" buttons were nested inside the "Detect" button's code. In Streamlit, clicking any button reruns the script with every other button reset, so the code that saved feedback could never run. The app also logged every message users typed, with no label, into a file committed to git.

## What I changed

- One training script, with the preprocessing built into the saved scikit-learn `Pipeline`, so the app cannot clean text differently from training.
- Duplicates removed before the split. The model is chosen by 5-fold cross-validation on the training data only, and the test set is used once.
- Calibrated probabilities, so "70% spam" means something.
- Metrics saved next to the model in `models/metrics.json`.
- Feedback that saves only when a user clicks, kept out of git, and a retraining script that replaces the live model only if the new one scores at least as well on the same test set.
- Pinned dependencies and 23 tests, including tests that drive the app's pages.

## Results

On 1,032 test messages the model never saw, at a 50% threshold:

| | Word model (main) | Letter-pattern model (second opinion) |
|---|---|---|
| Spam caught (recall) | 89.1% (114 of 128) | 93.8% (120 of 128) |
| Right when it says spam (precision) | 97.4% | 100% |
| Real messages wrongly flagged | 3 of 904 | 0 of 904 |
| Disguised spam caught | 78% | 86% |

The old "~98% accuracy" claim was close: the word model scores 98.4%. But accuracy hid the fact that it lets about 1 in 9 spam messages through.

## Three findings worth knowing

**The model learned a shortcut.** Two of its strongest "not spam" signals are the fragments *lt* and *gt*. They come from `&lt;#&gt;`, a placeholder the dataset's authors used to hide numbers. It appears in 214 normal messages and in no spam, so the model learned that the placeholder means "genuine". Real messages never contain it, so the signal is useless outside this dataset. Inspecting the model's weights, not its score, is what revealed this.

**Disguised spelling beats a word model.** I wrote a function that disguises spam the way spammers do ("fr33", "c.l.a.i.m", "n o w") and ran every test spam message through it. The word model's recall fell from 89% to 78%, because "pr1ze" is simply not a word it knows. A second model that reads 2–5 letter fragments still recognises it, and caught 86%. That model is also more accurate on normal text, but I kept the word model for the main verdict: its reasons are words a person can read, while the letter-pattern model's reasons are fragments like "ze!". The app shows both, and the "Try to fool it" page lets visitors find each model's weak spots themselves.

**Rewording fools both models.** Rewriting a prize scam as a friendly text ("Hey hun, your £900 is ready, just ring me on…") drops both models below 25%. Neither model knows what a message is *for*, only what it looks like. The same applies to modern scams: a parcel-delivery scam with a short link scores under 1%, because the 2011-era training data contains almost none.

## What I'd do next

1. **Test on email.** The model is trained on UK text messages. Measuring how far it falls on a real email dataset (such as SpamAssassin), then training on both, would make the "email" in the name honest.
2. **Keep feedback on the live site.** Streamlit Cloud wipes its files when the app restarts, so feedback should go to a small database.
3. **Compare against an LLM.** A zero-shot LLM classifier would likely handle reworded scams better. The question is whether that's worth the cost and speed compared with a small model that runs anywhere for free.

## Reproduce it

```bash
pip install -r requirements-dev.txt
python train_model.py      # trains both models, prints the metrics above
pytest                     # 23 tests
streamlit run streamlit_app.py
```

# Email / SMS Spam Detector
A machine learning web app that detects whether a message is Spam or Not Spam, built using Python, Scikit-learn, and Streamlit.
 Live Demo: Try it here https://email-spam-detector-ml-project.streamlit.app/

# Project Overview
Spam messages (junk mail) are unsolicited communications often containing scams, phishing attempts, or misleading information. This project demonstrates how Natural Language Processing (NLP) and Machine Learning can be used to automatically classify messages as Spam or Not Spam (Ham).

The model is trained on the [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection), so it works best on short, SMS-style messages. Long modern emails and phishing messages look quite different and may be misclassified.

# Key Features
•	 Data Preprocessing: Cleans text (lowercasing, punctuation and stopword removal, stemming) and removes duplicate messages
•	 Feature Extraction: TF-IDF on words and word pairs
•	 Model Selection: Multinomial Naive Bayes vs Logistic Regression, chosen by 5-fold cross-validation
•	 Calibrated Probabilities: the "spam probability" shown in the app is calibrated, not a raw model score
•	 Evaluation Metrics: Spam precision, recall, F1, average precision and accuracy on a held-out test set, saved to `models/metrics.json`
•	 Explained Predictions: every word in the message is highlighted by how much it pushed the verdict towards spam or not spam
•	 Adjustable Threshold: a slider shows, live, how many spam messages are caught and how many real messages are wrongly flagged at each setting
•	 Honest Error Analysis: a "How well it works" page lists every test message the model gets wrong, shows how confident it is, and points out a shortcut it learned from the dataset
•	 Web Interface: Streamlit app for real-time spam detection, with one-click example messages
•	 User Feedback System: Users can mark predictions as correct or incorrect
•	 Safe Retraining: Feedback is added to training data, and the new model only replaces the current one if it scores at least as well

# Project Structure
| File | Purpose |
|---|---|
| `streamlit_app.py` | The web app's entry point and page navigation |
| `views/check_message.py` | Page 1: check a message and see why |
| `views/model_report.py` | Page 2: test results, the words the model relies on, and its mistakes |
| `app_ui.py` | Shared app pieces: model loading, threshold slider, highlighting and styles |
| `spam_core.py` | Shared code: data loading, preprocessing, model, metrics, feedback |
| `train_model.py` | Trains the model from `spam.csv` |
| `update_model.py` | Retrains with user feedback and promotes the model only if it is not worse |
| `models/` | The trained pipeline (`spam_pipeline.joblib`) and its `metrics.json` |
| `spam.ipynb` | Exploratory analysis and comparison of 11 classifiers |
| `tests/` | Tests for the shared code and the app |

# How to Run Locally
1.	 Clone the repository: git clone https://github.com/Getrudetrudy01/Email-spam-detector-using-machine-learning-model.git && cd Email-spam-detector-using-machine-learning-model
2.	 Create and activate a virtual environment: py -m venv .venv && .venv\Scripts\activate
3.	 Install dependencies: pip install -r requirements.txt
4.	 Train the model (optional, a trained model is included): python train_model.py
5.	 Run the Streamlit app: streamlit run streamlit_app.py
6.	 Open your browser at your specific url displayed in the terminal of your editor.

To run the tests or the notebook, also install the development dependencies: pip install -r requirements-dev.txt, then run: pytest

Dependency versions are pinned in `requirements.txt` because a saved scikit-learn model is only guaranteed to load with the version that trained it. If you upgrade scikit-learn, retrain with `python train_model.py`.

# Deployment
The app is deployed on Streamlit Cloud. To deploy your own version, push your code to GitHub, visit https://share.streamlit.io, connect your repo, and select `streamlit_app.py` as the main file.

Note: Streamlit Cloud does not keep files between restarts, so feedback saved by the deployed app is lost when it restarts. To keep feedback from the live app, store it somewhere persistent such as a database or Google Sheet.

# Example Messages
# Spam Examples:
•	Congratulations! You’ve been selected to win a free iPhone. Click here to claim your prize!
•	You have won $5000 cash! Reply WIN to receive your reward.
•	Get cheap loans instantly! Apply now.
# Not Spam Examples:
•	Hey Getrude, are we still meeting tomorrow for the project discussion?
•	Your order has been shipped and will arrive by Friday.
•	Please find attached the updated report for your review.

# Feedback & Retraining
Users can mark each prediction as correct or incorrect. Only messages that receive feedback are stored, in `feedback_data.csv`, which is kept out of git because it contains what users typed.

Retrain with: python update_model.py

The script trains on the original training data plus feedback, then scores both the current and the new model on the same held-out test set. The new model replaces the current one only if its spam F1 does not drop (`--force` overrides this). The previous model is kept in `models/previous/`.

Feedback comes from anonymous users, so review `feedback_data.csv` before retraining. Someone could deliberately submit wrong labels.

# Model Performance
Results on a held-out test set of 1,032 messages (20% of 5,158 unique messages), with a 0.5 spam threshold:

| Metric | Score |
|---|---|
| Spam precision | 97.4% |
| Spam recall | 89.1% |
| Spam F1 | 93.1% |
| Average precision | 97.0% |
| Accuracy | 98.4% |

Out of 128 spam messages in the test set, 114 were caught and 14 were missed. Out of 904 normal messages, 3 were wrongly flagged as spam.

Accuracy looks high partly because 87% of messages are not spam: a model that always answered "not spam" would already score 87%. Spam recall and precision are the more useful numbers. The current figures are always in `models/metrics.json`.

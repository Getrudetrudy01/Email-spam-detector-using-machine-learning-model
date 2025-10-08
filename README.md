# Email / SMS Spam Detector
A machine learning web app that detects whether an email or SMS message is Spam or Not Spam, built using Python, Scikit-learn, and Streamlit.
 Live Demo: Try it here https://email-spam-detector-ml-project.streamlit.app/

# Project Overview
Spam messages (junk mail) are unsolicited communications often containing scams, phishing attempts, or misleading information. This project demonstrates how Natural Language Processing (NLP) and Machine Learning can be used to automatically classify messages as Spam or Not Spam (Ham).
Key Features
•	 Data Preprocessing: Cleans text (lowercasing, stopword removal, stemming)
•	 Feature Extraction: TF-IDF Vectorization
•	 Model Training: Multinomial Naive Bayes (optionally Logistic Regression or SVM)
•	 Evaluation Metrics: Accuracy, Precision, Recall, F1-score
•	 Model Persistence: Trained model & vectorizer saved using Joblib
•	 Web Interface: Streamlit app for real-time spam detection
•	 User Feedback System: Users can rate predictions as correct or incorrect
•	Retraining Support: Incorporates feedback for future model improvement


# How to Run Locally
1.	1. Clone the repository: git clone https://github.com/Getrudetrudy01/email-spam-detector.git
2.	2. Create and activate a virtual environment: py -m venv .venv && .venv\Scripts\activate
3.	3. Install dependencies: pip install -r requirements.txt
4.	4. Train the model (optional): python train_model.py
5.	5. Run the Streamlit app: streamlit run streamlit_app.py
6.	6. Open your browser at your specific url displayed in the terminal of your editor.
# Deployment
The app is deployed on Streamlit Cloud. To deploy your own version, push your code to GitHub, visit https://share.streamlit.io, connect your repo, and select `streamlit_app.py` as the main file.
Example Messages
Spam Examples:
•	Congratulations! You’ve been selected to win a free iPhone. Click here to claim your prize!
•	You have won $5000 cash! Reply WIN to receive your reward.
•	Get cheap loans instantly! Apply now.
Not Spam Examples:
•	Hey Getrude, are we still meeting tomorrow for the project discussion?
•	Your order has been shipped and will arrive by Friday.
•	Please find attached the updated report for your review.
# Feedback & Retraining
Users can rate each prediction as correct or incorrect. Feedback is stored in `feedback_data.csv`. Retrain the model later with: python update_model.py

# Model Performance
Accuracy: ~98%, Precision: High, Recall: High, F1-score: Excellent (values depend on retraining).



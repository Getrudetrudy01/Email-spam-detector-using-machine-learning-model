import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load the vectorizer and model
tfidf = pickle.load(open('vectorizer1.pkl','rb'))
model = pickle.load(open('model1.pkl','rb'))

# Streamlit UI
st.set_page_config(page_title="Email Spam Detector", page_icon="📧", layout="wide")

st.title("📧 Email Spam Detector")
st.markdown("### Detect spam emails using Machine Learning")

# Create two columns
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("#### Enter your email/SMS text below:")
    input_sms = st.text_area("", height=200, placeholder="Type or paste your message here...")

    if st.button('🔍 Predict', use_container_width=True):
        if input_sms:
            # 1. Preprocess
            transformed_sms = transform_text(input_sms)
            # 2. Vectorize
            vector_input = tfidf.transform([transformed_sms])
            # 3. Predict
            result = model.predict(vector_input)[0]
            # 4. Display

            if result == 1:
                st.error("### 🚨 SPAM DETECTED!")
                st.warning("This message appears to be spam. Please be cautious.")
            else:
                st.success("### ✅ NOT SPAM")
                st.info("This message appears to be legitimate.")
        else:
            st.warning("⚠️ Please enter some text to analyze")

with col2:
    st.markdown("#### About")
    st.info("""
    This spam detector uses:
    - **TF-IDF** vectorization
    - **MultinomialNB** classifier
    - **97%+ accuracy** on test data

    **How it works:**
    1. Text preprocessing (lowercase, tokenization, stemming)
    2. Feature extraction using TF-IDF
    3. Classification using Naive Bayes
    """)

    st.markdown("#### Examples")
    if st.button("Try Spam Example"):
        st.session_state.example = "WINNER!! You have won a £1000 prize! Click here to claim now!"

    if st.button("Try Ham Example"):
        st.session_state.example = "Hey, are we still meeting for lunch today at 1pm?"

    if 'example' in st.session_state:
        st.code(st.session_state.example)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit | Powered by scikit-learn")

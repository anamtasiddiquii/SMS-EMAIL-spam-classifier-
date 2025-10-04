import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download necessary data
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# Preprocessing function
def transform_text(input):
    input = input.lower()
    input = nltk.word_tokenize(input)

    y = []
    for i in input:
        if i.isalnum():
            y.append(i)

    input = y[:]
    y.clear()

    for i in input:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    input = y[:]
    y.clear()

    for i in input:
        y.append(ps.stem(i))

    return " ".join(y)


# Load vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# 💻 Web App UI
st.set_page_config(page_title="Spam Detector", page_icon="📩", layout="centered")

st.markdown("""
    <div style="text-align: center; padding: 10px;">
        <h1>📩 Email/SMS Spam Classifier</h1>
        <p style="font-size:18px;">🔎 Paste any message below and find out if it's <strong>Spam</strong> or <strong>Safe</strong>.</p>
    </div>
""", unsafe_allow_html=True)

# ✉️ Input Box
input_sms = st.text_area("✉️ Enter the message below:", placeholder="Type or paste a message here...")

# 📊 Prediction
if st.button("🚀 Predict"):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message before predicting.")
    else:
        # 1️⃣ Preprocess
        transformed_sms = transform_text(input_sms)
        # 2️⃣ Vectorize
        vector_input = tfidf.transform([transformed_sms])
        # 3️⃣ Predict
        result = model.predict(vector_input)[0]
        # 4️⃣ Show result
        if result == 1:
            st.error("🚨 This message is likely **SPAM**. Be careful!")
        else:
            st.success("✅ This message looks **SAFE**.")

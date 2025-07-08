# app.py

import streamlit as st
import pickle
import re
import nltk

# Download stopwords once
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load stopwords
stop_words = set(stopwords.words('english'))

# -------------------------------
# Load your model and vectorizer
# -------------------------------

model = pickle.load(open('sentiment_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# -------------------------------
# Clean function (same as training)
# -------------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# -------------------------------
# Streamlit UI
# -------------------------------

st.title("📝 Sentiment Analysis App")

st.write("Type your text below and click **Analyze**:")

user_input = st.text_area("Your Text", height=150)

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text!")
    else:
        cleaned_input = clean_text(user_input)
        input_vec = vectorizer.transform([cleaned_input])
        prediction = model.predict(input_vec)[0]

        st.success(f"**Sentiment:** {prediction.capitalize()} 🎉")



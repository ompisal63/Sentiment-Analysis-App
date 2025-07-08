import streamlit as st
import pickle

# Load model + vectorizer
model = pickle.load(open('sentiment_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Streamlit config
st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬", layout="centered")
st.title("Sentiment Analysis App")

# Text input
user_input = st.text_area("Enter your text:")

if st.button("Analyze"):
    if user_input:
        # Transform input
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]
        st.write(f"**Sentiment:** {prediction.capitalize()}")
    else:
        st.write("Please enter some text!")


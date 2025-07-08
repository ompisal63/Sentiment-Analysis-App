# app.py

import streamlit as st
import pickle

# -------------------------------
# Load your model and vectorizer
# -------------------------------

model = pickle.load(open('sentiment_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

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
        input_vec = vectorizer.transform([user_input])  # NO cleaning here!
        prediction = model.predict(input_vec)[0]

        st.success(f"**Sentiment:** {prediction.capitalize()} 🎉")




import streamlit as st
import pickle

st.title("🎉 Sentiment Analysis App")

# Load your trained model
model = pickle.load(open('sentiment_model.pkl', 'rb'))

# Input box
user_input = st.text_area("✍️ Enter your text:")

# Button
if st.button("🔍 Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        prediction = model.predict([user_input])[0]
        st.success(f"**Sentiment:** `{prediction.upper()}` ✅")

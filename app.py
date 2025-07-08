import streamlit as st
import pickle

# Load model & vectorizer
model = pickle.load(open('sentiment_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# UI
st.title("📝 Sentiment Analysis")

st.write("Type your text below:")

user_input = st.text_area("Your Text", height=150)

if st.button("Analyze"):
    # 1️⃣ Safe fallback: make sure it is always a string
    if not isinstance(user_input, str):
        st.error("Input must be text.")
    elif user_input.strip() == "":
        st.warning("⚠️ Please enter some text!")
    else:
        # 2️⃣ Always convert to string just to be safe
        safe_input = str(user_input).strip()

        # 3️⃣ Predict
        try:
            input_vec = vectorizer.transform([safe_input])
            prediction = model.predict(input_vec)[0]
            st.success(f"**Sentiment:** {prediction.capitalize()}")
        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")




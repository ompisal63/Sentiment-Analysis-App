import streamlit as st
import pickle

# Load the pipeline (vectorizer + model in one!)
model = pickle.load(open('sentiment_model.pkl', 'rb'))

# UI
st.title("📝 Sentiment Analysis")

st.write("Type your text below:")

user_input = st.text_area("Your Text", height=150)

if st.button("Analyze"):
    if not isinstance(user_input, str):
        st.error("Input must be text.")
    elif user_input.strip() == "":
        st.warning("⚠️ Please enter some text!")
    else:
        safe_input = str(user_input).strip()
        try:
            prediction = model.predict([safe_input])[0]
            st.success(f"**Sentiment:** {prediction.capitalize()}")
        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")





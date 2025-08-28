import streamlit as st
import joblib

# Load model + vectorizer with absolute paths
model = joblib.load(r"C:\Users\rahul\sentiment_model.pkl")
vectorizer = joblib.load(r"C:\Users\rahul\tfidf.pkl")

st.title("📊 Twitter Sentiment Analysis")

user_input = st.text_area("Enter a tweet or any text:")

if st.button("Analyze"):
    if user_input.strip():
        vec = vectorizer.transform([user_input])
        prediction = model.predict(vec)[0]
        
        if prediction == "positive":
            st.success("😊 Positive Sentiment")
        elif prediction == "neutral":
            st.warning("😐 Neutral Sentiment")
        else:
            st.error("😞 Negative Sentiment")
    else:
        st.write("⚠️ Please enter some text")

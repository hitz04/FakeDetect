# app.py

import streamlit as st
import pickle
import numpy as np

# Load model and vectorizer
try:
    with open("mmodel (1).pkl", "rb") as f:
        model = pickle.load(f)
    print("✅ Model loaded!")

    with open("vectorizer (1).pkl", "rb") as f:
        vectorizer = pickle.load(f)
    print("✅ Vectorizer loaded!")

except Exception as e:
    print("❌ Failed to load model or vectorizer:", e)


# App title
st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detector")
st.markdown("Paste a news article or story below to check if it's **Fake** or **Real**.")

# Text input
user_input = st.text_area("Enter News Article Text", height=250, placeholder="Paste your news article here...")

# Predict button
if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text before predicting.")
    else:
        try:
            X_new = vectorizer.transform([user_input])
            prediction = model.predict(X_new)[0]
            confidence = model.predict_proba(X_new).max()
            label = "🟢 Real" if prediction == 1 else "🔴 Fake"
            st.success(f"Prediction: {label}")
            st.info(f"Confidence: {confidence:.2%}")
        except Exception as e:
            st.error(f"Something went wrong during prediction: {e}")



import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load tokenizer and model
@st.cache_resource
def load_resources():
    with open("tokenizer.pkl", "rb") as handle:
        tokenizer = pickle.load(handle)
    model = tf.keras.models.load_model("sentiment_analysis_lstm_model.h5")
    return tokenizer, model

# Preprocessing function
def preprocess_input(text, tokenizer, max_length=100):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    return padded_sequence

# Main app
def main():
    st.title("IMDB Review Sentiment Analysis")
    st.write("This app predicts the sentiment of movie reviews as Positive or Negative using an LSTM model.")

    # Text input for review
    review = st.text_area("Enter a movie review:", placeholder="Type your review here...")
    
    if st.button("Predict Sentiment"):
        if review.strip() == "":
            st.warning("Please enter a valid review.")
        else:
            tokenizer, model = load_resources()
            processed_input = preprocess_input(review, tokenizer)
            prediction = model.predict(processed_input)
            sentiment = "Positive" if prediction >= 0.5 else "Negative"
            
            st.subheader("Prediction:")
            st.write(f"The sentiment of the review is **{sentiment}**.")
            st.write(f"Confidence: {prediction[0][0]:.2f}")

if __name__ == "__main__":
    main()

import streamlit as st
import joblib
import string


# Load the pre-trained model and vectorizer
model = joblib.load('sentiment_analysis_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Function to preprocess the input text
def preprocess_text(text):
    # Lowercase the text
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Return the preprocessed text
    return text

# Streamlit app
def main():
    st.title("Sentiment Analysis App")
    st.write("Analyze the sentiment of your text as positive or negative.")

    # Text input
    user_input = st.text_area("Enter your text for sentiment analysis:")

    if st.button("Analyze Sentiment"):
        if user_input.strip():
            # Preprocess and vectorize the input text
            processed_text = preprocess_text(user_input)
            
            # Transform using the pre-fitted vectorizer (no fitting needed)
            vectorized_text = vectorizer.transform([processed_text])  # Use transform, not fit_transform

            # Predict sentiment
            prediction = model.predict(vectorized_text)[0]
            probabilities = model.predict_proba(vectorized_text)[0]

            # Display sentiment
            if prediction == 1:
                st.success("The sentiment is Positive!")
            else:
                st.error("The sentiment is Negative!")

            # Display confidence scores
            
            st.write(f"Confidence: Positive - {probabilities[1] * 100:.2f}%, Negative - {probabilities[0] * 100:.2f}%")
        else:
            st.warning("Please enter some text for analysis.")

if __name__ == "__main__":
    main()

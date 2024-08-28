import streamlit as st
import joblib
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data (only needed once)
nltk.download('stopwords')
nltk.download('wordnet')

# Load your trained model and vectorizer
model = joblib.load('ensemble_model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Text preprocessing function
def preprocess_text(text):
    # Lowercase conversion
    text = text.lower()
    
    # Remove punctuation and special characters
    text = re.sub(f'[{string.punctuation}]', '', text)
    
    # Remove stopwords and lemmatize
    text = ' '.join(
        lemmatizer.lemmatize(word) 
        for word in text.split() 
        if word not in stop_words
    )
    
    return text

# Streamlit app
st.title("Sentiment Analysis App")
st.write("Enter a review text to predict its sentiment:")

# Text input
user_input = st.text_area("Enter your review here:")

# Predict button
if st.button("Predict Sentiment"):
    if user_input:
        # Preprocess the text
        processed_text = preprocess_text(user_input)
        
        # Vectorize the text
        vectorized_text = vectorizer.transform([processed_text])
        
        # Predict using the model
        prediction = model.predict(vectorized_text)
        
        # Display the result with an animated emoji
        if prediction[0] == 0:
            st.write("Sentiment: Negative")
            st.image("negative.gif", width=100)  # Display negative emoji
        elif prediction[0] == 1:
            st.write("Sentiment: Neutral")
            st.image("neutral.gif", width=100)  # Display neutral emoji
        else:
            st.write("Sentiment: Positive")
            st.image("positive.gif", width=100)  # Display positive emoji
    else:
        st.write("Please enter a review text.")

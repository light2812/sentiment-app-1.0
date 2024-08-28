import pickle
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the model
try:
    with open('E:/ictproj/app/ensemble_model .pkl', 'rb') as file:
        model = pickle.load(file)
    print("Model loaded successfully!")

except Exception as e:
    print("Error loading model:", e)
print(nltk.data.find('tokenizers/punkt'))
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Download NLTK data (only needed once)
nltk.download('punkt')
nltk.download('stopwords')

# Load model and vectorizer
with open("sentiment_model.pkl", "rb") as f:
    classifier = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load stopwords
stop_words = set(stopwords.words('english'))

# Get user input
user_input = input("Enter a movie review: ")

# Preprocess the input
tokens = word_tokenize(user_input)
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
processed_text = ' '.join(filtered_tokens)

# Vectorize the input
input_vector = vectorizer.transform([processed_text])

# Predict sentiment
prediction = classifier.predict(input_vector)[0]

# Show result
print(f"\nPredicted Sentiment: {prediction.upper()}")

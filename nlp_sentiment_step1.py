import nltk

nltk.download('movie_reviews')

from nltk.corpus import movie_reviews
# ---------------- STEP 2: Clean and Preprocess the Text Data ----------------
# ---------------- STEP 2: Clean and Preprocess the Text Data ----------------
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required resources
nltk.download('punkt')
nltk.download('stopwords')

# Preprocessing function
def preprocess_review(words):
    stop_words = set(stopwords.words('english'))
    cleaned = [
        word.lower()
        for word in words
        if word.isalpha() and word.lower() not in stop_words
    ]
    return cleaned

# Load and clean the movie reviews
documents = []
for fileid in movie_reviews.fileids():
    category = movie_reviews.categories(fileid)[0]  # 'pos' or 'neg'
    words = movie_reviews.words(fileid)
    cleaned_words = preprocess_review(words)
    documents.append((cleaned_words, category))

# Show the first cleaned example
print("Sample cleaned review:", documents[0])


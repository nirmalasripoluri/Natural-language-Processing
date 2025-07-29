import nltk
import random
import pickle
from nltk.corpus import movie_reviews, stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Download required datasets
nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('stopwords')

# Load stopwords
stop_words = set(stopwords.words('english'))

# Load data and preprocess
documents = []
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        words = movie_reviews.words(fileid)
        filtered_words = [w for w in words if w.lower() not in stop_words]
        document = ' '.join(filtered_words)
        documents.append((document, category))

# Shuffle data
random.shuffle(documents)

# Split data into features and labels
texts, labels = zip(*documents)

# Vectorize text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
y = labels

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Predict on test set
predictions = classifier.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Save the trained classifier
with open("sentiment_model.pkl", "wb") as f:
    pickle.dump(classifier, f)

# Save the vectorizer
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

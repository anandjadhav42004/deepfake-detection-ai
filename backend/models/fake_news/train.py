import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import os

# Create directory if it doesn't exist
os.makedirs('models/fake_news', exist_ok=True)

# Load Data
# Assuming data/news.csv exists. 
# For a real scenario, we might download a dataset here.
try:
    df = pd.read_csv('data/news.csv')
except FileNotFoundError:
    print("Error: data/news.csv not found. Please place a dataset there.")
    exit(1)

# Preprocessing (Simple for now)
# In a robust system, we would add stemming, stop-word removal here utilizing NLTK.
labels = df.label
x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7)

# Feature Extraction
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(x_train) 
tfidf_test = tfidf_vectorizer.transform(x_test)

# Model Training
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

# Evaluation
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f'Accuracy: {round(score*100,2)}%')

# Save Model and Vectorizer
with open('models/fake_news/model.pkl', 'wb') as model_file:
    pickle.dump(pac, model_file)

with open('models/fake_news/vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(tfidf_vectorizer, vec_file)

print("Model and vectorizer saved to models/fake_news/")

import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

# Corrected Paths (Relative to backend folder)
os.makedirs('models/fake_news', exist_ok=True)
os.makedirs('data', exist_ok=True)

data = {
    'text': [
        "NASA scientists confirm the Earth is round and the Moon landing was real.",
        "The industrial revolution transformed agrarian societies into industrial ones using steam power.",
        "Experts warn that drinking bleach is a cure for viruses. Doctors say it is a miracle secret.",
        "BREAKING: Massive solar flare to shut down internet. Governments preparing emergency rations.",
        "The capital of France is Paris. It is a historical city known for the Eiffel Tower."
    ],
    'label': [0, 0, 1, 1, 0] # 0 = Real, 1 = Fake
}
df = pd.DataFrame(data)
df.to_csv('data/news.csv', index=False)

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['text'])
y = df['label']

model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X, y)

with open('models/fake_news/model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('models/fake_news/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("✅ Linguistic Demo Model Ready!")

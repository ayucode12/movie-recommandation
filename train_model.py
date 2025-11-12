import pandas as pd
import numpy as np
import ast
import re
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
import joblib
import matplotlib.pyplot as plt

# ==================== DATA LOADING ====================
print("📥 Loading data...")
movies = pd.read_csv("data/tmdb_5000_movies.csv")
credits = pd.read_csv("data/tmdb_5000_credits.csv")

# Keep useful columns (no poster_path in this version)
movies = movies[['id', 'title', 'overview', 'genres', 'keywords']]

# Merge on title
df = movies.merge(credits, on='title')

# ==================== DATA CLEANING ====================
print("🧹 Cleaning data...")
drop_cols = ['homepage', 'status', 'tagline']
for c in drop_cols:
    if c in df.columns:
        df.drop(columns=c, inplace=True)

df.dropna(subset=['overview', 'genres', 'keywords'], inplace=True)
df.drop_duplicates(subset=['title'], inplace=True)

# ==================== FEATURE EXTRACTION ====================
print("🧠 Extracting text features...")
tqdm.pandas()

def parse_json_column(text):
    try:
        items = ast.literal_eval(text)
        names = [i.get('name', '') for i in items]
        return ' '.join(names)
    except Exception:
        return ''

def top_cast(text, n=3):
    try:
        items = ast.literal_eval(text)
        names = [i.get('name', '') for i in items][:n]
        return ' '.join(names)
    except Exception:
        return ''

df['genres_text'] = df['genres'].progress_apply(parse_json_column)
df['keywords_text'] = df['keywords'].progress_apply(parse_json_column)
df['cast_text'] = df['cast'].progress_apply(lambda x: top_cast(x, n=3))
df['crew_text'] = df['crew'].progress_apply(parse_json_column) if 'crew' in df.columns else ''
df['overview_text'] = df['overview'].fillna('')

def clean_text(s):
    s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    return s.lower().strip()

df['content'] = (
    df['genres_text'] + ' ' +
    df['keywords_text'] + ' ' +
    df['cast_text'] + ' ' +
    df['overview_text']
).progress_apply(clean_text)

print("✅ Final dataset size:", df.shape)

# ==================== TF-IDF + DIMENSIONALITY REDUCTION ====================
print("🔢 Converting text to vectors...")
vectorizer = TfidfVectorizer(max_features=20000, stop_words='english')
X = vectorizer.fit_transform(df['content'])

print("📉 Applying dimensionality reduction...")
svd = TruncatedSVD(n_components=50, random_state=42)
X_reduced = svd.fit_transform(X)

plt.figure(figsize=(8,6))
plt.scatter(X_reduced[:,0], X_reduced[:,1], s=3, alpha=0.5)
plt.title("Movies in reduced 2D space")
plt.savefig("pca_plot.png")

# ==================== MODEL TRAINING ====================
print("🎯 Training KNN model...")
knn = NearestNeighbors(n_neighbors=6, metric='cosine', algorithm='brute')
knn.fit(X)

print("✅ Model training complete!")

# ==================== SAVE ARTIFACTS ====================
print("💾 Saving model and data...")
joblib.dump(knn, "backend/model.joblib")
joblib.dump(vectorizer, "backend/vectorizer.joblib")
df.to_pickle("backend/movies_df.pkl")

print("✅ All artifacts saved successfully in 'backend/' folder!")

# ==================== TEST RECOMMENDER ====================
def recommend(title):
    if title not in df['title'].values:
        print(f"❌ Movie '{title}' not found!")
        return
    idx = df.index[df['title'] == title][0]
    vec = X[idx]
    distances, indices = knn.kneighbors(vec, n_neighbors=6)
    print(f"\n🎬 Movies similar to '{title}':")
    for i, (index, dist) in enumerate(zip(indices[0][1:], distances[0][1:]), 1):
        print(f"{i}. {df.iloc[index]['title']} (Similarity: {1 - dist:.2f})")

recommend("The Dark Knight")

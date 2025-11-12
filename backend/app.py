from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from difflib import get_close_matches

# Initialize Flask
app = Flask(__name__)
CORS(app)

# ---------------- Load Models & Data ----------------
try:
    vectorizer = joblib.load("backend/vectorizer.joblib")
    knn = joblib.load("backend/model.joblib")
    df = pd.read_pickle("backend/movies_df.pkl")
    print("✅ Model and data loaded successfully!")
except Exception as e:
    print("❌ Error loading model/data:", e)


# ---------------- Default Route ----------------
@app.route("/")
def home():
    return jsonify({"message": "🎬 Flask Movie Recommendation API is running successfully!"})


# ---------------- Recommendation Route ----------------
@app.route("/recommend", methods=["GET"])
def recommend():
    movie_title = request.args.get("movie", "").strip()

    if not movie_title:
        return jsonify({"error": "Please provide a movie title using ?movie=Title"}), 400

    # Fuzzy matching to handle typos and case differences
    all_titles = df['title'].astype(str).str.lower().tolist()
    match = get_close_matches(movie_title.lower(), all_titles, n=1, cutoff=0.6)

    if not match:
        return jsonify({"error": f"Movie '{movie_title}' not found in dataset."}), 404

    matched_title = match[0]
    idx = df.index[df['title'].str.lower() == matched_title][0]

    try:
        # Vectorize all movies
        X = vectorizer.transform(df['content'])
        vec = X[idx]
        distances, indices = knn.kneighbors(vec, n_neighbors=6)

        results = []
        for i, dist in zip(indices[0][1:], distances[0][1:]):  # skip itself
            movie = df.iloc[i]
            results.append({
                "title": movie['title'],
                "genres": movie['genres_text'],
                "similarity": round(1 - dist, 2)
            })

        return jsonify({
            "query": df.iloc[idx]['title'],
            "results": results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------- Run Server ----------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)

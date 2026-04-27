import os
import pandas as pd
from flask import Flask, request, jsonify, render_template
from google import genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# --- 1. GEMINI CONFIGURATION ---
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# --- 2. TRAIN MODEL ---
df = pd.read_csv("Volunteer_match.csv")
X = df["Skills"]
y = df["Type of Organization"]

model = Pipeline(steps=[
    ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 3))),
    ("classifier", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"))
])

model.fit(X, y)

# --- 3. ROUTES ---
@app.route("/")
def home():
    return render_template("gindex.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Age validation
        try:
            user_age = int(data.get("age", 0))
        except (ValueError, TypeError):
            return jsonify({"error": "Please enter a valid number for age."}), 400

        if user_age < 17:
            return jsonify({"error": "Not Eligible: You must be at least 17 years old to participate."}), 400

        if user_age > 75:
            return jsonify({"error": "Not Eligible: Maximum age for this program is 75."}), 400

        # Prediction
        skills = data.get("skills", "")
        prediction = model.predict([skills])[0]

        # Gemini explanation
        prompt = (
            f"Volunteer {data.get('name')} (Age: {user_age}, City: {data.get('city')}) "
            f"with skills '{skills}' matched to '{prediction}'. "
            f"Explain why in 2 short sentences."
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return jsonify({
            "match": str(prediction),
            "explanation": response.text
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "An internal error occurred. Please try again."}), 500


# --- 4. RUN APP ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

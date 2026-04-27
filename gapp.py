import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from google import genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# --- 1. GEMINI CONFIGURATION ---
client = genai.Client(api_key="AIzaSyAhnd9lFesGXsWyhobdACBmYQYq_uVtyDc")

# --- 2. TRAIN MODEL ---
# Using the original dataset as requested
df = pd.read_csv('Volunteer_match.csv')
X = df['Skills'] 
y = df['Type of Organization']

model = Pipeline(steps=[
    ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 3))),
    ('classifier', RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'))
])
model.fit(X, y)

@app.route('/')
def home():
    return send_from_directory('front_end', 'gindex.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # --- AGE LIMIT VALIDATION ---
        try:
            user_age = int(data.get('age', 0))
        except (ValueError, TypeError):
            return jsonify({'error': 'Please enter a valid number for age.'}), 400

        # Specific check for under 17
        if user_age < 17:
            return jsonify({'error': 'Not Eligible: You must be at least 17 years old to participate.'}), 400
        
        # Specific check for over 75
        if user_age > 75:
            return jsonify({'error': 'Not Eligible: Maximum age for this program is 75.'}), 400

        # 1. Prediction
        skills = data.get('skills', '')
        prediction = model.predict([skills])[0]
        
        # 2. Gemini Explanation
        prompt = f"Volunteer {data.get('name')} (Age: {user_age}, City: {data.get('city')}) with skills '{skills}' matched to '{prediction}'. Explain why in 2 short sentences."
        
        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=prompt
        )

        return jsonify({
            'match': str(prediction),
            'explanation': response.text
        })
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'An internal error occurred. Please try again.'}), 500

if __name__ == "__main__":
    app.run(debug=True)
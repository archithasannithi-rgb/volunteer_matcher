import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Load the dataset
df = pd.read_csv('Volunteer_match.csv')

# 2. Define Features and Target
# We exclude 'Volunteer Name' as it's a unique identifier and doesn't help in matching
X = df[['Age', 'Gender', 'Skills', 'Availability', 'Location']]
y = df['Type of Organization']

# 3. Preprocessing Setup
# We use TF-IDF for the 'Skills' text and OneHotEncoding for categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', ['Age']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Gender', 'Availability', 'Location']),
        ('skills', TfidfVectorizer(stop_words='english'), 'Skills')
    ])

# 4. Create the Model Pipeline
# Using Random Forest as it is robust for multi-class classification
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 5. Split Data into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train the Model
model_pipeline.fit(X_train, y_train)

# 7. Evaluate the Model
y_pred = model_pipeline.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8. Function to Match a New Volunteer
def match_volunteer(age, gender, skills, availability, location):
    new_volunteer = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'Skills': skills,
        'Availability': availability,
        'Location': location
    }])
    prediction = model_pipeline.predict(new_volunteer)
    return prediction[0]

# Example Test:
example_match = match_volunteer(24, 'Female', 'First aid, healthcare', 'Weekends', 'Delhi')
print(f"\nRecommended Organization for Example: {example_match}")
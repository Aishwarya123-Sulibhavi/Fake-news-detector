import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model
model = joblib.load("fake_news_model.pkl")

# Load the vectorizer
vectorizer = joblib.load("vectorizer.pkl")  # Ensure this file exists

# Example new data (news article)
new_data = ["This is an example of a new news article."]

# Preprocess the new data
new_data_transformed = vectorizer.transform(new_data)  # Use the same vectorizer as in training

# Make prediction using the trained model
prediction = model.predict(new_data_transformed)

# Output the result
print("Predicted label (0 = real, 1 = fake):", prediction)

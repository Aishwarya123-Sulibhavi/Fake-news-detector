# Import necessary libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import joblib

# Load the train and test datasets
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Step 1: TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train = vectorizer.fit_transform(train['text'])
X_test = vectorizer.transform(test['text'])

# Step 2: Define target labels
y_train = train['label']
y_test = test['label']

# Step 3: Train the Naïve Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 4: Predict on the test set
y_pred = model.predict(X_test)

# Step 5: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Print classification report (precision, recall, F1 score, etc.)
print(classification_report(y_test, y_pred))

# Step 6: Save the trained model
joblib.dump(model, "fake_news_model.pkl")
print("Model trained and saved successfully!")

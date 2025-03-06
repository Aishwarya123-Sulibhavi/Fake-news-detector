import os
import joblib
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load("fake_news_model.pkl")  # Ensure this model is well-trained
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    fake_news = []
    real_news = []

    if request.method == "POST":
        text_input = request.form.get("news_text")
        file = request.files.get("file")

        articles = []

        if text_input:
            articles.append(text_input.strip())

        if file and file.filename.endswith(".txt"):
            file_content = file.read().decode("utf-8").strip()
            articles.extend(file_content.split("\n"))

        articles = [article.strip() for article in articles if article.strip()]

        if articles:
            transformed_articles = vectorizer.transform(articles)
            predictions = model.predict(transformed_articles)
            probabilities = model.predict_proba(transformed_articles)  # Get confidence scores

            for article, label, prob in zip(articles, predictions, probabilities):
                confidence = round(max(prob) * 100, 2)  # Convert to percentage

                if label == 1:  # Fake News
                    fake_news.append(f"{article} (Confidence: {confidence}%)")
                else:  # Real News
                    real_news.append(f"{article} (Confidence: {confidence}%)")

    return render_template("index.html", fake_news=fake_news, real_news=real_news)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

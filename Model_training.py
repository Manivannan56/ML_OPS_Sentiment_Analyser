import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv("Data/amazon_reviews.csv")

# Convert star ratings into sentiment labels
def map_sentiment(rating):
    return "Negative" if rating <= 2 else "Neutral" if rating == 3 else "Positive"

df["label"] = df["star_rating"].apply(map_sentiment)

# Prepare Data
X = df["review_body"].astype(str)
y = df["label"].map({"Negative": 0, "Neutral": 1, "Positive": 2})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Initialize MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Sentiment Analysis")

with mlflow.start_run():
    # Model Training
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    # Predictions
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)

    # Log Metrics
    mlflow.log_param("model_type", "Naive Bayes")
    mlflow.log_metric("accuracy", accuracy)

    # Save Model & Vectorizer
    joblib.dump(model, "models/naive_bayes_sentiment.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")

    # Log Artifacts
    mlflow.log_artifact("models/naive_bayes_sentiment.pkl")
    mlflow.log_artifact("models/vectorizer.pkl")

print(f"Model trained and logged to MLflow with accuracy: {accuracy:.4f}")

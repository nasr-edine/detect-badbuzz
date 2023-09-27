import joblib
from flask import Flask, request, jsonify
from text_processing_utils import preprocess_text

# Define the Flask app
app = Flask(__name__)

# Define the route for the predictor
@app.route('/predict', methods=['POST'])
def predict():
    # Get the tweet from the request body
    tweet = request.json['tweet']

    # Preprocess the tweet
    tweet = preprocess_text(tweet)

    # Load the vectorizer from the file
    tfidf_vectorizer = joblib.load('./models/tfidf_vectorizer.pkl')

    # Vectorize the preprocessed tweet
    vectorized_tweet = tfidf_vectorizer.transform([tweet])

    # Load the model
    model = joblib.load('./models/logistic_model.pkl')

    # Make the prediction
    prediction = model.predict(vectorized_tweet)

    # Interpret the prediction
    sentiment = "Positive" if prediction == 1 else "Negative"

    # Return the prediction as JSON
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(debug=True)  # Change the port to a different value, e.g., 5001
    # app.run(debug=True, port=5002)

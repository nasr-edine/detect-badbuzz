import joblib
from flask import Flask, request, jsonify
from text_processing_utils import preprocess_text

# Define the Flask app
app = Flask(__name__)

# Define the route for the predictor
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the tweet from the request body
        tweet = request.json['tweet']

        # Preprocess the tweet
        tweet = preprocess_text(tweet)

        # Load the vectorizer and model from Azure Blob Storage or another production-ready location
        tfidf_vectorizer = load_vectorizer_from_production()  # Implement this function
        model = load_model_from_production()  # Implement this function

        # Vectorize the preprocessed tweet
        vectorized_tweet = tfidf_vectorizer.transform([tweet])

        # Make the prediction
        prediction = model.predict(vectorized_tweet)

        # Interpret the prediction
        sentiment = "Positive" if prediction == 1 else "Negative"

        # Return the prediction as JSON
        return jsonify({'sentiment': sentiment})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))

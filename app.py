import os
import joblib
from flask import Flask, request, jsonify
from text_processing_utils import preprocess_text

# Define the Flask app
app = Flask(__name__)

# Define the route for the predictor
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if the 'tweet' key is present in the request JSON
        if 'tweet' not in request.json:
            return jsonify({'error': 'Missing or invalid "tweet" key in the request'}), 400

        # Get the tweet from the request body
        tweet = request.json['tweet']

        # Check if 'tweet' is empty or None
        if not tweet:
            return jsonify({'error': 'Invalid tweet provided'}), 400

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

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))

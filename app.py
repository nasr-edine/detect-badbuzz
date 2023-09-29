import os
import joblib
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from flask import Flask, request, jsonify
from flask_cors import CORS

from text_processing_utils import preprocess_text

# Define the Flask app
app = Flask(__name__)
CORS(app)

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

        max_len = 150
        tokenizer = Tokenizer()

        text_seq = tokenizer.texts_to_sequences([tweet])
        text_seq = pad_sequences(text_seq, maxlen=max_len)

        # Load the model
        # model = joblib.load('./models/logistic_model.pkl')
        model = joblib.load('./models/lstm_model_glove.pkl')

        # Make the prediction
        prediction = model.predict(text_seq)[0]

        # Interpret the prediction
        sentiment = "Positive" if prediction == 1 else "Negative"
    
        response = jsonify({'sentiment': sentiment})

        # Set the Access-Control-Allow-Origin header to '*'
        # response.headers['Access-Control-Allow-Origin'] = '*'

        # Return the prediction as JSON
        return response

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8000)

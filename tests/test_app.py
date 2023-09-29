import os
import unittest
from flask import json
from app import app

class TestApp(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.app = app.test_client()

    def test_predict_invalid_tweet(self):
        """Test that the predictor returns an error for an invalid tweet."""

        tweet = ""

        # Make the prediction
        response = self.app.post('/predict', json={'tweet': tweet})

        # Check the response status code
        self.assertEqual(response.status_code, 400)

        # Check the response body
        response_body = json.loads(response.data)
        self.assertIn("error", response_body)

        # Test an invalid request (missing 'tweet' key)
        response = self.app.post('/predict', json={})
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 400) 
        self.assertTrue('error' in data)

    def test_predict_sentiment(self):
        # Test a request with a tweet
        data = {'tweet': 'I love this product!'}
        response = self.app.post('/predict', json=data)
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)  # Expect a 200 OK
        self.assertIn('sentiment', data)
        self.assertIn(data['sentiment'], ['Positive', 'Negative'])

if __name__ == '__main__':
    unittest.main()

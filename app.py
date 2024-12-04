from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests
    
# Load the pre-trained model in app context
with app.app_context():
    model = joblib.load('nba_playoff_prediction_model.pkl')

# Helper function to make predictions
def make_prediction(data):
    try:
        # Extract features from the request
        win_loss_record = float(data.get('win_loss_record'))  # Convert win-loss record to float
        ppg = float(data.get('ppg'))  # Convert Points Per Game to float
        oppg = float(data.get('oppg'))  # Convert Opponent Points Per Game to float
        offensive_rating = float(data.get('offensive_rating'))  # Convert Offensive Rating to float
        defensive_rating = float(data.get('defensive_rating'))  # Convert Defensive Rating to float
        games_missed = float(data.get('games_missed'))  # Convert Games Missed to float

        # Prepare the feature vector for prediction
        features = np.array([[win_loss_record, ppg, oppg, offensive_rating, defensive_rating, games_missed]])

        # Make a prediction
        prediction = model.predict(features)

        # Convert prediction to 'Qualified'/'Not Qualified' label
        prediction_label = 'Qualified' if prediction == 1 else 'Not Qualified'

        # Return the result
        result = {'prediction': prediction_label}
        return result, 200

    except KeyError as ke:
        return {'error': f'Missing key: {str(ke)}'}, 400
    except ValueError as ve:
        return {'error': f'Invalid value: {str(ve)}'}, 400
    except Exception as e:
        return {'error': str(e)}, 500

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    print("Received data:", data)  # Log received data for debugging
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    result, status_code = make_prediction(data)
    return jsonify(result), status_code

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

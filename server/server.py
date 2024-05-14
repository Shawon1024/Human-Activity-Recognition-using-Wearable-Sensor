import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

model = load_model('HAR_Model.keras')
encoder = joblib.load('Encoder.joblib')
scaler = joblib.load('Scaler.joblib')

@app.route('/data', methods=['POST'])
def handle_data():
    content = request.get_json()
    if not content or 'sequence' not in content:
        return jsonify({"error": "Missing sequence data"}), 400

    try:
        sequence = np.array(content['sequence'])
        if sequence.shape[-1] != 6:
            raise ValueError(f"Expected 3 features per time step, got {sequence.shape[-1]}")
        
        sequence = pd.DataFrame(sequence, columns=['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz'])
        sequence = scaler.transform(sequence)
        sequence = sequence.reshape(1, -1, 6)
        
        prediction = model.predict(sequence)
        predicted_index = np.argmax(prediction, axis=1)[0]
        predicted_activity = encoder.inverse_transform([predicted_index])[0]
        
        print("Predicted Activity:", predicted_activity)
        
        socketio.emit('activity_update', {'activity': predicted_activity})
        return jsonify({'activity': predicted_activity})
    except Exception as e:
        print(f"Error Processing Request: {e}")
        return jsonify({"error": "Error processing request"}), 500

@app.route('/')
def index():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)

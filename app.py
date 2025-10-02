from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__, static_folder='.', template_folder='.')
CORS(app)

# Load models
MODEL_PATH = 'models'
model = None
scaler = None
target_encoder = None
label_encoders = None
feature_columns = None

def load_models():
    """Load all saved models and encoders"""
    global model, scaler, target_encoder, label_encoders, feature_columns
    
    try:
        model = joblib.load(os.path.join(MODEL_PATH, 'nids_model.pkl'))
        scaler = joblib.load(os.path.join(MODEL_PATH, 'scaler.pkl'))
        target_encoder = joblib.load(os.path.join(MODEL_PATH, 'target_encoder.pkl'))
        label_encoders = joblib.load(os.path.join(MODEL_PATH, 'label_encoders.pkl'))
        feature_columns = joblib.load(os.path.join(MODEL_PATH, 'feature_columns.pkl'))
        print("✓ Models loaded successfully!")
        return True
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        return False

def preprocess_input(data):
    """Preprocess input data for prediction"""
    
    # Define all features with default values
    default_values = {
        'duration': 0, 'protocol_type': 'tcp', 'service': 'http', 'flag': 'SF',
        'src_bytes': 0, 'dst_bytes': 0, 'land': 0, 'wrong_fragment': 0,
        'urgent': 0, 'hot': 0, 'num_failed_logins': 0, 'logged_in': 1,
        'num_compromised': 0, 'root_shell': 0, 'su_attempted': 0, 'num_root': 0,
        'num_file_creations': 0, 'num_shells': 0, 'num_access_files': 0,
        'num_outbound_cmds': 0, 'is_host_login': 0, 'is_guest_login': 0,
        'count': 0, 'srv_count': 0, 'serror_rate': 0.0, 'srv_serror_rate': 0.0,
        'rerror_rate': 0.0, 'srv_rerror_rate': 0.0, 'same_srv_rate': 0.0,
        'diff_srv_rate': 0.0, 'srv_diff_host_rate': 0.0, 'dst_host_count': 0,
        'dst_host_srv_count': 0, 'dst_host_same_srv_rate': 0.0,
        'dst_host_diff_srv_rate': 0.0, 'dst_host_same_src_port_rate': 0.0,
        'dst_host_srv_diff_host_rate': 0.0, 'dst_host_serror_rate': 0.0,
        'dst_host_srv_serror_rate': 0.0, 'dst_host_rerror_rate': 0.0,
        'dst_host_srv_rerror_rate': 0.0
    }
    
    # Merge user input with defaults
    full_data = {**default_values, **data}
    
    # Create DataFrame
    df = pd.DataFrame([full_data])
    
    # Encode categorical features
    categorical_columns = ['protocol_type', 'service', 'flag']
    for col in categorical_columns:
        if col in label_encoders:
            try:
                df[col] = label_encoders[col].transform(df[col])
            except:
                # Handle unknown categories
                df[col] = 0
    
    # Ensure we have all feature columns in correct order
    df = df[feature_columns]
    
    # Convert to numeric
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Scale features
    scaled_data = scaler.transform(df)
    
    return scaled_data

@app.route('/')
def index():
    """Serve the main page"""
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Predict intrusion from network traffic data"""
    try:
        # Get input data
        data = request.get_json()
        
        # Preprocess input
        processed_data = preprocess_input(data)
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        prediction_proba = model.predict_proba(processed_data)[0]
        
        # Get attack type
        attack_type = target_encoder.inverse_transform([prediction])[0]
        confidence = float(max(prediction_proba))
        
        # Determine if it's an attack
        is_attack = attack_type != 'normal'
        
        # Get all class probabilities
        class_probabilities = {}
        for idx, class_name in enumerate(target_encoder.classes_):
            class_probabilities[class_name] = float(prediction_proba[idx])
        
        response = {
            'success': True,
            'is_attack': is_attack,
            'attack_type': attack_type,
            'confidence': confidence,
            'probabilities': class_probabilities,
            'recommendation': get_recommendation(attack_type, confidence)
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Predict multiple network traffic samples"""
    try:
        data = request.get_json()
        samples = data.get('samples', [])
        
        results = []
        for sample in samples:
            processed_data = preprocess_input(sample)
            prediction = model.predict(processed_data)[0]
            prediction_proba = model.predict_proba(processed_data)[0]
            
            attack_type = target_encoder.inverse_transform([prediction])[0]
            confidence = float(max(prediction_proba))
            
            results.append({
                'attack_type': attack_type,
                'confidence': confidence,
                'is_attack': attack_type != 'normal'
            })
        
        return jsonify({
            'success': True,
            'results': results,
            'total_samples': len(samples),
            'attacks_detected': sum(1 for r in results if r['is_attack'])
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    try:
        info = {
            'model_type': type(model).__name__,
            'num_features': len(feature_columns),
            'attack_types': target_encoder.classes_.tolist(),
            'categorical_features': list(label_encoders.keys())
        }
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def get_recommendation(attack_type, confidence):
    """Get security recommendation based on attack type"""
    recommendations = {
        'normal': 'Traffic appears normal. Continue monitoring.',
        'dos': 'Denial of Service attack detected! Block source IP immediately and increase bandwidth capacity.',
        'probe': 'Network probing detected! Review firewall rules and monitor for subsequent attacks.',
        'r2l': 'Remote to Local attack detected! Check authentication logs and reset compromised credentials.',
        'u2r': 'User to Root attack detected! Isolate affected system immediately and perform security audit.',
        'unknown': 'Unknown attack pattern detected! Flag for manual review by security team.'
    }
    
    return recommendations.get(attack_type, 'Monitor traffic closely and alert security team.')

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🛡️  Network Intrusion Detection System API")
    print("="*60)
    
    if load_models():
        print(f"\n✓ Attack types: {target_encoder.classes_.tolist()}")
        print(f"✓ Number of features: {len(feature_columns)}")
        print("\n🚀 Starting Flask server...")
        print("📡 Access the web interface at: http://localhost:5000")
        print("="*60 + "\n")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n✗ Failed to load models. Please run train_pipeline.py first!")
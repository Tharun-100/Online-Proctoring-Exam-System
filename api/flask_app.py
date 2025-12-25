"""
Flask API for Fraud Detection System with Image Upload
"""
from flask import Flask, request, render_template, jsonify, send_file
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import os
import sys
import io
import base64
from PIL import Image
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import FraudDetectionInference
from src.feature_extraction import FeatureExtractor
from config import MODEL_PATH, PREPROCESSOR_PATH

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for models
inference = None
feature_extractor = None


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def init_models():
    """Initialize models on startup"""
    global inference, feature_extractor
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH):
            inference = FraudDetectionInference()
            feature_extractor = FeatureExtractor()
            print("Models loaded successfully")
            return True
        else:
            print("Warning: Model files not found. Please train the model first.")
            return False
    except Exception as e:
        print(f"Error loading models: {e}")
        return False


# Initialize models on startup
init_models()


@app.route('/')
def index():
    """Main page with upload form"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, GIF, BMP'}), 400
    
    if inference is None or feature_extractor is None:
        return jsonify({'error': 'Models not loaded. Please train the model first.'}), 503
    
    try:
        # Read image file
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Failed to decode image'}), 400
        
        # Extract features from image
        features = feature_extractor.extract_features(image)
        
        # Make prediction
        prediction_result = inference.predict(features)
        
        # Draw annotations on image
        annotated_image = feature_extractor.draw_annotations(image, features)
        
        # Add prediction label to image
        label = prediction_result['prediction_label']
        probability = prediction_result['fraud_probability']
        color = (0, 0, 255) if prediction_result['prediction'] == 1 else (0, 255, 0)
        
        cv2.putText(annotated_image, f"Prediction: {label} ({probability:.2%})",
                   (10, annotated_image.shape[0] - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Convert annotated image to base64 for display
        _, buffer = cv2.imencode('.jpg', annotated_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Create classification report (simplified for single prediction)
        # In a real scenario with multiple predictions, you'd have true labels
        report_data = {
            'prediction': prediction_result['prediction'],
            'prediction_label': prediction_result['prediction_label'],
            'fraud_probability': prediction_result['fraud_probability'],
            'legitimate_probability': prediction_result['legitimate_probability'],
            'features': features
        }
        
        # Generate a text classification report
        report_text = generate_classification_report(prediction_result)
        
        return jsonify({
            'success': True,
            'image': img_base64,
            'prediction': prediction_result,
            'classification_report': report_text,
            'features': features
        })
    
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500


def generate_classification_report(prediction_result: dict) -> str:
    """Generate a classification report text"""
    label = prediction_result['prediction_label']
    fraud_prob = prediction_result['fraud_probability']
    legit_prob = prediction_result['legitimate_probability']
    
    report = f"""
=== CLASSIFICATION REPORT ===

Prediction: {label}
Confidence Score: {max(fraud_prob, legit_prob):.4f}

Class Probabilities:
  - Legitimate: {legit_prob:.4f} ({legit_prob*100:.2f}%)
  - Fraud:      {fraud_prob:.4f} ({fraud_prob*100:.2f}%)

Interpretation:
"""
    if prediction_result['prediction'] == 1:
        report += "  ⚠️  FRAUD DETECTED - Suspicious behavior detected in the image.\n"
        report += "     Please review the exam session for potential violations.\n"
    else:
        report += "  ✓  LEGITIMATE - No suspicious behavior detected.\n"
        report += "     The exam session appears normal.\n"
    
    report += "\n=== END REPORT ==="
    
    return report


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if (inference is not None and feature_extractor is not None) else 'models_not_loaded',
        'models_loaded': inference is not None and feature_extractor is not None
    })


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for programmatic access (returns JSON only)"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if inference is None or feature_extractor is None:
        return jsonify({'error': 'Models not loaded'}), 503
    
    try:
        # Read image
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Failed to decode image'}), 400
        
        # Extract features and predict
        features = feature_extractor.extract_features(image)
        prediction_result = inference.predict(features)
        
        return jsonify({
            'success': True,
            'prediction': prediction_result,
            'features': features
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("Starting Flask API server...")
    print("Access the web interface at: http://localhost:5000")
    print("API documentation: http://localhost:5000/health")
    app.run(debug=True, host='0.0.0.0', port=5000)




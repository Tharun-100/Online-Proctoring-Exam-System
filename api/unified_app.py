"""
Unified Flask API for Fraud Detection System
Combines image upload and feature-based prediction endpoints
"""
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
import cv2
import numpy as np
import os
import sys
import base64

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import FraudDetectionInference
from src.feature_extraction import FeatureExtractor
from config import MODEL_PATH, PREPROCESSOR_PATH

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

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
            print("✓ Models loaded successfully")
            return True
        else:
            print("⚠ Warning: Model files not found. Please train the model first.")
            return False
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        return False


# Initialize models on startup
init_models()


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


def make_prediction_with_report(features: dict):
    """Make prediction and generate report from features"""
    if inference is None:
        raise ValueError("Models not loaded. Please train the model first.")
    
    # Make prediction
    prediction_result = inference.predict(features)
    
    # Generate classification report
    report_text = generate_classification_report(prediction_result)
    
    return prediction_result, report_text


@app.route('/')
def index():
    """Main page with upload form"""
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if (inference is not None and feature_extractor is not None) else 'models_not_loaded',
        'models_loaded': inference is not None and feature_extractor is not None,
        'endpoints': {
            'web_interface': '/',
            'image_upload': '/predict/image',
            'feature_prediction': '/predict/features',
            'batch_prediction': '/predict/batch'
        }
    })


@app.route('/predict/image', methods=['POST'])
def predict_from_image():
    """
    Handle image upload, extract features, and make prediction
    
    Accepts: multipart/form-data with 'image' file
    Returns: JSON with prediction, classification report, annotated image, and extracted features
    """
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
        
        # Make prediction and generate report
        prediction_result, report_text = make_prediction_with_report(features)
        
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
        
        return jsonify({
            'success': True,
            'image': img_base64,
            'prediction': prediction_result,
            'classification_report': report_text,
            'features': features,
            'endpoint': 'predict/image'
        })
    
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500


@app.route('/predict/features', methods=['POST'])
def predict_from_features():
    """
    Make prediction from extracted features
    
    Accepts: JSON with feature values
    Returns: JSON with prediction and classification report
    
    Example request body:
    {
        "face_present": 1,
        "no_of_face": 1,
        "face_x": 262.5,
        ...
    }
    """
    if inference is None:
        return jsonify({'error': 'Models not loaded. Please train the model first.'}), 503
    
    try:
        # Get JSON data
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        features = request.get_json()
        
        if not features:
            return jsonify({'error': 'No features provided'}), 400
        
        # Validate required features (at least check if dict is not empty)
        if len(features) == 0:
            return jsonify({'error': 'Features dictionary is empty'}), 400
        
        # Make prediction and generate report
        prediction_result, report_text = make_prediction_with_report(features)
        
        return jsonify({
            'success': True,
            'prediction': prediction_result,
            'classification_report': report_text,
            'features': features,
            'endpoint': 'predict/features'
        })
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 503
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Make batch predictions from multiple feature sets
    
    Accepts: JSON with list of feature dictionaries
    Returns: JSON with list of predictions and reports
    
    Example request body:
    {
        "data": [
            {"face_present": 1, ...},
            {"face_present": 0, ...}
        ]
    }
    """
    if inference is None:
        return jsonify({'error': 'Models not loaded. Please train the model first.'}), 503
    
    try:
        # Get JSON data
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        
        if 'data' not in data:
            return jsonify({'error': 'Missing "data" field. Expected: {"data": [...]}'}), 400
        
        features_list = data['data']
        
        if not isinstance(features_list, list):
            return jsonify({'error': '"data" must be a list of feature dictionaries'}), 400
        
        if len(features_list) == 0:
            return jsonify({'error': 'Features list is empty'}), 400
        
        # Process each feature set
        results = []
        for idx, features in enumerate(features_list):
            try:
                prediction_result, report_text = make_prediction_with_report(features)
                results.append({
                    'index': idx,
                    'prediction': prediction_result,
                    'classification_report': report_text,
                    'features': features
                })
            except Exception as e:
                results.append({
                    'index': idx,
                    'error': f'Error processing item {idx}: {str(e)}',
                    'features': features
                })
        
        return jsonify({
            'success': True,
            'count': len(results),
            'results': results,
            'endpoint': 'predict/batch'
        })
    
    except Exception as e:
        return jsonify({'error': f'Batch prediction error: {str(e)}'}), 500


# Legacy endpoint for backward compatibility
@app.route('/predict', methods=['POST'])
def predict_legacy():
    """
    Legacy endpoint - detects if image or features are provided
    Routes to appropriate handler
    """
    # Check if image file is present
    if 'image' in request.files:
        return predict_from_image()
    # Otherwise, assume it's JSON features
    elif request.is_json:
        return predict_from_features()
    else:
        return jsonify({
            'error': 'Invalid request. Provide either an image file or JSON features.',
            'image_upload': 'POST /predict/image with multipart/form-data',
            'features': 'POST /predict/features with JSON'
        }), 400


# API documentation endpoint
@app.route('/api/docs', methods=['GET'])
def api_docs():
    """API documentation endpoint"""
    return jsonify({
        'title': 'Fraud Detection API Documentation',
        'version': '1.0.0',
        'endpoints': {
            'GET /': {
                'description': 'Web interface for image upload',
                'content_type': 'text/html'
            },
            'GET /health': {
                'description': 'Health check and endpoint information',
                'returns': 'JSON with status and available endpoints'
            },
            'POST /predict/image': {
                'description': 'Upload image, extract features, and get prediction',
                'content_type': 'multipart/form-data',
                'required_field': 'image (file)',
                'returns': 'JSON with prediction, report, annotated image, and features'
            },
            'POST /predict/features': {
                'description': 'Get prediction from extracted features',
                'content_type': 'application/json',
                'required_fields': 'All 37 feature values',
                'returns': 'JSON with prediction, report, and features'
            },
            'POST /predict/batch': {
                'description': 'Batch prediction from multiple feature sets',
                'content_type': 'application/json',
                'required_field': 'data (array of feature dictionaries)',
                'returns': 'JSON with array of predictions and reports'
            },
            'POST /predict': {
                'description': 'Legacy endpoint - auto-detects image or features',
                'returns': 'Routes to /predict/image or /predict/features'
            }
        },
        'feature_list': [
            'face_present', 'no_of_face', 'face_x', 'face_y', 'face_w', 'face_h',
            'left_eye_x', 'left_eye_y', 'right_eye_x', 'right_eye_y',
            'nose_tip_x', 'nose_tip_y', 'mouth_x', 'mouth_y', 'face_conf',
            'hand_count', 'left_hand_x', 'left_hand_y', 'right_hand_x', 'right_hand_y',
            'hand_obj_interaction', 'head_pose', 'head_pitch', 'head_yaw', 'head_roll',
            'phone_present', 'phone_loc_x', 'phone_loc_y', 'phone_conf',
            'gaze_on_script', 'gaze_direction', 'gazePoint_x', 'gazePoint_y',
            'pupil_left_x', 'pupil_left_y', 'pupil_right_x', 'pupil_right_y'
        ]
    })


if __name__ == '__main__':
    print("=" * 60)
    print("Fraud Detection System - Unified Flask API")
    print("=" * 60)
    print("Access the web interface at: http://localhost:5000")
    print("API documentation: http://localhost:5000/api/docs")
    print("Health check: http://localhost:5000/health")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)


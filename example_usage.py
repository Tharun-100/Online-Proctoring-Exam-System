"""
Example usage script for Fraud Detection System
"""
import pandas as pd
from src.inference import FraudDetectionInference

# Example: Load inference pipeline and make predictions
if __name__ == "__main__":
    print("Fraud Detection System - Example Usage")
    print("=" * 60)
    
    # Load inference pipeline (model must be trained first)
    try:
        inference = FraudDetectionInference()
        print("✓ Model loaded successfully\n")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("Please train the model first by running: python src/train.py")
        exit(1)
    
    # Example 1: Single prediction with sample data
    print("Example 1: Single Prediction")
    print("-" * 60)
    
    sample_data = {
        'face_present': 1,
        'no_of_face': 1,
        'face_x': 262.5,
        'face_y': 312.0,
        'face_w': 151.7,
        'face_h': 151.7,
        'left_eye_x': 307.9,
        'left_eye_y': 354.4,
        'right_eye_x': 373.0,
        'right_eye_y': 353.4,
        'nose_tip_x': 344.5,
        'nose_tip_y': 387.3,
        'mouth_x': 345.0,
        'mouth_y': 419.5,
        'face_conf': 87.7,
        'hand_count': 0,
        'left_hand_x': 0,
        'left_hand_y': 0,
        'right_hand_x': 0,
        'right_hand_y': 0,
        'hand_obj_interaction': 0,
        'head_pose': 'forward',
        'head_pitch': 0.018,
        'head_yaw': 0.015,
        'head_roll': -0.0005,
        'phone_present': 0,
        'phone_loc_x': 0,
        'phone_loc_y': 0,
        'phone_conf': 0,
        'gaze_on_script': 1,
        'gaze_direction': 'center',
        'gazePoint_x': 336,
        'gazePoint_y': 325,
        'pupil_left_x': 304,
        'pupil_left_y': 348,
        'pupil_right_x': 368,
        'pupil_right_y': 354
    }
    
    result = inference.predict(sample_data)
    print(f"Prediction: {result['prediction_label']}")
    print(f"Fraud Probability: {result['fraud_probability']:.4f}")
    print(f"Legitimate Probability: {result['legitimate_probability']:.4f}")
    
    # Example 2: Batch prediction from CSV
    print("\n" + "=" * 60)
    print("Example 2: Batch Prediction from CSV")
    print("-" * 60)
    print("To make batch predictions, use:")
    print("  python src/inference.py --input path/to/data.csv --output path/to/predictions.csv")
    
    # Example 3: API usage
    print("\n" + "=" * 60)
    print("Example 3: API Deployment")
    print("-" * 60)
    print("To start the API server, use:")
    print("  python api/app.py")
    print("\nThen access the API at: http://localhost:8000")
    print("API documentation: http://localhost:8000/docs")
    print("\n" + "=" * 60)




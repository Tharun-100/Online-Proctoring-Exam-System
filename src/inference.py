"""
Inference pipeline for Fraud Detection System
"""
import pandas as pd
import numpy as np
import joblib
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MODEL_PATH, PREPROCESSOR_PATH
from src.data_preprocessing import DataPreprocessor


class FraudDetectionInference:
    """Class for making fraud detection predictions"""
    
    def __init__(self, model_path=None, preprocessor_path=None):
        """
        Initialize inference pipeline
        
        Args:
            model_path: Path to trained model (default: from config)
            preprocessor_path: Path to preprocessor (default: from config)
        """
        self.model_path = model_path or MODEL_PATH
        self.preprocessor_path = preprocessor_path or PREPROCESSOR_PATH
        
        # Load model and preprocessor
        self.model = None
        self.preprocessor = None
        self.load_model()
    
    def load_model(self):
        """Load trained model and preprocessor"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}. Please train the model first.")
        if not os.path.exists(self.preprocessor_path):
            raise FileNotFoundError(f"Preprocessor not found at {self.preprocessor_path}. Please train the model first.")
        
        self.model = joblib.load(self.model_path)
        self.preprocessor = DataPreprocessor.load(self.preprocessor_path)
        print(f"Model and preprocessor loaded successfully from {os.path.dirname(self.model_path)}")
    
    def predict(self, data):
        """
        Make predictions on new data
        
        Args:
            data: DataFrame or dict with features, or list of dicts
            
        Returns:
            Dictionary with predictions and probabilities
        """
        # Convert input to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        elif isinstance(data, list):
            data = pd.DataFrame(data)
        elif not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a DataFrame, dict, or list of dicts")
        
        # Preprocess data
        X = self.preprocessor.transform(data)
        
        # Make predictions
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        # Format results
        results = []
        for i in range(len(predictions)):
            result = {
                'prediction': int(predictions[i]),
                'prediction_label': 'Fraud' if predictions[i] == 1 else 'Legitimate',
                'fraud_probability': float(probabilities[i][1]),
                'legitimate_probability': float(probabilities[i][0])
            }
            results.append(result)
        
        # Return single result if single input, else list
        if len(results) == 1:
            return results[0]
        return results
    
    def predict_batch(self, data_path, output_path=None):
        """
        Make predictions on a batch of data from CSV file
        
        Args:
            data_path: Path to CSV file with features
            output_path: Optional path to save predictions
            
        Returns:
            DataFrame with original data and predictions
        """
        # Load data
        df = pd.read_csv(data_path)
        
        # Make predictions
        results = self.predict(df)
        
        # Convert results to DataFrame if list
        if isinstance(results, list):
            results_df = pd.DataFrame(results)
        else:
            results_df = pd.DataFrame([results])
        
        # Combine with original data
        output_df = pd.concat([df, results_df], axis=1)
        
        # Save if output path provided
        if output_path:
            output_df.to_csv(output_path, index=False)
            print(f"Predictions saved to {output_path}")
        
        return output_df


def load_inference_pipeline(model_path=None, preprocessor_path=None):
    """
    Convenience function to load inference pipeline
    
    Args:
        model_path: Optional path to model
        preprocessor_path: Optional path to preprocessor
        
    Returns:
        FraudDetectionInference instance
    """
    return FraudDetectionInference(model_path, preprocessor_path)


if __name__ == "__main__":
    # Example usage
    import argparse
    parser = argparse.ArgumentParser(description="Fraud Detection Inference")
    parser.add_argument("--input", type=str, help="Path to input CSV file")
    parser.add_argument("--output", type=str, help="Path to save predictions")
    parser.add_argument("--sample", action="store_true", help="Run on sample data")
    args = parser.parse_args()
    
    # Load inference pipeline
    inference = load_inference_pipeline()
    
    if args.input:
        # Batch prediction
        inference.predict_batch(args.input, args.output)
    elif args.sample:
        # Sample prediction
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
        print("\nSample Prediction Result:")
        print(f"Prediction: {result['prediction_label']}")
        print(f"Fraud Probability: {result['fraud_probability']:.4f}")
        print(f"Legitimate Probability: {result['legitimate_probability']:.4f}")




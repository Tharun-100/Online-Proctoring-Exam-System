"""
Example usage of the Unified Fraud Detection API
Demonstrates both image upload and feature-based prediction
"""
import requests
import json

# Base URL
BASE_URL = "http://localhost:5000"

def example_image_upload():
    """Example: Upload image and get prediction"""
    print("=" * 60)
    print("Example 1: Image Upload")
    print("=" * 60)
    
    # Upload image file
    image_path = "path/to/your/exam_image.jpg"  # Replace with actual image path
    
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(f"{BASE_URL}/predict/image", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Prediction: {result['prediction']['prediction_label']}")
            print(f"  Fraud Probability: {result['prediction']['fraud_probability']:.2%}")
            print(f"\nClassification Report:")
            print(result['classification_report'])
            print(f"\nExtracted Features Count: {len(result['features'])}")
        else:
            print(f"✗ Error: {response.json()}")
    except FileNotFoundError:
        print(f"✗ Image file not found: {image_path}")
    except Exception as e:
        print(f"✗ Error: {e}")


def example_feature_prediction():
    """Example: Predict from extracted features"""
    print("\n" + "=" * 60)
    print("Example 2: Feature-Based Prediction")
    print("=" * 60)
    
    # Sample feature data (extracted from an image)
    features = {
        "face_present": 1,
        "no_of_face": 1,
        "face_x": 262.5,
        "face_y": 312.0,
        "face_w": 151.7,
        "face_h": 151.7,
        "left_eye_x": 307.9,
        "left_eye_y": 354.4,
        "right_eye_x": 373.0,
        "right_eye_y": 353.4,
        "nose_tip_x": 344.5,
        "nose_tip_y": 387.3,
        "mouth_x": 345.0,
        "mouth_y": 419.5,
        "face_conf": 87.7,
        "hand_count": 0,
        "left_hand_x": 0,
        "left_hand_y": 0,
        "right_hand_x": 0,
        "right_hand_y": 0,
        "hand_obj_interaction": 0,
        "head_pose": "forward",
        "head_pitch": 0.018,
        "head_yaw": 0.015,
        "head_roll": -0.0005,
        "phone_present": 0,
        "phone_loc_x": 0,
        "phone_loc_y": 0,
        "phone_conf": 0,
        "gaze_on_script": 1,
        "gaze_direction": "center",
        "gazePoint_x": 336,
        "gazePoint_y": 325,
        "pupil_left_x": 304,
        "pupil_left_y": 348,
        "pupil_right_x": 368,
        "pupil_right_y": 354
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict/features",
            json=features,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Prediction: {result['prediction']['prediction_label']}")
            print(f"  Fraud Probability: {result['prediction']['fraud_probability']:.2%}")
            print(f"\nClassification Report:")
            print(result['classification_report'])
        else:
            print(f"✗ Error: {response.json()}")
    except Exception as e:
        print(f"✗ Error: {e}")


def example_batch_prediction():
    """Example: Batch prediction from multiple feature sets"""
    print("\n" + "=" * 60)
    print("Example 3: Batch Prediction")
    print("=" * 60)
    
    # Multiple feature sets
    feature_sets = [
        {
            "face_present": 1,
            "no_of_face": 1,
            "face_x": 262.5,
            "face_y": 312.0,
            "face_w": 151.7,
            "face_h": 151.7,
            "left_eye_x": 307.9,
            "left_eye_y": 354.4,
            "right_eye_x": 373.0,
            "right_eye_y": 353.4,
            "nose_tip_x": 344.5,
            "nose_tip_y": 387.3,
            "mouth_x": 345.0,
            "mouth_y": 419.5,
            "face_conf": 87.7,
            "hand_count": 0,
            "left_hand_x": 0,
            "left_hand_y": 0,
            "right_hand_x": 0,
            "right_hand_y": 0,
            "hand_obj_interaction": 0,
            "head_pose": "forward",
            "head_pitch": 0.018,
            "head_yaw": 0.015,
            "head_roll": -0.0005,
            "phone_present": 0,
            "phone_loc_x": 0,
            "phone_loc_y": 0,
            "phone_conf": 0,
            "gaze_on_script": 1,
            "gaze_direction": "center",
            "gazePoint_x": 336,
            "gazePoint_y": 325,
            "pupil_left_x": 304,
            "pupil_left_y": 348,
            "pupil_right_x": 368,
            "pupil_right_y": 354
        },
        {
            "face_present": 1,
            "no_of_face": 1,
            "face_x": 280.0,
            "face_y": 300.0,
            "face_w": 140.0,
            "face_h": 140.0,
            "left_eye_x": 310.0,
            "left_eye_y": 340.0,
            "right_eye_x": 380.0,
            "right_eye_y": 340.0,
            "nose_tip_x": 350.0,
            "nose_tip_y": 380.0,
            "mouth_x": 345.0,
            "mouth_y": 410.0,
            "face_conf": 90.0,
            "hand_count": 1,
            "left_hand_x": 200.0,
            "left_hand_y": 400.0,
            "right_hand_x": 0,
            "right_hand_y": 0,
            "hand_obj_interaction": 1,
            "head_pose": "right",
            "head_pitch": 0.02,
            "head_yaw": 0.03,
            "head_roll": -0.001,
            "phone_present": 1,
            "phone_loc_x": 250.0,
            "phone_loc_y": 450.0,
            "phone_conf": 0.7,
            "gaze_on_script": 0,
            "gaze_direction": "right",
            "gazePoint_x": 400,
            "gazePoint_y": 350,
            "pupil_left_x": 315,
            "pupil_left_y": 345,
            "pupil_right_x": 385,
            "pupil_right_y": 345
        }
    ]
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict/batch",
            json={"data": feature_sets},
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Processed {result['count']} predictions")
            for idx, res in enumerate(result['results']):
                if 'error' not in res:
                    print(f"\n  Prediction {idx + 1}:")
                    print(f"    Label: {res['prediction']['prediction_label']}")
                    print(f"    Fraud Probability: {res['prediction']['fraud_probability']:.2%}")
                else:
                    print(f"\n  Prediction {idx + 1}: Error - {res['error']}")
        else:
            print(f"✗ Error: {response.json()}")
    except Exception as e:
        print(f"✗ Error: {e}")


def check_health():
    """Check API health and get endpoint information"""
    print("=" * 60)
    print("Health Check")
    print("=" * 60)
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            result = response.json()
            print(f"Status: {result['status']}")
            print(f"Models Loaded: {result['models_loaded']}")
            print(f"\nAvailable Endpoints:")
            for endpoint, url in result['endpoints'].items():
                print(f"  {endpoint}: {url}")
        else:
            print(f"✗ Error: {response.status_code}")
    except Exception as e:
        print(f"✗ Cannot connect to API: {e}")
        print(f"  Make sure the server is running at {BASE_URL}")


if __name__ == "__main__":
    # Check health first
    check_health()
    
    # Run examples
    print("\n")
    example_feature_prediction()
    example_batch_prediction()
    
    # Uncomment to test image upload (requires an actual image file)
    # example_image_upload()


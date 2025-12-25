# Fraud Detection System for Online Proctored Exams

A machine learning system for detecting fraudulent behavior in online proctored exams using XGBoost classifier.

## Project Structure

```
.
├── Dataset/
│   └── Students suspicious behaviors detection dataset_V1.csv
├── api/
│   ├── app.py                 # FastAPI deployment application
│   ├── flask_app.py           # Flask web application with image upload
│   └── templates/
│       └── index.html         # Web interface for image upload
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py  # Data preprocessing and feature engineering
│   ├── feature_extraction.py  # Feature extraction from images
│   ├── train.py              # Model training script
│   └── inference.py          # Inference pipeline
├── models/                    # Saved models and preprocessors (created after training)
├── config.py                  # Configuration file
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Features

- **XGBoost Classifier**: Trained on behavioral features from exam proctoring
- **Data Preprocessing**: Handles missing values, categorical encoding, and feature scaling
- **Feature Extraction**: Extracts 37 parameters from images using computer vision (face, eyes, hands, head pose, gaze, phone detection)
- **Inference Pipeline**: Easy-to-use API for making predictions
- **Flask Web API**: User-friendly web interface for image upload and fraud detection
- **FastAPI REST API**: RESTful API for programmatic access
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, and AUC-ROC

## Dataset

The system uses the "Students suspicious behaviors detection dataset_V1.csv" which contains features such as:
- Face detection metrics (presence, position, confidence)
- Eye tracking (pupil positions, gaze direction)
- Head pose (pitch, yaw, roll)
- Hand detection and interaction
- Phone detection
- Gaze analysis

Target variable: `label` (0 = Legitimate, 1 = Fraud)

## Installation

1. Clone or navigate to the project directory:
```bash
cd "E:\Projects in ML\FRAUD DETECTION SYSTEM FOR THE ONLINE PROCTORED EXAMS"
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Train the Model

Train the XGBoost model on the dataset:

```bash
python src/train.py
```

This will:
- Load and preprocess the data
- Split into train/test sets
- Train the XGBoost model
- Evaluate performance metrics
- Save the model and preprocessor to `models/` directory

### 2. Make Predictions (Inference)

#### Single Prediction (Python)

```python
from src.inference import FraudDetectionInference

# Load inference pipeline
inference = FraudDetectionInference()

# Prepare sample data
sample_data = {
    'face_present': 1,
    'no_of_face': 1,
    'face_x': 262.5,
    'face_y': 312.0,
    # ... (all other features)
}

# Make prediction
result = inference.predict(sample_data)
print(f"Prediction: {result['prediction_label']}")
print(f"Fraud Probability: {result['fraud_probability']:.4f}")
```

#### Batch Prediction from CSV

```bash
python src/inference.py --input path/to/data.csv --output path/to/predictions.csv
```

#### Sample Prediction

```bash
python src/inference.py --sample
```

### 3. Deploy API Server

#### Unified Flask Application (Recommended)

Start the unified Flask application that combines both image upload and feature-based prediction:

```bash
python api/unified_app.py
```

The web interface will be available at `http://localhost:5000`

**Available Endpoints:**
- `GET /` - Web interface for image upload
- `GET /health` - Health check and endpoint information
- `GET /api/docs` - API documentation
- `POST /predict/image` - Upload image, extract features, and get prediction
- `POST /predict/features` - Get prediction from extracted features (JSON)
- `POST /predict/batch` - Batch prediction from multiple feature sets
- `POST /predict` - Legacy endpoint (auto-detects image or features)

**Features:**
- Upload images via web interface
- View annotated images with detected features
- See classification results and reports
- Display extracted features
- Accept extracted features directly as JSON
- Batch prediction support

#### FastAPI REST API

Start the FastAPI server for programmatic access:

```bash
python api/app.py
```

Or using uvicorn directly:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

#### API Endpoints

1. **Root**: `GET /` - API information
2. **Health Check**: `GET /health` - Check if model is loaded
3. **Single Prediction**: `POST /predict` - Make a single prediction
4. **Batch Prediction**: `POST /batch_predict` - Make batch predictions
5. **API Documentation**: `GET /docs` - Interactive Swagger UI

#### Example API Requests

**1. Image Upload (Extract features and predict):**

```python
import requests

# Upload image file
with open('exam_image.jpg', 'rb') as f:
    files = {'image': f}
    response = requests.post("http://localhost:5000/predict/image", files=files)

result = response.json()
print(f"Prediction: {result['prediction']['prediction_label']}")
print(f"Fraud Probability: {result['prediction']['fraud_probability']}")
print(result['classification_report'])
# Access annotated image as base64: result['image']
```

**2. Feature-Based Prediction (Use extracted features):**

```python
import requests

# Provide extracted features directly
features = {
    "face_present": 1,
    "no_of_face": 1,
    "face_x": 262.5,
    # ... (all 37 features)
}

response = requests.post(
    "http://localhost:5000/predict/features",
    json=features
)

result = response.json()
print(f"Prediction: {result['prediction']['prediction_label']}")
print(result['classification_report'])
```

**3. Batch Prediction:**

```python
import requests

# Multiple feature sets
feature_sets = [features1, features2, features3]

response = requests.post(
    "http://localhost:5000/predict/batch",
    json={"data": feature_sets}
)

result = response.json()
for res in result['results']:
    print(f"Prediction: {res['prediction']['prediction_label']}")
```

See `api_examples.py` for complete examples.

**FastAPI (JSON Features):**

```python
import requests

# Single prediction with features
response = requests.post("http://localhost:8000/predict", json={
    "face_present": 1,
    "no_of_face": 1,
    "face_x": 262.5,
    # ... (all required features)
})

result = response.json()
print(f"Prediction: {result['prediction_label']}")
print(f"Fraud Probability: {result['fraud_probability']}")
```

#### Using curl

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "face_present": 1,
    "no_of_face": 1,
    ...
  }'
```

## Model Performance

After training, the model performance metrics are saved to `models/model_metrics.csv`. The training script displays:
- Training Accuracy
- Test Accuracy
- Precision, Recall, F1-Score
- AUC-ROC Score
- Classification Report
- Confusion Matrix

## Configuration

Edit `config.py` to customize:
- Dataset path
- Model parameters (XGBoost hyperparameters)
- Train/test split ratio
- Model save paths

## Dependencies

- pandas: Data manipulation
- numpy: Numerical computations
- scikit-learn: Preprocessing and metrics
- xgboost: Gradient boosting classifier
- joblib: Model serialization
- flask: Web framework for image upload interface
- opencv-python: Computer vision and image processing
- mediapipe: Face, hand, and pose detection
- pillow: Image processing
- fastapi: REST API framework
- uvicorn: ASGI server
- pydantic: Data validation

## Notes

- The model must be trained before running inference or the API
- Ensure all required features are provided in the correct format
- Categorical features (`head_pose`, `gaze_direction`) should match training data values
- Missing values are handled automatically by the preprocessor

## License

This project is for educational and research purposes.


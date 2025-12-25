# Quick Start Guide - Flask API with Image Upload

## Prerequisites

1. Train the model first:
```bash
python src/train.py
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Unified Flask Application

1. Start the unified Flask server:
```bash
python api/unified_app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. Upload an exam image using the web interface

4. View the results:
   - Annotated image with detected features (face, eyes, hands, gaze)
   - Classification result (Fraud/Legitimate)
   - Classification report
   - Extracted features

## Features Extracted from Images

The system automatically extracts 37 features from uploaded images:

- **Face Detection**: Presence, position, size, confidence
- **Eye Tracking**: Left/right eye positions, pupil coordinates
- **Head Pose**: Pitch, yaw, roll angles and direction
- **Hand Detection**: Hand positions and interactions
- **Gaze Analysis**: Gaze point, direction, script detection
- **Phone Detection**: Phone presence and location

## API Endpoints

- `GET /` - Web interface for image upload
- `GET /health` - Health check and endpoint information
- `GET /api/docs` - API documentation
- `POST /predict/image` - Upload image, extract features, and get prediction
- `POST /predict/features` - Get prediction from extracted features (JSON)
- `POST /predict/batch` - Batch prediction from multiple feature sets
- `POST /predict` - Legacy endpoint (auto-detects image or features)

## Examples: Using the API Programmatically

**Image Upload:**
```python
import requests

# Upload image
with open('exam_image.jpg', 'rb') as f:
    files = {'image': f}
    response = requests.post('http://localhost:5000/predict/image', files=files)

result = response.json()
print(f"Prediction: {result['prediction']['prediction_label']}")
print(result['classification_report'])
```

**Feature-Based Prediction:**
```python
import requests

# Use extracted features directly
features = {"face_present": 1, "no_of_face": 1, ...}  # All 37 features

response = requests.post(
    'http://localhost:5000/predict/features',
    json=features
)

result = response.json()
print(f"Prediction: {result['prediction']['prediction_label']}")
print(result['classification_report'])
```

See `api_examples.py` for more examples.

## Troubleshooting

- **Models not loaded**: Make sure you've trained the model first using `python src/train.py`
- **Image upload fails**: Check file size (max 16MB) and format (PNG, JPG, JPEG, GIF, BMP)
- **MediaPipe errors**: Ensure you have installed all dependencies from requirements.txt




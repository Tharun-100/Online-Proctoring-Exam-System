# Unified API Integration Guide

## Overview

The unified Flask application (`api/unified_app.py`) integrates both image upload functionality and feature-based prediction into a single API service.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│           Unified Flask Application                      │
│                  (unified_app.py)                        │
└─────────────────────────────────────────────────────────┘
           │                    │                    │
           ▼                    ▼                    ▼
    ┌──────────┐        ┌──────────┐        ┌──────────┐
    │  Image   │        │ Features │        │  Batch   │
    │  Upload  │        │ Prediction│       │Prediction │
    └──────────┘        └──────────┘        └──────────┘
           │                    │                    │
           ▼                    │                    │
    Feature Extraction          │                    │
    (MediaPipe)                 │                    │
           │                    │                    │
           └────────────────────┴────────────────────┘
                              │
                              ▼
                    XGBoost Model Inference
                              │
                              ▼
                    Classification Report
```

## Endpoints

### 1. Web Interface
- **GET /** - Interactive web interface for image upload
- Returns HTML page with drag-and-drop image upload

### 2. Image Upload & Prediction
- **POST /predict/image**
  - Accepts: `multipart/form-data` with `image` file
  - Returns: JSON with:
    - `prediction`: Prediction result (label, probabilities)
    - `classification_report`: Text report
    - `image`: Base64-encoded annotated image
    - `features`: Extracted 37 features
    - `success`: Boolean status

### 3. Feature-Based Prediction
- **POST /predict/features**
  - Accepts: JSON with 37 feature values
  - Returns: JSON with:
    - `prediction`: Prediction result
    - `classification_report`: Text report
    - `features`: Input features (echoed back)
    - `success`: Boolean status

### 4. Batch Prediction
- **POST /predict/batch**
  - Accepts: JSON with `{"data": [features1, features2, ...]}`
  - Returns: JSON with:
    - `results`: Array of predictions and reports
    - `count`: Number of predictions
    - `success`: Boolean status

### 5. Health Check
- **GET /health**
  - Returns: API status and available endpoints

### 6. API Documentation
- **GET /api/docs**
  - Returns: Complete API documentation in JSON format

### 7. Legacy Endpoint
- **POST /predict**
  - Auto-detects if request contains image or features
  - Routes to appropriate handler

## Workflow Examples

### Workflow 1: Image Upload → Extract Features → Predict

```
User uploads image
    ↓
POST /predict/image
    ↓
Feature Extraction (MediaPipe)
    ↓
Feature Preprocessing
    ↓
XGBoost Prediction
    ↓
Generate Classification Report
    ↓
Draw Annotations on Image
    ↓
Return JSON with results
```

### Workflow 2: Direct Feature Prediction

```
User provides extracted features
    ↓
POST /predict/features (JSON)
    ↓
Feature Preprocessing
    ↓
XGBoost Prediction
    ↓
Generate Classification Report
    ↓
Return JSON with results
```

## Usage Scenarios

### Scenario 1: Real-time Exam Monitoring
- Use `/predict/image` endpoint
- Upload frames from webcam/exam video
- Get real-time fraud detection results

### Scenario 2: Batch Processing
- Use `/predict/batch` endpoint
- Process multiple feature sets at once
- Efficient for analyzing multiple exam sessions

### Scenario 3: External Feature Extraction
- Extract features using custom methods
- Use `/predict/features` endpoint
- Integrate with existing systems

### Scenario 4: Web Application
- Access `GET /` for web interface
- Upload images through browser
- View results with visualizations

## Response Format

All endpoints return consistent JSON structure:

```json
{
  "success": true,
  "prediction": {
    "prediction": 0,
    "prediction_label": "Legitimate",
    "fraud_probability": 0.1234,
    "legitimate_probability": 0.8766
  },
  "classification_report": "=== CLASSIFICATION REPORT ===\n...",
  "features": { ... },
  "endpoint": "predict/image"
}
```

## Error Handling

All endpoints return appropriate HTTP status codes:
- `200`: Success
- `400`: Bad request (missing/invalid input)
- `500`: Server error (prediction failed)
- `503`: Service unavailable (models not loaded)

Error response format:
```json
{
  "error": "Error message description"
}
```

## Integration Benefits

1. **Unified Interface**: Single API for all prediction needs
2. **Flexibility**: Support both image upload and feature input
3. **Consistency**: Same classification report format across endpoints
4. **Scalability**: Batch processing support
5. **Developer-Friendly**: Clear endpoint structure and documentation
6. **User-Friendly**: Web interface for non-technical users

## Migration from Separate APIs

### From flask_app.py:
- `/predict` → `/predict/image`
- Functionality remains the same

### From app.py (FastAPI):
- `/predict` → `/predict/features`
- Request format: Same JSON structure
- Response format: Same structure with added `classification_report`

## Testing

Use the provided `api_examples.py` script to test all endpoints:

```bash
python api_examples.py
```

Or test manually:
```bash
# Health check
curl http://localhost:5000/health

# Feature prediction
curl -X POST http://localhost:5000/predict/features \
  -H "Content-Type: application/json" \
  -d @features.json
```


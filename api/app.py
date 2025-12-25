"""
FastAPI Deployment API for Fraud Detection System
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import FraudDetectionInference

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="API for detecting fraudulent behavior in online proctored exams",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load inference pipeline
inference = None

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global inference
    try:
        inference = FraudDetectionInference()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        inference = None


# Pydantic models for request/response
class FeatureInput(BaseModel):
    """Input features for fraud detection"""
    face_present: int = Field(..., description="Whether face is present (0 or 1)")
    no_of_face: int = Field(..., description="Number of faces detected")
    face_x: float = Field(..., description="Face x coordinate")
    face_y: float = Field(..., description="Face y coordinate")
    face_w: float = Field(..., description="Face width")
    face_h: float = Field(..., description="Face height")
    left_eye_x: float = Field(..., description="Left eye x coordinate")
    left_eye_y: float = Field(..., description="Left eye y coordinate")
    right_eye_x: float = Field(..., description="Right eye x coordinate")
    right_eye_y: float = Field(..., description="Right eye y coordinate")
    nose_tip_x: float = Field(..., description="Nose tip x coordinate")
    nose_tip_y: float = Field(..., description="Nose tip y coordinate")
    mouth_x: float = Field(..., description="Mouth x coordinate")
    mouth_y: float = Field(..., description="Mouth y coordinate")
    face_conf: float = Field(..., description="Face confidence score")
    hand_count: int = Field(..., description="Number of hands detected")
    left_hand_x: float = Field(..., description="Left hand x coordinate")
    left_hand_y: float = Field(..., description="Left hand y coordinate")
    right_hand_x: float = Field(..., description="Right hand x coordinate")
    right_hand_y: float = Field(..., description="Right hand y coordinate")
    hand_obj_interaction: int = Field(..., description="Hand object interaction (0 or 1)")
    head_pose: str = Field(..., description="Head pose direction (forward, left, right, down, up, None)")
    head_pitch: float = Field(..., description="Head pitch angle")
    head_yaw: float = Field(..., description="Head yaw angle")
    head_roll: float = Field(..., description="Head roll angle")
    phone_present: int = Field(..., description="Whether phone is present (0 or 1)")
    phone_loc_x: float = Field(..., description="Phone location x coordinate")
    phone_loc_y: float = Field(..., description="Phone location y coordinate")
    phone_conf: float = Field(..., description="Phone detection confidence")
    gaze_on_script: int = Field(..., description="Whether gaze is on script (0 or 1)")
    gaze_direction: str = Field(..., description="Gaze direction (center, top, bottom, left, right, etc.)")
    gazePoint_x: float = Field(..., description="Gaze point x coordinate")
    gazePoint_y: float = Field(..., description="Gaze point y coordinate")
    pupil_left_x: float = Field(..., description="Left pupil x coordinate")
    pupil_left_y: float = Field(..., description="Left pupil y coordinate")
    pupil_right_x: float = Field(..., description="Right pupil x coordinate")
    pupil_right_y: float = Field(..., description="Right pupil y coordinate")
    
    class Config:
        schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    """Prediction response"""
    prediction: int = Field(..., description="Predicted class (0=Legitimate, 1=Fraud)")
    prediction_label: str = Field(..., description="Predicted class label")
    fraud_probability: float = Field(..., description="Probability of fraud")
    legitimate_probability: float = Field(..., description="Probability of legitimate behavior")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    data: List[FeatureInput] = Field(..., description="List of feature inputs")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Fraud Detection API for Online Proctored Exams",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/batch_predict",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if inference is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "healthy",
        "model_loaded": True
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(input_data: FeatureInput):
    """
    Make a single prediction
    
    Args:
        input_data: Feature input data
        
    Returns:
        Prediction result
    """
    if inference is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
    
    try:
        # Convert input to dict
        input_dict = input_data.dict()
        
        # Make prediction
        result = inference.predict(input_dict)
        
        return PredictionResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def predict_batch(batch_request: BatchPredictionRequest):
    """
    Make batch predictions
    
    Args:
        batch_request: Batch prediction request with list of inputs
        
    Returns:
        Batch prediction results
    """
    if inference is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
    
    try:
        # Convert inputs to list of dicts
        input_list = [item.dict() for item in batch_request.data]
        
        # Make predictions
        results = inference.predict(input_list)
        
        # Ensure results is a list
        if not isinstance(results, list):
            results = [results]
        
        # Convert to response format
        predictions = [PredictionResponse(**result) for result in results]
        
        return BatchPredictionResponse(predictions=predictions)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)




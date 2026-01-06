"""
Training script for XGBoost Fraud Detection Model
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
import xgboost as xgb
import joblib
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DATASET_PATH, MODEL_PATH, PREPROCESSOR_PATH, MODEL_DIR,
    TEST_SIZE, RANDOM_STATE, XGBOOST_PARAMS, TARGET_COLUMN
)
from src.data_preprocessing import load_and_preprocess_data, DataPreprocessor


def train_model():
    """Train XGBoost model for fraud detection"""

    print("=" * 60)
    print("Fraud Detection System - Model Training")
    print("=" * 60)
    
    # Load and preprocess data
    print("\n[1/5] Loading and preprocessing data...")
    X, y, preprocessor = load_and_preprocess_data(
        DATASET_PATH,
        target_column=TARGET_COLUMN,
        fit_preprocessor=True
    )
    
    print(f"   Dataset shape: {X.shape}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Samples: {X.shape[0]}")
    print(f"   Class distribution:\n{y.value_counts().to_dict()}")
    
    # Split data
    print("\n[2/5] Splitting data into train/test sets...")
    X_train, X_t, y_train, y_t = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_test,X_valid, y_test, y_valid = train_test_split(
        X_t, y_t, test_size=0.5, random_state=RANDOM_STATE, stratify=y_t
    )
    
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")
    
    # Train XGBoost model
    print("\n[3/5] Training XGBoost model...")
    print(f"   Parameters: {XGBOOST_PARAMS}")
    
    model = xgb.XGBClassifier(**XGBOOST_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False
    )
    
    # Make predictions
    print("\n[4/5] Evaluating model...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]

    y_valid_pred = model.predict(X_valid)
    y_valid_proba = model.predict_proba(X_valid)[:, 1]

    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    valid_accuracy = accuracy_score(y_valid, y_valid_pred)
    valid_auc = roc_auc_score(y_valid, y_valid_proba)
    valid_precision = precision_score(y_valid, y_valid_pred, zero_division=0)
    valid_recall = recall_score(y_valid, y_valid_pred, zero_division=0)
    valid_f1 = f1_score(y_valid, y_valid_pred, zero_division=0)

    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
    test_auc = roc_auc_score(y_test, y_test_proba)
    
    print("\n" + "=" * 60)
    print("TRAINING RESULTS")
    print("=" * 60)
    print(f"Training Accuracy:  {train_accuracy:.4f}")
    print(f"Test Accuracy:      {test_accuracy:.4f}")
    print(f"Test Precision:     {test_precision:.4f}")
    print(f"Test Recall:        {test_recall:.4f}")
    print(f"Test F1-Score:      {test_f1:.4f}")
    print(f"Test AUC-ROC:       {test_auc:.4f}")
    
    print("\nValidation Set Metrics:")
    
    print(f"Validation Accuracy:  {valid_accuracy:.4f}")
    print(f"Validation Precision: {valid_precision:.4f}")
    print(f"Validation Recall:    {valid_recall:.4f}")
    print(f"Validation F1-Score:  {valid_f1:.4f}")
    print(f"Validation AUC-ROC:   {valid_auc:.4f}")

    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_test_pred, target_names=['Legitimate', 'Fraud']))
    print("\nConfusion Matrix (Test Set):")
    print(confusion_matrix(y_test, y_test_pred))

    # Save model and preprocessor
    print("\n[5/5] Saving model and preprocessor...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    joblib.dump(model, MODEL_PATH)
    preprocessor.save(PREPROCESSOR_PATH)
    
    print(f"   Model saved to: {MODEL_PATH}")
    print(f"   Preprocessor saved to: {PREPROCESSOR_PATH}")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
        'Train': [train_accuracy, None, None, None, None],
        'Test': [test_accuracy, test_precision, test_recall, test_f1, test_auc],
        'Validation': [valid_accuracy, valid_precision, valid_recall, valid_f1, valid_auc]
    })

    metrics_path = os.path.join(MODEL_DIR, 'model_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"   Metrics saved to: {metrics_path}")
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
    
    return model, preprocessor, {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'test_auc': test_auc
    }


if __name__ == "__main__":
    train_model()


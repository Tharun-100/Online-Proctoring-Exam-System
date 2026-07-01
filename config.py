"""
Configuration file for the Fraud Detection System
"""

import os

# Dataset path
DATASET_PATH = "Dataset/Students suspicious behaviors detection dataset_V1.csv"

# DATASET_PATH = "merged_features.csv"

# Model paths
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_fraud_detection_model.pkl")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.pkl")

# Training parameters
TEST_SIZE = 0.3
RANDOM_STATE = 42
VALIDATION_SIZE = 0.2  # From training set

# XGBoost parameters

XGBOOST_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 2,
    'learning_rate': 0.1,
    'n_estimators': 2,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# Feature columns (all except label)
FEATURE_COLUMNS = None  # Will be set dynamically
# Categorical feature handling

CATEGORICAL_COLUMNS = ['head_pose', 'gaze_direction']
AUTO_DETECT_CATEGORICALS = True

# Target column
TARGET_COLUMN = "label"


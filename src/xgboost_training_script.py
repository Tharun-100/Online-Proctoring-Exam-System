"""
Training script for XGBoost Fraud Detection Model
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    balanced_accuracy_score, matthews_corrcoef, average_precision_score
)

import xgboost as xgb
import joblib
import os
import sys
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DATASET_PATH, MODEL_PATH, PREPROCESSOR_PATH, MODEL_DIR,
    TEST_SIZE, RANDOM_STATE, XGBOOST_PARAMS, TARGET_COLUMN,
    CATEGORICAL_COLUMNS, AUTO_DETECT_CATEGORICALS
)

from src.data_preprocessing import load_and_preprocess_data, DataPreprocessor
def train_model():
    """Train XGBoost model for fraud detection"""

    print("=" * 60)
    print("Fraud Detection System - Model Training")
    print("=" * 60)

    # Load raw data first to avoid leakage during preprocessing
    print("\n[1/5] Loading data...")
    df = pd.read_csv(DATASET_PATH)

    print(f"   Dataset shape: {df.shape}")
    print(f"   Features: {df.shape[1] - 1}")
    print(f"   Samples: {df.shape[0]}")
    print(f"   Class distribution:\n{df[TARGET_COLUMN].value_counts().to_dict()}")

    # Split raw data before preprocessing
    print("\n[2/5] Splitting data into train/valid/test sets...")
    train_df, temp_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df[TARGET_COLUMN]
    )
    test_df, valid_df = train_test_split(
        temp_df, test_size=0.5, random_state=RANDOM_STATE, stratify=temp_df[TARGET_COLUMN]
    )

    # Preprocess data (fit on train only)
    preprocessor = DataPreprocessor(
        categorical_columns=CATEGORICAL_COLUMNS,
        auto_detect_categoricals=AUTO_DETECT_CATEGORICALS
    )
    X_train, y_train = preprocessor.fit_transform(train_df, target_column=TARGET_COLUMN)
    X_test, y_test = preprocessor.transform(test_df, target_column=TARGET_COLUMN)
    X_valid, y_valid = preprocessor.transform(valid_df, target_column=TARGET_COLUMN)

        # Shuffle labels test

    # y_train = np.random.permutation(y_train)

    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")

    # Train XGBoost model
    print("\n[3/5] Training XGBoost model...")
    print(f"   Parameters: {XGBOOST_PARAMS}")
    
    # Some older XGBoost versions don't accept eval_metric in fit(),
    # so set it on the estimator directly.
    model = xgb.XGBClassifier(**XGBOOST_PARAMS)
    model.set_params(eval_metric=["logloss", "auc", "error", "aucpr"])
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid), (X_test, y_test)],
        verbose=False
    )
    evals_result = model.evals_result()

    # Make predictions
    print("\n[4/5] Evaluating model...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]

    y_valid_pred = model.predict(X_valid)
    y_valid_proba = model.predict_proba(X_valid)[:, 1]

    # Calculate metrics
    def compute_specificity(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0

    def compute_metrics(y_true, y_pred, y_proba):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "specificity": compute_specificity(y_true, y_pred),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_proba),
            "pr_auc": average_precision_score(y_true, y_proba),
            "mcc": matthews_corrcoef(y_true, y_pred)
        }

    train_metrics = compute_metrics(y_train, y_train_pred, model.predict_proba(X_train)[:, 1])
    valid_metrics = compute_metrics(y_valid, y_valid_pred, y_valid_proba)
    test_metrics = compute_metrics(y_test, y_test_pred, y_test_proba)
    
    print("\n" + "=" * 60)
    print("TRAINING RESULTS")
    print("=" * 60)
    print(f"Training Accuracy:  {train_metrics['accuracy']:.4f}")
    print(f"Training AUC-ROC:   {train_metrics['roc_auc']:.4f}")
    print(f"Training PR-AUC:    {train_metrics['pr_auc']:.4f}")
    print(f"Training MCC:       {train_metrics['mcc']:.4f}")

    print(f"Test Accuracy:      {test_metrics['accuracy']:.4f}")
    print(f"Test Precision:     {test_metrics['precision']:.4f}")
    print(f"Test Recall:        {test_metrics['recall']:.4f}")
    print(f"Test Specificity:   {test_metrics['specificity']:.4f}")
    print(f"Test F1-Score:      {test_metrics['f1']:.4f}")
    print(f"Test AUC-ROC:       {test_metrics['roc_auc']:.4f}")
    print(f"Test PR-AUC:        {test_metrics['pr_auc']:.4f}")
    print(f"Test MCC:           {test_metrics['mcc']:.4f}")
    
    print("\nValidation Set Metrics:")

    print(f"Validation Accuracy:  {valid_metrics['accuracy']:.4f}")
    print(f"Validation Precision: {valid_metrics['precision']:.4f}")
    print(f"Validation Recall:    {valid_metrics['recall']:.4f}")
    print(f"Validation Specificity:{valid_metrics['specificity']:.4f}")
    print(f"Validation F1-Score:  {valid_metrics['f1']:.4f}")
    print(f"Validation AUC-ROC:   {valid_metrics['roc_auc']:.4f}")
    print(f"Validation PR-AUC:    {valid_metrics['pr_auc']:.4f}")
    print(f"Validation MCC:       {valid_metrics['mcc']:.4f}")
    
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_test_pred, target_names=['Legitimate', 'Fraud']))
    print("\nConfusion Matrix (Test Set):")
    print(confusion_matrix(y_test, y_test_pred))

    # Plot training curves
    print("\n[5/6] Plotting training curves...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    epochs = range(1, len(evals_result["validation_0"]["logloss"]) + 1)

    def to_accuracy(error_list):
        return [1.0 - e for e in error_list]

    # Accuracy vs epochs
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(epochs, to_accuracy(evals_result["validation_0"]["error"]), label="Train Accuracy")
    plt.plot(epochs, to_accuracy(evals_result["validation_1"]["error"]), label="Valid Accuracy")
    plt.plot(epochs, to_accuracy(evals_result["validation_2"]["error"]), label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epochs")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(epochs, evals_result["validation_0"]["auc"], label="Train AUC")
    plt.plot(epochs, evals_result["validation_1"]["auc"], label="Valid AUC")
    plt.plot(epochs, evals_result["validation_2"]["auc"], label="Test AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title("AUC vs Epochs")
    plt.legend()

    plot_path = os.path.join(MODEL_DIR, "pure_data_training_curves.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"   Accuracy/AUC curves saved to: {plot_path}")

    # Logloss and PR-AUC vs epochs
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(epochs, evals_result["validation_0"]["logloss"], label="Train Logloss")
    plt.plot(epochs, evals_result["validation_1"]["logloss"], label="Valid Logloss")
    plt.plot(epochs, evals_result["validation_2"]["logloss"], label="Test Logloss")
    plt.xlabel("Epoch")
    plt.ylabel("Logloss")
    plt.title("Logloss vs Epochs")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(epochs, evals_result["validation_0"]["aucpr"], label="Train PR-AUC")
    plt.plot(epochs, evals_result["validation_1"]["aucpr"], label="Valid PR-AUC")
    plt.plot(epochs, evals_result["validation_2"]["aucpr"], label="Test PR-AUC")
    plt.xlabel("Epoch")
    plt.ylabel("PR-AUC")
    plt.title("PR-AUC vs Epochs")
    plt.legend()

    plot_path_2 = os.path.join(MODEL_DIR, "pure_data_training_curves_logloss_pr_auc.png")
    plt.tight_layout()
    plt.savefig(plot_path_2, dpi=150)
    plt.close()
    print(f"   Logloss/PR-AUC curves saved to: {plot_path_2}")

    # Save model and preprocessor
    print("\n[6/6] Saving model and preprocessor...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    joblib.dump(model, MODEL_PATH)
    preprocessor.save(PREPROCESSOR_PATH)
    
    print(f"   Model saved to: {MODEL_PATH}")
    print(f"   Preprocessor saved to: {PREPROCESSOR_PATH}")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Metric': [
            'Accuracy', 'Balanced Accuracy', 'Precision', 'Recall', 'Specificity',
            'F1-Score', 'AUC-ROC', 'PR-AUC', 'MCC'
        ],
        'Train': [
            train_metrics['accuracy'], train_metrics['balanced_accuracy'],
            train_metrics['precision'], train_metrics['recall'], train_metrics['specificity'],
            train_metrics['f1'], train_metrics['roc_auc'], train_metrics['pr_auc'],
            train_metrics['mcc']
        ],
        'Test': [
            test_metrics['accuracy'], test_metrics['balanced_accuracy'],
            test_metrics['precision'], test_metrics['recall'], test_metrics['specificity'],
            test_metrics['f1'], test_metrics['roc_auc'], test_metrics['pr_auc'],
            test_metrics['mcc']
        ],
        'Validation': [
            valid_metrics['accuracy'], valid_metrics['balanced_accuracy'],
            valid_metrics['precision'], valid_metrics['recall'], valid_metrics['specificity'],
            valid_metrics['f1'], valid_metrics['roc_auc'], valid_metrics['pr_auc'],
            valid_metrics['mcc']
        ]
    })

    metrics_path = os.path.join(MODEL_DIR, 'model_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"   Metrics saved to: {metrics_path}")
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
    
    return model, preprocessor, {
        'train': train_metrics,
        'validation': valid_metrics,
        'test': test_metrics
    }


if __name__ == "__main__":
    train_model()

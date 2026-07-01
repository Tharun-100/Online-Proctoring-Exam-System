"""
Training script for XGBoost Fraud Detection Model
"""

from __future__ import annotations
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    balanced_accuracy_score, matthews_corrcoef, average_precision_score
)

from sklearn.metrics import precision_recall_curve

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


def _predict_at_threshold(y_proba: np.ndarray, threshold: float) -> np.ndarray:
    return (y_proba >= threshold).astype(int)


def _compute_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": _compute_specificity(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def _threshold_sweep_metrics(
    y_true: np.ndarray, y_proba: np.ndarray, thresholds: list[float]
) -> pd.DataFrame:
    rows = []
    for thr in thresholds:
        y_pred = _predict_at_threshold(y_proba, thr)
        rows.append(
            {
                "threshold": float(thr),
                "accuracy": accuracy_score(y_true, y_pred),
                "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "specificity": _compute_specificity(y_true, y_pred),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "mcc": matthews_corrcoef(y_true, y_pred),
            }
        )

    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def _plot_threshold_overview(metrics_df: pd.DataFrame, title: str, out_path: str) -> None:
    # Multi-panel view keeps things readable compared to plotting many metrics on one axis.
    thr = metrics_df["threshold"].to_numpy()
    thr_labels = [f"{t:.2f}".rstrip("0").rstrip(".") for t in thr]

    def annotate_points(ax, x, y, labels):
        # Small labels near points so exact thresholds are visible in the exported PNG.
        for xi, yi, lab in zip(x, y, labels):
            ax.annotate(
                lab,
                (xi, yi),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=8,
                color="black",
                alpha=0.9,
            )

    plt.figure(figsize=(14, 10))

    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(thr, metrics_df["accuracy"], marker="o", label="Accuracy")
    ax1.plot(thr, metrics_df["balanced_accuracy"], marker="o", label="Balanced Acc")
    annotate_points(ax1, thr, metrics_df["accuracy"].to_numpy(), thr_labels)
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Decision Threshold")
    ax1.set_ylabel("Score")
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(thr, metrics_df["precision"], marker="o", label="Precision")
    ax2.plot(thr, metrics_df["recall"], marker="o", label="Recall")
    ax2.plot(thr, metrics_df["f1"], marker="o", label="F1")
    annotate_points(ax2, thr, metrics_df["f1"].to_numpy(), thr_labels)
    ax2.set_title("Precision / Recall / F1")
    ax2.set_xlabel("Decision Threshold")
    ax2.set_ylabel("Score")
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(thr, metrics_df["specificity"], marker="o", label="Specificity (TNR)")
    annotate_points(ax3, thr, metrics_df["specificity"].to_numpy(), thr_labels)
    ax3.set_title("Specificity")
    ax3.set_xlabel("Decision Threshold")
    ax3.set_ylabel("Score")
    ax3.set_ylim(0.0, 1.0)
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(thr, metrics_df["mcc"], marker="o", label="MCC")
    annotate_points(ax4, thr, metrics_df["mcc"].to_numpy(), thr_labels)
    ax4.set_title("Matthews Corrcoef")
    ax4.set_xlabel("Decision Threshold")
    ax4.set_ylabel("Score")
    ax4.set_ylim(-1.0, 1.0)
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.suptitle(title, y=0.98)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def train_model(*, decision_threshold: float = 0.5, sweep_thresholds: list[float] | None = None,
):
    """Train XGBoost model for fraud detection.
    
    Notes:
    - Training is threshold-independent (probabilities are learned).
    - The threshold is used only for converting probabilities to hard labels for reporting.
    - Optionally performs a threshold sweep for sensitivity analysis.
    """

    print("=" * 60)
    print("Fraud Detection System - Model Training")
    print("=" * 60)
    
    # Load raw data first to avoid leakage during preprocessing
    print("\n[1/7] Loading data...")
    df = pd.read_csv(DATASET_PATH)

    print(f"   Dataset shape: {df.shape}")
    print(f"   Features: {df.shape[1] - 1}")
    print(f"   Samples: {df.shape[0]}")
    print(f"   Class distribution:\n{df[TARGET_COLUMN].value_counts().to_dict()}")

    # Split raw data before preprocessing
    print("\n[2/7] Splitting data into train/valid/test sets...")
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
    print(f"   Preprocessing Completed. Checking The First Elements: {X_train.head}")
        # Shuffle labels test
    # y_train = np.random.permutation(y_train)

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Train XGBoost model
    print("\n[3/7] Training XGBoost model...")
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
    print("Printing The Features in the model", model.feature_importances_)

    evals_result = model.evals_result()

    # Make predictions
    print("\n [4/7] Evaluating model...")

    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_valid_proba = model.predict_proba(X_valid)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]

    y_train_pred = _predict_at_threshold(y_train_proba, decision_threshold)
    y_valid_pred = _predict_at_threshold(y_valid_proba, decision_threshold)
    y_test_pred = _predict_at_threshold(y_test_proba, decision_threshold)

    train_metrics = _compute_metrics(y_train, y_train_pred, y_train_proba)
    valid_metrics = _compute_metrics(y_valid, y_valid_pred, y_valid_proba)
    test_metrics = _compute_metrics(y_test, y_test_pred, y_test_proba)

    print("\n" + "=" * 60)
    print("TRAINING RESULTS")
    print("=" * 60)

    print(f"Decision Threshold: {decision_threshold:.2f}")
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
    print("\n[5/7] Plotting training curves...")
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

    # Threshold sweep + precision-recall curve (sensitivity analysis)
    if sweep_thresholds:
        print("\n[6/7] Threshold sensitivity analysis...")
        os.makedirs(MODEL_DIR, exist_ok=True)

        valid_sweep_df = _threshold_sweep_metrics(y_valid, y_valid_proba, sweep_thresholds)
        test_sweep_df = _threshold_sweep_metrics(y_test, y_test_proba, sweep_thresholds)

        valid_sweep_path = os.path.join(MODEL_DIR, "threshold_sweep_metrics_valid.csv")
        test_sweep_path = os.path.join(MODEL_DIR, "threshold_sweep_metrics_test.csv")
        valid_sweep_df.to_csv(valid_sweep_path, index=False)
        test_sweep_df.to_csv(test_sweep_path, index=False)
        print(f"   Valid sweep metrics saved to: {valid_sweep_path}")
        print(f"   Test sweep metrics saved to:  {test_sweep_path}")

        best_valid_f1 = valid_sweep_df.iloc[int(valid_sweep_df["f1"].idxmax())]
        print(f"   Best valid F1: {best_valid_f1['f1']:.4f} at threshold={best_valid_f1['threshold']:.2f}")

        # Precision-recall curve (computed across all possible thresholds)
        precision, recall, pr_thresholds = precision_recall_curve(y_test, y_test_proba)
        plt.figure(figsize=(10, 7))
        plt.plot(recall, precision, label=f"PR curve (AP={average_precision_score(y_test, y_test_proba):.3f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision–Recall Curve (Test Set)")
        plt.grid(True, alpha=0.3)

        # Mark the sweep thresholds as reference points on the PR curve (nearest threshold)
        if pr_thresholds.size > 0:
            for thr in sweep_thresholds:
                idx = int(np.argmin(np.abs(pr_thresholds - thr)))
                # precision_recall_curve returns precision/recall arrays one longer than thresholds
                plt.scatter(recall[idx + 1], precision[idx + 1], s=25)
                lab = f"{thr:.2f}".rstrip("0").rstrip(".")
                plt.annotate(
                    lab,
                    (recall[idx + 1], precision[idx + 1]),
                    textcoords="offset points",
                    xytext=(6, 6),
                    ha="left",
                    fontsize=8,
                    alpha=0.9,
                )

        pr_plot_path = os.path.join(MODEL_DIR, "precision_recall_curve_test.png")
        plt.tight_layout()
        plt.savefig(pr_plot_path, dpi=150)
        plt.close()
        print(f"   PR curve saved to: {pr_plot_path}")

        # Unified threshold overview (valid + test)
        valid_overview_path = os.path.join(MODEL_DIR, "threshold_metrics_overview_valid.png")
        test_overview_path = os.path.join(MODEL_DIR, "threshold_metrics_overview_test.png")
        _plot_threshold_overview(valid_sweep_df, "Threshold Metrics Overview (Validation Set)", valid_overview_path)
        _plot_threshold_overview(test_sweep_df, "Threshold Metrics Overview (Test Set)", test_overview_path)
        print(f"   Threshold overview (valid) saved to: {valid_overview_path}")
        print(f"   Threshold overview (test) saved to:  {test_overview_path}")

        # Threshold tradeoff plot (test set)
        plt.figure(figsize=(12, 7))
        plt.plot(test_sweep_df["threshold"], test_sweep_df["precision"], marker="o", label="Precision")
        plt.plot(test_sweep_df["threshold"], test_sweep_df["recall"], marker="o", label="Recall")
        plt.plot(test_sweep_df["threshold"], test_sweep_df["f1"], marker="o", label="F1")
        for t, f1v in zip(test_sweep_df["threshold"].to_numpy(), test_sweep_df["f1"].to_numpy()):
            lab = f"{t:.2f}".rstrip("0").rstrip(".")
            plt.annotate(lab, (t, f1v), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
        plt.xlabel("Decision Threshold")
        plt.ylabel("Score")
        plt.title("Threshold Sensitivity (Test Set)")
        plt.ylim(0.0, 1.0)
        plt.grid(True, alpha=0.3)
        plt.legend()
        sweep_plot_path = os.path.join(MODEL_DIR, "threshold_sensitivity_test.png")
        plt.tight_layout()
        plt.savefig(sweep_plot_path, dpi=150)
        plt.close()
        print(f"   Threshold sensitivity plot saved to: {sweep_plot_path}")

    # Save model and preprocessor
    print("\n[7/7] Saving model and preprocessor...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    joblib.dump(model, MODEL_PATH)
    preprocessor.save(PREPROCESSOR_PATH)
    
    print(f"   Model saved to: {MODEL_PATH}")
    print(f"   Preprocessor saved to: {PREPROCESSOR_PATH}")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Metric': [
            'Decision Threshold',
            'Accuracy', 'Balanced Accuracy', 'Precision', 'Recall', 'Specificity',
            'F1-Score', 'AUC-ROC', 'PR-AUC', 'MCC'
        ],
        'Train': [
            decision_threshold,
            train_metrics['accuracy'], train_metrics['balanced_accuracy'],
            train_metrics['precision'], train_metrics['recall'], train_metrics['specificity'],
            train_metrics['f1'], train_metrics['roc_auc'], train_metrics['pr_auc'],
            train_metrics['mcc']
        ],
        'Test': [
            decision_threshold,
            test_metrics['accuracy'], test_metrics['balanced_accuracy'],
            test_metrics['precision'], test_metrics['recall'], test_metrics['specificity'],
            test_metrics['f1'], test_metrics['roc_auc'], test_metrics['pr_auc'],
            test_metrics['mcc']
        ],
        'Validation': [
            decision_threshold,
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
    parser = argparse.ArgumentParser(description="Train XGBoost model with optional threshold sweep.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for converting probabilities to labels (default: 0.5).",
    )

    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run threshold sensitivity analysis and save PR/thresh plots + CSVs.",
    )

    parser.add_argument("--sweep-min", type=float, default=0.1, help="Min threshold for sweep (default: 0.1).")
    parser.add_argument("--sweep-max", type=float, default=0.999, help="Max threshold for sweep (default: 0.9).")
    parser.add_argument("--sweep-step", type=float, default=0.1, help="Step size for sweep (default: 0.1).")
    
    parser.add_argument(
        "--thresholds",
        type=str,
        default=None,
        help="Optional explicit thresholds list, like '0.1,0.2,0.5,0.9' (overrides sweep range).",
    )

    args = parser.parse_args()

    sweep_thresholds = None
    if args.sweep or args.thresholds:
        if args.thresholds:
            sweep_thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
        else:
            # Include endpoint safely for common decimal steps
            sweep_thresholds = np.arange(args.sweep_min, args.sweep_max + 1e-9, args.sweep_step).tolist()
            sweep_thresholds = [float(f"{x:.10g}") for x in sweep_thresholds]

    train_model(decision_threshold=args.threshold, sweep_thresholds=sweep_thresholds)

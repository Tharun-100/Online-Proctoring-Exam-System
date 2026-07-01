"""
Train an XGBoost model on the window-level proctoring dataset.

Default input matches:
  New_setup/new_dataset_output/final_window_dataset.csv

This version uses:
  1. Train / validation / test split
  2. Group split by video_id to reduce leakage
  3. Early stopping on validation set
  4. Safe model saving in JSON format
  5. Separate metadata JSON for inference

Recommended split with defaults:
  test_size = 0.20
  val_size = 0.20 of remaining training data

Final percentage:
  Train = 64%
  Validation = 16%
  Test = 20%
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split


# -----------------------------
# Metrics helpers
# -----------------------------

def _predict_at_threshold(y_proba: np.ndarray, threshold: float) -> np.ndarray:
    return (y_proba >= threshold).astype(int)


def _compute_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def _safe_roc_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_proba))


def _safe_pr_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_proba))


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(_compute_specificity(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": _safe_roc_auc(y_true, y_proba),
        "pr_auc": _safe_pr_auc(y_true, y_proba),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def _save_training_evidence(
    *,
    model: xgb.XGBClassifier,
    metrics: dict[str, dict[str, float]],
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
    y_test_proba: np.ndarray,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_validation: pd.DataFrame,
    y_validation: np.ndarray,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    raw_history = model.evals_result()
    validation_history = raw_history["validation_0"]
    round_count = len(validation_history["logloss"])
    train_logloss = []
    train_pr_auc = []
    for end_round in range(1, round_count + 1):
        train_probability = model.predict_proba(
            X_train, iteration_range=(0, end_round)
        )[:, 1]
        train_logloss.append(float(log_loss(y_train, train_probability, labels=[0, 1])))
        train_pr_auc.append(float(average_precision_score(y_train, train_probability)))
    history = {
        "train": {"logloss": train_logloss, "aucpr": train_pr_auc},
        "validation": {
            "logloss": [float(value) for value in validation_history["logloss"]],
            "aucpr": [float(value) for value in validation_history["aucpr"]],
        },
        "raw_xgboost_history": raw_history,
    }
    with open(os.path.join(output_dir, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for key, label in (("train", "Train"), ("validation", "Validation")):
        rounds = np.arange(1, len(history[key]["logloss"]) + 1)
        axes[0].plot(rounds, history[key]["logloss"], label=label)
        axes[1].plot(rounds, history[key]["aucpr"], label=label)
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is not None:
        for axis in axes:
            axis.axvline(int(best_iteration) + 1, color="black", linestyle="--", alpha=0.6)
    axes[0].set_title("Window XGBoost Log-loss")
    axes[0].set_ylabel("Log-loss")
    axes[1].set_title("Window XGBoost PR-AUC")
    axes[1].set_xlabel("Boosting round")
    axes[1].set_ylabel("PR-AUC")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "training_curves.png"), dpi=180)
    plt.close(fig)

    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "roc_auc",
        "pr_auc",
        "mcc",
    ]
    metrics_df = pd.DataFrame(
        {
            split.title(): [metrics[split][name] for name in metric_names]
            for split in ("train", "validation", "test")
        },
        index=metric_names,
    )
    metrics_df.to_csv(os.path.join(output_dir, "metrics_comparison.csv"))
    axis = metrics_df.plot(kind="bar", figsize=(13, 6), ylim=(0, 1.05), rot=35)
    axis.set_title("Window XGBoost Performance by Dataset Split")
    axis.set_ylabel("Score")
    axis.set_xlabel("Metric")
    axis.grid(axis="y", alpha=0.25)
    axis.figure.tight_layout()
    axis.figure.savefig(os.path.join(output_dir, "metrics_comparison.png"), dpi=180)
    plt.close(axis.figure)

    cm = confusion_matrix(y_test, y_test_pred, labels=[0, 1])
    fig, axis = plt.subplots(figsize=(5.5, 5))
    image = axis.imshow(cm, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(cm[row, column]), ha="center", va="center", fontsize=13)
    axis.set_xticks([0, 1], labels=["Normal", "Suspicious"])
    axis.set_yticks([0, 1], labels=["Normal", "Suspicious"])
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title("Window XGBoost Test Confusion Matrix")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "test_confusion_matrix.png"), dpi=180)
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
    fig, axis = plt.subplots(figsize=(7, 5.5))
    axis.plot(recall, precision, label=f"PR-AUC = {metrics['test']['pr_auc']:.4f}")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Window XGBoost Test Precision-Recall Curve")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "test_precision_recall_curve.png"), dpi=180)
    plt.close(fig)

@dataclass(frozen=True)
class TrainArtifacts:
    model_path: str
    meta_path: str
    feature_columns: list[str]
    drop_columns: list[str]
    label_column: str
    group_column: str | None
    decision_threshold: float
    split_percentages: dict[str, float]
    split_counts: dict[str, int]
    label_distribution: dict[str, dict[str, int]]
    xgboost_params: dict[str, Any]
    best_iteration: int | None
    best_score: float | None
    metrics: dict[str, dict[str, float]]


# -----------------------------
# Model parameters
# -----------------------------

def _default_xgb_params(*, random_state: int, scale_pos_weight: float | None) -> dict[str, Any]:
    params: dict[str, Any] = {
        "objective": "binary:logistic",
        # Last metric is used by XGBoost early stopping.
        # We keep aucpr last because suspicious detection usually cares about positive class quality.
        "eval_metric": ["logloss", "auc", "error", "aucpr"],
        "n_estimators": 1000,
        "learning_rate": 0.03,
        "max_depth": 4,
        "min_child_weight": 1,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 2.0,
        "random_state": random_state,
        "n_jobs": -1,
        "tree_method": "hist",
    }

    if scale_pos_weight is not None:
        params["scale_pos_weight"] = float(scale_pos_weight)
    return params

def _auto_scale_pos_weight(y: pd.Series) -> float | None:
    counts = y.value_counts()
    if set(counts.index.astype(int).tolist()) != {0, 1}:
        return None
    n_neg = int(counts.get(0, 0))
    n_pos = int(counts.get(1, 0))
    if n_pos == 0 or n_neg == 0:
        return None
    return n_neg / n_pos


# -----------------------------
# Split helpers
# -----------------------------

def _group_or_stratified_split(
    df: pd.DataFrame,
    *,
    label_col: str,
    group_col: str | None,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe while trying to keep windows from same video together."""
    if group_col and group_col in df.columns:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        groups = df[group_col]
        train_idx, test_idx = next(splitter.split(df, df[label_col], groups=groups))
        return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[label_col],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _make_train_val_test_split(
    df: pd.DataFrame,
    *,
    label_col: str,
    group_col: str | None,
    test_size: float,
    val_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    First split: train_valid vs test.
    Second split: train vs validation.

    If test_size=0.20 and val_size=0.20, final split becomes:
      train = 64%, validation = 16%, test = 20%
    because validation is 20% of the remaining 80%.
    """
    train_valid_df, test_df = _group_or_stratified_split(
        df,
        label_col=label_col,
        group_col=group_col,
        test_size=test_size,
        random_state=random_state,
    )

    train_df, val_df = _group_or_stratified_split(
        train_valid_df,
        label_col=label_col,
        group_col=group_col,
        test_size=val_size,
        random_state=random_state + 1,
    )

    return train_df, val_df, test_df


def _label_distribution(df: pd.DataFrame, label_col: str) -> dict[str, int]:
    return {str(k): int(v) for k, v in df[label_col].value_counts().sort_index().to_dict().items()}


# -----------------------------
# Main training pipeline
# -----------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Train XGBoost on window-level CSV features.")
    parser.add_argument(
        "--data",
        default=os.path.join("New_setup", "new_data", "final_window_dataset.csv"),
        help="Path to window-level dataset CSV.",
    )
    parser.add_argument("--label-col", default="label", help="Target column name.")
    parser.add_argument("--group-col", default="video_id", help="Group column for split; prevents video-window leakage.")
    parser.add_argument("--no-group-split", action="store_true", help="Disable group split even if group column exists.")
    parser.add_argument("--test-size", type=float, default=0.20, help="Final test fraction. Default: 20%%.")
    parser.add_argument("--val-size",type=float,default=0.20,help="Validation fraction from remaining train_valid data. Default gives final 16%% validation.",)
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for reports/pred labels.")
    parser.add_argument(
        "--drop-cols",
        default="video_id,window_start_sec,window_end_sec,source_folder",
        help="Comma-separated columns to drop before training.",
    )

    parser.add_argument(
        "--model-out",
        default=os.path.join("models", "xgboost_window_model.json"),
        help="Where to save XGBoost model. JSON is recommended for XGBoost.",
    )

    parser.add_argument(
        "--meta-out",
        default=os.path.join("models", "xgboost_window_model_meta.json"),
        help="Where to save training metadata JSON.",
    )

    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=50,
        help="Stop if validation aucpr does not improve for this many rounds. Set 0 to disable.",
    )

    parser.add_argument(
        "--scale-pos-weight",
        default="auto",
        help='Use "auto" default, provide a float, or set "none" to disable.',
    )

    args = parser.parse_args()
    data_path = args.data
    label_col = args.label_col
    group_col = None if args.no_group_split else args.group_col
    test_size = float(args.test_size)
    val_size = float(args.val_size)
    random_state = int(args.random_state)
    threshold = float(args.threshold)
    drop_cols = [c.strip() for c in str(args.drop_cols).split(",") if c.strip()]
    model_out = args.model_out
    meta_out = args.meta_out

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    if label_col not in df.columns:
        raise ValueError(f"Missing label column '{label_col}'. Columns: {df.columns.tolist()}")

    print(f"Loaded: {data_path}")
    print(f"Shape: {df.shape}")
    print(f"Overall label distribution: {_label_distribution(df, label_col)}")

    unique_labels = sorted(df[label_col].dropna().unique().tolist())
    if len(unique_labels) < 2:
        print("\nERROR: Need at least 2 classes: 0=normal and 1=suspicious.")
        print(f"Found only: {unique_labels}")
        return 2

    if set(map(int, unique_labels)) != {0, 1}:
        print("\nERROR: Expected binary labels {0,1}.")
        print(f"Found labels: {unique_labels}")
        return 2

    # Basic cleanup: convert numeric strings to numbers where possible.
    for c in df.columns:
        if c == label_col:
            continue
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="ignore")

    train_df, val_df, test_df = _make_train_val_test_split(
        df,
        label_col=label_col,
        group_col=group_col,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )

    total_n = len(df)

    split_counts = {
        "train": int(len(train_df)),
        "validation": int(len(val_df)),
        "test": int(len(test_df)),
    }

    split_percentages = {k: round(v / total_n * 100, 2) for k, v in split_counts.items()}

    print("\nSplit counts:", split_counts)
    print("Split percentages:", split_percentages)
    print("Train label distribution:", _label_distribution(train_df, label_col))
    print("Validation label distribution:", _label_distribution(val_df, label_col))
    print("Test label distribution:", _label_distribution(test_df, label_col))

    # Build X/y
    feature_drop = [label_col] + [c for c in drop_cols if c in df.columns]
    X_train = train_df.drop(columns=feature_drop, errors="ignore")
    X_val = val_df.drop(columns=feature_drop, errors="ignore")
    X_test = test_df.drop(columns=feature_drop, errors="ignore")

    y_train = train_df[label_col].astype(int).to_numpy()
    y_val = val_df[label_col].astype(int).to_numpy()
    y_test = test_df[label_col].astype(int).to_numpy()

    # Keep numeric feature columns only.
    X_train = X_train.select_dtypes(include=[np.number])
    X_val = X_val[X_train.columns]
    X_test = X_test[X_train.columns]

    if X_train.shape[1] == 0:
        print("\nERROR: No numeric feature columns remain after dropping columns.")
        print(f"Drop cols: {drop_cols}")
        return 2
    
    if str(args.scale_pos_weight).lower() == "auto":
        spw = _auto_scale_pos_weight(train_df[label_col].astype(int))
    elif str(args.scale_pos_weight).lower() in {"none", "off", "false"}:
        spw = None
    else:
        spw = float(args.scale_pos_weight)

    xgb_params = _default_xgb_params(random_state=random_state, scale_pos_weight=spw)

    if args.early_stopping_rounds:
        xgb_params["early_stopping_rounds"] = int(args.early_stopping_rounds)

    model = xgb.XGBClassifier(**xgb_params)

    print("\nTraining...")
    print(f"Features ({len(X_train.columns)}): {list(X_train.columns)}")
    if spw is not None:
        print(f"scale_pos_weight={spw:.4f}")

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
        )

    # Evaluate using the best iteration automatically when available.
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_val_proba = model.predict_proba(X_val)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]

    y_train_pred = _predict_at_threshold(y_train_proba, threshold)
    y_val_pred = _predict_at_threshold(y_val_proba, threshold)
    y_test_pred = _predict_at_threshold(y_test_proba, threshold)

    train_metrics = _compute_metrics(y_train, y_train_pred, y_train_proba)
    val_metrics = _compute_metrics(y_val, y_val_pred, y_val_proba)
    test_metrics = _compute_metrics(y_test, y_test_pred, y_test_proba)

    print("\nMetrics at threshold={:.3f}".format(threshold))
    print("Train:     ", train_metrics)
    print("Validation:", val_metrics)
    print("Test:      ", test_metrics)
    print("\nClassification report - test set:")
    print(classification_report(y_test, y_test_pred, digits=4))
    print("Confusion matrix - test set [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_test, y_test_pred, labels=[0, 1]))

    best_iteration = getattr(model, "best_iteration", None)
    best_score = getattr(model, "best_score", None)
    print(f"\nBest iteration: {best_iteration}")
    print(f"Best validation score: {best_score}")

    # Save artifacts
    os.makedirs(os.path.dirname(model_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(meta_out) or ".", exist_ok=True)

    # JSON is the recommended durable XGBoost format.
    if not hasattr(model, "_estimator_type"):
        model._estimator_type = "classifier"
    model.save_model(model_out)

    artifacts = TrainArtifacts(
        model_path=model_out,
        meta_path=meta_out,
        feature_columns=list(X_train.columns),
        drop_columns=drop_cols,
        label_column=label_col,
        group_column=group_col if (group_col and group_col in df.columns) else None,
        decision_threshold=threshold,
        split_percentages=split_percentages,
        split_counts=split_counts,
        label_distribution={
            "overall": _label_distribution(df, label_col),
            "train": _label_distribution(train_df, label_col),
            "validation": _label_distribution(val_df, label_col),
            "test": _label_distribution(test_df, label_col),
        },
        xgboost_params=xgb_params,
        best_iteration=int(best_iteration) if best_iteration is not None else None,
        best_score=float(best_score) if best_score is not None else None,
        metrics={"train": train_metrics, "validation": val_metrics, "test": test_metrics},
    )

    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(asdict(artifacts), f, indent=2)

    evidence_dir = os.path.join(os.path.dirname(model_out) or ".", "training_evidence")
    _save_training_evidence(
        model=model,
        metrics={"train": train_metrics, "validation": val_metrics, "test": test_metrics},
        y_test=y_test,
        y_test_pred=y_test_pred,
        y_test_proba=y_test_proba,
        X_train=X_train,
        y_train=y_train,
        X_validation=X_val,
        y_validation=y_val,
        output_dir=evidence_dir,
    )

    print(f"\nSaved model: {model_out}")
    print(f"Saved meta:  {meta_out}")
    print(f"Saved training evidence: {evidence_dir}")
    print("\nUse the feature_columns from metadata during inference to keep column order identical.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Train and compare XGBoost models on the benchmark and merged datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
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
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.data_preprocessing import DataPreprocessor


RANDOM_STATE = 42
TARGET_COLUMN = "label"
CATEGORICAL_COLUMNS = ["head_pose", "gaze_direction"]
MODEL_PARAMS: dict[str, Any] = {
    "objective": "binary:logistic",
    "eval_metric": ["logloss", "auc", "error", "aucpr"],
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "tree_method": "hist",
}


def _split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create the intended 56/14/30 stratified train/validation/test split."""
    train_valid, test = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COLUMN],
    )
    train, validation = train_test_split(
        train_valid,
        test_size=0.20,
        random_state=RANDOM_STATE + 1,
        stratify=train_valid[TARGET_COLUMN],
    )
    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def _metrics(y_true: pd.Series, probability: np.ndarray) -> dict[str, Any]:
    prediction = (probability >= 0.5).astype(int)
    cm = confusion_matrix(y_true, prediction, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "mcc": float(matthews_corrcoef(y_true, prediction)),
        "confusion_matrix": [[int(v) for v in row] for row in cm],
    }


def _plot_learning_curves(evals: dict[str, Any], output_path: Path, title: str) -> None:
    rounds = np.arange(1, len(evals["validation_0"]["logloss"]) + 1)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for index, label in enumerate(("Train", "Validation")):
        key = f"validation_{index}"
        axes[0].plot(rounds, evals[key]["logloss"], label=label)
        axes[1].plot(rounds, evals[key]["aucpr"], label=label)
    axes[0].set_ylabel("Log-loss")
    axes[0].set_title(f"{title}: Log-loss")
    axes[1].set_xlabel("Boosting round")
    axes[1].set_ylabel("PR-AUC")
    axes[1].set_title(f"{title}: PR-AUC")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_confusion_matrix(matrix: list[list[int]], output_path: Path, title: str) -> None:
    cm = np.asarray(matrix)
    fig, axis = plt.subplots(figsize=(5.5, 5))
    image = axis.imshow(cm, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(cm[row, column]), ha="center", va="center", fontsize=13)
    axis.set_xticks([0, 1], labels=["Normal", "Suspicious"])
    axis.set_yticks([0, 1], labels=["Normal", "Suspicious"])
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title(title)
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _train_one(name: str, data_path: Path, output_root: Path) -> dict[str, Any]:
    dataset_dir = output_root / name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(data_path)
    if TARGET_COLUMN not in raw_df.columns:
        raise ValueError(f"{data_path} does not contain a '{TARGET_COLUMN}' column")

    rows_before = len(raw_df)
    duplicate_rows = int(raw_df.duplicated().sum())
    df = raw_df.drop_duplicates().reset_index(drop=True)
    labels = sorted(pd.to_numeric(df[TARGET_COLUMN], errors="raise").unique().tolist())
    if labels != [0, 1]:
        raise ValueError(f"{data_path} must contain binary labels 0 and 1; found {labels}")
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    train_df, validation_df, test_df = _split_data(df)
    preprocessor = DataPreprocessor(
        categorical_columns=CATEGORICAL_COLUMNS,
        auto_detect_categoricals=True,
    )
    X_train, y_train = preprocessor.fit_transform(train_df, target_column=TARGET_COLUMN)
    X_validation, y_validation = preprocessor.transform(
        validation_df, target_column=TARGET_COLUMN
    )
    X_test, y_test = preprocessor.transform(test_df, target_column=TARGET_COLUMN)

    model = xgb.XGBClassifier(**MODEL_PARAMS)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_validation, y_validation)],
        verbose=False,
    )

    probabilities = {
        "train": model.predict_proba(X_train)[:, 1],
        "validation": model.predict_proba(X_validation)[:, 1],
        "test": model.predict_proba(X_test)[:, 1],
    }
    metrics = {
        "train": _metrics(y_train, probabilities["train"]),
        "validation": _metrics(y_validation, probabilities["validation"]),
        "test": _metrics(y_test, probabilities["test"]),
    }

    model.save_model(dataset_dir / "model.json")
    joblib.dump(preprocessor, dataset_dir / "preprocessor.joblib")
    _plot_learning_curves(
        model.evals_result(), dataset_dir / "learning_curves.png", name.replace("_", " ").title()
    )
    _plot_confusion_matrix(
        metrics["test"]["confusion_matrix"],
        dataset_dir / "test_confusion_matrix.png",
        f"{name.replace('_', ' ').title()} Test Confusion Matrix",
    )

    result = {
        "dataset": name,
        "path": str(data_path),
        "rows_in_file": rows_before,
        "exact_duplicates_removed": duplicate_rows,
        "rows_used": len(df),
        "missing_values": int(raw_df.isna().sum().sum()),
        "class_distribution_used": {
            str(k): int(v) for k, v in df[TARGET_COLUMN].value_counts().sort_index().items()
        },
        "split_counts": {
            "train": len(train_df),
            "validation": len(validation_df),
            "test": len(test_df),
        },
        "processed_feature_count": int(X_train.shape[1]),
        "model_parameters": MODEL_PARAMS,
        "decision_threshold": 0.5,
        "metrics": metrics,
    }
    (dataset_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _write_comparison(results: list[dict[str, Any]], output_root: Path) -> None:
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
    rows = []
    for result in results:
        row = {
            "dataset": result["dataset"],
            "rows_in_file": result["rows_in_file"],
            "rows_used": result["rows_used"],
            "duplicates_removed": result["exact_duplicates_removed"],
            "processed_feature_count": result["processed_feature_count"],
        }
        row.update({metric: result["metrics"]["test"][metric] for metric in metric_names})
        rows.append(row)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_root / "test_metrics_comparison.csv", index=False)

    long_df = comparison.melt(
        id_vars="dataset",
        value_vars=["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"],
        var_name="metric",
        value_name="score",
    )
    pivot = long_df.pivot(index="metric", columns="dataset", values="score")
    axis = pivot.plot(kind="bar", figsize=(11, 6), ylim=(0, 1.05), rot=0)
    axis.set_title("Benchmark vs Merged Dataset: Test Metrics")
    axis.set_ylabel("Score")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(title="Dataset")
    axis.figure.tight_layout()
    axis.figure.savefig(output_root / "test_metrics_comparison.png", dpi=180)
    plt.close(axis.figure)

    benchmark, merged = results
    report = [
        "# Benchmark vs Merged Dataset Comparison",
        "",
        "## Protocol",
        "",
        "- Identical XGBoost hyperparameters for both datasets.",
        "- Stratified 56% training, 14% validation, and 30% test split.",
        "- Exact duplicate rows removed before splitting.",
        "- Imputation, scaling, and categorical encoding fitted only on each training split.",
        "- Decision threshold: 0.5.",
        "",
        "## Dataset audit",
        "",
        f"- Benchmark: {benchmark['rows_in_file']} rows in file; {benchmark['rows_used']} used.",
        f"- Merged: {merged['rows_in_file']} rows in file; {merged['rows_used']} used.",
        "- The current merged file does not contain the previously stated 8,000 rows.",
        "",
        "## Test metrics",
        "",
        comparison.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Interpretation constraint",
        "",
        "These are internal random-split results. They compare in-distribution tabular performance but do not, by themselves, establish generalization to independently recorded real exam videos.",
    ]
    (output_root / "comparison_report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("Dataset") / "Students suspicious behaviors detection dataset_V1.csv",
    )
    parser.add_argument("--merged", type=Path, default=Path("merged_features.csv"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("comparison_results") / "benchmark_vs_merged",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = [
        _train_one("benchmark", args.benchmark, args.output_dir),
        _train_one("merged", args.merged, args.output_dir),
    ]
    _write_comparison(results, args.output_dir)
    print(json.dumps({r["dataset"]: r["metrics"]["test"] for r in results}, indent=2))
    print(f"Artifacts saved to: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

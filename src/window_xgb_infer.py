"""
Run inference with a trained window-level XGBoost model on a CSV.

Produces an output CSV with:
  - xgb_window_proba
  - xgb_window_pred  (based on decision_threshold from meta, unless overridden)
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb


def main() -> int:
    parser = argparse.ArgumentParser(description="Infer window-level fraud probabilities from a CSV.")
    parser.add_argument(
        "--input",
        default=os.path.join("New_setup", "new_dataset_output", "final_window_dataset.csv"),
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("New_setup", "new_dataset_output", "final_window_dataset_with_xgb.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--model",
        default=os.path.join("models", "xgboost_window_model.json"),
        help="Trained XGBoost model path (JSON).",
    )
    parser.add_argument(
        "--meta",
        default=os.path.join("models", "xgboost_window_model_meta.json"),
        help="Model metadata JSON (feature columns, drop columns, threshold).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override decision threshold for xgb_window_pred (default from meta).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input CSV not found: {args.input}")
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not os.path.exists(args.meta):
        raise FileNotFoundError(f"Meta not found: {args.meta}")

    with open(args.meta, "r", encoding="utf-8") as f:
        meta = json.load(f)

    feature_cols = list(meta.get("feature_columns", []))
    drop_cols = list(meta.get("drop_columns", []))
    meta_threshold = float(meta.get("decision_threshold", 0.5))
    threshold = float(args.threshold) if args.threshold is not None else meta_threshold

    df = pd.read_csv(args.input)

    # Keep numeric columns; enforce expected feature set.
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    if "label" in X.columns:
        X = X.drop(columns=["label"], errors="ignore")

    for c in X.columns:
        if X[c].dtype == object:
            X[c] = pd.to_numeric(X[c], errors="ignore")

    X = X.select_dtypes(include=[np.number])

    missing = [c for c in feature_cols if c not in X.columns]
    extra = [c for c in X.columns if c not in feature_cols]
    if missing:
        print("ERROR: Missing required feature columns:", missing)
        return 2
    # Align to training feature order; ignore extras.
    X = X[feature_cols]

    model = xgb.XGBClassifier()
    model.load_model(args.model)

    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)

    out = df.copy()
    out["xgb_window_proba"] = proba
    out["xgb_window_pred"] = pred

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"Saved: {args.output}")
    if extra:
        print(f"Note: ignored extra numeric columns not in training features: {extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


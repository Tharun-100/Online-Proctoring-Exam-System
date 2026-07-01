"""Create an exact stratified 64/16/20 row split for the window dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


TRAIN_ROWS = 1765
VALIDATION_ROWS = 441
TEST_ROWS = 552
RANDOM_STATE = 42


def _label_counts(df: pd.DataFrame) -> dict[str, int]:
    return {
        str(label): int(count)
        for label, count in df["label"].value_counts().sort_index().items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("New_setup") / "new_data" / "final_window_dataset.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("New_setup") / "new_data" / "row_splits_64_16_20",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    if "label" not in df.columns:
        raise ValueError("Dataset must contain a 'label' column.")
    if len(df) != TRAIN_ROWS + VALIDATION_ROWS + TEST_ROWS:
        raise ValueError(
            f"Expected {TRAIN_ROWS + VALIDATION_ROWS + TEST_ROWS} rows, found {len(df)}. "
            "Update the fixed split counts before running on a different dataset."
        )
    labels = sorted(df["label"].dropna().astype(int).unique().tolist())
    if labels != [0, 1]:
        raise ValueError(f"Expected binary labels [0, 1], found {labels}.")

    all_indices = np.arange(len(df))
    train_validation_indices, test_indices = train_test_split(
        all_indices,
        test_size=TEST_ROWS,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )
    train_indices, validation_indices = train_test_split(
        train_validation_indices,
        test_size=VALIDATION_ROWS,
        random_state=RANDOM_STATE + 1,
        stratify=df.iloc[train_validation_indices]["label"],
    )

    split_indices = {
        "train": np.sort(train_indices),
        "validation": np.sort(validation_indices),
        "test": np.sort(test_indices),
    }
    split_frames = {
        name: df.iloc[indices].reset_index(drop=True)
        for name, indices in split_indices.items()
    }

    assigned = np.concatenate(list(split_indices.values()))
    if len(np.unique(assigned)) != len(df) or set(assigned.tolist()) != set(all_indices.tolist()):
        raise RuntimeError("Split integrity check failed: rows are missing or duplicated.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in split_frames.items():
        frame.to_csv(args.output_dir / f"{name}.csv", index=False)

    assignment_parts = []
    for name, indices in split_indices.items():
        part = pd.DataFrame(
            {
                "source_row_index": indices,
                "split": name,
                "label": df.iloc[indices]["label"].to_numpy(),
            }
        )
        if "video_id" in df.columns:
            part["video_id"] = df.iloc[indices]["video_id"].to_numpy()
        assignment_parts.append(part)
    assignments = pd.concat(assignment_parts, ignore_index=True).sort_values("source_row_index")
    assignments.to_csv(args.output_dir / "split_assignments.csv", index=False)

    video_sets = {
        name: set(frame["video_id"].astype(str))
        for name, frame in split_frames.items()
        if "video_id" in frame.columns
    }
    video_overlap = {}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if left in video_sets and right in video_sets:
            video_overlap[f"{left}_{right}"] = len(video_sets[left] & video_sets[right])

    manifest = {
        "source": str(args.data),
        "strategy": "stratified_row_split",
        "random_state": RANDOM_STATE,
        "total_rows": len(df),
        "splits": {
            name: {
                "rows": len(frame),
                "percentage": round(len(frame) / len(df) * 100, 4),
                "label_distribution": _label_counts(frame),
                "unique_videos": int(frame["video_id"].nunique())
                if "video_id" in frame.columns
                else None,
            }
            for name, frame in split_frames.items()
        },
        "video_overlap_counts": video_overlap,
        "warning": (
            "This split is row-based. Windows from the same video may occur in multiple subsets, "
            "so it must not be described as video-independent evaluation."
        ),
    }
    (args.output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(json.dumps(manifest, indent=2))
    print(f"Split files saved to: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

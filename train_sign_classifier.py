"""
Train a static hand-shape classifier from captured feature CSV.

Usage:
  python -m src.train_sign_classifier --data data/handshape_dataset.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from src.sign_features import FEATURES_PER_HAND


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/handshape_dataset.csv", help="Dataset CSV path")
    parser.add_argument("--output", default="models/handshape_classifier.joblib", help="Model output path")
    parser.add_argument("--min-samples-per-class", type=int, default=25)
    return parser.parse_args()


def load_dataset(data_path: Path) -> pd.DataFrame:
    import csv

    rows: list[dict[str, float | str | int]] = []
    with data_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Dataset is empty: {data_path}")

        for line_no, fields in enumerate(reader, start=2):
            if not fields:
                continue
            if len(fields) == FEATURES_PER_HAND + 2:
                # Legacy: label, handedness, f0..f(N-1)
                label, handedness = fields[0], fields[1]
                single_feats = np.array(fields[2:], dtype=np.float32)
                left = np.zeros((FEATURES_PER_HAND,), dtype=np.float32)
                right = np.zeros((FEATURES_PER_HAND,), dtype=np.float32)
                if handedness.lower() == "left":
                    left[: len(single_feats)] = single_feats
                    left_present, right_present, hands_required = 1, 0, "left"
                else:
                    right[: len(single_feats)] = single_feats
                    left_present, right_present, hands_required = 0, 1, "right"
                dual = np.concatenate([left, right])
                row = {
                    "label": label,
                    "hands_required": hands_required,
                    "left_present": left_present,
                    "right_present": right_present,
                }
                for i, val in enumerate(dual):
                    row[f"f{i}"] = float(val)
                rows.append(row)
            elif len(fields) == FEATURES_PER_HAND * 2 + 4:
                # Dual-hand: label, hands_required, left_present, right_present, f0..f(2N-1)
                label, hands_required = fields[0], fields[1]
                left_present, right_present = int(fields[2]), int(fields[3])
                row = {
                    "label": label,
                    "hands_required": hands_required,
                    "left_present": left_present,
                    "right_present": right_present,
                }
                for i, val in enumerate(fields[4:]):
                    row[f"f{i}"] = float(val)
                rows.append(row)
            else:
                raise ValueError(
                    f"Unsupported row at line {line_no}: expected "
                    f"{FEATURES_PER_HAND + 2} or {FEATURES_PER_HAND * 2 + 4} fields, got {len(fields)}"
                )

    if not rows:
        raise ValueError(f"No rows found in dataset: {data_path}")
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = load_dataset(data_path)

    counts = df["label"].value_counts()
    keep = counts[counts >= args.min_samples_per_class].index.tolist()
    df = df[df["label"].isin(keep)].copy()

    if df["label"].nunique() < 2:
        raise ValueError("Need at least 2 labels with enough samples to train.")

    feature_cols = [c for c in df.columns if c.startswith("f")]
    x = df[feature_cols].astype(np.float32).to_numpy()
    y = df["label"].astype(str).to_numpy()

    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=18,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(x_train, y_train)

    preds = model.predict(x_val)
    print(classification_report(y_val, preds, digits=3))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "classes": sorted(np.unique(y).tolist()),
        "feature_columns": feature_cols,
        "feature_layout": "dual_hand",
        "features_per_hand": FEATURES_PER_HAND,
    }
    joblib.dump(payload, output_path)
    print(f"[INFO] Saved trained classifier to {output_path}")


if __name__ == "__main__":
    main()

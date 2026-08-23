#!/usr/bin/env python
"""Train the lightweight TF-IDF + Ridge rating prediction model.

Usage::

    python training/train_ridge.py --data data/raw/sample_reviews.csv
    python training/train_ridge.py --data my_reviews.csv --alpha 1.5 --test-size 0.2

The trained pipeline (plus metadata and holdout metrics) is saved to
``models/ridge_model.pkl``.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.models.base import clamp_rating
from app.services.preprocessing import batch_clean

TEXT_COL = "text"
RATING_COL = "rating"


def build_pipeline(alpha: float) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=50_000,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )


def evaluate(pipeline: Pipeline, texts: np.ndarray, ratings: np.ndarray) -> dict:
    continuous = np.clip(pipeline.predict(texts), 1.0, 5.0)
    rounded = [clamp_rating(value) for value in continuous]
    return {
        "mae": round(float(mean_absolute_error(ratings, continuous)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(ratings, continuous))), 4),
        "r2": round(float(r2_score(ratings, continuous)), 4),
        "exact_accuracy": round(float(np.mean(np.array(rounded) == ratings)), 4),
        "within_one_accuracy": round(float(np.mean(np.abs(np.array(rounded) - ratings) <= 1)), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the TF-IDF + Ridge rating model.")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "raw" / "sample_reviews.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "ridge_model.pkl")
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge regularisation strength.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-processed",
        type=Path,
        default=None,
        help="Optionally save the cleaned training data to this CSV path.",
    )
    args = parser.parse_args()

    from training.data_utils import load_reviews_csv

    frame = load_reviews_csv(args.data)
    print(f"Loaded {len(frame)} reviews from {args.data}")
    print("Rating distribution:")
    print(frame[RATING_COL].value_counts().sort_index().to_string())

    texts = np.array(batch_clean(frame[TEXT_COL].tolist()))
    ratings = frame[RATING_COL].to_numpy(dtype=float)

    x_train, x_test, y_train, y_test = train_test_split(
        texts, ratings, test_size=args.test_size, random_state=args.seed
    )
    print(f"Training on {len(x_train)} rows, evaluating on {len(x_test)} rows")

    pipeline = build_pipeline(args.alpha)
    pipeline.fit(x_train, y_train)

    train_metrics = evaluate(pipeline, x_train, y_train)
    test_metrics = evaluate(pipeline, x_test, y_test)
    print(f"Train metrics: {json.dumps(train_metrics)}")
    print(f"Test metrics:  {json.dumps(test_metrics)}")

    if args.save_processed is not None:
        args.save_processed.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({TEXT_COL: texts, RATING_COL: ratings}).to_csv(args.save_processed, index=False)
        print(f"Saved cleaned dataset to {args.save_processed}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pipeline": pipeline,
        "metadata": {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "dataset": str(args.data),
            "rows": {"train": int(len(x_train)), "test": int(len(x_test))},
            "params": {"alpha": args.alpha, "ngram_range": [1, 2], "min_df": 2},
            "metrics": {"train": train_metrics, "test": test_metrics},
        },
    }
    joblib.dump(payload, args.output)
    print(f"Saved model artifact to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

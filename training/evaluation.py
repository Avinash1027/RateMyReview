#!/usr/bin/env python
"""Model evaluation pipeline.

Compares every available backend (DistilBERT, TF-IDF + Ridge, heuristic
baseline) on a labelled review dataset and prints a metrics table.

Usage::

    python training/evaluation.py --data data/raw/sample_reviews.csv
    python training/evaluation.py --data data/raw/held_out.csv --backends ridge heuristic
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from app.core.model_loader import try_build_predictor
from training.data_utils import load_reviews_csv

TEXT_COL = "text"
RATING_COL = "rating"


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    error = y_pred - y_true
    return {
        "mae": round(float(np.mean(np.abs(error))), 4),
        "rmse": round(float(np.sqrt(np.mean(error**2))), 4),
        "exact_acc": round(float(np.mean(error == 0)), 4),
        "within_one": round(float(np.mean(np.abs(error) <= 1)), 4),
    }


def format_table(results: dict[str, dict]) -> str:
    header = f"{'backend':<12} {'MAE':>7} {'RMSE':>7} {'exact':>7} {'within1':>8}"
    lines = [header, "-" * len(header)]
    for backend, metrics in results.items():
        lines.append(
            f"{backend:<12} {metrics['mae']:>7.3f} {metrics['rmse']:>7.3f} "
            f"{metrics['exact_acc']:>7.2%} {metrics['within_one']:>8.2%}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate available rating prediction backends.")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "raw" / "sample_reviews.csv")
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["auto"],
        help="Backends to evaluate (default: every available backend).",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Optionally evaluate on a random sample of N reviews (seeded).",
    )
    args = parser.parse_args()

    frame = load_reviews_csv(args.data)
    if args.sample is not None and args.sample < len(frame):
        frame = frame.sample(n=args.sample, random_state=42).reset_index(drop=True)
        print(f"[info] sampled {len(frame)} of the available reviews")
    # The predictor applies backend-specific preprocessing itself.
    texts = frame[TEXT_COL].tolist()
    y_true = frame[RATING_COL].to_numpy(dtype=int)

    candidates = args.backends
    if candidates == ["auto"]:
        candidates = ["distilbert", "ridge", "heuristic"]

    results: dict[str, dict] = {}
    for backend in candidates:
        predictor = try_build_predictor(backend)
        if predictor is None:
            print(f"[skip] backend '{backend}' is not available")
            continue
        predictions = predictor.predict_batch(texts)
        y_pred = np.array([p.rating for p in predictions], dtype=int)
        results[backend] = compute_metrics(y_true, y_pred)

    if not results:
        print("No backends could be evaluated.")
        return 1

    print(f"\nEvaluation on {len(frame)} reviews from {args.data}\n")
    print(format_table(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

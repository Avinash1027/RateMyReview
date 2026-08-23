#!/usr/bin/env python
"""Fine-tune DistilBERT for 5-class review rating prediction.

Class ids map linearly to star ratings: 0 -> 1 star ... 4 -> 5 stars.

Usage::

    python training/train_distilbert.py --data data/raw/reviews.csv --epochs 3
    python training/train_distilbert.py --max-rows 5000 --batch-size 32

Requires ``torch`` and ``transformers`` (see requirements.txt).
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import numpy as np
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )
except ImportError:
    print(
        "torch/transformers are required for transformer training.\n"
        "Install them with:  pip install torch transformers",
        file=sys.stderr,
    )
    raise SystemExit(1)

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from training.data_utils import load_reviews_csv

NUM_LABELS = 5
TEXT_COL = "text"
RATING_COL = "rating"


class ReviewDataset(Dataset):
    """Tokenised review dataset with star ratings shifted to 0-based labels."""

    def __init__(self, texts, ratings, tokenizer, max_length: int):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        self.labels = [int(rating) - 1 for rating in ratings]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        item = {key: torch.tensor(val[index]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    mae = mean_absolute_error(labels, predictions)
    return {
        "accuracy": float((predictions == labels).mean()),
        "mae": float(mae),
        "within_one": float((np.abs(predictions - labels) <= 1).mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for rating prediction.")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "raw" / "sample_reviews.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "distilbert_weights")
    parser.add_argument("--base-model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--max-rows", type=int, default=None, help="Optionally cap dataset size.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = load_reviews_csv(args.data)
    if args.max_rows is not None and len(frame) > args.max_rows:
        frame = frame.sample(n=args.max_rows, random_state=args.seed).reset_index(drop=True)
    print(f"Loaded {len(frame)} reviews from {args.data}")

    train_frame, val_frame = train_test_split(
        frame, test_size=args.test_size, random_state=args.seed, stratify=frame[RATING_COL]
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    train_dataset = ReviewDataset(train_frame[TEXT_COL], train_frame[RATING_COL], tokenizer, args.max_length)
    val_dataset = ReviewDataset(val_frame[TEXT_COL], val_frame[RATING_COL], tokenizer, args.max_length)
    print(f"Training on {len(train_dataset)} rows, validating on {len(val_dataset)} rows")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training device: {device}")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=NUM_LABELS, id2label={i: f"{i + 1}_stars" for i in range(NUM_LABELS)}
    )

    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=10,
        report_to="none",
        seed=args.seed,
        use_cpu=(device == "cpu"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    metrics = trainer.evaluate()
    print(f"Final validation metrics: {metrics}")

    args.output.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    print(f"Saved fine-tuned checkpoint to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

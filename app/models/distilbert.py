"""DistilBERT based rating prediction model.

Wraps a fine-tuned ``AutoModelForSequenceClassification`` checkpoint with five
classes that map linearly to star ratings: class id 0 -> 1 star ... id 4 -> 5
stars. Heavy dependencies (torch, transformers) are imported lazily so the
application can run in lightweight deployments without them installed.
"""

from pathlib import Path
from typing import Any, Dict, List

from app.models.base import BackendNotAvailableError, clamp_rating

try:  # pragma: no cover - exercised implicitly via TRANSFORMERS_AVAILABLE
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:  # torch/transformers not installed
    TRANSFORMERS_AVAILABLE = False

#: A checkpoint is only considered trained once this file exists.
WEIGHTS_MARKER = "config.json"

NUM_LABELS = 5
#: Class id -> star rating (1-5).
ID_TO_RATING = {i: i + 1 for i in range(NUM_LABELS)}


def weights_look_trained(weights_dir: Path) -> bool:
    """Return True if the directory contains a fine-tuned checkpoint."""
    return (Path(weights_dir) / WEIGHTS_MARKER).is_file()


def _require_transformers() -> None:
    if not TRANSFORMERS_AVAILABLE:
        raise BackendNotAvailableError(
            "torch/transformers are not installed. Install them with "
            "`pip install torch transformers` to enable the DistilBERT backend."
        )


def resolve_device(preference: str = "auto") -> "torch.device":
    """Resolve the torch device from a preference string."""
    _require_transformers()
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


class DistilBertRatingModel:
    """Fine-tuned DistilBERT classifier producing 1-5 star ratings."""

    def __init__(self, tokenizer: Any, model: Any, device: "torch.device", max_length: int = 256):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.max_length = max_length
        self.model.to(device)
        self.model.eval()

    @classmethod
    def load(
        cls,
        weights_dir: Path,
        device: str = "auto",
        max_length: int = 256,
    ) -> "DistilBertRatingModel":
        """Load a fine-tuned checkpoint from ``weights_dir``.

        Raises:
            BackendNotAvailableError: if dependencies are missing or the
                directory does not contain trained weights.
        """
        _require_transformers()
        weights_dir = Path(weights_dir)
        if not weights_look_trained(weights_dir):
            raise BackendNotAvailableError(
                f"No trained DistilBERT weights found in '{weights_dir}'. "
                "Fine-tune it first: python training/train_distilbert.py"
            )
        try:
            tokenizer = AutoTokenizer.from_pretrained(weights_dir)
            model = AutoModelForSequenceClassification.from_pretrained(weights_dir)
        except Exception as exc:  # noqa: BLE001 - surfaced as backend failure
            raise BackendNotAvailableError(
                f"Failed to load DistilBERT checkpoint from '{weights_dir}': {exc}"
            ) from exc
        return cls(tokenizer, model, resolve_device(device), max_length=max_length)

    def predict(self, texts: List[str], batch_size: int = 16) -> List[Dict[str, Any]]:
        """Predict ratings for a list of raw review texts.

        Returns one dict per text with the integer ``rating`` (1-5), the
        ``confidence`` (max class probability), the soft ``expected_value``
        and the full class ``probabilities`` keyed by star rating.
        """
        _require_transformers()
        if not texts:
            return []

        results: List[Dict[str, Any]] = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = [str(t) for t in texts[start : start + batch_size]]
                encoded = self.tokenizer(
                    batch,
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**encoded).logits
                probabilities = torch.softmax(logits, dim=-1)

                stars = torch.arange(
                    1, probabilities.shape[-1] + 1,
                    dtype=torch.float32,
                    device=probabilities.device,
                )
                for row in probabilities:
                    row = row.float()
                    expected = float((row * stars).sum().item())
                    results.append(
                        {
                            "rating": clamp_rating(expected),
                            "confidence": round(float(row.max().item()), 4),
                            "expected_value": round(expected, 4),
                            "probabilities": {
                                str(ID_TO_RATING[i]): round(float(p), 4)
                                for i, p in enumerate(row.tolist())
                            },
                        }
                    )
        return results

    def describe(self) -> Dict[str, Any]:
        return {
            "type": "distilbert",
            "device": str(self.device),
            "max_length": self.max_length,
            "num_labels": NUM_LABELS,
        }

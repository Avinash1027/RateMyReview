"""Lightweight TF-IDF + Ridge regression rating model.

The persisted artifact is a scikit-learn :class:`~sklearn.pipeline.Pipeline`
(TfidfVectorizer -> Ridge) optionally wrapped together with training metadata
in a dict payload. It provides CPU-efficient inference for resource
constrained deployments.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

from app.models.base import BackendNotAvailableError


class TfidfRidgeModel:
    """Wrapper around a persisted TfidfVectorizer + Ridge pipeline."""

    def __init__(self, pipeline: Any, metadata: Optional[Dict[str, Any]] = None):
        self.pipeline = pipeline
        self.metadata: Dict[str, Any] = metadata or {}

    @classmethod
    def load(cls, path: Path) -> "TfidfRidgeModel":
        """Load a trained pipeline from disk.

        Raises:
            BackendNotAvailableError: if the artifact does not exist or is
                corrupted.
        """
        path = Path(path)
        if not path.is_file():
            raise BackendNotAvailableError(
                f"Ridge model artifact not found at '{path}'. "
                "Train it first: python training/train_ridge.py"
            )
        try:
            payload = joblib.load(path)
        except Exception as exc:  # noqa: BLE001 - surfaced as backend failure
            raise BackendNotAvailableError(f"Failed to load ridge model from '{path}': {exc}") from exc

        if isinstance(payload, dict) and "pipeline" in payload:
            return cls(payload["pipeline"], payload.get("metadata", {}))
        # Plain pipeline objects are accepted for backwards compatibility.
        return cls(payload)

    def predict(self, texts: List[str]) -> np.ndarray:
        """Return continuous rating scores clipped to [1, 5]."""
        if not texts:
            return np.array([], dtype=float)
        scores = self.pipeline.predict(list(texts))
        return np.clip(np.asarray(scores, dtype=float), 1.0, 5.0)

    def describe(self) -> Dict[str, Any]:
        """Return a small summary of the loaded pipeline for diagnostics."""
        try:
            vectorizer = self.pipeline.named_steps.get("tfidf")
            vocabulary_size = len(getattr(vectorizer, "vocabulary_", {}))
        except Exception:  # noqa: BLE001 - diagnostics only
            vocabulary_size = None
        return {
            "type": "tfidf+ridge",
            "vocabulary_size": vocabulary_size,
            **({"metadata": self.metadata} if self.metadata else {}),
        }

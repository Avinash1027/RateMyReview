"""Prediction service layer.

:class:`PredictorService` exposes a uniform ``predict`` / ``predict_batch``
interface regardless of which backend is active:

* ``distilbert`` - fine-tuned transformer (highest accuracy, needs torch).
* ``ridge``      - TF-IDF + Ridge pipeline (CPU friendly).
* ``heuristic``  - lexicon based baseline that is always available so the
  service degrades gracefully before any model has been trained.
"""

import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.models.base import clamp_rating
from app.models.tfidf_ridge import TfidfRidgeModel
from app.services.preprocessing import batch_clean, minimal_clean

_TOKEN_RE = re.compile(r"[a-z']+")

_POSITIVE_WORDS = {
    "love", "loved", "loves", "amazing", "awesome", "excellent", "great",
    "fantastic", "wonderful", "perfect", "best", "good", "nice", "superb",
    "outstanding", "brilliant", "delightful", "delicious", "happy",
    "satisfied", "impressed", "recommend", "recommended", "flawless",
    "incredible", "friendly", "helpful", "beautiful", "spotless", "premium",
    "sturdy", "reliable", "tasty", "generous", "lovely", "enjoyable",
    "enjoyed", "liked", "like", "comfortable", "intuitive", "masterpiece",
    "gem", "fresh", "quick", "fast", "smooth", "works", "worked",
}
_NEGATIVE_WORDS = {
    "terrible", "awful", "horrible", "worst", "bad", "poor", "disappointed",
    "disappointing", "waste", "broken", "broke", "useless", "garbage",
    "scam", "defective", "cheap", "flimsy", "slow", "cold", "bland",
    "tasteless", "inedible", "rude", "dirty", "filthy", "damaged",
    "crashed", "crashes", "malfunctioned", "boring", "annoying",
    "overpriced", "misleading", "painful", "scratchy", "dated", "musty",
    "noisy", "hate", "hated", "avoid", "returned", "refund", "disgusting",
    "unusable", "wrong", "faulty", "leaking", "rusty", "delayed",
}
_NEGATORS = {"not", "no", "never", "nothing", "cannot", "none", "without"}
_INTENSIFIERS = {"very", "really", "extremely", "absolutely", "super",
                 "incredibly", "totally", "so", "quite"}


@dataclass
class Prediction:
    """Framework-agnostic prediction result."""

    rating: int
    confidence: float
    model_used: str
    expected_rating: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None
    processing_time_ms: float = 0.0


class PredictorService:
    """Facade that dispatches predictions to the active backend."""

    def __init__(self, backend: str, model: Any = None, batch_size: int = 32):
        self.backend = backend
        self.model = model
        self.batch_size = batch_size

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def predict(self, text: str) -> Prediction:
        """Predict a 1-5 star rating for a single review."""
        start = time.perf_counter()
        prediction = self._predict_texts([text])[0]
        prediction.processing_time_ms = self._elapsed_ms(start)
        return prediction

    def predict_batch(self, texts: List[str]) -> List[Prediction]:
        """Predict ratings for a batch of reviews."""
        start = time.perf_counter()
        predictions = self._predict_texts(list(texts))
        elapsed = self._elapsed_ms(start)
        for prediction in predictions:
            prediction.processing_time_ms = elapsed
        return predictions

    @staticmethod
    def _elapsed_ms(start: float) -> float:
        return round((time.perf_counter() - start) * 1000, 2)

    # ------------------------------------------------------------------ #
    # Backend dispatch
    # ------------------------------------------------------------------ #
    def _predict_texts(self, texts: List[str]) -> List[Prediction]:
        if self.backend == "distilbert":
            return self._predict_distilbert(texts)
        if self.backend == "ridge":
            return self._predict_ridge(texts)
        if self.backend == "heuristic":
            return [self._predict_heuristic(text) for text in texts]
        raise ValueError(f"Unknown prediction backend: {self.backend!r}")

    def _predict_distilbert(self, texts: List[str]) -> List[Prediction]:
        outputs = self.model.predict(
            [minimal_clean(text) for text in texts],
            batch_size=self.batch_size,
        )
        return [
            Prediction(
                rating=output["rating"],
                confidence=output["confidence"],
                model_used="distilbert",
                expected_rating=output.get("expected_value"),
                probabilities=output.get("probabilities"),
            )
            for output in outputs
        ]

    def _predict_ridge(self, texts: List[str]) -> List[Prediction]:
        scores = self.model.predict(batch_clean(texts))
        predictions: List[Prediction] = []
        for score in scores:
            rating = clamp_rating(float(score))
            # Certainty of the rounding decision: 1.0 when the regressor lands
            # exactly on a whole star, 0.0 at a rounding boundary.
            confidence = max(0.0, 1.0 - 2.0 * abs(float(score) - rating))
            predictions.append(
                Prediction(
                    rating=rating,
                    confidence=round(confidence, 4),
                    model_used="ridge",
                    expected_rating=round(float(score), 4),
                )
            )
        return predictions

    # ------------------------------------------------------------------ #
    # Heuristic fallback
    # ------------------------------------------------------------------ #
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return _TOKEN_RE.findall(text.lower())

    @classmethod
    def _heuristic_score(cls, text: str) -> float:
        """Score text sentiment in [-1, 1] with a small lexicon.

        Negators flip the polarity of a following sentiment word and
        intensifiers boost the following word's magnitude.
        """
        tokens = cls._tokenize(text)
        contributions: List[tuple[int, int, float]] = []  # (index, sign, magnitude)

        for index, token in enumerate(tokens):
            token = token.strip("'")
            if token in _POSITIVE_WORDS:
                sign, magnitude = 1, 1.0
            elif token in _NEGATIVE_WORDS:
                sign, magnitude = -1, 1.0
            else:
                continue

            # Look back up to three tokens for modifiers.
            lookback = [tokens[j].strip("'") for j in range(max(0, index - 3), index)]
            if any(word in _NEGATORS for word in lookback):
                sign = -sign
            if any(word in _INTENSIFIERS for word in lookback):
                magnitude *= 1.5
            contributions.append((index, sign, magnitude))

        if not contributions:
            return 0.0
        return sum(sign * magnitude for _, sign, magnitude in contributions) / len(contributions)

    def _predict_heuristic(self, text: str) -> Prediction:
        score = self._heuristic_score(text)
        rating = clamp_rating(3.0 + 2.0 * score)
        confidence = round(min(1.0, 0.35 + 0.65 * abs(score)), 4)
        return Prediction(
            rating=rating,
            confidence=confidence,
            model_used="heuristic",
            expected_rating=round(3.0 + 2.0 * score, 4),
        )

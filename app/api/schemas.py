"""Pydantic request/response models for the RateMyReview API."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

MAX_REVIEW_LENGTH = 5_000
MAX_BATCH_SIZE = 64


class ReviewRequest(BaseModel):
    """Single review prediction request."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_REVIEW_LENGTH,
        description="Raw review text to analyse.",
        examples=["Absolutely fantastic product, works flawlessly and arrived early!"],
    )

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class BatchReviewRequest(BaseModel):
    """Batch review prediction request."""

    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH_SIZE,
        description=f"1-{MAX_BATCH_SIZE} review texts.",
        examples=[["Terrible, broke after one use.", "It is okay, nothing special.", "Love it!"]],
    )

    @field_validator("texts")
    @classmethod
    def texts_not_blank(cls, values: List[str]) -> List[str]:
        if any(not value.strip() for value in values):
            raise ValueError("every text must be non-blank")
        return values


class PredictionResponse(BaseModel):
    """Rating prediction for a single review."""

    rating: int = Field(..., ge=1, le=5, description="Predicted star rating (1-5).")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence in the prediction.")
    model_used: str = Field(..., description="Backend that produced the prediction.")
    expected_rating: Optional[float] = Field(
        None, description="Continuous rating estimate before rounding, when available."
    )
    probabilities: Optional[Dict[str, float]] = Field(
        None, description="Per-star class probabilities (transformer backend only)."
    )
    processing_time_ms: float = Field(..., ge=0.0, description="Inference wall time in milliseconds.")


class BatchPredictionResponse(BaseModel):
    """Rating predictions for a batch of reviews."""

    total: int = Field(..., ge=1, description="Number of predictions returned.")
    model_used: str = Field(..., description="Backend that produced the predictions.")
    predictions: List[PredictionResponse]


class BackendStatus(BaseModel):
    available: bool
    detail: str


class ModelInfoResponse(BaseModel):
    """Diagnostics about the active backend and model artifacts."""

    active_backend: str = Field(..., description="Backend currently serving predictions.")
    requested_backend: str
    backends: Dict[str, BackendStatus] = Field(
        ..., description="Availability of every backend option."
    )


class HealthResponse(BaseModel):
    """Liveness probe payload."""

    status: str
    app: str
    version: str
    active_backend: str

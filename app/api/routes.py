"""API endpoints for review rating prediction."""

from typing import Dict

from fastapi import APIRouter, Depends

from app.api.schemas import (
    BackendStatus,
    BatchPredictionResponse,
    BatchReviewRequest,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    ReviewRequest,
)
from app.core.config import get_settings
from app.core.model_loader import get_predictor
from app.services.predictor import PredictorService

router = APIRouter(prefix="/api/v1", tags=["reviews"])


def get_predictor_service() -> PredictorService:
    """FastAPI dependency exposing the singleton predictor."""
    return get_predictor()


@router.post("/predict", response_model=PredictionResponse)
def predict_review(
    request: ReviewRequest,
    predictor: PredictorService = Depends(get_predictor_service),
) -> PredictionResponse:
    """Predict a 1-5 star rating for a single review."""
    result = predictor.predict(request.text)
    return PredictionResponse(
        rating=result.rating,
        confidence=result.confidence,
        model_used=result.model_used,
        expected_rating=result.expected_rating,
        probabilities=result.probabilities,
        processing_time_ms=result.processing_time_ms,
    )


@router.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_reviews_batch(
    request: BatchReviewRequest,
    predictor: PredictorService = Depends(get_predictor_service),
) -> BatchPredictionResponse:
    """Predict 1-5 star ratings for up to 64 reviews at once."""
    results = predictor.predict_batch(request.texts)
    return BatchPredictionResponse(
        total=len(results),
        model_used=results[0].model_used if results else predictor.backend,
        predictions=[
            PredictionResponse(
                rating=result.rating,
                confidence=result.confidence,
                model_used=result.model_used,
                expected_rating=result.expected_rating,
                probabilities=result.probabilities,
                processing_time_ms=result.processing_time_ms,
            )
            for result in results
        ],
    )


@router.get("/model/info", response_model=ModelInfoResponse)
def model_info(
    predictor: PredictorService = Depends(get_predictor_service),
) -> ModelInfoResponse:
    """Report the active backend and availability of every backend option."""
    settings = get_settings()

    from app.models.distilbert import TRANSFORMERS_AVAILABLE, weights_look_trained

    distilbert_trained = weights_look_trained(settings.distilbert_weights_dir)
    ridge_trained = settings.ridge_model_path.is_file()

    backends: Dict[str, BackendStatus] = {
        "distilbert": BackendStatus(
            available=bool(TRANSFORMERS_AVAILABLE and distilbert_trained),
            detail=(
                "ready"
                if TRANSFORMERS_AVAILABLE and distilbert_trained
                else "dependencies missing (pip install torch transformers)"
                if not TRANSFORMERS_AVAILABLE
                else "no trained weights - run training/train_distilbert.py"
            ),
        ),
        "ridge": BackendStatus(
            available=ridge_trained,
            detail="ready" if ridge_trained else "no artifact - run training/train_ridge.py",
        ),
        "heuristic": BackendStatus(available=True, detail="lexicon baseline, always available"),
    }
    return ModelInfoResponse(
        active_backend=predictor.backend,
        requested_backend=settings.model_backend,
        backends=backends,
    )


@router.get("/health", response_model=HealthResponse)
def health(
    predictor: PredictorService = Depends(get_predictor_service),
) -> HealthResponse:
    """Service liveness probe including the active model backend."""
    settings = get_settings()
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        version=settings.app_version,
        active_backend=predictor.backend,
    )

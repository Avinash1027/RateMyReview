"""FastAPI application entry point for RateMyReview.

Run locally with::

    uvicorn app.main:app --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import get_settings
from app.core.model_loader import get_predictor

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise the prediction backend on startup."""
    predictor = get_predictor()
    print(f"RateMyReview API ready - active backend: '{predictor.backend}'")
    yield
    print("RateMyReview API shutting down")


app = FastAPI(
    title=f"{settings.app_name} API",
    version=settings.app_version,
    description=(
        "Automated sentiment-based rating prediction. Submit a review and get "
        "a predicted 1-5 star rating. Backed by a fine-tuned DistilBERT model "
        "with a TF-IDF + Ridge fallback for CPU-only deployments."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", tags=["meta"])
def root() -> dict:
    """Service index with links to the documentation."""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/api/v1/health",
        "endpoints": {
            "predict": "POST /api/v1/predict",
            "predict_batch": "POST /api/v1/predict/batch",
            "model_info": "GET /api/v1/model/info",
        },
    }


@app.get("/health", tags=["meta"])
def health() -> dict:
    """Quick liveness check."""
    return {"status": "ok"}

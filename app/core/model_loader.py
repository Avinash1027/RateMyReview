"""Model initialisation and backend selection.

``auto`` mode picks the best available backend at startup:

1. ``distilbert`` - requires torch/transformers installed AND a fine-tuned
   checkpoint under ``models/distilbert_weights``.
2. ``ridge``      - requires ``models/ridge_model.pkl``.
3. ``heuristic``  - always available lexicon baseline.

An explicitly requested backend that cannot be loaded raises
:class:`~app.models.base.BackendNotAvailableError` at startup.
"""

from functools import lru_cache
from typing import Optional

from app.core.config import get_settings
from app.models.base import BackendNotAvailableError
from app.models.tfidf_ridge import TfidfRidgeModel
from app.services.predictor import PredictorService

AUTO_BACKEND_ORDER = ("distilbert", "ridge", "heuristic")


def build_predictor(backend: str) -> PredictorService:
    """Construct a :class:`PredictorService` for an explicit backend.

    Raises:
        BackendNotAvailableError: if the backend cannot be initialised.
    """
    settings = get_settings()

    if backend == "distilbert":
        from app.models.distilbert import DistilBertRatingModel  # lazy heavy import

        model = DistilBertRatingModel.load(
            settings.distilbert_weights_dir,
            device=settings.device,
            max_length=settings.max_sequence_length,
        )
        return PredictorService("distilbert", model=model, batch_size=settings.prediction_batch_size)

    if backend == "ridge":
        model = TfidfRidgeModel.load(settings.ridge_model_path)
        return PredictorService("ridge", model=model)

    if backend == "heuristic":
        return PredictorService("heuristic")

    raise ValueError(
        f"Unknown backend {backend!r}. Valid options: auto, distilbert, ridge, heuristic"
    )


def try_build_predictor(backend: str) -> Optional[PredictorService]:
    """Like :func:`build_predictor` but returns ``None`` instead of raising."""
    try:
        return build_predictor(backend)
    except BackendNotAvailableError as exc:
        print(f"[model-loader] backend '{backend}' unavailable: {exc}")
        return None


@lru_cache
def get_predictor() -> PredictorService:
    """Return the cached predictor, resolving ``auto`` to the best backend."""
    settings = get_settings()
    requested = settings.model_backend

    if requested != "auto":
        return build_predictor(requested)  # explicit failures surface at startup

    for backend in AUTO_BACKEND_ORDER:
        predictor = try_build_predictor(backend)
        if predictor is not None:
            print(f"[model-loader] auto model selection resolved to '{backend}'")
            return predictor

    raise BackendNotAvailableError("No prediction backend could be initialised.")


def reset_predictor() -> None:
    """Clear cached predictors (used for configuration reloads)."""
    get_predictor.cache_clear()

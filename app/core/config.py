"""Application configuration.

All settings can be overridden through environment variables prefixed with
``RMR_`` (e.g. ``RMR_MODEL_BACKEND=ridge``) or through a ``.env`` file at the
project root.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BACKEND_CHOICES = ("auto", "distilbert", "ridge", "heuristic")
DEVICE_CHOICES = ("auto", "cpu", "cuda", "mps")


class Settings(BaseSettings):
    """Central application settings."""

    model_config = SettingsConfigDict(
        env_prefix="RMR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -- Application -----------------------------------------------------
    app_name: str = "RateMyReview"
    app_version: str = "1.0.0"
    debug: bool = False

    # -- Model selection -------------------------------------------------
    # auto       -> best available: distilbert -> ridge -> heuristic
    # distilbert -> transformer only (fails if unavailable)
    # ridge      -> TF-IDF + Ridge only
    # heuristic  -> lexicon baseline, always available
    model_backend: str = "auto"

    # -- Model artifacts -------------------------------------------------
    distilbert_weights_dir: Path = PROJECT_ROOT / "models" / "distilbert_weights"
    ridge_model_path: Path = PROJECT_ROOT / "models" / "ridge_model.pkl"

    # -- Transformer settings ---------------------------------------------
    distilbert_base_model: str = "distilbert-base-uncased"
    max_sequence_length: int = 256
    device: str = "auto"  # auto | cpu | cuda | mps
    prediction_batch_size: int = 32

    # -- Data --------------------------------------------------------------
    data_raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    data_processed_dir: Path = PROJECT_ROOT / "data" / "processed"

    # -- API ---------------------------------------------------------------
    cors_origins: list[str] = ["*"]

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings instance."""
    return Settings()

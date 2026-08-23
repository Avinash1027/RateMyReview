"""Shared utilities for rating prediction models."""

import math

RATING_MIN = 1
RATING_MAX = 5


class BackendNotAvailableError(RuntimeError):
    """Raised when a prediction backend cannot be initialised."""


def round_half_up(value: float) -> int:
    """Round a float to the nearest integer, halves away from zero."""
    return math.floor(float(value) + 0.5)


def clamp_rating(value: float) -> int:
    """Convert a continuous score to an integer star rating in [1, 5]."""
    return min(max(round_half_up(value), RATING_MIN), RATING_MAX)

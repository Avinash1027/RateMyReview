"""Shared dataset loading helpers for the training scripts."""

import sys
from pathlib import Path

import pandas as pd

TEXT_COLUMNS = ("text", "review", "review_text", "review_body", "content", "body", "comment", "summary")
RATING_COLUMNS = ("rating", "score", "stars", "rate", "label", "overall")

#: Expected output columns of :func:`load_reviews_csv`.
REQUIRED_COLUMNS = ("text", "rating")


def _find_column(frame: pd.DataFrame, candidates) -> str | None:
    """Find the first matching column, case-insensitively."""
    lookup = {str(col).lower().strip(): col for col in frame.columns}
    for candidate in candidates:
        actual = lookup.get(candidate)
        if actual is not None:
            return actual
    return None


def _resolve_project_root() -> Path:
    """Ensure the project root is importable when scripts run directly."""
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def load_reviews_csv(path: str | Path) -> pd.DataFrame:
    """Load a reviews CSV into a DataFrame with ``text`` and ``rating`` columns.

    Column names are auto-detected from common aliases. Ratings are expected
    on a 1-5 scale; 0-4 labelled datasets are shifted automatically. Rows
    with missing or out-of-range ratings are dropped.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Place a CSV with review text and "
            "rating columns under data/raw/ (see README for the expected format)."
        )

    frame = pd.read_csv(path)
    text_col = _find_column(frame, TEXT_COLUMNS)
    rating_col = _find_column(frame, RATING_COLUMNS)
    if text_col is None or rating_col is None:
        raise ValueError(
            f"Could not detect text/rating columns in {path}. "
            f"Found: {list(frame.columns)}. Expected text in {TEXT_COLUMNS} "
            f"and rating in {RATING_COLUMNS} (case-insensitive)."
        )

    frame = frame.rename(columns={text_col: "text", rating_col: "rating"})
    frame["rating"] = pd.to_numeric(frame["rating"], errors="coerce")

    before = len(frame)
    # Drop missing values before casting so NaN never becomes the string "nan".
    frame = frame.dropna(subset=["text", "rating"])
    frame = frame[frame["text"].astype(str).str.strip().str.len() > 0]

    min_rating, max_rating = frame["rating"].min(), frame["rating"].max()
    if min_rating >= 0 and max_rating <= 4:
        print(f"warning: ratings look 0-indexed (range {min_rating}-{max_rating}); shifting to 1-5 scale")
        frame["rating"] = frame["rating"] + 1

    frame = frame[frame["rating"].between(1, 5)]
    dropped = before - len(frame)
    if dropped:
        print(f"warning: dropped {dropped} rows with missing or out-of-range values")

    frame["rating"] = frame["rating"].astype(int)
    return frame.reset_index(drop=True)[list(REQUIRED_COLUMNS)]

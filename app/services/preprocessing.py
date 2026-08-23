"""Text preprocessing utilities.

The lightweight TF-IDF backend consumes fully cleaned text, while DistilBERT
prefers mostly raw text (its tokenizer handles casing and sub-words), so only
minimal normalisation is applied for the transformer path.
"""

import html
import re
from typing import Iterable, List

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_REPEATED_CHAR_RE = re.compile(r"(\w)\1{2,}")
_NON_WORD_RE = re.compile(r"[^a-z0-9\s.,!?'\-]")

# A small, common set of English contractions (enough for review text).
_CONTRACTIONS = {
    "can't": "can not",
    "won't": "will not",
    "n't": " not",
    "'re": " are",
    "'ve": " have",
    "'ll": " will",
    "'m": " am",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "what's": "what is",
    "let's": "let us",
}

_CONTRACTION_RE = re.compile(
    r"\b(" + "|".join(re.escape(c) for c in sorted(_CONTRACTIONS, key=len, reverse=True)) + r")\b"
)


def expand_contractions(text: str) -> str:
    """Expand common English contractions (``don't`` -> ``do not``)."""
    def _replace(match: re.Match) -> str:
        token = match.group(0).lower()
        return _CONTRACTIONS.get(token, match.group(0))

    return _CONTRACTION_RE.sub(_replace, text)


def clean_text(
    text: str,
    lowercase: bool = True,
    remove_urls: bool = True,
    strip_html: bool = True,
) -> str:
    """Normalise a raw review for bag-of-words style features.

    Steps: HTML unescape and tag stripping, URL removal, contraction
    expansion, repeated-character squashing (``soooo`` -> ``soo``), symbol
    cleanup and whitespace normalisation.
    """
    if not text:
        return ""

    cleaned = html.unescape(text)
    if strip_html:
        cleaned = _HTML_TAG_RE.sub(" ", cleaned)
    if remove_urls:
        cleaned = _URL_RE.sub(" ", cleaned)

    if lowercase:
        cleaned = cleaned.lower()

    cleaned = expand_contractions(cleaned)
    cleaned = _REPEATED_CHAR_RE.sub(r"\1\1", cleaned)
    cleaned = _NON_WORD_RE.sub(" ", cleaned)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    return cleaned


def batch_clean(texts: Iterable[str], **kwargs) -> List[str]:
    """Apply :func:`clean_text` to a collection of reviews."""
    return [clean_text(text, **kwargs) for text in texts]


def minimal_clean(text: str) -> str:
    """Light normalisation used before transformer tokenisation."""
    if not text:
        return ""
    cleaned = html.unescape(text)
    cleaned = _HTML_TAG_RE.sub(" ", cleaned)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    return cleaned

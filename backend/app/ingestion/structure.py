"""Generic section-path detection from loader-provided heading metadata."""

import re

from app.core.config import get_settings
from app.ingestion.extractors import TextSegment

_NUMBERED_HEADING = re.compile(r"^(?:\d+(?:\.\d+)*[.)]?|[A-Z][.)]|\([a-z0-9]+\))\s+\S")


def is_heading(text: str, parser_heading: bool = False) -> bool:
    """Classify a short standalone heading using parser evidence or generic form."""
    value = text.strip().lstrip("#* ")
    if parser_heading:
        return True
    if not value or len(value) > get_settings().ingestion_heading_max_chars:
        return False
    if _NUMBERED_HEADING.match(value):
        return True
    if value.endswith((".", "?", "!", ",", ";")):
        return False
    words = value.split()
    letters = [char for char in value if char.isalpha()]
    if 1 <= len(words) <= 10 and letters and all(char.isupper() for char in letters):
        return True
    title_case = sum(word[:1].isupper() for word in words if word[:1].isalpha())
    return 2 <= len(words) <= 8 and bool(words) and title_case / len(words) >= 0.75


def section_paths(segments: list[TextSegment]) -> list[tuple[TextSegment, tuple[str, ...]]]:
    """Attach a generic heading hierarchy to each extracted segment in order."""
    path: list[str] = []
    results: list[tuple[TextSegment, tuple[str, ...]]] = []
    for segment in segments:
        value = segment.heading.strip() if segment.heading else ""
        if value and value not in path:
            path = [value]
        elif not value and is_heading(segment.text):
            path = [segment.text.strip()]
        results.append((segment, tuple(path)))
    return results

"""Conservative, document-agnostic normalization helpers for extracted text."""

import re
import unicodedata
from collections import defaultdict

from app.core.config import get_settings
from app.ingestion.extractors import TextSegment

_ZERO_WIDTH = re.compile("[\\u200b-\\u200f\\u202a-\\u202e\\u2060\\ufeff]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([,.;:!?])")
_BULLET = re.compile(r"^\s*(?:[-*\u2022\u25aa\u25e6\u2023\u2043\uf0b7]|\d+[.)])\s+")
_PAGE_NUMBER = re.compile(r"^\s*(?:page\s+)?\d+(?:\s*(?:of|/)\s*\d+)?\s*$", re.I)


def normalize_unicode(text: str) -> str:
    """Normalize Unicode compatibility forms and remove invisible controls."""
    value = unicodedata.normalize("NFKC", text)
    value = _ZERO_WIDTH.sub("", value)
    value = _CONTROL.sub("", value)
    value = value.translate(
        str.maketrans(
            {
                "\u201c": '"',
                "\u201d": '"',
                "\u2018": "'",
                "\u2019": "'",
                "\u2013": "-",
                "\u2014": "-",
                "\u2212": "-",
            }
        )
    )
    replacements = {
        "\u00e2\u20ac\u0153": '"',
        "\u00e2\u20ac\u009d": '"',
        "\u00e2\u20ac\u02dc": "'",
        "\u00e2\u20ac\u2122": "'",
        "\u00e2\u20ac\u201c": "-",
        "\u00e2\u20ac\u201d": "-",
        "\u00e2\u02c6\u2012": "-",
    }
    for malformed, corrected in replacements.items():
        value = value.replace(malformed, corrected)
    return value


def normalize_spacing(text: str) -> str:
    """Remove extraction whitespace defects without changing token order."""
    value = _SPACE_BEFORE_PUNCTUATION.sub(r"\1", text)
    value = re.sub(r"[ \t]+", " ", value)
    return value.strip()


def normalize_hyphenation(text: str) -> str:
    """Join likely line-wrap hyphenation while preserving established compounds."""
    value = re.sub(r"(?<![\w-])([A-Za-z]{5,})-\s+(?=[a-z])", r"\1", text)
    # Preserve short prefixes and internal compound hyphens (for example, post- or man-in-).
    value = re.sub(r"(?<=\w)-\s+(?=[a-z])", "-", value)
    return value


def _line_to_segments(line: str, page: int | None, heading: str | None) -> list[TextSegment]:
    """Split explicit bullets and question-like glued headings conservatively."""
    line = normalize_spacing(normalize_hyphenation(normalize_unicode(line)))
    if not line:
        return []
    bullet = _BULLET.match(line)
    if bullet:
        marker = "\u2022" if not bullet.group().lstrip()[0].isdigit() else bullet.group().strip()
        return [TextSegment(f"{marker} {line[bullet.end() :]}", page, heading)]
    labels = sorted(get_settings().ingestion_metadata_labels, key=len, reverse=True)
    if labels:
        label_pattern = re.compile(
            r"(?<!\w)(" + "|".join(re.escape(label) for label in labels) + r")(?!\w)",
            re.IGNORECASE,
        )
        matches = list(label_pattern.finditer(line))
        if len(matches) >= 2 and matches[0].start() == 0:
            fields = [
                (match.group(1), line[match.end() : matches[index + 1].start()].strip())
                for index, match in enumerate(matches[:-1])
            ]
            fields.append((matches[-1].group(1), line[matches[-1].end() :].strip()))
            if all(value for _label, value in fields):
                return [TextSegment(f"{label}: {value}", page, heading) for label, value in fields]
    # A question following prose on the same extracted line is a new unit.
    split = re.search(r"(?<=[.!?])\s+(?=(?:[A-Z][^.!?]{0,100}\?)$)", line)
    if split:
        return [
            TextSegment(line[: split.start()].strip(), page, heading),
            TextSegment(line[split.end() :].strip(), page, line[split.end() :].strip()),
        ]
    return [TextSegment(line, page, heading)]


def normalize_segments_with_removed(
    segments: list[TextSegment],
) -> tuple[list[TextSegment], list[str]]:
    """Normalize extracted content and return repeated page furniture removed."""
    settings = get_settings()
    normalized_lines: list[TextSegment] = []
    for segment in segments:
        for raw_line in segment.text.splitlines() or [segment.text]:
            normalized_lines.extend(
                _line_to_segments(raw_line, segment.page_number, segment.heading)
            )

    joined_lines: list[TextSegment] = []
    for item in normalized_lines:
        if joined_lines:
            previous = joined_lines[-1]
            if (
                previous.page_number == item.page_number
                and previous.heading == item.heading
                and previous.text.endswith("-")
                and item.text[:1].islower()
                and item.text[:1].isalpha()
            ):
                left = previous.text[:-1]
                word_fragment = re.search(r"([A-Za-z]+)$", left)
                joiner = "" if word_fragment and len(word_fragment.group(1)) >= 5 else "-"
                joined_lines[-1] = TextSegment(
                    left + joiner + item.text, previous.page_number, previous.heading
                )
                continue
        joined_lines.append(item)
    normalized_lines = joined_lines

    pages = {item.page_number for item in normalized_lines if item.page_number is not None}
    line_pages: dict[str, set[int]] = defaultdict(set)
    for item in normalized_lines:
        normalized_line = normalize_spacing(item.text).casefold()
        if normalized_line and item.page_number is not None:
            line_pages[normalized_line].add(item.page_number)
    repeat_min = max(2, int(len(pages) * settings.ingestion_header_repeat_fraction + 0.999))
    repeated = {
        line for line, source_pages in line_pages.items() if len(source_pages) >= repeat_min
    }
    output: list[TextSegment] = []
    removed: set[str] = set()
    for item in normalized_lines:
        text = normalize_spacing(item.text)
        if _PAGE_NUMBER.fullmatch(text) or text.casefold() in repeated:
            removed.add(text)
            continue
        if not text:
            continue
        output.append(TextSegment(text, item.page_number, item.heading))
    return output, sorted(removed)


def normalize_segments(segments: list[TextSegment]) -> list[TextSegment]:
    """Clean extracted segments and remove repeated page furniture conservatively."""
    return normalize_segments_with_removed(segments)[0]

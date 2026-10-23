"""Fixture-level invariants for supported document formats."""

import re
from collections import Counter
from pathlib import Path

import fitz
import pytest

from app.ingestion.chunking import chunk_segments
from app.ingestion.extractors import validate_file
from app.ingestion.loaders import get_loader
from app.ingestion.normalize import normalize_segments

FIXTURES = Path(__file__).parent / "fixtures"
WORD = re.compile(r"[\w]+", re.UNICODE)


@pytest.mark.parametrize(
    "filename",
    [
        "plain_unstructured.txt",
        "handbook.md",
        "policy.html",
        "headings.docx",
        "repeating_furniture.pdf",
        "hyphenated_wrap.pdf",
    ],
)
def test_fixture_words_and_chunk_metadata_are_preserved(filename: str) -> None:
    """Cleaned fixture words should remain in output chunks with source metadata."""
    path = FIXTURES / filename
    content = path.read_bytes()
    validate_file(path.name, content)
    parsed = get_loader(path.name, content).load(path.name, content)
    segments = normalize_segments(parsed.to_segments())
    if filename == "hyphenated_wrap.pdf":
        assert "reconnaissance" in " ".join(segment.text for segment in segments)
        assert "man-in-the-middle" in " ".join(segment.text for segment in segments)
    chunks = chunk_segments(segments)
    clean_words = Counter(WORD.findall(" ".join(segment.text for segment in segments).casefold()))
    chunk_words = Counter(WORD.findall(" ".join(chunk.text for chunk in chunks).casefold()))
    assert clean_words == chunk_words
    assert all(chunk.text.strip() for chunk in chunks)
    assert all(chunk.heading is None or isinstance(chunk.heading, str) for chunk in chunks)


def test_scanned_pdf_fixture_is_image_only_and_uses_ocr(monkeypatch) -> None:
    """The scanned fixture should exercise OCR detection rather than native text."""
    content = (FIXTURES / "scanned_only.pdf").read_bytes()
    with fitz.open(stream=content, filetype="pdf") as pdf:
        assert pdf.page_count == 1
        assert not pdf[0].get_text().strip()
        assert pdf[0].get_images()
    from app.ingestion import extractors

    monkeypatch.setattr(extractors, "_ocr_page", lambda *_args, **_kwargs: "Scanned fixture text")
    parsed = get_loader("scanned_only.pdf", content).load("scanned_only.pdf", content)
    assert parsed.ocr_pages == (1,)
    assert any("Scanned fixture text" in block.text for block in parsed.blocks)

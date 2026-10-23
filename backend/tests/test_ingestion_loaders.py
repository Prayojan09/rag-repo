"""Contract tests for format-specific loaders and generic type sniffing."""

import pytest

from app.ingestion.extractors import IngestionError
from app.ingestion.loaders import get_loader
from app.ingestion.loaders.base import HTMLLoader, MarkdownLoader, sniff_document_type


def test_markdown_loader_preserves_heading_and_list_structure() -> None:
    """Markdown headings and list items should remain identifiable blocks."""
    parsed = MarkdownLoader().load("guide.md", b"# Guide\n\nIntro text.\n\n- First\n- Second\n")
    assert [(block.type, block.heading_level) for block in parsed.blocks] == [
        ("heading", 1),
        ("paragraph", None),
        ("list_item", None),
        ("list_item", None),
    ]


def test_html_loader_extracts_document_content_but_discards_navigation() -> None:
    """HTML loader should keep semantic content and skip scripts and navigation."""
    html = (
        b"<nav>menu text</nav><main><h1>Policy</h1><p>Keep this text.</p>"
        b"<script>secret()</script></main>"
    )
    parsed = HTMLLoader().load("policy.html", html)
    assert [block.text for block in parsed.blocks] == ["Policy", "Keep this text."]
    assert parsed.blocks[0].heading_level == 1


def test_loader_selection_uses_content_sniffing() -> None:
    """Recognizable HTML content should select the HTML loader."""
    assert sniff_document_type("upload.bin", b"<!doctype html><p>text</p>") == ".html"
    assert isinstance(get_loader("upload.html", b"<html><p>text</p></html>"), HTMLLoader)


def test_unknown_content_type_is_rejected() -> None:
    """Unknown file suffixes must fail with an actionable validation error."""
    with pytest.raises(IngestionError, match="Unsupported document type"):
        get_loader("unknown.xyz", b"binary")


def test_mismatched_extension_and_content_are_rejected() -> None:
    """A supported suffix cannot disguise a different binary content type."""
    with pytest.raises(IngestionError, match="extension does not match"):
        from app.ingestion.extractors import validate_file

        validate_file("document.txt", b"%PDF-1.7")

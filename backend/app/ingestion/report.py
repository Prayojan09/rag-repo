"""Build and persist a compact per-document ingestion quality report."""

import json
import logging
import statistics
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.ingestion.chunking import ChunkData
from app.ingestion.extractors import TextSegment
from app.ingestion.loaders.base import ParsedDocument

logger = logging.getLogger(__name__)


def build_quality_report(
    parsed: ParsedDocument,
    segments: list[TextSegment],
    chunks: list[ChunkData],
    *,
    filename: str,
    removed_header_footer_lines: list[str] | None = None,
) -> dict[str, Any]:
    """Summarize extraction and chunk quality without inventing unavailable metrics."""
    pages = {segment.page_number for segment in segments if segment.page_number is not None}
    headings = {segment.heading for segment in segments if segment.heading}
    settings = get_settings()
    tokenizer_name = chunks[0].tokenizer if chunks else "not applicable"
    sizes = [chunk.token_count for chunk in chunks]
    text_char_count = sum(len(segment.text) for segment in segments)
    page_count = len(pages) or parsed.page_count
    warnings: list[str] = []
    if tokenizer_name != "tiktoken" and chunks:
        warnings.append("exact token counts unavailable; tiktoken is not installed")
    if not headings:
        warnings.append("no headings detected")
    if (
        sizes
        and sum(size < settings.ingestion_tiny_chunk_tokens for size in sizes) / len(sizes)
        >= settings.ingestion_tiny_chunk_fraction
    ):
        warnings.append("many tiny chunks")
    if page_count and text_char_count / page_count < settings.ingestion_low_text_chars_per_page:
        warnings.append("low text density")
    return {
        "filename": filename,
        "page_count": page_count,
        "parser_used": parsed.parser_used,
        "ocr_pages": list(parsed.ocr_pages) if parsed.ocr_pages is not None else None,
        "headings_detected": len(headings),
        "chunk_count": len(chunks),
        "tokenizer": tokenizer_name,
        "token_size_min_median_max": {
            "min": min(sizes) if sizes else 0,
            "median": statistics.median(sizes) if sizes else 0,
            "max": max(sizes) if sizes else 0,
        },
        "removed_header_footer_lines": removed_header_footer_lines or [],
        "warnings": warnings,
    }


def save_quality_report(document_id: str, report: dict[str, Any]) -> Path:
    """Write a JSON quality report beneath the configured reports directory."""
    destination_dir = Path(get_settings().ingestion_report_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{document_id}.json"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(destination)
    logger.info("Document quality report saved", extra={"document_id": document_id})
    return destination


def delete_quality_report(document_id: str) -> None:
    """Remove one persisted report when its source document is deleted."""
    report_path = Path(get_settings().ingestion_report_dir) / f"{document_id}.json"
    try:
        report_path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning(
            "Unable to remove document quality report (error_type=%s)",
            type(exc).__name__,
            extra={"document_id": document_id},
        )

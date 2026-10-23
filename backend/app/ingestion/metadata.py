"""Stable source metadata and content hashing for uploaded documents."""

import hashlib
import logging
from pathlib import Path

from app.core.config import get_settings

logger = logging.getLogger(__name__)
_tokenizer_warning_logged = False


def document_sha256(content: bytes) -> str:
    """Return the SHA-256 digest of the original uploaded bytes."""
    return hashlib.sha256(content).hexdigest()


def document_title(filename: str, headings: list[str] | None = None) -> str:
    """Choose the first available document heading, falling back to its filename."""
    if headings:
        title = next((heading.strip() for heading in headings if heading.strip()), None)
        if title:
            return title
    return Path(filename).stem


def count_tokens(text: str) -> tuple[int, str]:
    """Count tokens with the configured tiktoken encoding, falling back explicitly."""
    global _tokenizer_warning_logged
    try:
        import tiktoken

        encoder = tiktoken.get_encoding(get_settings().ingestion_token_encoding)
        return len(encoder.encode_ordinary(text)), "tiktoken"
    except Exception as exc:
        # tiktoken downloads its encoding table lazily. Offline or restricted
        # deployments must still be able to ingest documents safely.
        if not _tokenizer_warning_logged:
            logger.warning(
                "Configured tokenizer unavailable; using word-count fallback (error_type=%s)",
                type(exc).__name__,
            )
            _tokenizer_warning_logged = True
        return len(text.split()), "word-count fallback"

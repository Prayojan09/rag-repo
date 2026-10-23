"""BM25 keyword retrieval over ready PostgreSQL chunks."""

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import Document, DocumentChunk

TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Case-fold text and return Unicode word tokens, retaining numbers."""
    return TOKEN_PATTERN.findall(text.casefold())


@dataclass(frozen=True)
class KeywordMatch:
    chunk_id: str
    document_id: str
    score: float


class BM25CorpusLimitError(RuntimeError):
    """Raised rather than allocating an unbounded per-request keyword corpus."""


def search_bm25(
    db: Session,
    query: str,
    top_k: int,
    document_ids: list[str] | None = None,
    tenant_id: str = "default",
) -> list[KeywordMatch]:
    """Rank ready chunks with Okapi BM25 and optionally scope to documents."""
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    statement = (
        select(DocumentChunk, Document)
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(Document.status == "ready", Document.tenant_id == tenant_id)
        .order_by(Document.id, DocumentChunk.ordinal)
        .limit(get_settings().bm25_max_chunks + 1)
    )
    if document_ids is not None:
        if not document_ids:
            return []
        statement = statement.where(Document.id.in_(document_ids))

    rows = db.execute(statement).all()
    if len(rows) > get_settings().bm25_max_chunks:
        raise BM25CorpusLimitError(
            "Tenant BM25 corpus exceeds the configured per-request chunk limit"
        )
    if not rows:
        return []

    tokenized_corpus = [
        tokenize(chunk.embed_text or f"{chunk.heading or ''} {chunk.text}")
        for chunk, _document in rows
    ]
    if not any(tokenized_corpus):
        return []
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query_tokens)
    query_terms = set(query_tokens)
    ranked = [
        (float(score), chunk.id, document.id)
        for (chunk, document), tokens, score in zip(rows, tokenized_corpus, scores, strict=True)
        if query_terms.intersection(tokens)
    ]
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [
        KeywordMatch(chunk_id=chunk_id, document_id=document_id, score=score)
        for score, chunk_id, document_id in ranked[:top_k]
    ]

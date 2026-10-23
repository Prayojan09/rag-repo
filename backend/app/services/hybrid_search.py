"""Dense, BM25 keyword, and Reciprocal Rank Fusion search orchestration."""

from typing import Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Document, DocumentChunk
from app.retrieval.bm25 import KeywordMatch, search_bm25
from app.retrieval.dense import DenseVectorService
from app.schemas.documents import DenseSearchResult
from app.services.dense_search import dense_search

SearchMode = Literal["dense", "keyword", "hybrid"]


def search(
    db: Session,
    query: str,
    top_k: int,
    document_ids: list[str] | None = None,
    *,
    mode: SearchMode = "hybrid",
    candidate_multiplier: int = 4,
    candidate_max: int = 200,
    rrf_constant: int = 60,
    vector_service: DenseVectorService | None = None,
    tenant_id: str = "default",
) -> list[DenseSearchResult]:
    """Run one retrieval mode; hybrid mode fuses independently ranked lists."""
    if mode == "dense":
        return dense_search(db, query, top_k, document_ids, vector_service, tenant_id)

    if mode == "keyword":
        keyword_matches = search_bm25(db, query, top_k, document_ids, tenant_id)
        if not keyword_matches:
            return []
        sources = _resolve_sources(db, keyword_matches, tenant_id)
        results = []
        for rank, match in enumerate(keyword_matches, start=1):
            source = sources.get(match.chunk_id)
            if source is None or source.document_id != match.document_id:
                continue
            results.append(
                source.model_copy(
                    update={
                        "score": match.score,
                        "keyword_score": match.score,
                        "keyword_rank": rank,
                    }
                )
            )
        return results

    candidate_k = min(max(top_k, top_k * candidate_multiplier), candidate_max)
    dense_results = dense_search(db, query, candidate_k, document_ids, vector_service, tenant_id)
    keyword_matches = search_bm25(db, query, candidate_k, document_ids, tenant_id)
    keyword_sources = _resolve_sources(db, keyword_matches, tenant_id)

    dense_by_id: dict[str, tuple[int, DenseSearchResult]] = {
        result.chunk_id: (rank, result) for rank, result in enumerate(dense_results, start=1)
    }
    keyword_by_id: dict[str, tuple[int, DenseSearchResult, float]] = {}
    for rank, match in enumerate(keyword_matches, start=1):
        source = keyword_sources.get(match.chunk_id)
        if source and source.document_id == match.document_id:
            keyword_by_id[match.chunk_id] = (rank, source, match.score)

    fused: list[tuple[float, DenseSearchResult]] = []
    for chunk_id in dense_by_id.keys() | keyword_by_id.keys():
        dense = dense_by_id.get(chunk_id)
        keyword = keyword_by_id.get(chunk_id)
        fused_score = 0.0
        if dense:
            fused_score += 1 / (rrf_constant + dense[0])
        if keyword:
            fused_score += 1 / (rrf_constant + keyword[0])
        source = dense[1] if dense else keyword[1]
        fused.append(
            (
                fused_score,
                source.model_copy(
                    update={
                        "score": fused_score,
                        "dense_score": dense[1].score if dense else None,
                        "keyword_score": keyword[2] if keyword else None,
                        "dense_rank": dense[0] if dense else None,
                        "keyword_rank": keyword[0] if keyword else None,
                    }
                ),
            )
        )
    fused.sort(key=lambda item: (-item[0], item[1].chunk_id))
    return [result for _score, result in fused[:top_k]]


def _resolve_sources(
    db: Session, matches: list[KeywordMatch], tenant_id: str = "default"
) -> dict[str, DenseSearchResult]:
    """Load authoritative text and metadata for ranked chunk IDs."""
    chunk_ids = [match.chunk_id for match in matches]
    if not chunk_ids:
        return {}
    rows = db.execute(
        select(DocumentChunk, Document)
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(
            DocumentChunk.id.in_(chunk_ids),
            Document.status == "ready",
            Document.tenant_id == tenant_id,
        )
    ).all()
    return {
        chunk.id: DenseSearchResult(
            chunk_id=chunk.id,
            document_id=document.id,
            filename=document.filename,
            version=document.version,
            ordinal=chunk.ordinal,
            page_number=chunk.page_number,
            heading=chunk.heading,
            text=chunk.text,
            score=0.0,
        )
        for chunk, document in rows
    }

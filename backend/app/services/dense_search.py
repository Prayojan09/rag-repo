"""Dense retrieval with SQL-backed source validation for Qdrant hits."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Document, DocumentChunk
from app.retrieval.dense import DenseVectorService, get_dense_vector_service
from app.schemas.documents import DenseSearchResult


def dense_search(
    db: Session,
    query: str,
    top_k: int,
    document_ids: list[str] | None = None,
    vector_service: DenseVectorService | None = None,
    tenant_id: str = "default",
) -> list[DenseSearchResult]:
    service = vector_service or get_dense_vector_service()
    allowed_document_ids = document_ids
    if document_ids is not None:
        allowed_document_ids = list(
            db.scalars(
                select(Document.id).where(
                    Document.id.in_(document_ids),
                    Document.status == "ready",
                    Document.tenant_id == tenant_id,
                )
            ).all()
        )
        if not allowed_document_ids:
            return []

    matches = service.search(query, top_k, allowed_document_ids, tenant_id)
    if not matches:
        return []
    chunks_by_id = {
        chunk.id: (chunk, document)
        for chunk, document in db.execute(
            select(DocumentChunk, Document)
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(
                DocumentChunk.id.in_([match.chunk_id for match in matches]),
                Document.status == "ready",
                Document.tenant_id == tenant_id,
            )
        ).all()
    }
    results = []
    for match in matches:
        source = chunks_by_id.get(match.chunk_id)
        if source is None:
            continue
        chunk, document = source
        if document.id != match.document_id:
            continue
        results.append(
            DenseSearchResult(
                chunk_id=chunk.id,
                document_id=document.id,
                filename=document.filename,
                version=document.version,
                ordinal=chunk.ordinal,
                page_number=chunk.page_number,
                heading=chunk.heading,
                text=chunk.text,
                score=match.score,
            )
        )
    return results

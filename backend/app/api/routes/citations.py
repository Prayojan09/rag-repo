"""Resolve a citation chunk ID to its authoritative indexed source."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.security import APIPrincipal, require_api_token
from app.db.models import Document, DocumentChunk
from app.db.session import get_db
from app.schemas.documents import CitationDetail

router = APIRouter(
    prefix="/api/v1/citations",
    tags=["citations"],
    dependencies=[Depends(require_api_token)],
)


@router.get("/{citation_id}", response_model=CitationDetail)
def get_citation(
    citation_id: UUID,
    db: Annotated[Session, Depends(get_db)],
    principal: Annotated[APIPrincipal, Depends(require_api_token)],
) -> CitationDetail:
    row = db.execute(
        select(DocumentChunk, Document)
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(
            DocumentChunk.id == str(citation_id),
            Document.status == "ready",
            Document.tenant_id == principal.tenant_id,
        )
    ).one_or_none()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Citation not found")

    chunk, document = row
    return CitationDetail(
        chunk_id=chunk.id,
        document_id=document.id,
        filename=document.filename,
        version=document.version,
        ordinal=chunk.ordinal,
        page_number=chunk.page_number,
        heading=chunk.heading,
        text=chunk.text,
    )

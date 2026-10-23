"""Document upload, status, retry, listing, and deletion endpoints."""

from pathlib import Path
from typing import Annotated

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.core.config import get_settings
from app.core.security import APIPrincipal, require_api_token
from app.db.models import Document
from app.db.session import get_db
from app.ingestion.extractors import IngestionError
from app.schemas.documents import DocumentDetail, DocumentList, DocumentSummary
from app.services.document_ingestion import (
    DuplicateDocumentError,
    delete_document,
    get_document,
    process_queued_document,
    queue_document,
    reindex_document,
    retry_document,
)

router = APIRouter(
    prefix="/api/v1/documents",
    tags=["documents"],
    dependencies=[Depends(require_api_token)],
)


def _storage_root() -> Path:
    return Path(get_settings().upload_dir).resolve()


@router.post("", response_model=DocumentSummary, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: Annotated[UploadFile, File()],
    background_tasks: BackgroundTasks,
    db: Annotated[Session, Depends(get_db)],
    principal: Annotated[APIPrincipal, Depends(require_api_token)],
) -> Document:
    settings = get_settings()
    filename = file.filename or ""
    content = await file.read(settings.max_upload_bytes + 1)
    await file.close()
    try:
        document = await run_in_threadpool(
            queue_document,
            db=db,
            filename=filename,
            content=content,
            storage_root=_storage_root(),
            max_bytes=settings.max_upload_bytes,
            tenant_id=principal.tenant_id,
        )
        background_tasks.add_task(
            process_queued_document,
            document.id,
            _storage_root(),
            settings.chunk_size_chars,
            settings.chunk_overlap_chars,
        )
        return document
    except DuplicateDocumentError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": "An identical file is already indexed",
                "document_id": exc.document_id,
            },
        ) from exc
    except IngestionError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc


@router.get("", response_model=DocumentList)
def list_documents(
    db: Annotated[Session, Depends(get_db)],
    principal: Annotated[APIPrincipal, Depends(require_api_token)],
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DocumentList:
    tenant_filter = Document.tenant_id == principal.tenant_id
    total = db.scalar(select(func.count()).select_from(Document).where(tenant_filter)) or 0
    documents = db.scalars(
        select(Document)
        .where(tenant_filter)
        .order_by(Document.created_at.desc(), Document.id)
        .limit(limit)
        .offset(offset)
    ).all()
    return DocumentList(items=documents, total=total, limit=limit, offset=offset)


@router.get("/{document_id}", response_model=DocumentDetail)
def read_document(
    document_id: str,
    db: Annotated[Session, Depends(get_db)],
    principal: Annotated[APIPrincipal, Depends(require_api_token)],
) -> Document:
    document = get_document(db, document_id, principal.tenant_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


@router.post("/{document_id}/retry", response_model=DocumentSummary)
def retry_ingestion(
    document_id: str,
    db: Annotated[Session, Depends(get_db)],
    principal: Annotated[APIPrincipal, Depends(require_api_token)],
) -> Document:
    document = db.scalar(
        select(Document).where(
            Document.id == document_id, Document.tenant_id == principal.tenant_id
        )
    )
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    settings = get_settings()
    try:
        return retry_document(
            db,
            document,
            _storage_root(),
            settings.chunk_size_chars,
            settings.chunk_overlap_chars,
        )
    except IngestionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/{document_id}/reindex", response_model=DocumentSummary)
def reindex_existing_document(
    document_id: str,
    db: Annotated[Session, Depends(get_db)],
    principal: Annotated[APIPrincipal, Depends(require_api_token)],
) -> Document:
    document = db.scalar(
        select(Document).where(
            Document.id == document_id, Document.tenant_id == principal.tenant_id
        )
    )
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    try:
        return reindex_document(db, document)
    except IngestionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_document(
    document_id: str,
    db: Annotated[Session, Depends(get_db)],
    principal: Annotated[APIPrincipal, Depends(require_api_token)],
) -> None:
    document = db.scalar(
        select(Document).where(
            Document.id == document_id, Document.tenant_id == principal.tenant_id
        )
    )
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    delete_document(db, document, _storage_root())

"""Typed adapter between ingestion orchestration and the vector-store interface."""

from collections.abc import Sequence
from typing import Protocol

from app.db.models import DocumentChunk


class VectorStore(Protocol):
    """Interface required by the ingestion indexer for dense vector writes."""

    def index_document(
        self,
        document_id: str,
        filename: str,
        version: int,
        chunks: Sequence[DocumentChunk],
        tenant_id: str = "default",
    ) -> None:
        """Replace a document's vectors with the supplied current chunk set."""


class DocumentIndexer:
    """Injectable dense-vector indexing coordinator."""

    def __init__(self, vector_store: VectorStore) -> None:
        self._vector_store = vector_store

    def index(
        self,
        document_id: str,
        filename: str,
        version: int,
        chunks: Sequence[DocumentChunk],
        tenant_id: str = "default",
    ) -> None:
        """Write the current document chunk set to its vector store."""
        self._vector_store.index_document(
            document_id=document_id,
            filename=filename,
            version=version,
            chunks=chunks,
            tenant_id=tenant_id,
        )

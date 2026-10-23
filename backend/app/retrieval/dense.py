"""Sentence-transformer embeddings and Qdrant dense vector operations."""

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from threading import RLock
from typing import Protocol

from qdrant_client import QdrantClient, models

from app.core.config import get_settings

logger = logging.getLogger(__name__)
QDRANT_UPSERT_BATCH_SIZE = 256


class Embedder(Protocol):
    model_name: str

    def dimension(self) -> int: ...

    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    """Lazy-load the configured Sentence Transformers model on first use."""

    def __init__(self) -> None:
        settings = get_settings()
        self.model_name = settings.embedding_model
        self._revision = settings.embedding_model_revision
        self._device = settings.embedding_device
        self._batch_size = settings.embedding_batch_size
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name,
                revision=self._revision,
                device=self._device,
            )
        return self._model

    def dimension(self) -> int:
        dimension = self._get_model().get_sentence_embedding_dimension()
        if not dimension:
            raise RuntimeError("The configured embedding model has no sentence embedding dimension")
        return int(dimension)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._get_model().encode(
            list(texts),
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()


@dataclass(frozen=True)
class VectorMatch:
    chunk_id: str
    document_id: str
    score: float


class DenseVectorService:
    """Index chunks and retrieve their nearest neighbors from Qdrant."""

    def __init__(
        self,
        client: QdrantClient | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        settings = get_settings()
        self.client = client or QdrantClient(
            url=settings.qdrant_url,
            api_key=(
                settings.qdrant_api_key.get_secret_value()
                if settings.qdrant_api_key
                else None
            ),
            timeout=settings.qdrant_timeout_seconds,
        )
        self.embedder = embedder or SentenceTransformerEmbedder()
        model_key = f"{settings.embedding_model}@{settings.embedding_model_revision}"
        model_suffix = sha256(model_key.encode("utf-8")).hexdigest()[:10]
        self.collection_name = f"{settings.qdrant_collection}_{model_suffix}"
        self._collection_lock = RLock()

    @property
    def model_name(self) -> str:
        return self.embedder.model_name

    def _ensure_collection(self) -> None:
        with self._collection_lock:
            if self.client.collection_exists(self.collection_name):
                info = self.client.get_collection(self.collection_name)
                vector_config = info.config.params.vectors
                dimension = getattr(vector_config, "size", None)
                if dimension is None and isinstance(vector_config, dict):
                    dimension = getattr(vector_config.get(""), "size", None)
                if dimension != self.embedder.dimension():
                    raise RuntimeError(
                        "Qdrant collection dimensions do not match the configured embedding model; "
                        "use a new collection name or reindex into an empty collection"
                    )
                self._ensure_payload_indexes(info.payload_schema)
                return
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.embedder.dimension(), distance=models.Distance.COSINE
                ),
            )
            self._ensure_payload_indexes({})

    def _ensure_payload_indexes(self, existing_indexes: Mapping[str, object]) -> None:
        """Create indexes required by document and tenant filters, if missing."""
        for field_name in ("document_id", "tenant_id"):
            if field_name not in existing_indexes:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                    wait=True,
                )

    def index_document(
        self,
        document_id: str,
        filename: str,
        version: int,
        chunks: Sequence[object],
        tenant_id: str = "default",
    ) -> None:
        started_at = time.perf_counter()
        self._ensure_collection()
        collection_ready_at = time.perf_counter()
        self.delete_document(document_id, tenant_id)
        delete_done_at = time.perf_counter()
        batch_size = get_settings().embedding_batch_size
        embedding_seconds = 0.0
        qdrant_write_seconds = 0.0
        pending_points: list[models.PointStruct] = []

        def flush_points() -> None:
            nonlocal qdrant_write_seconds
            if not pending_points:
                return
            write_started = time.perf_counter()
            self.client.upsert(
                collection_name=self.collection_name,
                points=pending_points,
                wait=True,
            )
            qdrant_write_seconds += time.perf_counter() - write_started
            pending_points.clear()

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            encode_started = time.perf_counter()
            vectors = self.embedder.encode(
                [str(getattr(chunk, "embed_text", None) or chunk.text) for chunk in batch]
            )
            embedding_seconds += time.perf_counter() - encode_started
            pending_points.extend(
                models.PointStruct(
                    id=str(chunk.id),
                    vector=vector,
                    payload={
                        "document_id": document_id,
                        "tenant_id": tenant_id,
                        "chunk_id": str(chunk.id),
                        "filename": filename,
                        "version": version,
                    },
                )
                for chunk, vector in zip(batch, vectors, strict=True)
            )
            if len(pending_points) >= QDRANT_UPSERT_BATCH_SIZE:
                flush_points()
        flush_points()
        logger.info(
            "Dense vector indexing timings (chunks=%d collection_setup_seconds=%.2f "
            "delete_seconds=%.2f embedding_seconds=%.2f qdrant_write_seconds=%.2f "
            "total_seconds=%.2f)",
            len(chunks),
            collection_ready_at - started_at,
            delete_done_at - collection_ready_at,
            embedding_seconds,
            qdrant_write_seconds,
            time.perf_counter() - started_at,
            extra={"document_id": document_id},
        )

    def delete_document(self, document_id: str, tenant_id: str = "default") -> None:
        if not self.client.collection_exists(self.collection_name):
            return
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id", match=models.MatchValue(value=document_id)
                        ),
                        self._tenant_condition(tenant_id),
                    ]
                )
            ),
            wait=True,
        )

    def search(
        self,
        query: str,
        top_k: int,
        document_ids: Sequence[str] | None = None,
        tenant_id: str = "default",
    ) -> list[VectorMatch]:
        self._ensure_collection()
        must_conditions = [self._tenant_condition(tenant_id)]
        if document_ids is not None:
            if not document_ids:
                return []
            must_conditions.append(
                models.FieldCondition(
                    key="document_id", match=models.MatchAny(any=list(document_ids))
                )
            )
        query_filter = models.Filter(must=must_conditions)
        query_vector = self.embedder.encode([query])[0]
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
        return [
            VectorMatch(
                chunk_id=str(point.payload["chunk_id"]),
                document_id=str(point.payload["document_id"]),
                score=float(point.score),
            )
            for point in response.points
            if point.payload and point.payload.get("chunk_id") and point.payload.get("document_id")
        ]

    @staticmethod
    def _tenant_condition(tenant_id: str):
        tenant_match = models.FieldCondition(
            key="tenant_id", match=models.MatchValue(value=tenant_id)
        )
        if tenant_id == "default":
            # Existing default-tenant vectors predate tenant payloads; they remain
            # visible only to this tenant and are still checked against PostgreSQL.
            return models.Filter(
                should=[
                    tenant_match,
                    models.IsEmptyCondition(is_empty=models.PayloadField(key="tenant_id")),
                ]
            )
        return tenant_match


@lru_cache
def get_dense_vector_service() -> DenseVectorService:
    return DenseVectorService()

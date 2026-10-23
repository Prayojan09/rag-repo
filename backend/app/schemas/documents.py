"""Request and response schemas for the document library."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DocumentSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    media_type: str
    sha256: str
    size_bytes: int
    version: int
    status: str
    retry_count: int
    error_message: str | None
    created_at: datetime
    updated_at: datetime
    quality_report: dict | None = None


class ChunkResponse(BaseModel):
    id: str
    ordinal: int
    text: str
    page_number: int | None
    heading: str | None
    section_path: list[str] | None = None
    token_count: int | None = None


class CitationDetail(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    version: int
    ordinal: int
    page_number: int | None
    heading: str | None
    text: str


class DocumentDetail(DocumentSummary):
    structured_content: str | None = None
    chunks: list[ChunkResponse] = Field(default_factory=list)


class DocumentList(BaseModel):
    items: list[DocumentSummary]
    total: int
    limit: int
    offset: int


class DenseSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=100)
    document_ids: list[str] | None = Field(default=None, max_length=100)
    mode: Literal["dense", "keyword", "hybrid"] = "hybrid"
    rerank: bool = False

    @field_validator("query")
    @classmethod
    def require_non_whitespace_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Query must contain non-whitespace characters")
        return value


class DenseSearchResult(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    version: int
    ordinal: int
    page_number: int | None
    heading: str | None
    text: str
    score: float
    dense_score: float | None = None
    keyword_score: float | None = None
    dense_rank: int | None = None
    keyword_rank: int | None = None
    retrieval_score: float | None = None
    rerank_score: float | None = None


class DenseSearchResponse(BaseModel):
    query: str
    mode: str = "dense"
    embedding_model: str | None
    reranker_model: str | None = None
    results: list[DenseSearchResult]

"""Structured chat request, model payload, and citation response schemas."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.schemas.documents import DenseSearchResult


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    model: str | None = Field(default=None, min_length=1, max_length=200)
    document_ids: list[str] | None = Field(default=None, max_length=100)
    mode: Literal["dense", "keyword", "hybrid"] = "hybrid"
    top_k: int | None = Field(default=None, ge=1, le=30)
    rerank: bool = False
    conversation_id: str | None = Field(default=None, min_length=36, max_length=36)

    @model_validator(mode="after")
    def trim_and_validate_question(self) -> "ChatRequest":
        self.question = self.question.strip()
        if not self.question:
            raise ValueError("Question must contain non-whitespace characters")
        return self


class GroundedClaim(BaseModel):
    text: str = Field(min_length=1, max_length=3000)
    source_chunk_ids: list[str] = Field(min_length=1, max_length=20)


class EvidenceConflict(BaseModel):
    description: str = Field(min_length=1, max_length=2000)
    source_chunk_ids: list[str] = Field(min_length=2, max_length=20)


class GeneratedAnswer(BaseModel):
    answer: str = Field(min_length=1, max_length=10000)
    abstained: bool
    abstention_reason: str | None = Field(default=None, max_length=1000)
    claims: list[GroundedClaim] = Field(default_factory=list, max_length=50)
    conflicts: list[EvidenceConflict] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def validate_answer_state(self) -> "GeneratedAnswer":
        if self.abstained and not self.abstention_reason:
            raise ValueError("Abstained answers must include an abstention reason")
        if not self.abstained and not self.claims:
            raise ValueError("Non-abstained answers must include at least one cited claim")
        return self


class ChatCitation(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    version: int
    ordinal: int
    page_number: int | None
    heading: str | None
    text: str

    @classmethod
    def from_search_result(cls, source: DenseSearchResult) -> "ChatCitation":
        return cls(
            chunk_id=source.chunk_id,
            document_id=source.document_id,
            filename=source.filename,
            version=source.version,
            ordinal=source.ordinal,
            page_number=source.page_number,
            heading=source.heading,
            text=source.text,
        )


class ChatResponse(BaseModel):
    conversation_id: str | None = None
    answer_id: str
    question: str
    answer: str
    abstained: bool
    abstention_reason: str | None
    claims: list[GroundedClaim]
    conflicts: list[EvidenceConflict]
    citations: list[ChatCitation]
    evidence_verified: bool
    verification_model: str | None
    unverified_claim_count: int = Field(default=0, ge=0)
    llm_prompt_tokens: int | None = Field(default=None, ge=0)
    llm_completion_tokens: int | None = Field(default=None, ge=0)
    retrieval_mode: str
    reranker_model: str | None
    llm_provider: str
    llm_model: str | None

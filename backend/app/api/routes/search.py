"""Semantic dense search endpoint."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.security import APIPrincipal, require_api_token
from app.db.session import get_db
from app.retrieval.dense import get_dense_vector_service
from app.retrieval.reranker import get_reranker
from app.schemas.documents import DenseSearchRequest, DenseSearchResponse
from app.services.hybrid_search import search

router = APIRouter(
    prefix="/api/v1/search",
    tags=["search"],
    dependencies=[Depends(require_api_token)],
)
logger = logging.getLogger(__name__)


@router.post("", response_model=DenseSearchResponse)
def search_documents(
    http_request: Request,
    payload: DenseSearchRequest,
    db: Annotated[Session, Depends(get_db)],
    principal: Annotated[APIPrincipal, Depends(require_api_token)],
) -> DenseSearchResponse:
    settings = get_settings()
    top_k = payload.top_k or settings.search_default_top_k
    if top_k > settings.search_max_top_k:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"top_k must not exceed {settings.search_max_top_k}",
        )
    try:
        service = get_dense_vector_service() if payload.mode != "keyword" else None
        retrieval_k = top_k
        if payload.rerank:
            retrieval_k = min(
                max(top_k, settings.reranker_candidate_k),
                settings.search_candidate_max,
            )
        results = search(
            db,
            payload.query,
            retrieval_k,
            payload.document_ids,
            mode=payload.mode,
            candidate_multiplier=settings.search_candidate_multiplier,
            candidate_max=settings.search_candidate_max,
            rrf_constant=settings.rrf_constant,
            vector_service=service,
            tenant_id=principal.tenant_id,
        )
        reranker_model = None
        if payload.rerank and results:
            reranker = get_reranker()
            results = reranker.rerank(payload.query, results, top_k)
            reranker_model = reranker.model_name
        return DenseSearchResponse(
            query=payload.query,
            mode=payload.mode,
            embedding_model=service.model_name if service else None,
            reranker_model=reranker_model,
            results=results,
        )
    except Exception as exc:
        request_id = getattr(http_request.state, "request_id", "unknown")
        logger.exception(
            "Search failed (request_id=%s error_type=%s)",
            request_id,
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Search is temporarily unavailable. Check the backend terminal for "
                f"request ID {request_id}."
            ),
        ) from exc

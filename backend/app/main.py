"""FastAPI application entry point."""

import logging
import re
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.api.routes.auth import router as auth_router
from app.api.routes.chat import router as chat_router
from app.api.routes.citations import router as citations_router
from app.api.routes.conversations import router as conversations_router
from app.api.routes.documents import router as documents_router
from app.api.routes.feedback import router as feedback_router
from app.api.routes.health import router as health_router
from app.api.routes.search import router as search_router
from app.core.config import get_settings
from app.core.errors import register_error_handlers
from app.core.logging import configure_logging
from app.retrieval.dense import get_dense_vector_service

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load the embedding model before the first upload or search needs it."""
    try:
        await run_in_threadpool(get_dense_vector_service().embedder.dimension)
        logger.info("Embedding model warmup completed")
    except Exception as exc:
        # Keep health/docs available if model hosting is temporarily unavailable.
        logger.warning("Embedding model warmup failed (error_type=%s)", type(exc).__name__)
    yield


app = FastAPI(
    title="Document Intelligence API",
    description="Document management and ingestion API for evidence-grounded question answering.",
    version="0.2.0",
    lifespan=lifespan,
    docs_url="/docs" if get_settings().environment == "development" else None,
    redoc_url="/redoc" if get_settings().environment == "development" else None,
    openapi_url="/openapi.json" if get_settings().environment == "development" else None,
)
if get_settings().environment in {"staging", "production"}:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=[
            host.strip() for host in get_settings().allowed_hosts.split(",") if host.strip()
        ],
    )
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in get_settings().cors_allowed_origins.split(",")
        if origin.strip()
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID", "X-CSRF-Token"],
)
app.include_router(health_router)
app.include_router(auth_router)
app.include_router(documents_router)
app.include_router(search_router)
app.include_router(chat_router)
app.include_router(conversations_router)
app.include_router(citations_router)
app.include_router(feedback_router)
register_error_handlers(app)


@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    supplied = request.headers.get("X-Request-ID", "")
    request_id = supplied if re.fullmatch(r"[A-Za-z0-9._-]{1,64}", supplied) else str(uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response

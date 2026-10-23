"""Liveness and dependency readiness endpoints."""

import logging

import httpx
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.core.config import get_settings
from app.db.session import engine

router = APIRouter(tags=["system"])
logger = logging.getLogger(__name__)


@router.get("/health", summary="Application liveness")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready", summary="Dependency readiness")
def readiness() -> JSONResponse:
    settings = get_settings()
    checks: dict[str, str] = {}
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        checks["postgres"] = "ok"
    except Exception:
        logger.warning("PostgreSQL readiness check failed")
        checks["postgres"] = "unavailable"

    headers = (
        {"api-key": settings.qdrant_api_key.get_secret_value()}
        if settings.qdrant_api_key
        else {}
    )
    try:
        response = httpx.get(
            f"{settings.qdrant_url.rstrip('/')}/readyz",
            headers=headers,
            timeout=settings.request_timeout_seconds,
        )
        response.raise_for_status()
        checks["qdrant"] = "ok"
    except (httpx.HTTPError, ValueError):
        logger.warning("Qdrant readiness check failed")
        checks["qdrant"] = "unavailable"

    ready = all(value == "ok" for value in checks.values())
    return JSONResponse(
        status_code=200 if ready else 503,
        content={"status": "ready" if ready else "not_ready", "checks": checks},
    )

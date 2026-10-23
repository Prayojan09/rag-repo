"""Bearer authentication and tenant scoping for API endpoints."""

import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import AuthSession, AuthUser
from app.db.session import get_db

bearer_scheme = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class APIPrincipal:
    """Authenticated tenant scope derived only from configured credentials."""

    tenant_id: str
    user_id: str | None = None
    email: str | None = None
    display_name: str | None = None
    picture_url: str | None = None
    csrf_token: str | None = None


def require_api_token(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(bearer_scheme),
    ],
    request: Request,
    db: Annotated[Session, Depends(get_db)],
) -> APIPrincipal:
    """Authenticate a bearer token or secure browser session and scope its tenant."""
    settings = get_settings()
    supplied_token = credentials.credentials if credentials else ""

    matched_tenant = None
    for tenant_id, tenant_token in settings.api_tenant_tokens.items():
        if secrets.compare_digest(supplied_token, tenant_token.get_secret_value()):
            matched_tenant = tenant_id
    if matched_tenant is not None:
        return APIPrincipal(tenant_id=matched_tenant)

    if settings.api_access_token and secrets.compare_digest(
        supplied_token, settings.api_access_token.get_secret_value()
    ):
        return APIPrincipal(tenant_id="default")

    raw_session = request.cookies.get(settings.auth_session_cookie_name, "")
    if raw_session and settings.google_auth_enabled:
        token_hash = sha256(raw_session.encode("utf-8")).hexdigest()
        session = db.get(AuthSession, token_hash)
        now = datetime.now(UTC)
        if session is not None:
            expires_at = session.expires_at
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=UTC)
            user = db.get(AuthUser, session.user_id)
            if user is not None and expires_at > now:
                if request.method.upper() not in {"GET", "HEAD", "OPTIONS"}:
                    allowed_origins = {
                        origin.strip()
                        for origin in settings.cors_allowed_origins.split(",")
                        if origin.strip()
                    }
                    origin = request.headers.get("origin", "")
                    csrf = request.headers.get("x-csrf-token", "")
                    if origin not in allowed_origins or not secrets.compare_digest(
                        csrf, session.csrf_token
                    ):
                        raise HTTPException(
                            status_code=status.HTTP_403_FORBIDDEN,
                            detail="Browser request verification failed",
                        )
                return APIPrincipal(
                    tenant_id=user.tenant_id,
                    user_id=user.id,
                    email=user.email,
                    display_name=user.display_name,
                    picture_url=user.picture_url,
                    csrf_token=session.csrf_token,
                )

    if (
        not settings.api_access_token
        and not settings.api_tenant_tokens
        and not settings.google_auth_enabled
    ):
        if settings.environment.lower() == "development":
            return APIPrincipal(tenant_id="default")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API authentication is not configured",
        )

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing bearer token",
        headers={"WWW-Authenticate": "Bearer"},
    )

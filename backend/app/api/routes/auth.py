"""Google sign-in endpoints and revocable browser sessions."""

import logging
import secrets
from datetime import UTC, datetime, timedelta
from typing import Annotated
from urllib.parse import urlencode, urlsplit
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from fastapi.responses import RedirectResponse
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.security import APIPrincipal, require_api_token
from app.db.models import AuthSession, AuthUser, OAuthLoginState
from app.db.session import get_db
from app.services.google_oidc import (
    GoogleOIDCError,
    GoogleOIDCProvider,
    create_code_verifier,
    hash_secret,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/auth", tags=["authentication"])
oidc_provider = GoogleOIDCProvider()


def _frontend_redirect(*, outcome: str) -> RedirectResponse:
    """Redirect to the configured UI with a coarse, non-sensitive result code."""
    settings = get_settings()
    response = RedirectResponse(
        f"{settings.auth_frontend_url.rstrip('/')}?{urlencode({'auth': outcome})}",
        status_code=status.HTTP_303_SEE_OTHER,
    )
    response.delete_cookie(
        key=f"{settings.auth_session_cookie_name}_oauth",
        path="/api/v1/auth/google/callback",
        secure=settings.auth_session_cookie_secure,
        httponly=True,
        samesite="lax",
    )
    return response


@router.get("/config")
def auth_config() -> dict[str, bool | str | None]:
    """Return public authentication capabilities without revealing secrets."""
    settings = get_settings()
    redirect_uri = urlsplit(settings.google_redirect_uri or "")
    login_url = (
        f"{redirect_uri.scheme}://{redirect_uri.netloc}/api/v1/auth/google/start"
        if settings.google_auth_enabled
        else None
    )
    return {
        "authentication_required": (
            settings.environment != "development" or settings.google_auth_enabled
        ),
        "google_enabled": settings.google_auth_enabled,
        "google_login_url": login_url,
    }


def _failed_callback(reason: str) -> RedirectResponse:
    """Log a safe failure category and return a generic UI result."""
    logger.warning("Google sign-in callback rejected (reason=%s)", reason)
    return _frontend_redirect(outcome="failed")


@router.get("/google/start")
def google_start(
    db: Annotated[Session, Depends(get_db)],
    login_hint: Annotated[str | None, Query(max_length=320)] = None,
) -> RedirectResponse:
    """Create single-use OAuth state and redirect to Google's consent screen."""
    settings = get_settings()
    if not settings.google_auth_enabled:
        raise HTTPException(status_code=503, detail="Google sign-in is not configured")
    now = datetime.now(UTC)
    db.execute(delete(OAuthLoginState).where(OAuthLoginState.expires_at < now))
    state = secrets.token_urlsafe(32)
    nonce = secrets.token_urlsafe(32)
    verifier = create_code_verifier()
    browser_binding = secrets.token_urlsafe(32)
    db.add(
        OAuthLoginState(
            state_hash=hash_secret(state),
            nonce=nonce,
            code_verifier=verifier,
            browser_binding_hash=hash_secret(browser_binding),
            expires_at=now + timedelta(seconds=settings.google_oauth_state_ttl_seconds),
        )
    )
    db.commit()
    response = RedirectResponse(
        oidc_provider.authorization_url(
            client_id=settings.google_client_id or "",
            redirect_uri=settings.google_redirect_uri or "",
            state=state,
            nonce=nonce,
            code_verifier=verifier,
            login_hint=login_hint.strip() if login_hint else None,
        ),
        status_code=status.HTTP_302_FOUND,
    )
    response.set_cookie(
        key=f"{settings.auth_session_cookie_name}_oauth",
        value=browser_binding,
        max_age=settings.google_oauth_state_ttl_seconds,
        httponly=True,
        secure=settings.auth_session_cookie_secure,
        samesite="lax",
        path="/api/v1/auth/google/callback",
    )
    return response


@router.get("/google/callback")
def google_callback(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    code: str | None = Query(default=None, max_length=4096),
    state: str | None = Query(default=None, max_length=256),
    error: str | None = Query(default=None, max_length=100),
) -> RedirectResponse:
    """Validate a Google response, provision the user, and issue a session cookie."""
    settings = get_settings()
    if not settings.google_auth_enabled:
        return _frontend_redirect(outcome="unavailable")
    if error or not code or not state:
        return _failed_callback("provider_denied_or_missing_parameters")
    state_row = db.get(OAuthLoginState, hash_secret(state))
    now = datetime.now(UTC)
    state_expiry = state_row.expires_at if state_row is not None else None
    if state_expiry is not None and state_expiry.tzinfo is None:
        state_expiry = state_expiry.replace(tzinfo=UTC)
    if state_row is None or state_expiry <= now:
        if state_row is not None:
            db.delete(state_row)
            db.commit()
        return _failed_callback("invalid_or_expired_state")
    binding_cookie = request.cookies.get(f"{settings.auth_session_cookie_name}_oauth", "")
    if not secrets.compare_digest(hash_secret(binding_cookie), state_row.browser_binding_hash):
        return _failed_callback("browser_binding_mismatch")
    nonce, verifier = state_row.nonce, state_row.code_verifier
    db.delete(state_row)
    db.commit()  # Consume state before the remote exchange to prevent replay.
    try:
        id_token = oidc_provider.exchange_code(
            code=code,
            client_id=settings.google_client_id or "",
            client_secret=settings.google_client_secret.get_secret_value(),
            redirect_uri=settings.google_redirect_uri or "",
            code_verifier=verifier,
        )
        claims = oidc_provider.verify_identity(
            id_token,
            client_id=settings.google_client_id or "",
            clock_skew_in_seconds=settings.google_token_clock_skew_seconds,
        )
    except GoogleOIDCError as exc:
        cause_type = type(exc.__cause__).__name__ if exc.__cause__ else "n/a"
        logger.warning(
            "Google sign-in provider validation failed "
            "(error_type=%s cause_type=%s failure_category=%s)",
            type(exc).__name__,
            cause_type,
            exc.failure_category,
        )
        return _frontend_redirect(outcome="failed")
    if not secrets.compare_digest(str(claims.get("nonce", "")), nonce):
        return _failed_callback("nonce_mismatch")

    google_sub = str(claims["sub"])
    email = str(claims.get("email", "")).strip().lower()
    if not email or len(email) > 320:
        return _failed_callback("verified_email_missing")
    user = db.scalar(select(AuthUser).where(AuthUser.google_sub == google_sub))
    if user is None:
        default_email = (settings.google_default_tenant_email or "").strip().lower()
        tenant_id = (
            "default"
            if default_email and email == default_email
            else f"g_{hash_secret(google_sub)[:60]}"
        )
        user = AuthUser(
            id=str(uuid4()),
            google_sub=google_sub,
            email=email,
            email_verified=True,
            display_name=str(claims.get("name", ""))[:200],
            picture_url=str(claims.get("picture"))[:2048] if claims.get("picture") else None,
            tenant_id=tenant_id,
        )
        db.add(user)
    else:
        user.email = email
        user.email_verified = True
        user.display_name = str(claims.get("name", ""))[:200]
        user.picture_url = str(claims.get("picture"))[:2048] if claims.get("picture") else None
        user.last_login_at = now

    # Revoke expired sessions opportunistically and issue an unpredictable cookie.
    db.execute(delete(AuthSession).where(AuthSession.expires_at < now))
    raw_token = secrets.token_urlsafe(48)
    csrf_token = secrets.token_urlsafe(32)
    session = AuthSession(
        token_hash=hash_secret(raw_token),
        user_id=user.id,
        csrf_token=csrf_token,
        expires_at=now + timedelta(seconds=settings.auth_session_ttl_seconds),
    )
    db.add(session)
    db.commit()

    response = _frontend_redirect(outcome="success")
    response.set_cookie(
        key=settings.auth_session_cookie_name,
        value=raw_token,
        max_age=settings.auth_session_ttl_seconds,
        httponly=True,
        secure=settings.auth_session_cookie_secure,
        samesite="lax",
        path="/",
    )
    return response


@router.get("/me")
def current_user(
    principal: Annotated[APIPrincipal, Depends(require_api_token)],
) -> dict[str, str | None]:
    """Return the authenticated identity and its per-session CSRF token."""
    if principal.user_id is None:
        raise HTTPException(status_code=401, detail="Google sign-in is required")
    return {
        "email": principal.email,
        "display_name": principal.display_name,
        "picture_url": principal.picture_url,
        "csrf_token": principal.csrf_token,
    }


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(
    response: Response,
    request: Request,
    principal: Annotated[APIPrincipal, Depends(require_api_token)],
    db: Annotated[Session, Depends(get_db)],
) -> None:
    """Revoke the current browser session and clear its cookie."""
    settings = get_settings()
    cookie = request.cookies.get(settings.auth_session_cookie_name, "")
    if principal.user_id:
        session = db.get(AuthSession, hash_secret(cookie))
        if session is not None:
            db.delete(session)
            db.commit()
    response.delete_cookie(
        key=settings.auth_session_cookie_name,
        path="/",
        secure=settings.auth_session_cookie_secure,
        httponly=True,
        samesite="lax",
    )

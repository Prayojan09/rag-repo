"""Security and lifecycle tests for Google browser authentication."""

from datetime import UTC, datetime, timedelta
from urllib.parse import parse_qs, urlparse

import pytest
from pydantic import SecretStr
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.routes import auth as auth_routes
from app.core import security
from app.core.config import Settings
from app.db.models import AuthSession, AuthUser, OAuthLoginState
from app.services.google_oidc import hash_secret


def test_google_sign_in_provisions_tenant_and_requires_csrf(
    client, test_engine, monkeypatch
) -> None:
    """Complete a mocked OIDC exchange, then verify tenant and CSRF enforcement."""
    settings = Settings(_env_file=None).model_copy(
        update={
            "google_client_id": "client-id",
            "google_client_secret": SecretStr("client-secret"),
            "google_redirect_uri": "http://localhost:8001/api/v1/auth/google/callback",
            "auth_frontend_url": "http://localhost:5173",
            "cors_allowed_origins": "http://localhost:5173",
        }
    )
    monkeypatch.setattr(auth_routes, "get_settings", lambda: settings)
    monkeypatch.setattr(security, "get_settings", lambda: settings)

    start = client.get(
        "/api/v1/auth/google/start",
        params={"login_hint": "person@example.com"},
        follow_redirects=False,
    )
    assert start.status_code == 302
    params = parse_qs(urlparse(start.headers["location"]).query)
    assert params["login_hint"] == ["person@example.com"]
    assert "prompt" not in params
    state = params["state"][0]
    with Session(test_engine) as db:
        state_row = db.get(OAuthLoginState, hash_secret(state))
        assert state_row is not None
        nonce = state_row.nonce

    monkeypatch.setattr(auth_routes.oidc_provider, "exchange_code", lambda **_kwargs: "id-token")
    monkeypatch.setattr(
        auth_routes.oidc_provider,
        "verify_identity",
        lambda *_args, **_kwargs: {
            "sub": "google-sub-123",
            "email": "person@example.com",
            "email_verified": True,
            "name": "Test Person",
            "nonce": nonce,
        },
    )
    callback = client.get(
        "/api/v1/auth/google/callback",
        params={"code": "one-time-code", "state": state},
        follow_redirects=False,
    )
    assert callback.status_code == 303
    assert "httponly" in callback.headers["set-cookie"].lower()

    profile = client.get("/api/v1/auth/me")
    assert profile.status_code == 200
    identity = profile.json()
    assert identity["email"] == "person@example.com"
    assert identity["csrf_token"]

    denied = client.post(
        "/api/v1/conversations?title=Private", headers={"Origin": "http://localhost:5173"}
    )
    assert denied.status_code == 403
    accepted = client.post(
        "/api/v1/conversations?title=Private",
        headers={
            "Origin": "http://localhost:5173",
            "X-CSRF-Token": identity["csrf_token"],
        },
    )
    assert accepted.status_code == 201

    with Session(test_engine) as db:
        user = db.scalar(select(AuthUser).where(AuthUser.google_sub == "google-sub-123"))
        assert user is not None
        assert user.tenant_id.startswith("g_")
        assert db.scalar(select(AuthSession)) is not None

    logout = client.post(
        "/api/v1/auth/logout",
        headers={"Origin": "http://localhost:5173", "X-CSRF-Token": identity["csrf_token"]},
    )
    assert logout.status_code == 204
    assert client.get("/api/v1/auth/me").status_code == 401


def test_google_start_requires_complete_client_configuration(client, monkeypatch) -> None:
    """Return an explicit unavailable error when Google credentials are unset."""
    settings = Settings(_env_file=None)
    monkeypatch.setattr(auth_routes, "get_settings", lambda: settings)
    response = client.get("/api/v1/auth/google/start")
    assert response.status_code == 503


def test_auth_config_uses_backend_host_for_oauth_start(client, monkeypatch) -> None:
    """Return an absolute OAuth start URL on the callback's host for browser cookies."""
    settings = Settings(_env_file=None).model_copy(
        update={
            "google_client_id": "client-id",
            "google_client_secret": SecretStr("client-secret"),
            "google_redirect_uri": "http://localhost:8001/api/v1/auth/google/callback",
        }
    )
    monkeypatch.setattr(auth_routes, "get_settings", lambda: settings)

    response = client.get("/api/v1/auth/config")

    assert response.status_code == 200
    assert response.json()["google_login_url"] == "http://localhost:8001/api/v1/auth/google/start"


def test_oauth_state_is_single_use(client, monkeypatch) -> None:
    """Reject a callback whose OAuth state was not issued by the service."""
    settings = Settings(_env_file=None).model_copy(
        update={
            "google_client_id": "client-id",
            "google_client_secret": SecretStr("client-secret"),
            "google_redirect_uri": "http://localhost:8001/api/v1/auth/google/callback",
        }
    )
    monkeypatch.setattr(auth_routes, "get_settings", lambda: settings)
    response = client.get(
        "/api/v1/auth/google/callback",
        params={"code": "one-time-code", "state": "not-issued"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert "auth=failed" in response.headers["location"]


def test_oauth_state_must_match_the_starting_browser(client, monkeypatch) -> None:
    """Reject OAuth state when the browser-binding cookie is absent."""
    settings = Settings(_env_file=None).model_copy(
        update={
            "google_client_id": "client-id",
            "google_client_secret": SecretStr("client-secret"),
            "google_redirect_uri": "http://localhost:8001/api/v1/auth/google/callback",
        }
    )
    monkeypatch.setattr(auth_routes, "get_settings", lambda: settings)
    start = client.get("/api/v1/auth/google/start", follow_redirects=False)
    state = parse_qs(urlparse(start.headers["location"]).query)["state"][0]
    client.cookies.clear()

    response = client.get(
        "/api/v1/auth/google/callback",
        params={"code": "one-time-code", "state": state},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert "auth=failed" in response.headers["location"]


def test_expired_google_session_is_rejected(client, test_engine, monkeypatch) -> None:
    """Refuse browser sessions after their server-side expiration time."""
    settings = Settings(_env_file=None).model_copy(
        update={
            "google_client_id": "client-id",
            "google_client_secret": "client-secret",
            "google_redirect_uri": "http://localhost:8001/api/v1/auth/google/callback",
        }
    )
    monkeypatch.setattr(security, "get_settings", lambda: settings)
    with Session(test_engine) as db:
        user = AuthUser(
            id="user-1",
            google_sub="google-sub-1",
            email="one@example.com",
            email_verified=True,
            display_name="One",
            tenant_id="g_one",
        )
        db.add(user)
        db.add(
            AuthSession(
                token_hash=hash_secret("expired-cookie"),
                user_id=user.id,
                csrf_token="csrf",
                expires_at=datetime.now(UTC) - timedelta(minutes=1),
            )
        )
        db.commit()
    client.cookies.set("rag_session", "expired-cookie")
    assert client.get("/api/v1/auth/me").status_code == 401


def test_google_identity_verification_uses_client_audience(monkeypatch) -> None:
    """Delegate signed token validation to Google's library with the configured audience."""
    from google.oauth2 import id_token

    from app.services.google_oidc import GoogleOIDCProvider

    calls = {}

    def verify(_token, _request, *, audience, clock_skew_in_seconds):
        calls["audience"] = audience
        calls["clock_skew_in_seconds"] = clock_skew_in_seconds
        return {"sub": "google-sub", "email_verified": True}

    monkeypatch.setattr(id_token, "verify_oauth2_token", verify)
    claims = GoogleOIDCProvider().verify_identity("signed-token", client_id="expected-client")

    assert calls["audience"] == "expected-client"
    assert calls["clock_skew_in_seconds"] == 60
    assert claims["sub"] == "google-sub"


def test_google_identity_verification_rejects_unverified_email(monkeypatch) -> None:
    """Reject identities whose Google token does not mark the email verified."""
    from google.oauth2 import id_token

    from app.services.google_oidc import GoogleOIDCError, GoogleOIDCProvider

    monkeypatch.setattr(
        id_token,
        "verify_oauth2_token",
        lambda *_args, **_kwargs: {"sub": "google-sub", "email_verified": False},
    )

    with pytest.raises(GoogleOIDCError, match="verified email"):
        GoogleOIDCProvider().verify_identity("signed-token", client_id="expected-client")


def test_google_identity_verification_classifies_audience_failure(monkeypatch) -> None:
    """Expose only a safe category when Google rejects the configured client ID."""
    from google.auth.exceptions import InvalidValue
    from google.oauth2 import id_token

    from app.services.google_oidc import GoogleOIDCError, GoogleOIDCProvider

    def reject_audience(*_args, **_kwargs):
        raise InvalidValue("Token has wrong audience: sensitive-value")

    monkeypatch.setattr(id_token, "verify_oauth2_token", reject_audience)

    with pytest.raises(GoogleOIDCError) as error:
        GoogleOIDCProvider().verify_identity("signed-token", client_id="expected-client")

    assert error.value.failure_category == "client_id_audience_mismatch"
    assert "sensitive-value" not in str(error.value)


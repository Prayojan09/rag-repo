"""Regression coverage for production configuration and offline token counting."""

import pytest
from pydantic import SecretStr, ValidationError

from app.core.config import Settings
from app.ingestion.metadata import count_tokens


def test_production_settings_require_authentication() -> None:
    """Reject production mode when API authentication is missing."""
    with pytest.raises(ValidationError, match="Configure API_ACCESS_TOKEN"):
        Settings(
            _env_file=None,
            environment="production",
            database_url="postgresql+psycopg://rag:strong-password@db:5432/rag",
            embedding_model_revision="a" * 40,
            reranker_model_revision="b" * 40,
        )


def test_production_settings_reject_floating_model_revisions() -> None:
    """Require immutable model revisions for reproducible production indexing."""
    with pytest.raises(ValidationError, match="Pin EMBEDDING_MODEL_REVISION"):
        Settings(
            _env_file=None,
            environment="production",
            api_access_token=SecretStr("t" * 40),
            database_url="postgresql+psycopg://rag:strong-password@db:5432/rag",
        )


def test_production_settings_accept_explicit_secure_configuration() -> None:
    """Accept production settings when secrets, hosts, origins, and revisions are set."""
    settings = Settings(
        _env_file=None,
        environment="production",
        api_access_token=SecretStr("t" * 40),
        database_url="postgresql+psycopg://rag:strong-password@db:5432/rag",
        qdrant_url="https://qdrant.example.com",
        qdrant_api_key=SecretStr("q" * 40),
        cors_allowed_origins="https://app.example.com",
        allowed_hosts="api.example.com",
        embedding_model_revision="a" * 40,
        reranker_model_revision="b" * 40,
    )

    assert settings.environment == "production"


def test_production_settings_accept_google_auth_without_api_token() -> None:
    """Allow secure interactive Google authentication as the production login."""
    settings = Settings(
        _env_file=None,
        environment="production",
        google_client_id="client-id",
        google_client_secret=SecretStr("client-secret"),
        google_redirect_uri="https://api.example.com/api/v1/auth/google/callback",
        auth_frontend_url="https://app.example.com",
        auth_session_cookie_secure=True,
        database_url="postgresql+psycopg://rag:strong-password@db:5432/rag",
        cors_allowed_origins="https://app.example.com",
        allowed_hosts="api.example.com",
        embedding_model_revision="a" * 40,
        reranker_model_revision="b" * 40,
    )

    assert settings.google_auth_enabled


def test_google_oauth_settings_must_be_configured_together() -> None:
    """Reject a partial Google OAuth configuration rather than silently disabling it."""
    with pytest.raises(ValidationError, match="Configure GOOGLE_CLIENT_ID"):
        Settings(_env_file=None, google_client_id="client-id")


def test_token_counting_falls_back_when_encoding_cannot_load(monkeypatch) -> None:
    """Keep chunking functional when tiktoken cannot download its encoding data."""
    import tiktoken

    def fail_to_load(_encoding: str):
        raise OSError("network unavailable")

    monkeypatch.setattr(tiktoken, "get_encoding", fail_to_load)

    assert count_tokens("one two three") == (3, "word-count fallback")

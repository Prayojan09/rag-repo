"""Environment-backed application settings."""

import re
from functools import lru_cache
from urllib.parse import urlsplit

from pydantic import AliasChoices, Field, PostgresDsn, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Document Intelligence API"
    environment: str = "development"
    log_level: str = "INFO"
    cors_allowed_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    allowed_hosts: str = "localhost,127.0.0.1,testserver"
    database_url: PostgresDsn = "postgresql+psycopg://rag:rag@localhost:5432/rag"
    database_connect_timeout_seconds: int = Field(default=10, ge=1, le=60)
    database_pool_size: int = Field(default=5, ge=1, le=50)
    database_max_overflow: int = Field(default=10, ge=0, le=100)
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    request_timeout_seconds: float = Field(default=3.0, gt=0, le=30)
    upload_dir: str = "data/uploads"
    max_upload_bytes: int = Field(default=100 * 1024 * 1024, gt=0, le=100 * 1024 * 1024)
    ocr_language: str = "eng"
    ocr_dpi: int = Field(default=200, ge=72, le=300)
    ocr_fallback_page_segmentation_mode: int = Field(default=6, ge=3, le=13)
    handwriting_ocr_model: str = "microsoft/trocr-small-handwritten"
    handwriting_ocr_min_confidence: float = Field(default=75.0, ge=0, le=100)
    handwriting_ocr_batch_size: int = Field(default=16, ge=1, le=32)
    handwriting_ocr_cpu_threads: int = Field(default=4, ge=1, le=32)
    handwriting_line_ink_fraction: float = Field(default=0.01, gt=0, le=0.2)
    handwriting_line_min_ink_pixels: int = Field(default=8, ge=1, le=100)
    handwriting_line_merge_gap_fraction: float = Field(default=0.006, gt=0, le=0.1)
    tesseract_cmd: str | None = None
    chunk_size_chars: int = Field(default=1200, gt=100)
    chunk_overlap_chars: int = Field(default=150, ge=0)
    ingestion_heading_max_chars: int = Field(default=140, ge=20, le=500)
    ingestion_header_repeat_fraction: float = Field(default=0.5, gt=0, le=1)
    ingestion_metadata_labels: tuple[str, ...] = (
        "Duration",
        "Level",
        "Price",
        "Date",
        "Location",
        "Author",
        "Version",
        "Format",
        "Provider",
        "Certification",
        "Status",
        "Owner",
        "Category",
    )
    ingestion_report_dir: str = "data/reports"
    ingestion_token_encoding: str = "cl100k_base"
    ingestion_tiny_chunk_tokens: int = Field(default=20, ge=1, le=200)
    ingestion_tiny_chunk_fraction: float = Field(default=0.33, gt=0, le=1)
    ingestion_low_text_chars_per_page: int = Field(default=100, ge=1, le=2_000)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model_revision: str = "main"
    embedding_device: str = "cpu"
    embedding_batch_size: int = Field(default=64, ge=1, le=256)
    qdrant_collection: str = "rag_chunks"
    search_default_top_k: int = Field(default=10, ge=1, le=100)
    search_max_top_k: int = Field(default=50, ge=1, le=100)
    search_candidate_multiplier: int = Field(default=4, ge=1, le=20)
    search_candidate_max: int = Field(default=200, ge=1, le=1000)
    bm25_max_chunks: int = Field(default=50000, ge=100, le=500000)
    rrf_constant: int = Field(default=60, ge=1, le=1000)
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_model_revision: str = "main"
    reranker_device: str = "cpu"
    reranker_batch_size: int = Field(default=32, ge=1, le=256)
    reranker_candidate_k: int = Field(default=40, ge=1, le=200)
    llm_provider: str = "disabled"
    llm_base_url: str = Field(
        default="https://api.openai.com/v1",
        validation_alias=AliasChoices("LLM_GATEWAY_BASE_URL", "LLM_BASE_URL"),
    )
    llm_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("LLM_GATEWAY_API_KEY", "LLM_API_KEY"),
    )
    llm_model: str | None = None
    llm_available_models: str = ""
    llm_timeout_seconds: float = Field(default=60.0, gt=0, le=300)
    llm_max_retries: int = Field(default=2, ge=0, le=3)
    llm_max_tokens: int = Field(default=2048, ge=64, le=8192)
    document_structuring_enabled: bool = True
    document_structuring_max_chars: int = Field(default=40_000, ge=1_000, le=200_000)
    document_structuring_max_tokens: int = Field(default=8192, ge=256, le=32_768)
    chat_context_k: int = Field(default=5, ge=1, le=30)
    api_access_token: SecretStr | None = None
    api_tenant_tokens: dict[str, SecretStr] = Field(default_factory=dict)
    google_client_id: str | None = None
    google_client_secret: SecretStr | None = None
    google_redirect_uri: str | None = None
    auth_frontend_url: str = "http://localhost:5173"
    auth_session_cookie_name: str = "rag_session"
    auth_session_cookie_secure: bool = False
    auth_session_ttl_seconds: int = Field(default=43_200, ge=300, le=2_592_000)
    google_oauth_state_ttl_seconds: int = Field(default=600, ge=60, le=1800)
    google_token_clock_skew_seconds: int = Field(default=60, ge=0, le=300)
    google_default_tenant_email: str | None = None

    @property
    def google_auth_enabled(self) -> bool:
        """Whether all required Google OAuth client settings are configured."""
        return bool(
            self.google_client_id and self.google_client_secret and self.google_redirect_uri
        )

    @property
    def available_llm_models(self) -> tuple[str, ...]:
        """Return the configured model allowlist, including the default model."""
        models = [item.strip() for item in self.llm_available_models.split(",") if item.strip()]
        if self.llm_model and self.llm_model not in models:
            models.insert(0, self.llm_model)
        return tuple(dict.fromkeys(models))

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        return value.upper()

    @field_validator("environment")
    @classmethod
    def normalize_environment(cls, value: str) -> str:
        """Normalize and validate the deployment environment name."""
        normalized = value.strip().lower()
        if normalized not in {"development", "test", "staging", "production"}:
            raise ValueError("ENVIRONMENT must be development, test, staging, or production")
        return normalized

    @model_validator(mode="after")
    def validate_chunk_overlap(self) -> "Settings":
        if self.chunk_overlap_chars >= self.chunk_size_chars:
            raise ValueError("CHUNK_OVERLAP_CHARS must be smaller than CHUNK_SIZE_CHARS")
        if self.search_default_top_k > self.search_max_top_k:
            raise ValueError("SEARCH_DEFAULT_TOP_K cannot exceed SEARCH_MAX_TOP_K")
        if self.reranker_candidate_k > self.search_candidate_max:
            raise ValueError("RERANKER_CANDIDATE_K cannot exceed SEARCH_CANDIDATE_MAX")
        if self.llm_provider not in {"disabled", "openai_compatible"}:
            raise ValueError("LLM_PROVIDER must be disabled or openai_compatible")
        google_oauth_values = (
            self.google_client_id,
            self.google_client_secret,
            self.google_redirect_uri,
        )
        if any(google_oauth_values) and not all(google_oauth_values):
            raise ValueError(
                "Configure GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, and GOOGLE_REDIRECT_URI together"
            )
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", self.auth_session_cookie_name):
            raise ValueError("AUTH_SESSION_COOKIE_NAME must be a safe cookie name")
        if self.api_access_token and len(self.api_access_token.get_secret_value()) < 32:
            raise ValueError("API_ACCESS_TOKEN must be at least 32 characters long")
        tenant_token_values = [
            token.get_secret_value() for token in self.api_tenant_tokens.values()
        ]
        if len(set(tenant_token_values)) != len(tenant_token_values):
            raise ValueError("Each tenant must have a distinct API token")
        if (
            self.api_access_token
            and self.api_access_token.get_secret_value() in tenant_token_values
        ):
            raise ValueError("API_ACCESS_TOKEN must differ from tenant-specific tokens")
        for tenant_id, token in self.api_tenant_tokens.items():
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", tenant_id):
                raise ValueError("API_TENANT_TOKENS keys must be safe tenant IDs")
            if len(token.get_secret_value()) < 32:
                raise ValueError("Each API_TENANT_TOKENS value must be at least 32 characters")
        if self.environment in {"staging", "production"}:
            if (
                not self.api_access_token
                and not self.api_tenant_tokens
                and not self.google_auth_enabled
            ):
                raise ValueError(
                    "Configure API_ACCESS_TOKEN or API_TENANT_TOKENS or Google OAuth "
                    "outside development"
                )
            if self.google_auth_enabled:
                redirect = urlsplit(self.google_redirect_uri or "")
                frontend = urlsplit(self.auth_frontend_url)
                if not self.auth_session_cookie_secure:
                    raise ValueError(
                        "AUTH_SESSION_COOKIE_SECURE must be true for Google OAuth "
                        "outside development"
                    )
                if redirect.scheme != "https" or frontend.scheme != "https":
                    raise ValueError(
                        "Google OAuth redirect and frontend URLs must use HTTPS outside development"
                    )
                configured_origins = {
                    origin.strip()
                    for origin in self.cors_allowed_origins.split(",")
                    if origin.strip()
                }
                if self.auth_frontend_url.rstrip("/") not in configured_origins:
                    raise ValueError("AUTH_FRONTEND_URL must be included in CORS_ALLOWED_ORIGINS")
                configured_hosts = {
                    host.strip() for host in self.allowed_hosts.split(",") if host.strip()
                }
                if redirect.hostname not in configured_hosts:
                    raise ValueError("GOOGLE_REDIRECT_URI host must be included in ALLOWED_HOSTS")
            database_password = urlsplit(str(self.database_url)).password
            if (
                not database_password
                or "REPLACE_WITH" in database_password.upper()
                or database_password.lower()
                in {
                    "rag",
                    "password",
                    "postgres",
                }
            ):
                raise ValueError("Set a non-default PostgreSQL password outside development")
            qdrant_host = urlsplit(self.qdrant_url).hostname
            if not qdrant_host:
                raise ValueError("QDRANT_URL must be an absolute URL")
            if qdrant_host not in {"localhost", "127.0.0.1", "::1", "qdrant"}:
                if not self.qdrant_api_key:
                    raise ValueError("Configure QDRANT_API_KEY for a remote Qdrant service")
                if "REPLACE_WITH" in self.qdrant_api_key.get_secret_value().upper():
                    raise ValueError("Replace the QDRANT_API_KEY example value")
                if urlsplit(self.qdrant_url).scheme != "https":
                    raise ValueError("Remote Qdrant connections must use HTTPS")
            if not re.fullmatch(r"[0-9a-fA-F]{40}", self.embedding_model_revision):
                raise ValueError("Pin EMBEDDING_MODEL_REVISION outside development")
            if not re.fullmatch(r"[0-9a-fA-F]{40}", self.reranker_model_revision):
                raise ValueError("Pin RERANKER_MODEL_REVISION outside development")
            configured_secrets = [
                self.api_access_token.get_secret_value() if self.api_access_token else "",
                self.google_client_secret.get_secret_value() if self.google_client_secret else "",
                *(token.get_secret_value() for token in self.api_tenant_tokens.values()),
            ]
            if any("REPLACE_WITH" in secret.upper() for secret in configured_secrets):
                raise ValueError("Replace all example API token values before deployment")
            if self.google_client_id and "REPLACE_WITH" in self.google_client_id.upper():
                raise ValueError("Replace the example GOOGLE_CLIENT_ID before deployment")
            allowed_hosts = [host.strip() for host in self.allowed_hosts.split(",") if host.strip()]
            if not allowed_hosts or "*" in allowed_hosts:
                raise ValueError("Set explicit ALLOWED_HOSTS outside development")
            origins = [
                origin.strip() for origin in self.cors_allowed_origins.split(",") if origin.strip()
            ]
            if not origins or "*" in origins:
                raise ValueError("Set explicit CORS_ALLOWED_ORIGINS outside development")
            for origin in origins:
                parsed_origin = urlsplit(origin)
                is_local = parsed_origin.hostname in {"localhost", "127.0.0.1", "::1"}
                if parsed_origin.scheme != "https" and not is_local:
                    raise ValueError("Non-local CORS origins must use HTTPS")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()

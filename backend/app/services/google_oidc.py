"""Google OpenID Connect authorization-code client and identity validation."""

import base64
import hashlib
import secrets
from typing import Any
from urllib.parse import urlencode

import httpx


class GoogleOIDCError(RuntimeError):
    """Raised when Google authorization or identity validation fails."""

    def __init__(self, message: str, *, failure_category: str = "provider_error") -> None:
        """Keep a safe diagnostic category separate from provider error text."""
        super().__init__(message)
        self.failure_category = failure_category


class GoogleOIDCProvider:
    """Perform Google OAuth code exchange and verify signed ID tokens."""

    authorization_endpoint = "https://accounts.google.com/o/oauth2/v2/auth"
    token_endpoint = "https://oauth2.googleapis.com/token"

    def authorization_url(
        self,
        *,
        client_id: str,
        redirect_uri: str,
        state: str,
        nonce: str,
        code_verifier: str,
        login_hint: str | None = None,
    ) -> str:
        """Build a Google authorization URL with nonce and PKCE protections."""
        challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode("ascii")).digest())
            .rstrip(b"=")
            .decode("ascii")
        )
        params = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": "openid email profile",
            "state": state,
            "nonce": nonce,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
        if login_hint:
            params["login_hint"] = login_hint
        else:
            params["prompt"] = "select_account"
        return f"{self.authorization_endpoint}?{urlencode(params)}"

    def exchange_code(
        self,
        *,
        code: str,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        code_verifier: str,
    ) -> str:
        """Exchange an authorization code and return its signed ID token."""
        try:
            response = httpx.post(
                self.token_endpoint,
                data={
                    "code": code,
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "redirect_uri": redirect_uri,
                    "grant_type": "authorization_code",
                    "code_verifier": code_verifier,
                },
                timeout=10.0,
            )
            response.raise_for_status()
            token = response.json().get("id_token")
        except (httpx.HTTPError, ValueError, AttributeError) as exc:
            raise GoogleOIDCError("Google sign-in could not be completed") from exc
        if not isinstance(token, str) or not token:
            raise GoogleOIDCError("Google did not return a valid identity token")
        return token

    def verify_identity(
        self, id_token: str, *, client_id: str, clock_skew_in_seconds: int = 60
    ) -> dict[str, Any]:
        """Verify token signature, issuer, expiry, and audience with google-auth."""
        try:
            from google.auth.transport.requests import Request
            from google.oauth2 import id_token as google_id_token

            claims = google_id_token.verify_oauth2_token(
                id_token,
                Request(),
                audience=client_id,
                clock_skew_in_seconds=clock_skew_in_seconds,
            )
        except Exception as exc:
            message = str(exc).lower()
            if "audience" in message:
                category = "client_id_audience_mismatch"
            elif "too early" in message or "clock" in message:
                category = "system_clock_invalid"
            elif "expired" in message:
                category = "identity_token_expired"
            else:
                category = "identity_token_invalid"
            raise GoogleOIDCError(
                "Google identity token validation failed", failure_category=category
            ) from exc
        if not claims.get("sub") or claims.get("email_verified") is not True:
            raise GoogleOIDCError("Google account must have a verified email address")
        return claims


def create_code_verifier() -> str:
    """Create a high-entropy PKCE verifier accepted by the OAuth standard."""
    return secrets.token_urlsafe(64)


def hash_secret(value: str) -> str:
    """Return a one-way SHA-256 digest suitable for opaque random credentials."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

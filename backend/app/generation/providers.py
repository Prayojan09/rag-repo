"""Provider-neutral LLM interface with an OpenAI-compatible HTTP adapter."""

import time
from functools import lru_cache
from typing import Protocol

import httpx

from app.core.config import Settings, get_settings


class LLMProviderError(RuntimeError):
    """Raised for provider configuration, transport, or response failures."""


class LLMProvider(Protocol):
    provider_name: str
    model_name: str

    def generate_json(
        self, system_prompt: str, user_prompt: str, *, max_tokens: int | None = None
    ) -> str: ...


class OpenAICompatibleProvider:
    """Call a Chat Completions-compatible API without binding to one SDK."""

    provider_name = "openai_compatible"

    def __init__(
        self,
        settings: Settings,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not settings.llm_api_key:
            raise LLMProviderError("LLM_API_KEY is not configured")
        if not settings.llm_model:
            raise LLMProviderError("LLM_MODEL is not configured")
        self.model_name = settings.llm_model
        self._api_key = settings.llm_api_key.get_secret_value()
        self._base_url = settings.llm_base_url.rstrip("/")
        self._timeout = settings.llm_timeout_seconds
        self._max_retries = settings.llm_max_retries
        self._max_tokens = settings.llm_max_tokens
        self._transport = transport
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._usage_complete = True

    def reset_usage(self) -> None:
        """Reset token accounting before one chat request (generation plus verification)."""
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._usage_complete = True

    @property
    def token_usage(self) -> tuple[int, int] | None:
        if not self._usage_complete:
            return None
        return self._prompt_tokens, self._completion_tokens

    def generate_json(
        self, system_prompt: str, user_prompt: str, *, max_tokens: int | None = None
    ) -> str:
        url = self._completion_url()
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "max_tokens": max_tokens or self._max_tokens,
            "response_format": {"type": "json_object"},
        }
        try:
            with httpx.Client(
                timeout=self._timeout,
                transport=self._transport,
            ) as client:
                for attempt in range(self._max_retries + 1):
                    try:
                        response = client.post(
                            url,
                            headers={
                                "Authorization": f"Bearer {self._api_key}",
                                "Content-Type": "application/json",
                            },
                            json=payload,
                        )
                    except (httpx.TimeoutException, httpx.TransportError):
                        if attempt >= self._max_retries:
                            raise
                        time.sleep(min(0.2 * (2**attempt), 0.8))
                        continue

                    if response.status_code == 429 or response.status_code >= 500:
                        if attempt >= self._max_retries:
                            response.raise_for_status()
                        time.sleep(min(0.2 * (2**attempt), 0.8))
                        continue
                    response.raise_for_status()
                    data = response.json()
                    break
                else:
                    raise LLMProviderError("LLM provider request failed after retries")
        except httpx.HTTPStatusError as exc:
            # Record only the HTTP status; provider response bodies may expose details.
            raise LLMProviderError(
                f"LLM provider returned HTTP {exc.response.status_code}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise LLMProviderError("LLM provider request timed out") from exc
        except httpx.ConnectError as exc:
            raise LLMProviderError("Could not connect to the LLM provider") from exc
        except httpx.HTTPError as exc:
            # Avoid returning upstream bodies that might contain sensitive details.
            raise LLMProviderError("LLM provider request failed") from exc
        except ValueError as exc:
            raise LLMProviderError("LLM provider returned invalid JSON") from exc

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMProviderError("LLM provider response is missing message content") from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMProviderError("LLM provider returned empty message content")
        usage = data.get("usage")
        if isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
                self._prompt_tokens += prompt_tokens
                self._completion_tokens += completion_tokens
            else:
                self._usage_complete = False
        else:
            self._usage_complete = False
        return content

    def _completion_url(self) -> str:
        if self._base_url.endswith("/chat/completions"):
            return self._base_url
        if self._base_url.endswith("/v1"):
            return f"{self._base_url}/chat/completions"
        return f"{self._base_url}/v1/chat/completions"


@lru_cache
def get_llm_provider() -> LLMProvider:
    settings = get_settings()
    if settings.llm_provider == "disabled":
        raise LLMProviderError("LLM_PROVIDER is disabled")
    if settings.llm_provider == "openai_compatible":
        return OpenAICompatibleProvider(settings)
    raise LLMProviderError("Configured LLM provider is not supported")

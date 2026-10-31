import json

import httpx
import pytest
from pydantic import SecretStr

from app.core.config import get_settings
from app.generation.providers import LLMProviderError, OpenAICompatibleProvider
from app.retrieval.reranker import CrossEncoderReranker
from app.schemas.chat import ChatRequest
from app.schemas.documents import DenseSearchResult
from app.services import chat as chat_service
from app.services.chat import ChatGenerationError, answer_question


def _source(chunk_id: str = "chunk-1", score: float = 0.7) -> DenseSearchResult:
    return DenseSearchResult(
        chunk_id=chunk_id,
        document_id="doc-1",
        filename="policy.txt",
        version=1,
        ordinal=0,
        page_number=2,
        heading="Leave",
        text="Eligible employees receive annual leave.",
        score=score,
    )


class FakeCrossEncoder:
    def predict(self, sentences, *, batch_size, show_progress_bar):
        assert batch_size == 32
        assert show_progress_bar is False
        return [0.1, 0.9][: len(sentences)]


def test_cross_encoder_reranks_and_preserves_original_score():
    reranker = CrossEncoderReranker(model=FakeCrossEncoder())
    first = _source("chunk-1", score=0.2)
    second = _source("chunk-2", score=0.9)

    results = reranker.rerank("annual leave", [first, second], top_k=2)

    assert [result.chunk_id for result in results] == ["chunk-2", "chunk-1"]
    assert results[0].score == 0.9
    assert results[0].retrieval_score == 0.9
    assert results[0].rerank_score == 0.9


def test_openai_compatible_provider_sends_json_mode_and_parses_content():
    observed = {}
    settings = get_settings().model_copy(
        update={
            "llm_api_key": SecretStr("test-key"),
            "llm_model": "test-model",
            "llm_base_url": "https://llm.example/v1",
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        observed["url"] = str(request.url)
        observed["authorization"] = request.headers["Authorization"]
        observed["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"answer":"ok"}'}}]},
        )

    provider = OpenAICompatibleProvider(settings, transport=httpx.MockTransport(handler))

    output = provider.generate_json("system", "user")

    assert output == '{"answer":"ok"}'
    assert observed["url"] == "https://llm.example/v1/chat/completions"
    assert observed["authorization"] == "Bearer test-key"
    assert observed["payload"]["response_format"] == {"type": "json_object"}
    assert observed["payload"]["model"] == "test-model"


def test_openai_compatible_provider_reports_usage_for_request_pair():
    outputs = iter(
        [
            {
                "choices": [{"message": {"content": '{"answer":"ok"}'}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10},
            },
            {
                "choices": [{"message": {"content": '{"results":[]}'}}],
                "usage": {"prompt_tokens": 200, "completion_tokens": 20},
            },
        ]
    )
    settings = get_settings().model_copy(
        update={"llm_api_key": SecretStr("test-key"), "llm_model": "test-model"}
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=next(outputs))

    provider = OpenAICompatibleProvider(settings, transport=httpx.MockTransport(handler))
    provider.generate_json("system", "answer")
    provider.generate_json("verify", "claim")

    assert provider.token_usage == (300, 30)


def test_openai_compatible_provider_retries_transient_server_error():
    calls = 0
    settings = get_settings().model_copy(
        update={
            "llm_api_key": SecretStr("test-key"),
            "llm_model": "test-model",
            "llm_max_retries": 1,
        }
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"answer":"ok"}'}}]},
        )

    provider = OpenAICompatibleProvider(settings, transport=httpx.MockTransport(handler))

    assert provider.generate_json("system", "user") == '{"answer":"ok"}'
    assert calls == 2


def test_openai_compatible_provider_reports_safe_http_status():
    settings = get_settings().model_copy(
        update={
            "llm_api_key": SecretStr("test-key"),
            "llm_model": "test-model",
            "llm_max_retries": 0,
        }
    )
    provider = OpenAICompatibleProvider(
        settings,
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(401, text="secret provider response")
        ),
    )

    with pytest.raises(LLMProviderError, match="HTTP 401") as error:
        provider.generate_json("system", "user")

    assert "secret provider response" not in str(error.value)


class StaticProvider:
    provider_name = "test"
    model_name = "test-model"

    def __init__(self, output: str | list[str]):
        self.outputs = output if isinstance(output, list) else [output]

    def generate_json(self, _system_prompt, _user_prompt):
        return self.outputs.pop(0)


def _verification(verdict: str = "supported") -> str:
    return json.dumps({"results": [{"index": 0, "verdict": verdict}]})


def test_chat_returns_only_validated_retrieved_citations(monkeypatch):
    source = _source()
    monkeypatch.setattr(chat_service, "search", lambda *_args, **_kwargs: [source])
    provider = StaticProvider(
        [
            json.dumps(
                {
                    "answer": "Eligible employees receive annual leave.",
                    "abstained": False,
                    "abstention_reason": None,
                    "claims": [
                        {
                            "text": "Eligible employees receive annual leave.",
                            "source_chunk_ids": ["chunk-1"],
                        }
                    ],
                    "conflicts": [],
                }
            ),
            _verification(),
        ]
    )

    response = answer_question(
        None,
        ChatRequest(question="How much leave is provided?", mode="keyword", rerank=False),
        provider=provider,
    )

    assert response.abstained is False
    assert response.evidence_verified is True
    assert response.verification_model == "test-model"
    assert response.claims[0].source_chunk_ids == ["chunk-1"]
    assert response.citations[0].page_number == 2
    assert response.citations[0].filename == "policy.txt"


def test_chat_does_not_multiply_reranker_candidate_pool(monkeypatch):
    """Bound hybrid retrieval to the configured reranker pool for faster chat."""
    observed = {}
    settings = get_settings().model_copy(
        update={"reranker_candidate_k": 40, "search_candidate_max": 200}
    )
    monkeypatch.setattr(chat_service, "get_settings", lambda: settings)

    def record_search(_db, _query, top_k, _document_ids, **kwargs):
        observed["top_k"] = top_k
        observed["candidate_multiplier"] = kwargs["candidate_multiplier"]
        return [_source()]

    class PassthroughReranker:
        model_name = "test-reranker"

        def rerank(self, _query, candidates, top_k):
            return list(candidates[:top_k])

    monkeypatch.setattr(chat_service, "search", record_search)
    answer_question(
        None,
        ChatRequest(question="How much leave is provided?", mode="hybrid", rerank=True),
        provider=StaticProvider(
            [
                json.dumps(
                    {
                        "answer": "Eligible employees receive annual leave.",
                        "abstained": False,
                        "abstention_reason": None,
                        "claims": [
                            {
                                "text": "Eligible employees receive annual leave.",
                                "source_chunk_ids": ["chunk-1"],
                            }
                        ],
                        "conflicts": [],
                    }
                ),
                _verification(),
            ]
        ),
        reranker=PassthroughReranker(),
    )

    assert observed == {"top_k": 40, "candidate_multiplier": 1}


def test_chat_rejects_citations_not_in_retrieved_evidence(monkeypatch):
    monkeypatch.setattr(chat_service, "search", lambda *_args, **_kwargs: [_source()])
    provider = StaticProvider(
        json.dumps(
            {
                "answer": "Unsupported citation.",
                "abstained": False,
                "abstention_reason": None,
                "claims": [{"text": "Unsupported citation.", "source_chunk_ids": ["unknown"]}],
                "conflicts": [],
            }
        )
    )

    response = answer_question(
        None,
        ChatRequest(question="Question?", mode="keyword", rerank=False),
        provider=provider,
    )

    assert response.abstained is True
    assert response.citations == []
    assert "validate the citations" in response.answer


def test_chat_abstains_when_claim_is_not_supported(monkeypatch):
    monkeypatch.setattr(chat_service, "search", lambda *_args, **_kwargs: [_source()])
    provider = StaticProvider(
        [
            json.dumps(
                {
                    "answer": "Employees receive 30 days of annual leave.",
                    "abstained": False,
                    "abstention_reason": None,
                    "claims": [
                        {
                            "text": "Employees receive 30 days of annual leave.",
                            "source_chunk_ids": ["chunk-1"],
                        }
                    ],
                    "conflicts": [],
                }
            ),
            _verification("unsupported"),
        ]
    )

    response = answer_question(
        None,
        ChatRequest(question="How much leave?", mode="keyword", rerank=False),
        provider=provider,
    )

    assert response.abstained is True
    assert response.evidence_verified is False
    assert response.claims == []
    assert response.citations == []
    assert "No generated claim" in response.abstention_reason


def test_chat_fails_closed_when_verifier_returns_malformed_output(monkeypatch):
    monkeypatch.setattr(chat_service, "search", lambda *_args, **_kwargs: [_source()])
    provider = StaticProvider(
        [
            json.dumps(
                {
                    "answer": "Eligible employees receive annual leave.",
                    "abstained": False,
                    "abstention_reason": None,
                    "claims": [
                        {
                            "text": "Eligible employees receive annual leave.",
                            "source_chunk_ids": ["chunk-1"],
                        }
                    ],
                    "conflicts": [],
                }
            ),
            "not json",
        ]
    )

    response = answer_question(
        None,
        ChatRequest(question="What leave?", mode="keyword", rerank=False),
        provider=provider,
    )

    assert response.abstained is True
    assert response.evidence_verified is False
    assert response.claims == []
    assert response.citations == []
    assert "could not be verified" in response.abstention_reason


def test_chat_abstains_without_calling_provider_when_retrieval_is_empty(monkeypatch):
    monkeypatch.setattr(chat_service, "search", lambda *_args, **_kwargs: [])

    response = answer_question(
        None,
        ChatRequest(question="Question?", mode="keyword", rerank=False),
    )

    assert response.abstained is True
    assert response.evidence_verified is False
    assert response.citations == []
    assert response.llm_provider == "none"


def test_chat_endpoint_returns_abstention_when_no_passages_exist(client):
    response = client.post(
        "/api/v1/chat",
        json={"question": "What is the travel policy?", "mode": "keyword"},
    )

    assert response.status_code == 200
    assert response.json()["abstained"] is True
    assert response.json()["citations"] == []


def test_chat_endpoint_returns_request_id_and_logs_safe_provider_error(client, monkeypatch, caplog):
    from app.api.routes import chat as chat_routes

    def fail_with_provider_error(*_args, **_kwargs):
        response = httpx.Response(
            401,
            request=httpx.Request("POST", "https://llm.example.test/v1/chat/completions"),
        )
        cause = httpx.HTTPStatusError(
            "unauthorized", request=response.request, response=response
        )
        raise LLMProviderError("LLM provider returned HTTP 401") from cause

    monkeypatch.setattr(chat_routes, "answer_question", fail_with_provider_error)
    response = client.post(
        "/api/v1/chat",
        headers={"X-Request-ID": "chat-debug-123"},
        json={"question": "Question?", "mode": "keyword", "rerank": False},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "I couldn’t complete that answer just now. Please try again shortly."
    )
    assert "HTTP 401" in caplog.text
    assert "chat-debug-123" in caplog.text


def test_chat_endpoint_shows_friendly_message_for_provider_rate_limit(client, monkeypatch):
    from app.api.routes import chat as chat_routes

    def fail_with_rate_limit(*_args, **_kwargs):
        response = httpx.Response(
            429,
            request=httpx.Request("POST", "https://llm.example.test/v1/chat/completions"),
        )
        cause = httpx.HTTPStatusError(
            "rate limited", request=response.request, response=response
        )
        raise LLMProviderError("LLM provider returned HTTP 429") from cause

    monkeypatch.setattr(chat_routes, "answer_question", fail_with_rate_limit)
    response = client.post(
        "/api/v1/chat",
        json={"question": "Question?", "mode": "keyword", "rerank": False},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "The AI service is busy right now. Please wait a moment and try again."
    )


def test_chat_model_allowlist_selects_only_configured_model(monkeypatch):
    from pydantic import SecretStr

    from app.core.config import get_settings
    from app.generation import providers

    settings = get_settings().model_copy(
        update={
            "llm_provider": "openai_compatible",
            "llm_api_key": SecretStr("test-key"),
            "llm_model": "provider/default",
            "llm_available_models": "provider/default,provider/alternate",
        }
    )
    monkeypatch.setattr(providers, "get_settings", lambda: settings)

    selected = providers.get_llm_provider("provider/alternate")
    assert selected.model_name == "provider/alternate"
    with pytest.raises(LLMProviderError, match="not available"):
        providers.get_llm_provider("provider/unlisted")

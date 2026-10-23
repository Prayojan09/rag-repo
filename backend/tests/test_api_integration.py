from uuid import uuid4

from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.core import security
from app.core.config import get_settings


def test_citation_endpoint_returns_source_metadata(client: TestClient, monkeypatch, upload_dir):
    from app.api.routes import documents as document_routes

    settings = get_settings().model_copy(update={"upload_dir": str(upload_dir)})
    monkeypatch.setattr(document_routes, "get_settings", lambda: settings)
    uploaded = client.post(
        "/api/v1/documents",
        files={"file": ("citation.txt", b"# Safety\n\nWear eye protection.", "text/plain")},
    )
    assert uploaded.status_code == 201
    document = client.get(f"/api/v1/documents/{uploaded.json()['id']}").json()
    chunk_id = document["chunks"][0]["id"]

    response = client.get(f"/api/v1/citations/{chunk_id}")

    assert response.status_code == 200
    assert response.json()["filename"] == "citation.txt"
    assert response.json()["heading"] == "Safety"
    assert response.json()["text"] == "Wear eye protection."


def test_citation_endpoint_hides_missing_or_non_uuid_citations(client: TestClient):
    assert client.get(f"/api/v1/citations/{uuid4()}").status_code == 404
    assert client.get("/api/v1/citations/not-a-uuid").status_code == 422


def test_feedback_endpoint_persists_valid_feedback(client: TestClient):
    answer_id = uuid4()
    response = client.post(
        "/api/v1/feedback",
        json={"answer_id": str(answer_id), "rating": "not_helpful", "comment": "  Wrong page.  "},
    )

    assert response.status_code == 201
    assert response.json()["answer_id"] == str(answer_id)
    assert response.json()["rating"] == "not_helpful"
    assert response.json()["comment"] == "Wrong page."


def test_feedback_rejects_invalid_rating(client: TestClient):
    response = client.post(
        "/api/v1/feedback",
        json={"answer_id": str(uuid4()), "rating": "maybe"},
    )
    assert response.status_code == 422


def test_chat_saves_user_and_structured_assistant_history(client: TestClient, monkeypatch):
    from app.api.routes import chat as chat_routes
    from app.schemas.chat import ChatResponse

    def fake_answer(_db, request, **_kwargs):
        return ChatResponse(
            answer_id=str(uuid4()),
            question=request.question,
            answer="Saved response",
            abstained=True,
            abstention_reason="No indexed evidence.",
            claims=[],
            conflicts=[],
            citations=[],
            evidence_verified=False,
            verification_model=None,
            retrieval_mode="keyword",
            reranker_model=None,
            llm_provider="none",
            llm_model=None,
        )

    monkeypatch.setattr(chat_routes, "answer_question", fake_answer)
    response = client.post("/api/v1/chat", json={"question": "What is saved?", "mode": "keyword"})

    assert response.status_code == 200
    conversation_id = response.json()["conversation_id"]
    history = client.get(f"/api/v1/conversations/{conversation_id}")
    assert history.status_code == 200
    messages = history.json()["messages"]
    assert [(item["role"], item["content"]) for item in messages] == [
        ("user", "What is saved?"),
        ("assistant", "Saved response"),
    ]
    assert messages[1]["response_data"]["answer_id"] == response.json()["answer_id"]


def test_configured_bearer_token_protects_api_routes(client: TestClient, monkeypatch):
    settings = get_settings().model_copy(update={"api_access_token": SecretStr("a" * 40)})
    monkeypatch.setattr(security, "get_settings", lambda: settings)

    assert client.get("/api/v1/documents").status_code == 401
    assert (
        client.get("/api/v1/documents", headers={"Authorization": "Bearer wrong"}).status_code
        == 401
    )
    assert (
        client.get("/api/v1/documents", headers={"Authorization": f"Bearer {'a' * 40}"}).status_code
        == 200
    )
    # Health probes stay public for orchestrator liveness/readiness checks.
    assert client.get("/health").status_code == 200


def test_document_access_and_search_are_isolated_between_tenants(
    client: TestClient, monkeypatch, upload_dir
):
    from app.api.routes import documents as document_routes
    from app.core import security

    settings = get_settings().model_copy(
        update={
            "upload_dir": str(upload_dir),
            "api_access_token": None,
            "api_tenant_tokens": {
                "tenant-a": SecretStr("a" * 40),
                "tenant-b": SecretStr("b" * 40),
            },
        }
    )
    monkeypatch.setattr(security, "get_settings", lambda: settings)
    monkeypatch.setattr(document_routes, "get_settings", lambda: settings)
    headers_a = {"Authorization": f"Bearer {'a' * 40}"}
    headers_b = {"Authorization": f"Bearer {'b' * 40}"}
    payload = b"# Hiring\n\nCompany hiring requires a structured interview."

    uploaded_a = client.post(
        "/api/v1/documents",
        files={"file": ("policy.txt", payload, "text/plain")},
        headers=headers_a,
    )
    uploaded_b = client.post(
        "/api/v1/documents",
        files={"file": ("policy.txt", payload, "text/plain")},
        headers=headers_b,
    )
    assert uploaded_a.status_code == uploaded_b.status_code == 201
    doc_a = uploaded_a.json()
    doc_b = uploaded_b.json()
    assert doc_a["id"] != doc_b["id"]
    assert doc_a["version"] == doc_b["version"] == 1

    assert client.get("/api/v1/documents", headers=headers_a).json()["total"] == 1
    assert client.get("/api/v1/documents", headers=headers_b).json()["total"] == 1
    assert client.get(f"/api/v1/documents/{doc_a['id']}", headers=headers_b).status_code == 404
    details_a = client.get(f"/api/v1/documents/{doc_a['id']}", headers=headers_a).json()
    citation_id = details_a["chunks"][0]["id"]
    assert client.get(f"/api/v1/citations/{citation_id}", headers=headers_b).status_code == 404

    search_response = client.post(
        "/api/v1/search",
        json={"query": "structured interview", "mode": "keyword"},
        headers=headers_b,
    )
    assert search_response.status_code == 200
    assert {row["document_id"] for row in search_response.json()["results"]} == {doc_b["id"]}

    assert client.delete(f"/api/v1/documents/{doc_a['id']}", headers=headers_b).status_code == 404

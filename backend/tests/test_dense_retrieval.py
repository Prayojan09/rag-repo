from dataclasses import dataclass

from qdrant_client import QdrantClient

from app.retrieval.dense import DenseVectorService


class DeterministicEmbedder:
    model_name = "test/deterministic"

    def dimension(self):
        return 3

    def encode(self, texts):
        vectors = []
        for text in texts:
            normalized = text.lower()
            if any(term in normalized for term in ("vacation", "leave", "paid time off")):
                vectors.append([1.0, 0.0, 0.0])
            elif any(term in normalized for term in ("security", "firewall")):
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return vectors


@dataclass
class Chunk:
    id: str
    text: str


def test_qdrant_indexes_searches_filters_and_deletes_points():
    service = DenseVectorService(
        client=QdrantClient(":memory:"),
        embedder=DeterministicEmbedder(),
    )
    vacation_id = "a2a3d712-31b1-4be9-b77d-9980f71c0d61"
    security_id = "b2b3d712-31b1-4be9-b77d-9980f71c0d61"
    service.index_document(
        "doc-vacation",
        "leave-policy.txt",
        1,
        [Chunk(vacation_id, "Annual leave and vacation policy")],
    )
    service.index_document(
        "doc-security",
        "security-policy.txt",
        1,
        [Chunk(security_id, "Firewall security requirements")],
    )

    results = service.search("paid time off", top_k=2)
    assert results[0].chunk_id == vacation_id
    assert results[0].document_id == "doc-vacation"
    security_results = service.search("security", top_k=2, document_ids=["doc-security"])
    assert security_results[0].chunk_id == security_id

    service.delete_document("doc-vacation")
    assert all(match.document_id != "doc-vacation" for match in service.search("paid time off", 2))


def test_collection_setup_creates_required_keyword_payload_indexes():
    service = DenseVectorService(
        client=QdrantClient(":memory:"),
        embedder=DeterministicEmbedder(),
    )
    indexed_fields = []
    create_payload_index = service.client.create_payload_index

    def record_index(*, collection_name, field_name, field_schema, wait):
        indexed_fields.append((field_name, field_schema))
        return create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=field_schema,
            wait=wait,
        )

    service.client.create_payload_index = record_index
    service._ensure_collection()

    assert indexed_fields == [
        ("document_id", "keyword"),
        ("tenant_id", "keyword"),
    ]


def test_qdrant_search_and_delete_are_scoped_by_tenant():
    service = DenseVectorService(
        client=QdrantClient(":memory:"),
        embedder=DeterministicEmbedder(),
    )
    leave_id_a = "c2a3d712-31b1-4be9-b77d-9980f71c0d61"
    leave_id_b = "d2a3d712-31b1-4be9-b77d-9980f71c0d61"
    service.index_document(
        "same-document-id-a",
        "leave-a.txt",
        1,
        [Chunk(leave_id_a, "annual leave policy")],
        tenant_id="tenant-a",
    )
    service.index_document(
        "same-document-id-b",
        "leave-b.txt",
        1,
        [Chunk(leave_id_b, "annual leave policy")],
        tenant_id="tenant-b",
    )

    assert [match.chunk_id for match in service.search("leave", 5, tenant_id="tenant-a")] == [
        leave_id_a
    ]
    assert [match.chunk_id for match in service.search("leave", 5, tenant_id="tenant-b")] == [
        leave_id_b
    ]

    service.delete_document("same-document-id-a", tenant_id="tenant-a")
    assert [match.chunk_id for match in service.search("leave", 5, tenant_id="tenant-b")] == [
        leave_id_b
    ]


def test_search_endpoint_returns_database_source_metadata(client, monkeypatch, upload_dir):
    from app.api.routes import documents as document_routes
    from app.api.routes import search as search_routes
    from app.core.config import get_settings

    settings = get_settings().model_copy(update={"upload_dir": str(upload_dir)})
    monkeypatch.setattr(document_routes, "get_settings", lambda: settings)
    uploaded = client.post(
        "/api/v1/documents",
        files={
            "file": (
                "leave-policy.txt",
                b"# Leave\n\nEmployees receive annual vacation leave.",
                "text/plain",
            )
        },
    )
    assert uploaded.status_code == 201
    document = client.get(f"/api/v1/documents/{uploaded.json()['id']}").json()
    chunk = document["chunks"][0]
    vector_service = DenseVectorService(
        client=QdrantClient(":memory:"),
        embedder=DeterministicEmbedder(),
    )
    vector_service.index_document(
        document_id=document["id"],
        filename=document["filename"],
        version=document["version"],
        chunks=[Chunk(chunk["id"], chunk["text"])],
    )
    monkeypatch.setattr(search_routes, "get_dense_vector_service", lambda: vector_service)

    response = client.post(
        "/api/v1/search",
        json={"query": "paid time off", "document_ids": [document["id"]], "mode": "dense"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "dense"
    assert payload["results"][0]["document_id"] == document["id"]
    assert payload["results"][0]["filename"] == "leave-policy.txt"
    assert payload["results"][0]["text"] == chunk["text"]


def test_keyword_and_hybrid_modes_return_rank_details(client, monkeypatch, upload_dir):
    from app.api.routes import documents as document_routes
    from app.api.routes import search as search_routes
    from app.core.config import get_settings
    from app.schemas.documents import DenseSearchResult
    from app.services import hybrid_search

    settings = get_settings().model_copy(update={"upload_dir": str(upload_dir)})
    monkeypatch.setattr(document_routes, "get_settings", lambda: settings)
    uploaded = client.post(
        "/api/v1/documents",
        files={
            "file": (
                "benefits.txt",
                b"Eligible employees may request a sabbatical leave after five years.",
                "text/plain",
            )
        },
    )
    assert uploaded.status_code == 201
    unrelated = client.post(
        "/api/v1/documents",
        files={
            "file": (
                "remote-work.txt",
                b"Employees can work remotely with manager approval.",
                "text/plain",
            )
        },
    )
    assert unrelated.status_code == 201
    document = client.get(f"/api/v1/documents/{uploaded.json()['id']}").json()
    chunk = document["chunks"][0]
    dense_result = DenseSearchResult(
        chunk_id=chunk["id"],
        document_id=document["id"],
        filename=document["filename"],
        version=document["version"],
        ordinal=chunk["ordinal"],
        page_number=chunk["page_number"],
        heading=chunk["heading"],
        text=chunk["text"],
        score=0.87,
    )
    monkeypatch.setattr(hybrid_search, "dense_search", lambda *_args, **_kwargs: [dense_result])
    monkeypatch.setattr(
        search_routes,
        "get_dense_vector_service",
        lambda: type("VectorService", (), {"model_name": "test-model"})(),
    )

    keyword_response = client.post(
        "/api/v1/search", json={"query": "sabbatical", "mode": "keyword"}
    )
    assert keyword_response.status_code == 200
    keyword_payload = keyword_response.json()
    assert keyword_payload["mode"] == "keyword"
    assert keyword_payload["embedding_model"] is None
    assert keyword_payload["results"][0]["keyword_rank"] == 1
    assert keyword_payload["results"][0]["dense_rank"] is None

    hybrid_response = client.post("/api/v1/search", json={"query": "sabbatical", "mode": "hybrid"})
    assert hybrid_response.status_code == 200
    hybrid_payload = hybrid_response.json()
    result = hybrid_payload["results"][0]
    assert hybrid_payload["mode"] == "hybrid"
    assert result["dense_rank"] == 1
    assert result["keyword_rank"] == 1
    assert result["score"] == 2 / 61


def test_search_endpoint_can_apply_cross_encoder_reranking(client, monkeypatch, upload_dir):
    from app.api.routes import documents as document_routes
    from app.api.routes import search as search_routes
    from app.core.config import get_settings
    from app.services import hybrid_search

    settings = get_settings().model_copy(update={"upload_dir": str(upload_dir)})
    monkeypatch.setattr(document_routes, "get_settings", lambda: settings)
    uploaded = client.post(
        "/api/v1/documents",
        files={"file": ("policy.txt", b"Annual leave is available to employees.", "text/plain")},
    )
    assert uploaded.status_code == 201
    document = client.get(f"/api/v1/documents/{uploaded.json()['id']}").json()
    chunk = document["chunks"][0]
    from app.schemas.documents import DenseSearchResult

    result = DenseSearchResult(
        chunk_id=chunk["id"],
        document_id=document["id"],
        filename=document["filename"],
        version=document["version"],
        ordinal=chunk["ordinal"],
        page_number=chunk["page_number"],
        heading=chunk["heading"],
        text=chunk["text"],
        score=0.4,
    )
    monkeypatch.setattr(hybrid_search, "search_bm25", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(hybrid_search, "dense_search", lambda *_args, **_kwargs: [result])

    class TestReranker:
        model_name = "test-cross-encoder"

        def rerank(self, _query, candidates, _top_k):
            return [
                candidate.model_copy(update={"score": 0.95, "rerank_score": 0.95})
                for candidate in candidates
            ]

    class TestVectorService:
        model_name = "test-embedder"

    monkeypatch.setattr(search_routes, "get_dense_vector_service", lambda: TestVectorService())
    monkeypatch.setattr(search_routes, "get_reranker", lambda: TestReranker())
    response = client.post(
        "/api/v1/search",
        json={"query": "annual leave", "mode": "hybrid", "rerank": True, "top_k": 1},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["reranker_model"] == "test-cross-encoder"
    assert payload["results"][0]["score"] == 0.95

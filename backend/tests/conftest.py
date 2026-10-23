import shutil
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core import security as security_core
from app.db.models import Base
from app.db.session import get_db
from app.main import app


@pytest.fixture
def vector_service_stub(monkeypatch):
    from app.services import document_ingestion

    class VectorServiceStub:
        model_name = "test-embedder"

        def __init__(self):
            self.indexed_documents = {}
            self.deleted_documents = []

        def index_document(self, document_id, filename, version, chunks, tenant_id="default"):
            self.indexed_documents[document_id] = {
                "filename": filename,
                "version": version,
                "tenant_id": tenant_id,
                "chunks": list(chunks),
            }

        def delete_document(self, document_id, tenant_id="default"):
            self.deleted_documents.append(document_id)
            self.indexed_documents.pop(document_id, None)

        def search(self, _query, _top_k, _document_ids=None):
            return []

    vector_service = VectorServiceStub()
    monkeypatch.setattr(document_ingestion, "get_dense_vector_service", lambda: vector_service)
    return vector_service


@pytest.fixture
def upload_dir():
    path = Path(__file__).parent / f".upload-test-{uuid4().hex}"
    path.mkdir()
    yield path
    shutil.rmtree(path)


@pytest.fixture
def test_engine():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture
def db_override(test_engine):
    test_session = sessionmaker(bind=test_engine, autoflush=False, expire_on_commit=False)

    def override_get_db():
        session = test_session()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def client(db_override, vector_service_stub, test_engine, monkeypatch, tmp_path) -> TestClient:
    from app.api.routes import documents as document_routes
    from app.core.config import get_settings
    from app.db.models import Document
    from app.ingestion import report as ingestion_report
    from app.services import document_ingestion

    test_settings = get_settings().model_copy(
        update={
            "ingestion_report_dir": str(tmp_path / "reports"),
            "environment": "development",
            "api_access_token": None,
            "api_tenant_tokens": {},
            "google_client_id": None,
            "google_client_secret": None,
            "google_redirect_uri": None,
        }
    )
    monkeypatch.setattr(security_core, "get_settings", lambda: test_settings)
    monkeypatch.setattr(ingestion_report, "get_settings", lambda: test_settings)

    test_session = sessionmaker(bind=test_engine, autoflush=False, expire_on_commit=False)

    def process_test_document(
        document_id: str, storage_root: Path, chunk_size: int, chunk_overlap: int
    ) -> None:
        with test_session() as session:
            document = session.get(Document, document_id)
            if document is None or document.status != "queued":
                return
            path = storage_root / document.storage_key
            content = path.read_bytes()
            document_ingestion._run_ingestion(session, document, content, chunk_size, chunk_overlap)

    monkeypatch.setattr(document_routes, "process_queued_document", process_test_document)
    return TestClient(app)


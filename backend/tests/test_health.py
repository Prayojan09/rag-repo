def test_health_reports_liveness(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.headers.get("x-request-id")
    assert response.json() == {"status": "ok"}


def test_frontend_origin_is_allowed_for_browser_api_requests(client):
    response = client.get("/health", headers={"Origin": "http://localhost:5173"})
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_readiness_reports_dependency_status(client, monkeypatch):
    from app.api.routes import health

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def execute(self, _query):
            return None

    class FakeEngine:
        def connect(self):
            return FakeConnection()

    class FakeResponse:
        def raise_for_status(self):
            return None

    monkeypatch.setattr(health, "engine", FakeEngine())
    monkeypatch.setattr(health.httpx, "get", lambda *_args, **_kwargs: FakeResponse())
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready", "checks": {"postgres": "ok", "qdrant": "ok"}}


def test_readiness_reports_unavailable_dependencies(client, monkeypatch):
    from app.api.routes import health

    class FailedEngine:
        def connect(self):
            raise RuntimeError("private connection details")

    def fail_request(*_args, **_kwargs):
        raise health.httpx.ConnectError("private endpoint details")

    monkeypatch.setattr(health, "engine", FailedEngine())
    monkeypatch.setattr(health.httpx, "get", fail_request)
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "checks": {"postgres": "unavailable", "qdrant": "unavailable"},
    }

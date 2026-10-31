"""Compare frontend search options without printing retrieved text or secrets."""

import httpx

from app.core.config import get_settings

settings = get_settings()
token = (
    settings.api_access_token.get_secret_value()
    if settings.api_access_token
    else next(iter(settings.api_tenant_tokens.values())).get_secret_value()
    if settings.api_tenant_tokens
    else None
)
headers = {"Authorization": f"Bearer {token}"} if token else {}
for mode, rerank in (("hybrid", True), ("hybrid", False), ("dense", True), ("keyword", False)):
    try:
        response = httpx.post(
            "http://127.0.0.1:8001/api/v1/search",
            headers=headers,
            json={"query": "test", "mode": mode, "top_k": 3, "rerank": rerank},
            timeout=120,
        )
        print(f"{mode} rerank={rerank}: HTTP {response.status_code}")
        if response.is_success:
            payload = response.json()
            print("  results:", len(payload.get("results", [])), "reranker:", payload.get("reranker_model"))
        else:
            payload = response.json()
            print("  detail:", payload.get("detail", "No detail"))
    except httpx.HTTPError as exc:
        print(f"{mode} rerank={rerank}: request error {type(exc).__name__}")

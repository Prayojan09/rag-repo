# Progress

## Phase 0 — inspection and plan

- Inspected the project directory: it was empty.
- Environment: Python 3.13.2 available; FastAPI, SQLAlchemy, Alembic, pytest, Ruff, and psycopg not installed. Docker CLI unavailable in the current shell.
- Chosen modular monolith, PostgreSQL as metadata store, Qdrant as future vector store, environment-based settings, and a migration baseline without speculative business tables.

## Phase 1 — backend foundation

- Added FastAPI app, settings, logging/error handling, SQLAlchemy session infrastructure, Alembic baseline, and health/readiness routes.
- Added Docker Compose for PostgreSQL, Qdrant, and the API container, plus setup docs and initial health tests.
- Verification is pending in this environment because Docker is unavailable and Python dependencies are not installed. No later implementation phases have started.

## Phase 2 — document ingestion

- Added PDF, DOCX, and UTF-8 TXT validation and extraction, with page/heading provenance where available.
- Added configurable paragraph-aware chunking, deterministic chunk IDs, PostgreSQL document/chunk models, and an Alembic migration.
- Added size limits, SHA-256 duplicate detection, filename versioning, status/error fields, retry endpoint, safe generated storage keys, and cascading metadata/source deletion.
- Added document upload/list/detail/delete API and integration/unit tests using SQLite. Original files persist in a Compose named volume.
- Verification: 11 tests passed and Ruff checks passed. Alembic history loads the new migration. Live PostgreSQL migration and Docker Compose startup remain unverified because Docker is unavailable in this environment.

## Phase 3 — dense vector retrieval

- Added configurable Sentence Transformers embedding model/device/batch size and Qdrant collection settings.
- Added Qdrant chunk indexing during upload/retry, model-specific collection isolation, document deletion cleanup, and per-document reindexing.
- Added `POST /api/v1/search` for cosine dense search with optional ready-document filtering and SQL-backed source validation.
- Added in-memory Qdrant API tests using deterministic test embeddings; production uses the real Sentence Transformers model.
- Verification: 13 tests passed, including actual in-memory Qdrant indexing, vector search, metadata filtering, deletion, and API source resolution. Ruff passed. Installed `qdrant-client` 1.19.1 and `sentence-transformers` 6.1.0. The default transformer weights were not downloaded and live Qdrant/PostgreSQL integration was not exercised because Docker is unavailable; the configured production embedder uses Sentence Transformers directly and loads weights lazily on first request.

## Phase 4 â€” BM25 and hybrid retrieval

- Added Okapi BM25 keyword search over ready PostgreSQL chunks, including extracted headings, with document filters and Unicode-aware case-insensitive tokenization.
- Added dense-only, keyword-only, and hybrid modes to `POST /api/v1/search`; hybrid search fuses independently ranked candidates with configurable RRF and exposes component ranks/scores.
- Added configurable candidate depth and RRF constant; documented the current per-request BM25 corpus scan as a scale trade-off.
- Verification: 14 tests passed and Ruff checks passed. Tests cover real BM25 ranking from the SQLite document corpus, dense retrieval through in-memory Qdrant, and RRF rank fusion. Live Postgres/Qdrant service validation is still unavailable because Docker is not installed.

## Phase 5 â€” cross-encoder reranking

- Added a lazy-loaded Sentence Transformers CrossEncoder service, configurable model/device/batch size, stable reranked results with original retrieval scores preserved, and optional reranking on `POST /api/v1/search`.
- Added `reranker_model`, `rerank_score`, and `retrieval_score` response fields. Model weights load on first reranked request.
- Verification: the full suite passes (19 tests) and Ruff passes. Reranking order and score preservation are tested with a scorer fixture; the default model weights are lazy-loaded and were not downloaded during this verification. The production path uses Sentence Transformers CrossEncoder.

## Phase 6 â€” LLM answer generation

- Added a provider protocol and configurable OpenAI-compatible Chat Completions adapter using JSON mode; API key, base URL, model, timeout, and output size are environment settings.
- Added `POST /api/v1/chat`, which retrieves, reranks, sends passages as untrusted evidence to the model, and returns structured claims, citation sources, conflicts, and abstention metadata.
- Added response schema validation and rejects model citations that reference chunks outside the retrieved candidate set. Full claim-to-evidence semantic verification remains Phase 7.
- Chat can abstain without an LLM call when retrieval produces no passages. Upload and search remain usable with `LLM_PROVIDER=disabled`.
- Verification: the full suite passes (19 tests) and Ruff passes. The provider adapter is tested through an HTTP mock transport; grounded response parsing, citation ID restriction, and no-evidence abstention are tested. A live LLM provider call was not made because no LLM credentials/model were configured.

## Phase 7 — citation and evidence-support verification

- Added source ID and metadata validation before citation resolution; citation document/page values are taken from retrieved database records.
- Added a structured support-verification pass over each generated claim and conflict, using only the passages cited for that statement. Unsupported and uncertain statements are dropped, and returned answer prose is rebuilt only from statements marked supported.
- Added fail-closed behavior: malformed, incomplete, or failed verifier output returns an abstention with no unchecked claims or citations. The API response exposes `evidence_verified` and `verification_model`.
- The configured LLM performs this second call; it increases latency and can add API cost. Semantic judgments remain fallible and need evaluation against labeled evidence examples.
- Verification: `backend/.venv/Scripts/python.exe -m pytest` passed (23 tests); `backend/.venv/Scripts/ruff.exe check app tests alembic` passed. Tests cover supported claims, unsupported claim abstention, malformed verifier output, source citation rejection, and empty-retrieval abstention. No live LLM or database service was used.

## Phase 8 — API integration

- Completed the v1 endpoint set: document upload/list/detail/delete, search, chat, citation lookup, and feedback. Document listing remains bounded and paginated; request payloads use validated Pydantic schemas.
- Added `GET /api/v1/citations/{citation_id}`, which resolves UUID chunk IDs to authoritative source metadata only while the parent document is ready.
- Added `POST /api/v1/feedback`, answer IDs in chat responses, and a PostgreSQL feedback table with an Alembic migration. Feedback stores rating and optional comment without persisting prompts or generated answers.
- Added bearer token authentication on all `/api/v1` endpoints using `API_ACCESS_TOKEN`, constant-time comparisons, minimum token length, public health/readiness probes, and required token configuration outside development. Development remains open when the token is unset.
- This is a shared single-workspace token; per-user roles, document ownership, and tenant isolation remain Phase 10 work.
- Verification: `backend/.venv/Scripts/python.exe -m pytest` passed (28 tests); `backend/.venv/Scripts/ruff.exe check app tests alembic` passed. Tests cover citation resolution, feedback validation/storage, and bearer access control. `alembic upgrade head` succeeded against the configured PostgreSQL database, and `alembic current` confirmed `0003_feedback` is at head.

## Phase 9 — evaluation (in progress)

- Added a versioned fictional policy corpus with 10 labeled questions, source chunk IDs, key answer facts, and one unanswerable question.
- Added reusable Recall@K, MRR@K, and nDCG@K metrics plus a runner comparing dense, BM25, hybrid RRF, and hybrid with the configured cross-encoder. It uses temporary SQLite and in-memory Qdrant and reports warm-query mean/p95 latency separately from model-load/index setup.
- Added opt-in chat evaluation for key-fact recall, gold citation precision, abstention accuracy, verifier rejection rate, end-to-end latency, provider-reported token counts, and cost when the operator supplies current rates. It does not run unless `--include-chat` is passed because that calls the configured provider and may incur cost.
- Added prompt/completion token accounting across generation and evidence-verifier calls in the OpenAI-compatible adapter and exposed token counts plus unverified claim counts in chat responses.
- Actual BM25 measurement on this synthetic corpus (`K=5`, nine answerable queries): Recall@5 `0.889`, MRR@5 `0.806`, nDCG@5 `0.826`, mean query latency `1.677 ms`, p95 `4.710 ms`. Full per-query output is saved at `backend/evaluation/results/keyword-baseline.json`. One labeled query about the expense-report deadline missed its relevant passage; the unanswerable query returned keyword matches, so abstention behavior must be assessed through chat evaluation.
- Dense and reranker weights were not found in standard local Hugging Face cache directories, so those paths were not run here. No paid LLM chat evaluation was run. The dataset is synthetic and small; its numbers are smoke-test measurements, not a production quality claim.
- Verification: full suite `31 passed`; Ruff passed for app, tests, migrations, and scripts. The keyword benchmark executed successfully using the actual BM25 implementation and temporary SQLite corpus.

## Phase 10 — production hardening (in progress)

- Added tenant IDs to documents and feedback, a migration that places existing rows in `default`, tenant-local duplicate/version constraints, per-tenant bearer token mapping, and SQL/Qdrant tenant filters. Legacy Qdrant points without tenant metadata are only visible to `default` during migration compatibility.
- Added cross-tenant API coverage for document list/detail/delete, citation resolution, and keyword search; added Qdrant tenant filter coverage.
- Added request IDs, generic client errors with server-side error-type logging, LLM retry limits for transient failures, explicit untrusted-document prompt instructions, bounded BM25 corpus loading, SQLAlchemy connection-pool controls, and upload execution in a worker thread.
- Hardened the API image to run as an unprivileged user, added a readiness healthcheck, and configured Compose restart policies. Updated setup/security documentation with tenant token configuration and deployment limitations.
- Verification: full test suite passed (`34 passed`); Ruff passed for app, tests, migrations, and scripts. PostgreSQL migration `0004_tenant_isolation` was applied and `alembic current` confirmed it is at head. Docker is not installed in this shell, so the image build and Compose startup could not be verified. The readiness endpoint returns HTTP 503 when either dependency is unavailable, which lets container/orchestrator healthchecks detect failure.

## Next

Phase 11 frontend is under implementation. Phase 10 still needs deployment-specific verification (tested backups, TLS/ingress, secret rotation, and measured resource limits). Phase 9's dense/hybrid/reranked and optional paid chat comparisons also remain incomplete because model weights and provider configuration were unavailable.

## Phase 11 — web frontend

- Added a responsive React, TypeScript, Vite, and Tailwind frontend with document uploads/listing, retry/re-index/delete actions, extracted passage preview, search mode/reranker controls, chat, citation lookup, feedback, and connection settings.
- Added a local Vite proxy to the FastAPI backend and configurable CORS origins for direct deployments. The UI keeps the bearer token in browser local storage and warns that frontend build variables must not contain secrets.
- Verification: `npm run build` passed (TypeScript check and Vite production bundle); the built page was served by Vite preview and returned HTTP 200. Backend suite passed (`35 passed`) and Ruff passed, including the configured frontend CORS-origin check. The Vite development optimizer hit this environment's filesystem boundary while scanning dependencies; the production build and preview work. Docker is not required to run the web application.
- Security update: npm audit initially found seven Tailwind CSS 3 dependency advisories (five high, two moderate). Migrated the PostCSS integration to Tailwind CSS 4.3.3 and updated the lockfile. `npm audit --json` now reports zero vulnerabilities; the frontend production build still passes.
- Upload diagnosis: the reported PDFs extracted successfully, but Qdrant collection creation returned HTTP 403 (`Global manage access is required`). Added safe, actionable status text and logging for Qdrant permission/credential/connectivity failures. The operator must create the expected collection or grant suitable Qdrant access before retrying.
- Qdrant Cloud upload follow-up: the configured collection was created with 384-dimensional cosine vectors. Actual indexing exposed missing keyword payload indexes required by Qdrant for `document_id` and `tenant_id` filters. Added automatic index creation in the backend, created both indexes on the configured cloud collection, and retried the latest PDF successfully (19 vectors; document status `ready`).
- Search error diagnostics: live `/ready` reported PostgreSQL and Qdrant healthy, and live search requests succeeded in hybrid, dense, and keyword modes, including hybrid with reranking. Updated the frontend to preserve backend 503 details and the backend to log search exceptions with a request ID so errors from a different/stale API instance can be traced.
- Chat error diagnostics: confirmed the local LLM provider, model ID, and key are configured, but did not send document passages to the external LLM provider as a diagnostic call. Chat failures now return a request ID; logs record only safe error types and provider HTTP status or connection/timeout categories, never prompts, outputs, response bodies, or credentials. Provider/chat tests passed (`12 passed`); Ruff passed.
- Added a rotating local backend diagnostic log at `backend/data/logs/backend.log` in addition to terminal output, so chat request IDs can be looked up after a request. The user’s previous request ID predates this file logging change and is not recoverable from the workspace.
- Latest verification before the payload-index fix: backend `36 passed`, Ruff passed, frontend production build passed, and npm audit reported zero vulnerabilities.

## Upload, OCR, and chat history update
- Chat follow-ups now include bounded context from the same tenant-scoped conversation for retrieval and generation; other conversation histories are not loaded, and history is not treated as factual evidence.

- Raised the default upload cap to 200 MiB in settings and the example environment file; updated the local backend `.env` limit without exposing its other values.
- Added PNG/JPG/TIFF/BMP uploads and OCR for image files and textless PDF pages using PyMuPDF plus Tesseract. Added page/pixel limits, language/DPI settings, Docker Tesseract packages, and setup documentation.
- Added PostgreSQL conversation/message models and Alembic migration `0005_conversations`; added tenant-scoped conversation create/list/read/delete endpoints and automatic chat message persistence with structured citations.
- Added a recent chat list and history restore to the frontend; user messages are right aligned and assistant messages remain left aligned.
- Verification pending in this update.
- OCR follow-up: the current Windows process PATH did not include Tesseract even though it was installed under `C:\Program Files\Tesseract-OCR`. The extractor now discovers the standard install path and accepts an optional `TESSERACT_CMD`. Verified OCR on the latest failed five-page PDF: 56,065 characters extracted across five pages. Reindexing could not complete from this execution environment because outbound Qdrant sockets are blocked (`WinError 10013`); retry indexing from the user's local backend after restart.
- Retry follow-up: logs show OCR completed and Qdrant accepted the point-delete request, but the subsequent vector upsert failed. Added a dedicated 30-second `QDRANT_TIMEOUT_SECONDS` (instead of reusing the 3-second readiness timeout) and more specific safe error reporting for Qdrant connection/response failures. The local `.env` was updated without reading or exposing its secrets. Retrying the write from this execution environment remains blocked by Windows socket restrictions.

## Upload latency update

- Changed uploads to return after validation, duplicate checks, and durable source-file storage. Extraction, OCR, chunking, embeddings, and Qdrant indexing now run as a FastAPI background task using a separate SQLAlchemy session.
- The frontend now explains that processing continues after upload, polls queued/indexing documents, and lets users retry queued or interrupted documents.
- Increased embedding batch size from 32 to 64 to reduce model and Qdrant batch overhead; reduced default OCR resolution from 150 to 120 DPI for faster scanned-file processing. Both settings remain configurable; higher DPI may improve small-text OCR accuracy.
- Verification pending in this update.

- Large scanned-PDF follow-up: raised the OCR page limit from 200 to 500 (covers the reported 490-page PDF) and changed generic OCR errors to distinguish a missing executable, missing language pack, or page-specific runtime failure. Verified this environment finds 	esseract.exe and has eng/osd data; the uploaded 87.9 MiB file was not present in this checkout, so its actual failure cause cannot be confirmed here. Restart the backend and retry to receive the more useful error if it still fails.
- Verification: backend tests passed (43 passed; one Qdrant local-payload-index warning), Ruff passed, and frontend production build passed.

## Indexing latency follow-up

- Backend log for a 247 KB PDF showed 32 extracted segments, 19 chunks, 0.26 seconds of extraction, negligible chunking, and 45.47 seconds inside dense vector indexing. The log showed Sentence Transformer loading during the indexing path.
- Added a best-effort embedding model warmup during FastAPI startup so cold model initialization happens before the first document is indexed. Startup may take longer on a cold cache; health/docs remain available if warmup fails.
- Added separate duration logs for collection setup, old-vector deletion, embedding, and Qdrant writes to identify any remaining bottleneck.
- Verification pending; next restart will be the first runtime check of startup warmup and the new timings.

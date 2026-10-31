# Document Intelligence RAG Backend

Backend foundation, document ingestion, hybrid retrieval, reranking, and grounded answer generation for a production-oriented document intelligence service. The API accepts PDF, DOCX, UTF-8 TXT, Markdown, HTML, and common image files (PNG, JPG, TIFF, BMP), stores originals on disk, extracts text, preserves page/heading provenance where available, and stores versioned document metadata and stable chunks in PostgreSQL. Search can use Sentence Transformers dense vectors, BM25 keyword retrieval, or combine both rankings with Reciprocal Rank Fusion (RRF), then optionally rerank candidates with a cross-encoder. Chat generates structured, cited answers from retrieved passages through a configurable OpenAI-compatible LLM endpoint and checks cited claims against their passages before returning them. Bearer credentials scope documents and feedback to configured tenant IDs.

## Requirements

- Docker Desktop with Docker Compose v2
- Python 3.11+ (the container uses 3.12)

## Run locally

From the project root:

```powershell
Copy-Item backend/.env.example backend/.env
docker compose up -d postgres qdrant
python -m venv backend/.venv
backend/.venv/Scripts/Activate.ps1
python -m pip install -r backend/requirements-dev.txt
Set-Location backend
alembic upgrade head
uvicorn app.main:app --reload --port 8001
```

Open `http://localhost:8001/docs`; liveness is `GET /health`, readiness is `GET /ready`. The local LLM gateway can remain on port 8000. Or run all services in containers with `docker compose up --build` after copying the env file. Docker Compose publishes the API on port 8001 and persists PostgreSQL, Qdrant, and uploaded source files in named volumes. The Compose backend reaches a host-installed local gateway at `host.docker.internal:8000`; set `DOCKER_LLM_GATEWAY_BASE_URL` if your gateway uses another address.

The backend logs to its terminal and to `backend/data/logs/backend.log` (rotated at 2 MiB, with three backups). Chat failures include a request ID and safe provider status details; prompts, document passages, and credentials are not written to this diagnostic log.

## Frontend (Phase 11)

The React web app lives in `frontend/`. It provides document upload and management, hybrid/semantic/keyword search, tenant-scoped saved conversation history, right-aligned user messages and left-aligned assistant responses, source citations, citation previews, feedback, and backend connection settings. Docker is optional.

Start PostgreSQL and Qdrant using your local installations. In one terminal, start the backend from `backend/` using the virtual environment commands above. In another terminal, run:

```powershell
Set-Location frontend
npm install
npm run dev
```

Open the local URL Vite prints (normally `http://127.0.0.1:5173`). During local development Vite forwards API requests to `http://127.0.0.1:8001`. The page does not collect or store API credentials: configure the API origin through `VITE_API_BASE_URL` for a deployed frontend, and configure authentication at the deployment or backend layer. Add the frontend origin to `CORS_ALLOWED_ORIGINS` in `backend/.env`. Keep API and LLM secrets out of `VITE_` variables because frontend variables are visible in the browser. LLM provider credentials belong in `backend/.env`.

Build the static frontend bundle with `npm run build`; the output is written to `frontend/dist/`. Production hosting must serve that directory and forward `/api`, `/health`, and `/ready` to the FastAPI service (or set the API URL and allowed CORS origin).

To point the backend container at the Qdrant Cloud URL configured in `backend/.env`, pass that file for Compose variable interpolation: `docker compose --env-file backend/.env up --build`. The API container reads the API key from the same ignored env file. Without this option, Compose uses its local Qdrant container.

For a manually created Qdrant Cloud collection, use the configured collection name shown by the backend logs/settings, one unnamed dense vector of size 384 with Cosine distance, and no sparse vector. The backend also requires keyword payload indexes for `document_id` and `tenant_id`; it creates these automatically when missing. The API key therefore needs permission to create payload indexes and read/write points in the collection. These indexes support document cleanup and tenant-isolated search.

### Google sign-in

The web app supports password-free Google sign-up and login. On a user's first successful Google sign-in, the backend creates an identity from Google's verified, stable account subject and assigns a private tenant. Later sign-ins resolve to that same tenant, even if the user's email address changes. The app does not store a Google password. It keeps an opaque, expiring, HttpOnly session cookie; the session token is hashed in PostgreSQL, and browser write requests also use a CSRF token.

To configure it, create a Google Cloud OAuth client with application type **Web application**. Add the frontend origin under **Authorized JavaScript origins** (local development: `http://localhost:5173`) and the exact backend callback under **Authorized redirect URIs** (`http://localhost:8001/api/v1/auth/google/callback`). While the OAuth consent screen is in testing mode, add the accounts that should be able to sign in as test users. Copy the client ID and secret into `backend/.env` as `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET`, set `GOOGLE_REDIRECT_URI` to that exact callback, and set `AUTH_FRONTEND_URL` to the frontend origin. Restart the backend and run `cd backend; alembic upgrade head`. With Google OAuth configured, the UI requires sign-in even in development. For production, use HTTPS URLs, set `AUTH_SESSION_COOKIE_SECURE=true`, add the frontend origin to `CORS_ALLOWED_ORIGINS`, and the API hostname to `ALLOWED_HOSTS`. `backend/.env.production.example` contains the deployment variables.

An optional `GOOGLE_DEFAULT_TENANT_EMAIL` maps one verified Google email to the legacy `default` tenant so that account can access data created before individual identities were added. Leave it unset for fresh, isolated Google-user tenants. API bearer tokens remain available for non-browser integrations; the browser uses Google sessions.

## Document ingestion API

- `POST /api/v1/documents` — multipart upload using the `file` field; returns status and version. An identical SHA-256 upload returns `409` with the existing document ID.
- `GET /api/v1/documents?limit=50&offset=0` — paginated metadata.
- `GET /api/v1/documents/{document_id}` — metadata and extracted chunks.
- `POST /api/v1/documents/{document_id}/retry` — retry a failed/queued ingestion.
- `POST /api/v1/documents/{document_id}/reindex` — rebuild vectors from the document's stored chunks.
- `DELETE /api/v1/documents/{document_id}` — remove metadata, chunks, and stored source file.

`/api/v1/conversations` provides tenant-scoped create, list, read, and delete operations for saved chat history. Chat messages and structured citation data are stored in PostgreSQL. Each request uses up to the latest eight messages from only its selected conversation to interpret follow-ups; the backend fetches this history by tenant-scoped conversation ID and does not accept history from the browser. Prior turns provide context but are not citation evidence. Apply migrations with `alembic upgrade head` before using the new history endpoints.

`POST /api/v1/search` accepts JSON such as `{"query":"annual leave policy","top_k":5}` and defaults to `hybrid` mode. Set `mode` to `dense`, `keyword`, or `hybrid` to evaluate each retriever independently. Set `rerank: true` to score a larger candidate set with a cross-encoder and return the best passages. Hybrid results use RRF over the two ranked candidate lists; the response includes component ranks and scores, while `score` is the mode's dense similarity, BM25 score, fused RRF score, or cross-encoder score when reranking is enabled. Set `document_ids` to restrict search to selected ready documents. BM25 currently builds its corpus from PostgreSQL chunks for each request, so it is simple and always reflects ready documents, but request cost grows with corpus size; a persistent/incremental keyword index is a later scaling improvement. The default CPU embedding model is `sentence-transformers/all-MiniLM-L6-v2`; it downloads on first use and is cached. The default cross-encoder is `cross-encoder/ms-marco-MiniLM-L-6-v2` and also downloads on first rerank. Docker Compose persists the model cache in `models_cache`. Pin `EMBEDDING_MODEL_REVISION` for repeatable embedding deployments; use a new Qdrant collection name when intentionally changing embedding models. Search candidate and RRF settings are configurable in `.env`.

`POST /api/v1/chat` runs hybrid retrieval, cross-encoder reranking, structured answer generation, and claim-level evidence verification. The response includes an `answer_id`, only claims the configured LLM verifier marked supported by their cited passages, resolved document/page metadata, and an `evidence_verified` flag. Submit a rating and that `answer_id` to `POST /api/v1/feedback`. `GET /api/v1/citations/{citation_id}` resolves a chunk ID to its current ready document source. Page and document metadata come from indexed source records, not the LLM. The verifier uses the same selected model and makes a second API call when there are claims or conflicts, which adds latency and may add provider cost. Model-based support checks can still make mistakes; malformed or unavailable verifier responses cause a safe abstention. Configure `LLM_PROVIDER=openai_compatible`, a gateway URL (`LLM_GATEWAY_BASE_URL`, or `LLM_BASE_URL`), a key (`LLM_GATEWAY_API_KEY`, or `LLM_API_KEY`), `LLM_MODEL`, and comma-separated `LLM_AVAILABLE_MODELS` in the ignored `backend/.env`. The chat model picker only exposes IDs in that allowlist; it never receives the gateway URL or API key. The gateway must implement the OpenAI Chat Completions-compatible API. No key is required for ingestion or search. Retrieved passage text is treated as untrusted prompt data. Questions with no retrieved passages abstain without calling the LLM.

All `/api/v1` routes require bearer authentication outside development. `API_ACCESS_TOKEN` is the legacy/default-tenant credential; for separate workspaces, configure `API_TENANT_TOKENS` as a JSON object mapping tenant IDs to distinct random tokens of at least 32 characters. Each API request is assigned to the tenant belonging to its credential, and document listing, reads, deletion, search, citations, and feedback are scoped to that tenant in PostgreSQL and Qdrant. Existing records are migrated into tenant `default`. During development only, no configured token leaves the API open to the `default` tenant. `/health` and `/ready` remain public for probes. This is tenant-level bearer access, not individual user identity, roles, or per-document permissions; distribute each tenant token only to trusted clients.

PDF text extraction preserves one-based page numbers and detects likely headings by font size, numbering, and short standalone title-like lines. DOCX heading styles, HTML semantic headings, and Markdown heading markers are retained as section context. Normalization handles Unicode variants, stray punctuation spaces, line-wrap hyphenation, common list markers, repeated page furniture, and configurable metadata labels conservatively. Citation text remains clean; contextual section text is stored separately for embeddings and BM25. PDFs use PyMuPDF with pdfplumber as a fallback; sparse or scanned pages use Tesseract OCR. For uncertain pages, the backend compares Tesseract's contrast-enhanced and adaptive-threshold results before invoking the handwriting model, so TrOCR is less likely to replace readable printed text with a poorer guess. DOCX paragraphs, tables, and headings are extracted; DOCX page numbers are unavailable from this extractor. HTML navigation, footer, script, and style elements are omitted. Markdown fenced code, lists, and tables are retained as blocks. For handwriting, the local `microsoft/trocr-small-handwritten` model is downloaded from Hugging Face on first use and cached locally. TrOCR can make errors with cursive or unusual writing, so extracted handwriting should be reviewed. Install Tesseract OCR with English language data and Python dependencies from `backend/requirements.txt`. When running directly on Windows, the backend searches the standard `C:\Program Files\Tesseract-OCR` location; otherwise put `tesseract` on PATH or set `TESSERACT_CMD` to the executable path. The Docker image installs Tesseract automatically. OCR is limited to 500 pages per file, with configurable language (`OCR_LANGUAGE`), resolution (`OCR_DPI`), and fallback page segmentation (`OCR_FALLBACK_PAGE_SEGMENTATION_MODE`). Uploads are limited to 100 MB per file (`MAX_UPLOAD_BYTES`). Chunk sizing, overlap, and embedding batch size can be adjusted in `.env`. The upload request returns after the file is stored; the Documents page tracks background extraction and indexing until status becomes Indexed. Per-document quality reports are persisted under `backend/data/reports/` and included in document responses after processing completes.

To inspect extraction locally before indexing, run from `backend/`:

```powershell
python -m app.ingestion.inspect "C:\path\to\document.pdf" --show-clean --show-chunks
```

The command prints raw extraction, cleaned text, detected section paths, chunks, and a quality report. Keep this output private when inspecting confidential documents.

For example, with the backend running:

```powershell
curl.exe -F "file=@C:\path\to\policy.pdf" http://localhost:8001/api/v1/documents
```

Uploads are authenticated and tenant-scoped with the same bearer credential as the other API operations. Each document is limited to 100 MB; larger files are rejected by both the webpage and API. Filenames are sanitized, and stored paths use generated identifiers. Keep the upload volume private and use a reverse proxy or gateway for TLS and network access controls.

For handwritten scans, TrOCR runs line recognition in batches of 16 and uses up to 4 CPU threads by default. These can be adjusted with `HANDWRITING_OCR_BATCH_SIZE` and `HANDWRITING_OCR_CPU_THREADS`. The backend logs elapsed OCR time per PDF page so long scans show progress in the backend terminal.

When an LLM provider is configured, ingestion can also request a structured Markdown rendering of the extracted text, following the document-structuring rules in `backend/app/ingestion/structuring.py`. It is enabled by default and can be turned off with `DOCUMENT_STRUCTURING_ENABLED=false`. The rendering appears as a separate preview; original extracted chunks remain the source for retrieval and citations. The full extracted text (up to `DOCUMENT_STRUCTURING_MAX_CHARS`, 40,000 by default) is sent to the configured LLM provider for this rendering, which can add latency and provider cost. If structuring is disabled, the provider is unavailable, or the text exceeds the limit, indexing continues with the original extraction. Apply the database update with `cd backend; alembic upgrade head` before restarting the API. Existing documents need to be reprocessed to generate the new preview.

## Tests and lint

```powershell
Set-Location backend
python -m pytest -q
python -m ruff check app tests alembic
```

## Evaluation

The versioned smoke corpus and labels are in `backend/evaluation/dataset.json`. Run from `backend/`:

```powershell
python scripts/evaluate.py --modes keyword --output evaluation/results/keyword-baseline.json
python scripts/evaluate.py --modes dense keyword hybrid hybrid_reranked --output evaluation/results/full.json
```

The full retrieval comparison loads the configured Sentence Transformers embedding and cross-encoder weights, indexes the sample corpus in an in-memory Qdrant, and reports Recall@K, MRR@K, nDCG@K, warm-query mean/p95 latency, and model-load/index setup time. It does not touch your PostgreSQL documents or Qdrant Cloud collection. To evaluate chat, abstention, citation precision against the labeled sources, exact key-fact recall, token usage, and estimated cost, add `--include-chat`. That flag makes real calls to the configured LLM and may incur charges. Pass current provider rates with `--input-cost-per-1m` and `--output-cost-per-1m`; without both rates, the report includes token counts but leaves cost blank. The included fictional corpus is a smoke test, not a substitute for a representative, human-reviewed company dataset.

Readiness checks PostgreSQL and Qdrant with bounded timeouts and reports dependency statuses; it returns HTTP 503 if either check fails. The backend container runs as an unprivileged user and its image healthcheck uses `/ready`. The default `docker-compose.yml` is for local development only: it includes local database credentials and publishes PostgreSQL and Qdrant ports.

## Deployment preparation

The app-only `docker-compose.prod.yml` targets a single backend instance connected to managed PostgreSQL and Qdrant. It publishes the API on loopback only, runs migrations as a separate one-shot service before the API starts, and leaves TLS termination to a reverse proxy or hosting ingress. It is a deployment starting point, not a turnkey multi-user production service.

1. Copy `backend/.env.production.example` to `backend/.env.production`. This file is ignored by Git. Replace every placeholder, use managed PostgreSQL with TLS and Qdrant Cloud or another private Qdrant service, and set exact HTTPS origins and hostnames.
2. Configure Google OAuth using the values in `backend/.env.production.example`; the web app then gives each Google account an isolated tenant. Pin both model revisions to real immutable Hugging Face commit IDs. For non-browser service integrations, optionally set a unique random `API_ACCESS_TOKEN` or distinct `API_TENANT_TOKENS`. `ENVIRONMENT=production` refuses missing authentication, insecure OAuth cookie settings, weak default database passwords, remote Qdrant without an API key/TLS, wildcard origins/hosts, and floating model revisions. Keep all secrets in the host's secret manager where supported.
3. Put an HTTPS reverse proxy or hosting ingress in front of `127.0.0.1:8001`, serve the frontend on the configured HTTPS origin, and route the Google callback to the backend. If the frontend and API use different hosts (for example `app.example.com` and `api.example.com`), build the frontend with `VITE_API_BASE_URL=https://api.example.com` so browser requests send the API-host session cookie. The browser uses HttpOnly Google sessions and CSRF checks; service integrations can use bearer tokens. An optional `GOOGLE_DEFAULT_TENANT_EMAIL` maps one verified account to legacy default-tenant data. This application has tenant isolation, but it does not yet implement organization roles or invitation-based access control.
4. Set up automated PostgreSQL, Qdrant, and uploaded-file backups and test restore procedures. Mount persistent encrypted storage for uploads. Restrict network access to the database/vector services. Apply proxy-level request size limits, timeouts, and rate limits; this repo does not implement a distributed rate limiter.
5. Build and start:

```powershell
Copy-Item backend/.env.production.example backend/.env.production
docker compose -f docker-compose.prod.yml config
docker compose -f docker-compose.prod.yml build
docker compose -f docker-compose.prod.yml up -d
docker compose -f docker-compose.prod.yml ps
```

`docker compose ... config` verifies Compose structure, not the external credentials or provider connectivity. Confirm `/ready` through the private API route after launch. The service downloads and warms the configured embedding model at startup; allow outbound access to Hugging Face or pre-populate the persistent model cache. Configure an LLM provider in the secret environment file to enable chat. Ingestion currently uses in-process FastAPI background tasks; for multi-replica deployments and guaranteed recovery after a process crash, a durable job queue and worker service must be added before production traffic.

On 2026-10-23, the installed Python environment reported no known vulnerabilities with `pip-audit --local`, and `npm audit` reported zero advisories for the frontend lockfile. Rerun both scans in CI on every release. The Docker image and operating-system packages were not scanned because Docker is unavailable in this workspace. Also conduct load tests with representative PDF/OCR files and validate managed-service backups, resource limits, and proxy rules in the target environment.

The backend container runs as an unprivileged user. Its production configuration disables interactive API documentation and adds trusted-host and response security headers. The development Compose credentials and exposed database/vector ports are not suitable for deployment. `docker compose down -v` removes development named volumes, including uploaded files.

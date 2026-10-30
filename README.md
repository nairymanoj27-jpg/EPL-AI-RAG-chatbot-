# Premier League RAG Chatbot

A production-shaped retrieval-augmented generation (RAG) service over an **existing**
English Premier League Pinecone index. It embeds questions with Pinecone's hosted
`multilingual-e5-large` model, retrieves passages, and asks Google Gemini to answer
**only from that retrieved evidence**, with citations such as `[S1]`.

The application is read-only with respect to Pinecone: it never creates, upserts,
updates, deletes, or re-embeds your vectors.

> **Live integration status:** the implementation, 130 automated tests, and an
> HTTP smoke test were run locally without real credentials. The Pinecone and
> Gemini network paths are implemented against the pinned SDKs but must be
> validated against your own index by following the steps below.

---

## Contents

- [Features](#features)
- [Architecture](#architecture)
- [Project layout](#project-layout)
- [Requirements](#requirements)
- [Quick start on Windows 11](#quick-start-on-windows-11)
- [Index and schema discovery](#index-and-schema-discovery)
- [Configuration](#configuration)
- [Running the app](#running-the-app)
- [API](#api)
- [Conversation memory](#conversation-memory)
- [Frontend](#frontend)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Costs](#costs)
- [Security](#security)
- [Design decisions and limitations](#design-decisions-and-limitations)
- [Production checklist](#production-checklist)

---

## Features

- **Existing vectors only.** Query-only Pinecone wrapper with no mutation methods.
- **Configurable index contract.** Index name, namespace, host, text metadata
  field, dimension, and metric all come from validated configuration.
- **Correct E5 query convention.** Uses Pinecone Inference with
  `input_type="query"`; no manual `query:` prefix (which would double-prefix).
- **Hard dimension checks.** A model/dimension mismatch fails loudly instead of
  silently returning meaningless neighbours.
- **Grounded Gemini answers.** Retrieved text is treated as untrusted data,
  escaped, and placed in explicit `<evidence>` blocks. The system prompt forbids
  outside knowledge, invented stats, and fabricated citations.
- **Honest fallback.** If retrieval returns nothing usable, the API answers with a
  fixed "not enough information" message without calling Gemini.
- **Stable source mapping.** Each citation `[Sn]` maps to a returned source with a
  Pinecone vector ID, score, text, and metadata.
- **Multi-turn memory.** SQLite-backed, isolated by conversation ID, with a
  server-side clear endpoint and a UI "New chat" button.
- **Defensive HTTP layer.** Optional `X-API-Key`, configured CORS, request body
  limits, upstream timeouts, sanitized errors, and request IDs in logs/responses.
- **Readiness and liveness.** `/health` is dependency-free; `/ready` validates
  Pinecone dimension, metric, namespace, and Gemini access with a short cache.
- **Self-contained frontend.** Plain HTML/CSS/JS served at `/ui`; no build step,
  no CDN, no third-party scripts.
- **Windows-first docs.** Exact PowerShell commands and a `run.ps1` launcher.

---

## Architecture

```text
Browser (/ui)
    │  POST /chat {question, conversation_id?}
    ▼
FastAPI route ──► ChatService ──► SQLite conversation history (optional)
    │                 │
    │                 ├─► RetrievalService
    │                 │      ├─ Pinecone Inference embed_query(input_type="query")
    │                 │      └─ Pinecone Index.query(namespace, top_k, metadata)
    │                 │
    │                 ├─ PromptBuilder (escape + budget + citation IDs)
    │                 ├─ GeminiService (google-genai, grounded generation)
    │                 └─ Citation validator (unknown / missing citations)
    ▼
ChatResponse {answer, conversation_id, sources[], cited_source_ids[], warnings[]}
```

Request flow in detail:

1. Validate the request and resolve `top_k` against server limits.
2. Load recent history for the conversation ID (history is context, never evidence).
3. Embed the question with `multilingual-e5-large` (`input_type="query"`).
4. Query the configured namespace and discard matches below the optional
   `MIN_RELEVANCE_SCORE`.
5. Extract chunk text from `PINECONE_TEXT_FIELD` (with documented fallbacks).
6. Build a bounded prompt: escaped evidence, metadata, source IDs `S1..Sn`, and
   recent conversation turns.
7. Generate with Gemini at low temperature.
8. Validate citations, attach warnings for unsupported or missing citations, and
   save the exchange.
9. Return the answer plus the exact source passages used.

The app never treats a previous assistant message as evidence. A new retrieval is
performed for every question.

---

## Project layout

```text
p4/
├─ app/
│  ├─ asgi.py                 # uvicorn entry: app.asgi:app
│  ├─ main.py                 # app factory, lifespan, middleware, error handlers
│  ├─ config.py               # typed settings + validation + safe startup warnings
│  ├─ schemas.py              # request/response models (extra fields forbidden)
│  ├─ dependencies.py         # service dependencies + optional API-key auth
│  ├─ errors.py               # safe application error hierarchy
│  ├─ middleware.py           # request IDs + body-size limit
│  ├─ routes/
│  │  ├─ system.py            # /, /health, /ready
│  │  └─ chat.py              # /chat, /retrieve, DELETE /conversations/{id}
│  ├─ services/
│  │  ├─ embeddings.py        # E5 query embedding + dimension validation
│  │  ├─ pinecone_service.py  # read-only query / stats wrapper
│  │  ├─ retrieval.py         # embed → query → normalize chunks
│  │  ├─ prompt_builder.py    # grounding prompt, escaping, citation parsing
│  │  ├─ gemini_service.py    # Gemini client + error mapping
│  │  ├─ conversation_store.py# SQLite memory
│  │  ├─ chat_service.py      # orchestration
│  │  ├─ readiness.py         # cached readiness checks
│  │  └─ upstream_errors.py   # Pinecone exception translation
│  └─ utils/logging_config.py # request-ID logging, no secret logging
├─ frontend/                  # index.html, styles.css, app.js (served at /ui)
├─ scripts/inspect_index.py   # read-only index/namespace/metadata discovery
├─ tests/                     # 130 offline tests with fake Pinecone/Gemini
├─ .env.example
├─ requirements.txt
├─ requirements-dev.txt
├─ pyproject.toml            # Ruff lint configuration
├─ pytest.ini
└─ run.ps1
```

---

## Requirements

- Windows 11 with PowerShell 5.1 or PowerShell 7+
- Python **3.11+**; this project was built and tested with **Python 3.14.7**
- An existing Pinecone index containing EPL vectors
- A Pinecone API key that can **read** that index
- A Google Gemini API key (Google AI Studio)
- Internet access to Pinecone and Google

The app uses `multilingual-e5-large` through **Pinecone Inference**. It does not
download or run an embedding model locally.

---

## Quick start on Windows 11

Open **PowerShell** and run each block from the project directory:

```powershell
cd "C:\Users\nairy\OneDrive\Desktop\Projects\p4"
```

### 1. Create the virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

Verify the interpreter:

```powershell
.\.venv\Scripts\python.exe --version
```

### 2. Create your local configuration

```powershell
Copy-Item .env.example .env
notepad .env
```

Fill in at minimum:

```dotenv
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=...
GEMINI_API_KEY=...
```

Add `PINECONE_HOST` and `PINECONE_NAMESPACE` if your index uses them. **Never share
these values in chat, screenshots, or commits.** `.env` is git-ignored.

### 3. Discover and validate the index

```powershell
.\.venv\Scripts\python.exe scripts\inspect_index.py
```

Then inspect a few metadata records (read-only):

```powershell
.\.venv\Scripts\python.exe scripts\inspect_index.py --sample 3
```

Use the output to set `PINECONE_NAMESPACE`, `PINECONE_TEXT_FIELD`,
`EMBEDDING_DIMENSION`, and `EXPECTED_METRIC`. See
[Index and schema discovery](#index-and-schema-discovery).

### 4. Start the server

```powershell
.\run.ps1
```

If PowerShell blocks the script for the current session only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\run.ps1
```

Open:

- UI: <http://127.0.0.1:8000/ui/>
- API docs: <http://127.0.0.1:8000/docs>
- Readiness: <http://127.0.0.1:8000/ready>

`run.ps1` options:

```powershell
.\run.ps1 -Reload
.\run.ps1 -BindHost 0.0.0.0 -Port 8080
```

For a first end-to-end check, open `/ready`. It should return HTTP 200 and show
Pinecone dimension, metric, namespace count, and Gemini access. If it returns 503,
read the `detail` fields; they are designed to name the misconfigured setting.

> `/ready` checks metadata and connectivity only; it does not consume Pinecone
> Inference tokens. A monthly embedding quota can therefore block `/retrieve` and
> `/chat` while `/ready` still returns 200. In that case the API returns
> `429 upstream_rate_limited` with a quota-specific message.

---

## Index and schema discovery

### What the app assumes

- The index already contains the chunked EPL vectors.
- Vectors were produced with Pinecone Inference using
  `multilingual-e5-large` and `input_type="passage"` at ingestion time.
- Metadata contains the chunk text (default field: `text`).
- The configured namespace contains those vectors.

If ingestion used a different embedding model or a manual `passage:`/`query:`
convention, query vectors may not be comparable even when dimensions match.
Pinecone index stats cannot reveal the embedding model, so validate retrieval
quality with `/retrieve` before trusting answers.

### Using `scripts/inspect_index.py`

The script is read-only. It calls `describe_index_stats`, and with `--sample` it
uses `list` + `fetch` only. It does **not** require a Gemini key.

Typical output:

```text
Index:        epl-stats
Host:         https://epl-stats-abc123.svc.aped-1234.pinecone.io
Dimension:    1024
Metric:       cosine
Total vectors: 18432
Namespace:    'epl-2024'

Namespaces:
  'epl-2024': 18432 vector(s)

Checks:
  OK    Dimension, metric, and namespace are consistent with .env.
  NOTE  Pinecone does not report the embedding model used at ingestion...
```

With `--sample 3`, each vector's metadata keys are printed, plus whether
`PINECONE_TEXT_FIELD` is present and its character length. To print a 160-char
preview of the stored text:

```powershell
.\.venv\Scripts\python.exe scripts\inspect_index.py --sample 3 --preview 160
```

### Required settings from discovery

| Discovery result | Setting |
|---|---|
| Index name | `PINECONE_INDEX_NAME` |
| Index host (recommended) | `PINECONE_HOST` |
| Namespace name (`""` for default) | `PINECONE_NAMESPACE` |
| Index dimension (usually 1024) | `EMBEDDING_DIMENSION` |
| Index metric | `EXPECTED_METRIC` |
| Metadata key holding text | `PINECONE_TEXT_FIELD` |
| Embedding model used at ingestion | `PINECONE_EMBED_MODEL` |

`PINECONE_HOST` accepts either `my-index.svc.pinecone.io` or
`https://my-index.svc.pinecone.io`; config normalizes it and **rejects plain HTTP**.

The SDK uses `""` for the default namespace. `__default__` is accepted and
normalized to `""`.

### If multiple namespaces exist

Querying the wrong namespace is the most common cause of "no evidence" answers.
Run the inspect script, compare vector counts, and set `PINECONE_NAMESPACE`
explicitly. `/ready` lists the namespaces it can see when the configured one is
missing. Requests cannot choose a namespace; it is server-side configuration only.

### Text-field fallbacks

If `PINECONE_TEXT_FIELD` is missing, the app tries
`PINECONE_TEXT_FIELD_FALLBACKS` (`chunk_text,content,page_content` by default) and
adds a warning to the response. Use this only as a diagnostic; set the correct
field for production.

---

## Configuration

All settings are environment variables or entries in `.env`. See `.env.example` for
the full annotated template. Invalid values fail startup with a message naming the
field, never echoing secrets.

### Required

| Variable | Purpose |
|---|---|
| `PINECONE_API_KEY` | Pinecone data-plane key with read access |
| `PINECONE_INDEX_NAME` | Existing index name |
| `GEMINI_API_KEY` | Google Gemini API key |

### Pinecone

| Variable | Default | Notes |
|---|---|---|
| `PINECONE_HOST` | empty | Recommended data-plane host; HTTPS only |
| `PINECONE_NAMESPACE` | `""` | `""`/`__default__` = default namespace |
| `PINECONE_EMBED_MODEL` | `multilingual-e5-large` | Must match ingestion |
| `PINECONE_TEXT_FIELD` | `text` | Metadata key with chunk text |
| `PINECONE_TEXT_FIELD_FALLBACKS` | `chunk_text,content,page_content` | Comma-separated |
| `EMBEDDING_DIMENSION` | `1024` | `none` disables the dimension check (not recommended) |
| `EXPECTED_METRIC` | `cosine` | Mismatch is a readiness warning; `none` skips |

### Gemini and retrieval

| Variable | Default | Notes |
|---|---|---|
| `GEMINI_MODEL` | `gemini-flash-latest` | Any model available to your key |
| `GEMINI_TEMPERATURE` | `0.2` | 0–2 |
| `GEMINI_MAX_OUTPUT_TOKENS` | `1024` | 1–65536 |
| `TOP_K` | `5` | Default chunks to retrieve |
| `MAX_TOP_K` | `20` | Upper bound for request `top_k` |
| `MIN_RELEVANCE_SCORE` | empty | Optional floor; **calibrate before enabling**; `none` disables |
| `MAX_CONTEXT_CHUNKS` | `5` | Maximum chunks placed in the prompt |
| `MAX_CONTEXT_CHARACTERS` | `12000` | Prompt context budget |
| `MAX_QUESTION_CHARACTERS` | `2000` | Server-side question cap |
| `REQUEST_TIMEOUT_SECONDS` | `30` | Pinecone/Gemini timeout |
| `MAX_REQUEST_BYTES` | `65536` | Hard HTTP body limit |

`MIN_RELEVANCE_SCORE` is ignored for the `euclidean` metric (lower is closer) and a
warning is returned. Leaving it empty sends every top-k hit to Gemini.

### HTTP, memory, and operations

| Variable | Default | Notes |
|---|---|---|
| `APP_ENV` | `development` | `development`, `test`, or `production` |
| `LOG_LEVEL` | `INFO` | Standard logging levels |
| `CORS_ORIGINS` | localhost:8000 | Comma-separated exact origins |
| `APP_API_KEY` | empty | When set, protected endpoints require `X-API-Key` |
| `ENABLE_RETRIEVE_ENDPOINT` | `true` | Disable to avoid exposing raw text |
| `SERVE_FRONTEND` | `true` | Serve `frontend/` at `/ui` |
| `CONVERSATION_STORAGE` | `sqlite` | `sqlite` or `none` |
| `CONVERSATION_DB_PATH` | `data/conversations.sqlite3` | Absolute or project-relative |
| `CONVERSATION_MAX_TURNS` | `6` | Recent messages included as context |
| `CONVERSATION_MAX_CHARACTERS` | `6000` | History character budget |
| `READINESS_CACHE_SECONDS` | `60` | Cache for `/ready`; `0` disables |

---

## Running the app

Direct uvicorn command (equivalent to `run.ps1`):

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.asgi:app --host 127.0.0.1 --port 8000
```

Development reload:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.asgi:app --reload --reload-dir app --port 8000
```

Do not use `--reload` in production. Terminate with `Ctrl+C`.

### Endpoints

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `GET` | `/` | No | App name, version, environment, UI/docs links |
| `GET` | `/health` | No | Liveness; no external calls |
| `GET` | `/ready` | No | Readiness; 200 or 503, cached |
| `POST` | `/chat` | Optional | Retrieve + generate + cite |
| `POST` | `/retrieve` | Optional | Retrieval-only debugging; disable in production |
| `DELETE` | `/conversations/{id}` | Optional | Clear one conversation's history |
| `GET` | `/docs` | No | Swagger UI |
| `GET` | `/openapi.json` | No | OpenAPI schema |
| `GET` | `/ui/` | No | Static chat UI |

"Optional" auth means `X-API-Key` is required **only when `APP_API_KEY` is set**.
Health, root, docs, and the UI remain public so probes and documentation work.

---

## API

All errors use one envelope:

```json
{
  "error": {
    "code": "invalid_request",
    "message": "The request is invalid.",
    "request_id": "8d3c0f2a91b44e77"
  }
}
```

Validation errors add `details`. Upstream error messages are never passed through
verbatim.

### `POST /chat`

```powershell
$body = @{
  question = "Which player scored the most goals in a single Premier League season?"
  conversation_id = $null
} | ConvertTo-Json

$response = Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/chat" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body

$response.answer
$response.conversation_id
$response.sources | Format-Table citation_id, id, score
```

With application auth enabled:

```powershell
$headers = @{ "X-API-Key" = $env:APP_API_KEY }
Invoke-RestMethod -Uri "http://127.0.0.1:8000/chat" -Method Post `
  -Headers $headers -ContentType "application/json" -Body $body
```

Request fields:

| Field | Type | Required | Notes |
|---|---|---|---|
| `question` | string | Yes | Trimmed, 1..`MAX_QUESTION_CHARACTERS` |
| `conversation_id` | string | No | 8–64 chars, alphanumeric/`_`/`-` |
| `top_k` | integer | No | 1..`MAX_TOP_K` |

Unknown fields are rejected with HTTP 422, so clients cannot inject
`namespace`, `system_prompt`, or model settings.

Response (abridged):

```json
{
  "answer": "Alan Shearer and Andy Cole share the record with 34 goals [S1].",
  "conversation_id": "0f9a6d5e2c1b4a7f8e3d9c0b1a2f4e6d",
  "sources": [
    {
      "id": "epl-chunk-0042",
      "citation_id": "S1",
      "rank": 1,
      "score": 0.873,
      "text": "Alan Shearer (1994-95) and Andy Cole (1993-94) scored 34 goals...",
      "metadata": {"season": "1994-95", "team": "Blackburn Rovers"}
    }
  ],
  "cited_source_ids": ["S1"],
  "evidence_found": true,
  "model": "gemini-flash-latest",
  "warnings": []
}
```

`warnings` can report fallback metadata fields, truncated context, unknown
citations, missing citations, or memory-save failures. Treat warnings as signals to
inspect configuration or output quality.

### `POST /retrieve`

Returns the same source objects without calling Gemini. Useful for calibrating
`MIN_RELEVANCE_SCORE` and checking namespace/text-field configuration.

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/retrieve" -Method Post `
  -ContentType "application/json" `
  -Body (@{ question = "Arsenal 2003-04 season"; top_k = 5 } | ConvertTo-Json) |
  ConvertTo-Json -Depth 6
```

This endpoint exposes raw indexed text. In production, either set
`ENABLE_RETRIEVE_ENDPOINT=false` (returns 404) or require `APP_API_KEY`.

### `DELETE /conversations/{conversation_id}`

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/conversations/0f9a6d5e2c1b4a7f8e3d9c0b1a2f4e6d" -Method Delete
# { "conversation_id": "0f9a...", "cleared": true }
```

Clearing an unknown conversation returns `cleared: false` with HTTP 200. If
`CONVERSATION_STORAGE=none`, the endpoint returns HTTP 409.

### `GET /ready`

```json
{
  "ready": true,
  "version": "1.0.0",
  "checks": [
    {"name": "configuration", "ok": true, "required": true, "detail": "..."},
    {"name": "pinecone_dimension", "ok": true, "required": true, "detail": "Index dimension is 1024."},
    {"name": "pinecone_metric", "ok": true, "required": true, "detail": "Index metric is cosine."},
    {"name": "pinecone_namespace", "ok": true, "required": true, "detail": "Namespace 'epl-2024' contains 18432 vector(s)."},
    {"name": "gemini", "ok": true, "required": true, "detail": "Gemini model 'gemini-flash-latest' is accessible."}
  ]
}
```

A required failure returns HTTP 503 with `ready: false`. Results are cached for
`READINESS_CACHE_SECONDS`; restart or wait out the cache after changing `.env`.

---

## Conversation memory

- Stored in SQLite at `CONVERSATION_DB_PATH` (default
  `data/conversations.sqlite3`) with WAL mode and a connection per operation.
- Each conversation is isolated by ID. IDs are random 32-character hex values
  generated server-side when omitted.
- Only the last `CONVERSATION_MAX_TURNS` messages and
  `CONVERSATION_MAX_CHARACTERS` characters are loaded.
- History is used for two things: a lightweight query rewrite for short follow-ups
  and extra context in the prompt. It is explicitly labelled "not evidence".
- Retrieval is always re-run for the current question.
- A failed history load or save never loses the generated answer; the response
  includes a warning instead.
- Set `CONVERSATION_STORAGE=none` for a fully stateless deployment.

There is no global in-memory conversation dictionary. The SQLite store is suitable
for a single-process student/demo deployment, not for horizontally scaled or
multi-tenant production use. Replace it with a shared database (with proper
authorization) before scaling out.

Conversation IDs behave like bearer tokens: anyone who knows an ID can continue
that conversation. Use `APP_API_KEY`, TLS, and access controls for sensitive data.
The UI keeps the application API key in `sessionStorage` only.

---

## Frontend

Served at `/ui/` from `frontend/`. Features:

- Loading indicator and disabled send button while a request is in flight.
- Distinct error banner with HTTP-status-aware titles, request details, and retry.
- Source expanders showing citation ID, Pinecone vector ID, score, chunk text, and
  filtered metadata. Inline `[S1]` citations jump to the matching source.
- Follow-up questions in the same conversation; "New chat" clears UI and stored
  history.
- Example questions, character counter, auto-growing input, copy-answer button.
- API key dialog (session-only storage) for servers with `APP_API_KEY`.
- No `innerHTML` with server data; retrieved text cannot inject markup.
- Responsive layout, dark-mode support, reduced-motion support, keyboard focus
  states, and `aria-live` updates.

To serve the UI from a different origin, set this tag in `frontend/index.html`:

```html
<meta name="api-base" content="https://your-api.example.com" />
```

and add that frontend origin to `CORS_ORIGINS` on the API. Do not use a wildcard
with credentials; the app disables credentialed CORS and warns on `*`.

---

## Testing

Run the full offline suite:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Expected result at the time of writing:

```text
130 passed, 1 warning
```

The warning is a Starlette deprecation notice about its `httpx`-based test client;
it does not affect the application.

The tests mock Pinecone and Gemini and never make paid or network calls. Coverage
includes:

- API routes, validation, auth, CORS, body limits, error envelopes, request IDs.
- Chat grounding, insufficient-evidence handling, citations, warnings, isolation.
- Follow-up contextualization and its guard against unrelated short questions.
- Retrieval thresholds, metric handling, fallbacks, missing text, invalid `top_k`.
- E5 query convention and dimension mismatch.
- Prompt escaping/injection containment, budgets, history separation, citations.
- SQLite memory limits, clearing, isolation, and re-opening.
- Readiness dimension/namespace/metric logic and cache behavior.
- Gemini configuration, empty output, timeout, and HTTP-error mapping.
- Configuration validation and safe error messages.

Static checks:

```powershell
.\.venv\Scripts\python.exe -m compileall -q app scripts tests
.\.venv\Scripts\ruff.exe check app tests scripts
```

Expected lint output: `All checks passed!`.

Manual smoke test (with dummy credentials, no external calls):

```powershell
$env:PINECONE_API_KEY="dummy"; $env:PINECONE_INDEX_NAME="dummy"; $env:GEMINI_API_KEY="dummy"
.\.venv\Scripts\python.exe -m uvicorn app.asgi:app --port 8765
```

Then verify `/health`, `/`, `/docs`, `/ui/`, and `/openapi.json`. Chat requires real
credentials.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ConfigurationError: Invalid or missing configuration for: PINECONE_API_KEY...` | `.env` missing/blank | Copy `.env.example` to `.env`, fill required values |
| `PINECONE_HOST must use https://` | Plain HTTP host | Use the HTTPS host from the Pinecone console |
| `401 Invalid or missing API key` | `APP_API_KEY` is set | Send `X-API-Key`, or enter it in the UI dialog |
| `/ready` 503 with dimension mismatch | Embedding model/dimension mismatch | Set `EMBEDDING_DIMENSION` to the index dimension **only if** the same model produced the vectors |
| `/ready` 503 "Namespace ... was not found" | Wrong namespace | Set `PINECONE_NAMESPACE` to one listed in the detail |
| `/ready` 503 "contains no vectors" | Empty/wrong namespace | Re-check discovery; do not re-upload accidentally |
| Ready never recovers after a config change | Cached readiness | Wait `READINESS_CACHE_SECONDS` or restart |
| Answer always "not enough information" | Wrong namespace, text field, threshold too high, or embedding mismatch | Call `/retrieve`; inspect warnings and scores |
| Warning: configured text field missing | Metadata schema differs | Set `PINECONE_TEXT_FIELD` from `inspect_index.py --sample` |
| Warning: no citation | Model omitted `[S1]` | Inspect answer; consider lowering temperature or tightening prompts |
| Warning: unknown citation | Model invented `[S9]` | Treat as unsupported; inspect source text |
| `422 invalid_request` for `namespace` | Extra field rejected | Namespace is server-side configuration only |
| `413 request_too_large` | Body over `MAX_REQUEST_BYTES` | Shorten question or raise the limit deliberately |
| `429 upstream_rate_limited` | Gemini/Pinecone rate limit | Wait; reduce traffic; check quotas |
| `429` mentioning embedding token quota | Monthly Pinecone Inference limit for the model is exhausted | Check Pinecone usage/billing; upgrade or wait for the quota to reset. Do **not** switch embedding models mid-index |
| `502 upstream_error` | Upstream failure | Check server logs for exception type; verify keys/network |
| `504 upstream_timeout` | Slow upstream | Raise `REQUEST_TIMEOUT_SECONDS` cautiously or retry |
| `404 feature_disabled` on `/retrieve` | Endpoint disabled | Set `ENABLE_RETRIEVE_ENDPOINT=true` if intended |
| `409 conversation_memory_disabled` | `CONVERSATION_STORAGE=none` | Expected; history endpoint unavailable |
| `database is locked` | Concurrent SQLite writes, OneDrive sync, antivirus | Use a local non-synced path; reduce concurrency; replace SQLite for scale |
| CORS error in browser console | Origin not allowed | Add exact origin to `CORS_ORIGINS`; do not use `*` casually |
| `run.ps1` blocked | Execution policy | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| Gemini 404 / model rejected | Model unavailable to key | Set `GEMINI_MODEL` to a model your key can use |
| `ssl_verify` errors behind a proxy | Corporate TLS interception | Configure the OS trust store; **do not disable TLS verification** |

Useful checks:

```powershell
# Does the app start and validate config?
.\.venv\Scripts\python.exe -c "from app.config import get_settings; print(get_settings().pinecone_index_name)"

# Which app process is using the port?
Get-NetTCPConnection -LocalPort 8000 -State Listen

# Read-only index report
.\.venv\Scripts\python.exe scripts\inspect_index.py --sample 3
```

Logs include a request ID but never request bodies, API keys, retrieved text, or
full prompts. Use the `X-Request-ID` response header to correlate an error with the
server log.

---

## Costs

This app makes paid API calls when you use it:

1. **Pinecone Inference** — one embedding per question (plus one query against the
   index). `multilingual-e5-large` is billed by tokens/requests per Pinecone's
   current pricing.
2. **Pinecone reads** — each `/chat`, `/retrieve`, and uncached `/ready` performs a
   query or stats lookup.
3. **Gemini generation** — input tokens (system prompt + history + evidence) and
   output tokens for each answer. `MAX_CONTEXT_CHARACTERS`,
   `MAX_CONTEXT_CHUNKS`, `CONVERSATION_MAX_TURNS`, and
   `GEMINI_MAX_OUTPUT_TOKENS` are the main cost controls.

Practical cost controls:

- Calibrate and set `MIN_RELEVANCE_SCORE` so weak matches do not reach Gemini.
- Keep `TOP_K`/`MAX_CONTEXT_CHUNKS` small (3–5 is often enough).
- Keep history short.
- Use a Flash-class Gemini model for chat.
- `/ready` uses a metadata lookup rather than a generation call and is cached.
- Monitor provider dashboards; rates change. Check current
  [Pinecone pricing](https://www.pinecone.io/pricing/) and
  [Gemini API pricing](https://ai.google.dev/pricing) before budgeting.

A rough per-question estimate is: one embedding + one vector query + one short
Gemini request. Multiply by expected daily questions and add retries/regeneration.
Automated tests in this repository make **no** paid calls.

---

## Security

### Secrets

- Secrets are read only from environment variables/`.env`; `.env` is git-ignored.
- Config errors name variables, never values; secret fields use `SecretStr`.
- The browser never receives `PINECONE_API_KEY` or `GEMINI_API_KEY`.
- The optional `APP_API_KEY` is compared with `secrets.compare_digest` and is only
  stored in the UI's `sessionStorage`.
- Do not put secrets in query strings, logs, screenshots, or issue reports.

### Grounding and prompt injection

- Retrieved text is untrusted. It is control-character-stripped, HTML-escaped, and
  wrapped in `<text>`/`<source>` tags.
- The system instruction explicitly forbids following instructions found in
  evidence and forbids outside knowledge or fabricated citations.
- Conversation history is labelled as context, not evidence.
- The app does not execute, render, or fetch anything found in retrieved text.
- Citation validation flags unknown or missing source IDs.
- No prompt or retrieval content is logged.
- These measures reduce risk but cannot guarantee a language model never errs.
  Keep human review for high-stakes use.

### HTTP hardening

- Request bodies are capped before parsing, including chunked bodies without
  `Content-Length`.
- CORS uses explicit configured origins, no credentialed wildcard, and only the
  methods/headers the app needs.
- Upstream calls have timeouts and bounded retries (Gemini SDK retry options).
- Errors are mapped to safe codes/messages; stack traces stay server-side.
- Request IDs correlate logs without exposing internals.
- No SSL verification bypass is implemented or documented as an option.
- Pinecone access is query/stats only. There is no code path that writes.

### Deployment

- Put the API behind HTTPS (reverse proxy or platform ingress).
- Add authentication at the proxy if the app is public; `APP_API_KEY` is a simple
  shared-secret control, not a user system.
- Add rate limiting at the proxy; the app does not implement distributed rate
  limiting.
- Restrict `CORS_ORIGINS` to your real frontend origin.
- Disable `/retrieve` or protect it if indexed text is sensitive. Note that
  `/chat` also returns the source passages it used so the UI can show citations.
- Bind to `127.0.0.1` unless you intentionally expose the service.
- Keep dependencies patched and review provider security guidance.
- Back up and protect the SQLite file if conversations are sensitive; it is not
  encrypted at rest by this app.

---

## Design decisions and limitations

- **FastAPI, not Flask.** The implementation follows the specified app structure,
  async I/O, typed models, and OpenAPI support.
- **Pinecone Inference for query embeddings.** This avoids a local model download
  and guarantees the same provider family as ingestion, assuming ingestion used
  Pinecone's `multilingual-e5-large`. No fallback model is attempted.
- **No manual E5 prefix.** Pinecone applies the asymmetric `query:`/`passage:`
  convention via `input_type`; adding prefixes manually would double-prefix.
- **Per-response citation IDs.** `S1..Sn` are stable within a response and map to
  returned Pinecone vector IDs. The same vector can receive a different `Sn` on a
  later turn.
- **Conservative no-evidence behavior.** If every match is filtered or lacks text,
  Gemini is not called and a fixed honest message is returned.
- **Heuristic query rewrite.** Short/pronoun-heavy follow-ups prepend the previous
  user question for retrieval. This is cheap and deterministic; it is not an extra
  LLM call and does not change the question shown to the answer model.
- **No streaming.** Responses are returned after completion, which keeps citation
  validation and error handling simple.
- **SQLite memory.** Appropriate for one process/demo. Conversation IDs are bearer
  tokens; there is no per-user identity layer.
- **No reranker.** Retrieval is dense top-k plus optional score threshold.
- **No write operations.** Re-ingestion, re-embedding, and index management are
  intentionally out of scope.
- **Model output can still be wrong.** Warnings and citations support review; they
  do not prove factual correctness.

---

## Production checklist

- [ ] `.env` populated and excluded from version control
- [ ] `inspect_index.py` confirms namespace, dimension, metric, text field
- [ ] `/ready` returns 200 with real credentials
- [ ] `/retrieve` returns relevant chunks for 10–20 representative questions
- [ ] `MIN_RELEVANCE_SCORE` calibrated, or left empty intentionally
- [ ] `APP_API_KEY` set and delivered over TLS
- [ ] `CORS_ORIGINS` restricted to the real frontend origin
- [ ] `ENABLE_RETRIEVE_ENDPOINT=false` unless needed and protected
- [ ] `APP_ENV=production`, `LOG_LEVEL=INFO`, `--reload` disabled
- [ ] Reverse proxy provides TLS, rate limiting, request logging, and auth
- [ ] SQLite path on fast local storage, backed up if conversations matter
- [ ] Provider quotas/billing alerts configured
- [ ] `pytest` passes on the deployment Python version
- [ ] A sampled set of answers reviewed against cited sources

---

## What was verified in this workspace

- Python 3.14.7 virtual environment with pinned dependencies.
- `130 passed` offline tests covering the API, retrieval, prompts, memory,
  readiness, Gemini error mapping, and configuration.
- `compileall` succeeds for `app`, `scripts`, and `tests`.
- A real uvicorn process was started with dummy credentials and smoke-tested:
  `/health`, `/`, `/docs`, `/ui/`, `/openapi.json`, and request validation all
  returned expected results. The process was then stopped.
- **Not verified:** live Pinecone retrieval and live Gemini generation, because no
  credentials or index details were available in this environment. Use `/ready`,
  `/retrieve`, then `/chat` with your `.env` to complete validation.

# Approach: options analysed, decisions taken, work broken down

This file is the "thinking out loud" companion to `README.md`. It records the
alternatives considered for every major design axis, why one was chosen, and the
ordered list of small tasks that were then implemented.

---

## 1. Problem restatement

"ChatGPT for your documents" as a backend:

1. Users upload PDF / DOCX / TXT / MD files.
2. Files are processed **asynchronously**: text extraction → chunking → embeddings →
   AI summary / insights / tags.
3. Users open chat sessions over one or many documents and ask questions;
   answers are grounded in retrieved chunks and carry **citations**.
4. Metrics endpoints expose document statistics and processing / AI-cost metrics.

Scoring is equally weighted across AI-first development, product thinking and
technical implementation, so the solution has to be *balanced*: a solid RAG core,
sensible API ergonomics, and production hygiene (async, caching, limits, health,
observability), not a science project on any one axis.

---

## 2. Options analysed

### 2.1 Web framework
| Option | Pros | Cons |
|---|---|---|
| **FastAPI** ✅ | Native async, Pydantic validation, auto OpenAPI docs, SSE friendly | — |
| Flask | Familiar | Sync-first, no built-in validation/docs |
| Django + DRF | Batteries included | Heavy for an API-only RAG service; async story weaker |

**Decision:** FastAPI. Auto-generated `/docs` doubles as API documentation (a
deliverable).

### 2.2 Task queue
| Option | Pros | Cons |
|---|---|---|
| **Celery + Redis** ✅ | Mature; retries, `acks_late`, time limits, prefetch control, concurrency knobs, monitoring (Flower) | Sync worker model — async code must be bridged |
| RQ | Very simple | No native retries/back-off, single-threaded workers |
| arq | Asyncio native, tiny | Less known, fewer operational features |
| FastAPI `BackgroundTasks` | Zero infra | Runs in the API process — dies with it, no retries, no horizontal scaling. Not acceptable for "1000s of documents" |

**Decision:** Celery + Redis. The worker runs a **synchronous** SQLAlchemy engine
(psycopg 3) while the API uses the **async** engine (asyncpg). Both share one
declarative model module, so there is exactly one schema definition. Bridging
async LLM clients into a prefork worker adds fragility for no benefit — OpenAI and
Anthropic SDKs ship sync clients.

### 2.3 Vector store
| Option | Pros | Cons |
|---|---|---|
| **pgvector (in PostgreSQL)** ✅ | One database → transactional consistency between chunks, metadata and status; HNSW index; enables **hybrid search** by joining with `tsvector` full-text search; zero extra infra | Not the fastest at 100M+ vectors (irrelevant here) |
| Pinecone / Weaviate / Qdrant | Purpose-built, scale to billions | Extra service, eventual consistency with metadata, cost, two sources of truth |
| FAISS / Chroma in-process | Simple | Not shared across API + worker processes; no persistence story |

**Decision:** pgvector. Document deletion, versioning, and per-user filtering
become plain SQL `WHERE` clauses, and hybrid retrieval is a single query.

### 2.4 RAG framework
| Option | Pros | Cons |
|---|---|---|
| LangChain / LlamaIndex | Many integrations, fast to start | Opaque prompts, heavy dependency tree, abstractions leak when you need hybrid search + custom citations + cost tracking; version churn |
| **Thin hand-rolled RAG layer** ✅ | Every prompt visible and testable; exact control of chunk boundaries, retrieval fusion and citation format; ~500 lines | Must write chunker/retriever ourselves |

**Decision:** Hand-rolled, deliberately small, with the *ideas* borrowed from the
frameworks (recursive token-aware chunking, condense-question for multi-turn,
reciprocal-rank fusion). The case study asks to "show your thought process for AI
prompts" — that is easier when prompts live in one readable module
(`app/ai/prompts.py`) instead of behind a chain object.

### 2.5 LLM + embeddings provider
Requirement says "any provider". Evaluators may have different keys, and CI has
none. So:

* `LLMProvider` protocol with **OpenAI**, **Anthropic** and **Fake** implementations.
* `EmbeddingProvider` protocol with **OpenAI** and **Fake** (deterministic hashed
  bag-of-words vector — enough to make retrieval *behave* in tests).
* Provider chosen by env (`LLM_PROVIDER`, `EMBEDDING_PROVIDER`).

The Fake providers make the whole pipeline — upload → process → chat with
citations — runnable and testable **without any API key**, and let the test
suite run in seconds.

### 2.6 Chunking strategy
| Option | Verdict |
|---|---|
| Fixed character windows | Splits mid-sentence, ignores structure |
| Sentence/paragraph only | Wildly variable sizes → poor retrieval |
| **Recursive, token-aware, structure-first** ✅ | Split on headings → paragraphs → sentences until each piece ≤ `CHUNK_TOKENS` (default 400) with `CHUNK_OVERLAP_TOKENS` (default 60) carried between neighbours; page numbers preserved for citations |

### 2.7 Retrieval strategy
Pure dense retrieval misses exact identifiers (invoice numbers, names, codes);
pure keyword search misses paraphrase. **Hybrid = pgvector cosine top-k ∪
PostgreSQL full-text top-k, merged with Reciprocal Rank Fusion**, then trimmed to
a token budget. Multi-turn questions are first rewritten into a standalone query
("condense question") so "what about the second one?" retrieves correctly.

### 2.8 Storage
`Storage` protocol with `LocalStorage` (default) and `S3Storage` (boto3, optional).
Objects are keyed by content hash → identical uploads are deduplicated for free.

### 2.9 Multi-tenancy / auth
Out of scope to build real auth; but every table carries `owner_id` and every
query filters on it. Identity comes from `X-User-Id` header (documented default
`demo-user`). Swapping in JWT later touches one dependency function.

---

## 3. Production concerns designed in

| Concern | What was done |
|---|---|
| Concurrency | Async API (asyncpg pool), N Celery workers with `acks_late=True`, `worker_prefetch_multiplier=1`, per-task soft/hard time limits |
| Idempotency | SHA-256 of file bytes; re-uploading identical content returns the existing document; same filename + new content ⇒ new **version** |
| Retries | Exponential back-off on transient LLM/embedding errors (tenacity) inside the task; Celery `autoretry_for` on infra errors, max 3 |
| Failure isolation | Every stage writes its status + error into `processing_jobs`; a failed doc never blocks others; `POST /documents/{id}/reprocess` |
| Latency | Embedding batch calls (up to 100 texts per request); Redis cache for embeddings (`sha256(model+text)`) and for question answers (`sha256(doc_set+question)`); SSE streaming so first token arrives early |
| Cost | Every LLM/embedding call records provider, model, input/output tokens, computed USD, latency into `ai_usage`; exposed via `/metrics/processing` and `/metrics/costs` |
| Rate limiting | Redis sliding-window per user on chat + upload endpoints (`RATE_LIMIT_*`), returns 429 with `Retry-After` |
| Limits | Max upload size, allowed MIME types (sniffed from magic bytes, not just extension), max pages, max chunks per doc |
| Observability | structlog JSON logs with request id; Prometheus `/metrics/prometheus`; `/health/live` and `/health/ready` (checks DB + Redis) |
| Schema evolution | Alembic migrations |
| Scale path | Stateless API + workers scale horizontally; pgvector HNSW index; partition `chunks` by `document_id` hash and move to dedicated vector DB only if p95 retrieval > budget |

---

## 4. Task breakdown (implemented in this order)

Each task is small enough to finish and verify independently.

1. **Scaffold** – repo layout, settings, logging, docker-compose (Postgres+pgvector, Redis, API, worker), Dockerfile, Makefile.
2. **Data model** – `documents`, `document_versions`, `chunks` (vector + tsvector), `document_insights`, `chat_sessions`, `chat_messages`, `processing_jobs`, `ai_usage`; Alembic initial migration.
3. **Storage** – `LocalStorage` / `S3Storage`, hash-addressed.
4. **Upload API** – validation (size, MIME sniff), dedup, version detection, enqueue pipeline; batch upload.
5. **Extraction** – PDF (pypdf, per-page), DOCX (python-docx, headings preserved), TXT/MD; page map for citations.
6. **Chunking** – recursive token-aware splitter with overlap and page tracking; unit-tested.
7. **AI providers** – `LLMProvider` / `EmbeddingProvider` protocols; OpenAI, Anthropic, Fake; usage + cost recorder.
8. **Pipeline task** – Celery chain: extract → chunk → embed → analyse; per-stage status, timings, retries.
9. **Analysis prompts** – summary (customisable length/focus/tone), key insights, sentiment, category + tags, suggested questions; JSON-schema-constrained outputs.
10. **Retrieval** – hybrid search (vector + FTS + RRF), owner/document filters, token budget trimming.
11. **Chat API** – sessions over 1..N docs, ask (sync + SSE stream), condense-question for multi-turn, citations, follow-up suggestions, history.
12. **Document APIs** – list/filter/paginate, get, status, insights, regenerate summary with options, compare two docs, delete.
13. **Metrics APIs** – document stats, processing metrics (throughput, p50/p95 per stage, failure rate), AI cost, Prometheus.
14. **Caching & rate limiting** – Redis embedding cache, answer cache, sliding-window limiter.
15. **Health & ops** – live/ready, request-id middleware, structured logging.
16. **Tests** – unit (chunker, RRF, cost, prompts, fake providers) + API integration (fake providers, real Postgres).
17. **Minimal dashboard** – single static page: upload, status polling, chat with streaming + citations.
18. **Docs** – README, AI_USAGE.md, DESIGN.html (Mermaid, print-to-PDF).

Second iteration (after review):

19. **Evaluation harness** – golden YAML suites (documents + Q&A + expected sources + abstention), runner scoring answer accuracy / citation precision & recall / abstention / latency / cost, thresholds per provider kind, `scripts/eval.py`, GitHub Actions (strict on push with Fake, nightly with a real provider).
20. **Reranker behind a flag** – cross-encoder / LLM / lexical scorers, blended with RRF, safe fallback when the model can't load; retrieval fetches 20 candidates untrimmed when enabled.
21. **Structured extraction per category** – schema registry, verbatim-value prompt, Pydantic normalisation, fifth non-fatal pipeline stage, `GET/POST /documents/{id}/extraction`, schema listing, enum migration.
22. **Real-time status** – `GET /documents/{id}/status/stream` (SSE) alongside polling; dashboard shows extracted fields.

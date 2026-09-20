# AI_USAGE.md — how AI was used to build this, and how AI is used inside it

## 1. AI coding tools during development

The project was built in a single pair-programming session with **Claude Code**
(Anthropic's agentic CLI), driving the whole loop: analysis → design → code → run →
debug → tests → docs. Roughly how the work split:

| Phase | What the AI did | What I did |
|---|---|---|
| Analysis | Compared frameworks / queues / vector stores / RAG approaches and wrote `APPROACH.md` with trade-off tables | Chose the constraints (must run with no API key; hybrid retrieval; pgvector) |
| Scaffolding | Generated settings, engines, models, Alembic env, docker-compose, Dockerfile, Makefile | Reviewed schema decisions (versioning, per-stage job rows, JSONB insights) |
| Core code | Wrote chunker, extraction, providers, prompts, retrieval, RAG, Celery pipeline, routers | Directed the order (small verifiable tasks), reviewed diffs |
| Debugging | Read logs/tracebacks and fixed: Celery config not loaded in the API process; prefork pool crash on macOS + Py3.13; tiktoken download blocked by corporate TLS; `str.format` vs JSON braces in a prompt; failed stage rows lost to transaction rollback | Reproduced via the smoke script, confirmed fixes |
| Tests | 61 tests / 83% coverage (unit + API integration with inline pipeline) + golden eval harness | Asked for edge cases: dedup, versioning, tenant isolation, SSE, blocked-until-ready |
| Docs | README, this file, DESIGN.html with Mermaid diagrams | Structure and emphasis |

**What made the AI effective**

* *Small, verifiable tasks.* Each task in `APPROACH.md §4` ended with a runnable check
  (import, migration, curl smoke test, pytest). Bugs were found within minutes of
  being written.
* *A smoke script as the feedback loop.* Instead of describing failures, the agent ran
  `smoke.sh` after every change and read the JSON responses / stage errors itself.
* *Offline Fake providers.* Designing `FakeLLM`/`FakeEmbeddings` first meant the
  agent could exercise the entire system — including RAG citations and metrics —
  without waiting on keys or spending money.

**Where the AI needed correction**

* It initially used `@shared_task` with the API importing tasks *without* the Celery
  app, so the API tried to publish to RabbitMQ. Reading the traceback surfaced it.
* It first wrote the pipeline as one transaction; a later failure rolled back earlier
  stage records. The fix (commit per stage) came from watching `/status` during a
  forced failure.
* Prompt templates with literal JSON braces broke `str.format`. The unit test
  `test_analysis_prompt_interpolates_options_without_breaking_json_braces` now guards it.
* With a real model, multi-turn answers started "correcting" citation numbers from
  earlier turns (each turn renumbers its passages). Fix: strip `[n]` markers from the
  history we send and add an explicit rule; caught only by running the real demo.

---

## 2. AI inside the product

### 2.1 Where AI is used

| Purpose | Model tier | Prompt | Output handling |
|---|---|---|---|
| Structured extraction per category | main | `structured._SYSTEM` + schema JSON | JSON → `StructuredExtraction` (verbatim values, confidence) → JSONB |
| Passage reranking (flag) | fast / local cross-encoder | `rerank._RERANK_SYSTEM` | scores blended with RRF |
| Document analysis (summary, key points, category, tags, sentiment, entities, language, suggested questions) | main | `ANALYSIS_SYSTEM` | JSON → Pydantic `DocumentAnalysis` (enums, caps, dedup) → JSONB |
| Custom summary (length / tone / focus) | main | same template, parameters interpolated | cached per options hash |
| Chat answer | main | `CHAT_SYSTEM` | citations verified programmatically |
| Condense follow-up into standalone question | **fast** | `CONDENSE_SYSTEM` | JSON; falls back to raw question on error |
| Follow-up suggestions | **fast** | `FOLLOWUP_SYSTEM` | JSON; best-effort |
| Document comparison | main | `COMPARE_SYSTEM` | JSON → `Comparison` |
| Embeddings (chunks, queries) | embedding | — | Redis cache; batched |
| *(any of the above via `claude-cli`)* | sonnet / haiku through the local CLI | same prompts | same parsing; cost from the CLI envelope |

### 2.2 Prompt engineering — the thinking

**Grounding and citations (`CHAT_SYSTEM`).**
The single biggest failure mode of document chat is confident answers from outside
the documents. The prompt therefore:

1. numbers every passage `[n] (file, p.X)` and says *use ONLY these*;
2. requires a citation after **every factual sentence** — this makes hallucinated
   sentences stand out because they have nothing to cite;
3. gives an explicit "say you don't know" path with a suggestion of what would help,
   so the model has a sanctioned alternative to guessing;
4. demands verbatim numbers/dates/identifiers — paraphrased figures are the most
   common subtle error in summaries of financial or legal text.

Then the code **verifies** rather than trusts: `verify_citations()` drops any `[n]`
that does not correspond to a supplied passage and renumbers the rest. The API
returns only citations the answer actually used, each with file, page range,
section and snippet so a UI can deep-link.

**Condense-question for multi-turn.**
Retrieval quality collapses on follow-ups like "and the second one?". Rewriting the
follow-up into a standalone question *before* retrieval fixes this cheaply. The
prompt is explicit that it must not *answer*, must keep constraints, and must return
the question unchanged if already self-contained (prevents drift). It runs on the
fast model — this is a ~50-token job.

**Schema-constrained analysis (`ANALYSIS_SYSTEM`).**
One call returns everything the UI needs. Every key has a type, a length cap and,
where sensible, an enum (`category`, `sentiment.label`). Enums make downstream
filtering (`GET /documents?category=invoice`) reliable; caps keep the JSONB small.
The Pydantic model normalises anyway (lowercases tags, dedups, clamps to the enum),
so a slightly non-compliant model still yields usable data instead of a 500.

**Customisation via parameters.** Length (`short|medium|long|bullets`), tone
(`neutral|executive|casual|technical`) and a free-text focus are interpolated into
*one* template as short "hints". This avoids a matrix of prompt variants and makes
the cache key (`options_hash`) trivial.

**Representative excerpt, not map-reduce.** For analysis we send the first chunks
(title/abstract/intro carry most signal) plus evenly spaced samples up to ~6 k
tokens. It is one call instead of N+1, and the chat path still has full-fidelity
retrieval. The trade-off — a fact buried mid-document might not reach the summary —
is acceptable for *metadata*.

**JSON mode where available, tolerant parsing everywhere.** OpenAI gets
`response_format=json_object`; Anthropic gets an instruction appended to the system
prompt. `parse_json_response()` strips code fences and leading prose before failing.

**Temperature.** 0.1 for analysis (consistency), 0.2 for answers (slight fluency),
0.0 for condense (determinism → cache hits), 0.7 for follow-ups (variety wanted).

### 2.3 Cost & latency controls

* Every call records provider, model, tokens, USD (from a price table), latency and
  whether it was a cache hit → `GET /metrics/costs`.
* Embeddings: batches of 100, Redis cache keyed by `sha256(model + text)` — re-uploads
  and shared boilerplate cost nothing.
* Answers: cache keyed by `(user, document versions, standalone question, model)`
  with a 1 h TTL.
* Context budget: retrieval returns at most 8 chunks / 3 000 tokens, so prompt size
  is bounded regardless of corpus size.
* Cheap model for condense and follow-ups; main model only where quality matters.

### 2.4a Claude CLI provider — real answers without an API key

`LLM_PROVIDER=claude-cli` shells out to the logged-in Claude Code binary
(`claude -p … --output-format json --system-prompt … --tools ""`). Passing our own
system prompt and an empty tool list cuts the per-call context from ~30 k to
~1.5 k tokens; the JSON envelope gives token counts and the provider-reported cost,
which we store as-is. Two gotchas found while wiring it: the CLI inherits the
developer's plugins (a "caveman" style plugin was making every summary terse — the
provider now sets `CAVEMAN_DEFAULT_MODE=off` and pins the register in the system
prompt), and it must not run inside another Claude Code session (env scrubbed).

### 2.4 Fake providers — why they matter

`FakeEmbeddings` is a hashed bag-of-words vector (normalised), so lexically similar
texts are close — retrieval *behaves*. `FakeLLM` answers chat prompts by selecting
context sentences with the highest word overlap with the question and cites their
passage numbers, and fills the JSON schemas for analysis prompts. This lets the
full pipeline, the citation verifier, caching, SSE and every metric run in CI with
zero credentials. Outputs are clearly labelled (`tags: ["fake-provider"]`) so nobody
mistakes them for real analysis.

### 2.5 Reranking (feature flag)

Hybrid + RRF is a recall stage; `RERANK_ENABLED=true` adds a precision stage over
the top-20 fused candidates. Three interchangeable scorers (`app/ai/rerank.py`):
a local **cross-encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`, no API cost,
lazy optional import with safe fallback), an **LLM reranker** (fast model scores
passages 0–10 in one JSON call — the prompt insists on *relevance only*), and a
**lexical** scorer used with the Fake provider. Scores are min-max normalised and
blended with the RRF score (`RERANK_WEIGHT`, default 0.6) so a passage both stages
like wins. The eval harness is how you decide whether to turn it on.

### 2.6 Structured extraction per category

The analysis stage's `category` selects a schema in `app/ai/structured.py`
(invoice, contract, legal, financial, report, policy, research, generic). The
prompt shows the schema as JSON with type hints, demands **verbatim** values,
empty strings for absent fields, and a `confidence` score; Pydantic normalises
the result. It runs as a fifth, non-fatal pipeline stage and on demand via
`POST /documents/{id}/extraction` with any schema (cached per schema). Adding a
category is a data change — a new dict entry — not a code change.

### 2.7 Evaluation harness

`evals/golden/*.yaml` bundles small documents with questions, acceptable answer
substrings (`expect_any` groups), expected citation sources (file, page) and
abstention cases. `evals/runner.py` drives the real API in-process, runs the
pipeline inline, and reports **answer accuracy, citation precision, citation
recall, abstention accuracy, p50/p95 latency and cost**. Thresholds live in
`evals/thresholds.yaml` — strict for the deterministic Fake provider (runs on
every push in CI), looser for the nightly run against a real model. Multi-turn
cases (`turns:`) exercise condense-question. This is the loop for prompt changes:
edit a prompt, run `python scripts/eval.py`, watch citation precision.

### 2.8 Ideas not built (and where they would go)

* **Query expansion / HyDE** for very short queries — `rag.retrieve()`.
* **Per-page extraction** for long, table-heavy invoices — `structured.py`, map over pages.
* **OCR** for scanned PDFs — a new extractor in `services/extraction.py`.

# RAG Engine
> A multi-tenant document question-answering backend. It uses hybrid retrieval, reranking and cited answers, and it ships with an evaluation harness that measures quality instead of assuming it.

## 1. Problem & why it exists
Most "RAG demos" work like this: upload one PDF, split it every 1,000 characters, put it in a vector database, and ask a question. That falls apart with real documents and real users:
- Keyword-heavy questions fail on pure vector search. Examples: invoice numbers, drug names, error codes.
- There are no citations, so nobody can check an answer.
- There is no access control, so tenant A can retrieve tenant B's documents.
- Re-uploading a document duplicates or corrupts the index.
- Nobody measures anything. "It seems to work" is not an engineering claim.

This project builds the full pipeline the way a production team would: layered ingestion, hybrid retrieval, reranking, grounded answers with citations, tenant isolation, and **an evaluation system that makes quality measurable and regressions visible**.

## 2. What this proves to an employer
| Skill | Target job requirement it maps to |
|---|---|
| RAG pipeline design end to end | EaseOps (RAG, vector DBs), Teal (RAG, LlamaIndex), Zeko (LLM workflows) |
| Embeddings, pgvector, BM25, hybrid retrieval, reranking | Teal (vector DBs, NLP), EaseOps (vector DBs) |
| LLM evaluation (retrieval metrics, faithfulness, citation accuracy) | Razorpay (evaluation), Google (model evaluation) |
| FastAPI, async Python, PostgreSQL, Redis | Zeko, EaseOps |
| LLM cost/latency tracking | EaseOps (LLM cost/performance) |
| Multi-tenant access control | EaseOps (auth, security), Atlassian/Amazon (backend) |
| LlamaIndex (as a comparison milestone) | Teal (explicitly lists LlamaIndex) |
| Streaming APIs (SSE) | general AI backend |

## 3. Scope
### In scope (v1)
- Ingest PDF, DOCX, Markdown, HTML and TXT.
- Collections with per-collection access control inside a tenant.
- Document versioning: re-uploading replaces an active version atomically, and old versions stay queryable by ID.
- Chunking implemented by hand: recursive structural chunking and token-window chunking, with overlap.
- Metadata extraction: title, headings path, page number, source, mime type, user tags.
- Embeddings through a provider interface (a hosted API or a local sentence-transformers model).
- pgvector storage with an HNSW index.
- BM25 keyword retrieval, first implemented by hand in Python and then compared against Postgres full-text search (`tsvector`).
- Hybrid fusion with Reciprocal Rank Fusion (RRF), implemented by hand.
- Reranking through an interface: a local cross-encoder, or a hosted rerank API.
- Metadata filters on queries (collection, tags, date, source type).
- Context construction: token budget, deduplication, ordering, and source labels.
- LLM answering with **structured output** (answer, citations[], confidence, `insufficient_context` flag).
- Refusal when retrieved evidence is below a threshold, instead of hallucinating.
- Streaming answers over Server-Sent Events.
- Conversation history, with follow-up question rewriting ("condense question").
- **The evaluation subsystem and the `rag-eval` CLI. Required, not optional.**
- Per-request token, latency and estimated-cost logging.
- A LlamaIndex implementation of the same pipeline, built as a **comparison milestone**.

### Out of scope (explicitly)
- A frontend UI (a minimal demo page is allowed at most).
- OCR of scanned PDFs (text-layer PDFs only in v1; noted as a limitation).
- Fine-tuning embedding models.
- Agentic multi-step retrieval (that belongs to `ai-agent-platform`).
- Billing.

## 4. Architecture
```
                    ┌─────────────────────────── FastAPI (async) ───────────────────────────┐
 Client ──HTTP──▶   │  /documents   /collections   /query (SSE)   /conversations   /eval     │
                    └───────┬───────────────────────────────┬──────────────────────────────┘
                            │ upload                        │ query
                            ▼                               ▼
                 ┌────────────────────┐          ┌──────────────────────────┐
                 │ Object storage     │          │ Query pipeline           │
                 │ (local FS / S3)    │          │ 1 authz → allowed colls  │
                 └─────────┬──────────┘          │ 2 condense follow-up     │
                           │ enqueue job         │ 3 BM25 ─┐                │
                           ▼                     │ 4 vector ┴─▶ RRF fuse     │
                 ┌────────────────────┐          │ 5 rerank top-N           │
                 │ Redis queue        │          │ 6 context builder        │
                 └─────────┬──────────┘          │ 7 LLM (structured)       │
                           ▼                     │ 8 citation validation    │
                 ┌────────────────────┐          │ 9 stream + log usage     │
                 │ Ingestion worker   │          └────────────┬─────────────┘
                 │ parse → clean →    │                       │
                 │ chunk → metadata → │                       │
                 │ embed → upsert     │                       │
                 └─────────┬──────────┘                       │
                           ▼                                  ▼
                 ┌──────────────────────────────────────────────────────┐
                 │ PostgreSQL 16 + pgvector                             │
                 │ tenants, users, collections, documents, versions,    │
                 │ chunks(embedding vector, tsv tsvector), conversations,│
                 │ messages, query_logs, eval_runs                      │
                 └──────────────────────────────────────────────────────┘
                 Redis: job queue, embedding cache, rate limits
```

Why each component exists:
- **FastAPI (async):** most time goes to I/O (LLM calls, embedding calls, the DB), so async gives high concurrency without threads. Pydantic models become the API contract and the OpenAPI docs.
- **Object storage:** original files are kept so a document can be re-parsed when the chunking strategy changes. Changing the pipeline must never require users to upload again.
- **Redis queue and ingestion worker:** parsing and embedding a 200-page PDF takes seconds to minutes. That can't happen inside an HTTP request, so upload returns `202 Accepted` with a job ID. Doing this also teaches retries and idempotency, since re-running a job must not duplicate chunks.
- **PostgreSQL + pgvector:** one database for relational data, vectors and full-text search. Tenant filters and vector search happen in *one query*, which keeps isolation simple and correct. A separate vector database (Qdrant) is an alternative considered in section 5.
- **The BM25 and vector retrievers run separately, then RRF fuses them:** each has different failure modes. Vectors capture meaning; BM25 catches exact tokens. RRF needs no score calibration between the two.
- **Reranker:** first-stage retrievers are tuned for recall; a cross-encoder rescoring the top 20–50 results improves precision. The gain gets measured, not assumed.
- **Context builder:** decides *what the model sees*. It enforces a token budget, removes near-duplicate chunks, orders by relevance or by document position, and labels each chunk `[S1]…[Sn]` so citations can be checked.
- **Citation validation:** every citation the model returns must point to a chunk that was actually in the context. Citations that fail are removed and counted, which feeds the evaluation metrics.
- **query_logs:** every query records its retrieved chunk IDs, scores, token counts, latency per stage, and cost. This is what makes debugging and evaluation possible.

## 5. Tech stack & justification
| Choice | Why | Alternatives considered |
|---|---|---|
| Python 3.12, FastAPI, Pydantic v2 | matches the target jobs (Zeko, EaseOps); async; typed contracts | Flask (sync, no native schema), Django (heavier) |
| PostgreSQL 16 + pgvector (HNSW) | one store, transactional, tenant filter + ANN in one query | Qdrant (better at huge scale, but a second system to keep in sync); Pinecone (managed, costs money, lock-in) |
| SQLAlchemy 2.0 async + Alembic | industry standard, migrations | raw asyncpg (used for hot paths if profiling shows a need) |
| Redis + arq (or Celery) | simple async job queue, also used for caching | Celery (heavier, sync-first); RabbitMQ (another component to run) |
| pypdf / pdfplumber, python-docx, BeautifulSoup, markdown-it | parser per format behind one `Parser` protocol | unstructured.io (convenient, but hides what it's doing; possibly a later comparison) |
| tiktoken or the provider's tokenizer | token-accurate chunking and budgeting | character counts (inaccurate) |
| Embeddings: provider interface, e.g. a hosted embedding model or local `bge-small`/`all-MiniLM` | swappable; the local option keeps evaluation runs cheap | hardcoding one vendor |
| Reranker: local cross-encoder (e.g. `bge-reranker-base`) or a hosted rerank API | the local one is free for experiments | skipping reranking (that decision should be measured) |
| LLM: provider interface (Gemini / OpenAI / Anthropic, configurable) | model names live in config; later routed via `ai-gateway` | one hardcoded SDK |
| LlamaIndex (milestone M7 only) | Teal lists it; building by hand first means the comparison teaches something | LangChain (similar; pick one) |
| pytest, httpx, testcontainers | real Postgres/Redis in tests | mocks everywhere (hides SQL bugs) |

## 6. Data model
```
tenants(id uuid pk, name, created_at)
users(id uuid pk, tenant_id fk, email unique, password_hash, role enum[admin,member,viewer], created_at)
collections(id pk, tenant_id fk, name, description, created_by, created_at, UNIQUE(tenant_id,name))
collection_members(collection_id fk, user_id fk, permission enum[read,write,admin], PK(collection_id,user_id))
documents(id pk, tenant_id fk, collection_id fk, title, source_uri, mime_type, tags text[],
          active_version_id fk null, created_by, created_at, deleted_at null)
document_versions(id pk, document_id fk, version int, content_sha256, storage_key, status enum[pending,processing,ready,failed],
                  error text null, parser, chunker_config jsonb, embedding_model, chunk_count, created_at,
                  UNIQUE(document_id,version), UNIQUE(document_id,content_sha256))
chunks(id pk, tenant_id, collection_id, document_id, version_id fk, ordinal int, text,
       token_count int, heading_path text[], page int null, metadata jsonb,
       embedding vector(D), tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED)
conversations(id pk, tenant_id, user_id, collection_ids uuid[], created_at)
messages(id pk, conversation_id fk, role enum[user,assistant], content, citations jsonb, query_log_id fk null, created_at)
query_logs(id pk, tenant_id, user_id, query, rewritten_query, filters jsonb, retrieved jsonb /*chunk_id,bm25_rank,vec_rank,rrf,rerank*/,
           context_chunk_ids uuid[], model, prompt_tokens, completion_tokens, est_cost_usd numeric,
           latency_ms jsonb /*per stage*/, refused bool, created_at)
ingestion_jobs(id pk, version_id fk, attempts, last_error, idempotency_key unique, status, created_at, updated_at)
eval_runs(id pk, dataset_name, dataset_sha, config jsonb, metrics jsonb, report_path, git_sha, created_at)
```
Indexes and constraints:
- `chunks`: HNSW on `embedding` (`vector_cosine_ops`), GIN on `tsv`, B-tree on `(tenant_id, collection_id, version_id)`.
- Retrieval joins only `chunks` whose `version_id = documents.active_version_id`. Old versions stay stored but are not searched by default.
- `UNIQUE(document_id, content_sha256)` makes re-uploading an identical file a no-op. That's idempotency at the data layer.
- A tenant ID lives on every row that queries touch, and **every repository method requires `tenant_id`**. Optional stretch goal: Postgres row-level security as defence in depth.

## 7. API / interface design
```
POST   /v1/auth/login                         → {access_token, refresh_token}
POST   /v1/collections                        {name, description}
GET    /v1/collections
POST   /v1/collections/{id}/members           {user_id, permission}
POST   /v1/collections/{id}/documents         multipart file + tags → 202 {document_id, version_id, job_id}
GET    /v1/documents/{id}                     → metadata + versions + status
GET    /v1/documents/{id}/versions/{v}/chunks → paginated chunks (debugging)
DELETE /v1/documents/{id}                     soft delete
GET    /v1/jobs/{job_id}                      → {status, attempts, error}

POST   /v1/query                              (SSE when Accept: text/event-stream)
  {question, collection_ids[], filters:{tags[], source_types[], date_from},
   conversation_id?, top_k?, rerank?:bool, debug?:bool}
  → events:  retrieval {chunks:[{id,score,title,page}]} (debug only)
             token     {text}
             final     {answer, citations:[{source_label, chunk_id, document_id, page, quote}],
                        insufficient_context, usage:{prompt_tokens, completion_tokens, est_cost_usd}, latency_ms}
POST   /v1/conversations                      → {id}
GET    /v1/conversations/{id}/messages
GET    /v1/health  /v1/ready  /metrics
```
Evaluation CLI (a separate entry point in the same package):
```
rag-eval run datasets/handbook_v1.json [--config configs/hybrid_rerank.yaml] [--k 5] [--judge-model <cfg>] [--out reports/]
rag-eval compare reports/run_A.json reports/run_B.json      # per-metric diff table, flags regressions
rag-eval ingest-fixtures datasets/corpus/handbook/          # loads the fixed eval corpus into an isolated tenant
```

## 8. Key engineering problems
1. **Chunking quality.** Fixed-size splits cut tables and sentences in half. The plan is a recursive structural splitter (headings → paragraphs → sentences) with a token ceiling and overlap. Chunk size and overlap become evaluation variables, not hunches.
2. **Hybrid fusion without score calibration.** BM25 and cosine scores sit on incomparable scales. RRF handles this: `score(d) = Σ 1/(k + rank_i(d))`, with k = 60 as the default to be tuned. Weighted linear fusion gets implemented too and compared in evaluation.
3. **Tenant isolation inside ANN search.** HNSW with a `WHERE tenant_id = … AND collection_id = ANY(...)` filter can return fewer than k results (post-filtering recall loss). Mitigations: raise `hnsw.ef_search`, use pgvector iterative index scans if available, or use partial indexes per large tenant. Recall under filters will be measured.
4. **Document version atomicity.** A new version gets ingested fully first. Then one transaction flips `active_version_id`, so queries never see half-ingested versions. Failed ingestion leaves the old version active.
5. **Idempotent, retryable ingestion.** The job is keyed by `version_id`. The worker deletes that version's chunks and reinserts them in a transaction, so a retry after a crash is safe. Exponential backoff, with a max-attempts limit before the job is marked failed.
6. **Grounding and refusal.** If the top rerank score is below a threshold, or the model returns `insufficient_context = true`, the system answers "I don't have enough information" and cites nothing. The threshold is tuned on the evaluation set with unanswerable questions included.
7. **Citation correctness.** The model sometimes invents `[S7]` when only S1–S5 exist. Server-side validation removes and counts invalid citations. Optional: check that a quoted span actually appears in the cited chunk.
8. **Follow-up questions.** "What about the second one?" has to be rewritten into a standalone query using conversation history before retrieval runs. The rewrite gets logged.
9. **Prompt injection through documents.** A retrieved chunk might say "ignore previous instructions". Context goes inside clearly delimited data blocks, the system prompt says document text is untrusted data, and the evaluation set includes adversarial documents.
10. **Cost and latency.** Stage-level timing (embed query, BM25, vector, rerank, LLM first token, LLM total). Query embeddings are cached in Redis keyed by `(model, sha(query))`.

## 9. Milestones
**M1 — Ingestion core (by hand).**
- Goal: files in, chunks out, stored.
- Deliverables: `Parser` protocol plus parsers for 5 formats, cleaning, two chunkers (token window, recursive structural), metadata extraction, the Postgres schema, and an Alembic migration. A CLI `rag ingest <path>` with no API yet.
- Learn: tokenization, parsing edge cases, SQLAlchemy async, migrations.
- Accept: unit tests per parser and chunker (chunk boundaries, overlap, token ceilings), plus golden-file tests.

**M2 — Embeddings + vector search.**
- Deliverables: `Embedder` interface with local and hosted implementations, batching, a Redis embedding cache, an HNSW index, and `rag search "<q>"` returning the top-k with scores.
- Learn: embeddings, cosine similarity, ANN/HNSW parameters.
- Accept: an integration test on testcontainers Postgres; batching and caching tested.

**M3 — BM25 + hybrid + rerank.**
- Deliverables: a hand-written BM25 (inverted index, IDF, k1/b parameters) validated against Postgres `ts_rank` on a small corpus, RRF fusion, weighted fusion, and a `Reranker` interface with a cross-encoder implementation.
- Learn: IR fundamentals, why hybrid beats either method alone.
- Accept: tests on hand-computed BM25 scores and RRF ordering.

**M4 — Evaluation subsystem v1 (retrieval metrics). Built *before* the LLM layer, on purpose.**
- Deliverables: `evaluation/` layout, the dataset schema, 50+ hand-labelled questions, and `rag-eval run` computing precision@k, recall@k, MRR and hit-rate for BM25-only, vector-only, hybrid, and hybrid + rerank.
- Accept: a report in `evaluation/reports/` with **measured** numbers and the config and git SHA recorded.

**M5 — Answer generation + citations + refusal.**
- Deliverables: context builder, prompt templates, structured-output parsing, citation validation, refusal logic, and query_logs with tokens, cost and latency.
- Accept: tests for context budget truncation, invalid-citation stripping, and the refusal path.

**M6 — API, auth, multi-tenancy, versioning, streaming, conversations.**
- Deliverables: FastAPI routes, JWT, collection permissions, the async ingestion worker, versioning with the atomic flip, SSE streaming, and conversation rewriting.
- Accept: authz tests (tenant A cannot retrieve tenant B's chunks, *including* through vector search), version-flip tests, and idempotent re-upload tests.

**M7 — Evaluation v2 (generation metrics) + LlamaIndex comparison.**
- Deliverables: LLM-as-judge metrics for faithfulness, answer relevance and context relevance (with the judge prompt versioned), exact citation-correctness scoring, and a `rag-eval compare` diff.
- Also: re-implement the same pipeline in LlamaIndex and evaluate both on the same dataset.
- Accept: `docs/llamaindex-comparison.md` covering quality, latency, code size, control, and when to use which. Measured values only.

**M8 — Production hardening.**
- Deliverables: Dockerfile, docker-compose (api, worker, postgres, redis), GitHub Actions (ruff, mypy, pytest with services, a docker build, plus an eval smoke run on a 10-question subset with a regression threshold), structured logs, Prometheus metrics, rate limiting, and a README.
- Accept: the CI pipeline goes green, and the eval smoke test fails the build if recall@5 drops by more than X% against the stored baseline.

**M9 (optional) — Deploy via `cloud-infra-lab`.** Low-cost AWS deployment, plus a short public demo with a sample corpus.

## 10. Testing strategy
- **Unit tests:** parsers, chunkers, BM25 math, RRF, context builder, citation validator, and the cost calculator.
- **Integration tests:** testcontainers Postgres (pgvector image) and Redis, covering the repositories, tenant-filtered retrieval, and the ingestion worker's retry/idempotency.
- **API tests:** httpx AsyncClient covering authz matrices and SSE event order.
- **LLM calls:** a fake provider returning scripted outputs for deterministic tests. Real-provider tests are marked `@pytest.mark.live` and excluded from default CI.
- **Evaluation as a test:** the smoke eval in CI with a fixed local embedder, which makes it deterministic and free.
- **Property tests (Hypothesis):** chunk reassembly covers the whole text, and no chunk exceeds the token limit.

## 11. Observability
- Structured JSON logs carrying `request_id`, `tenant_id`, `query_log_id`.
- Prometheus metrics:
  - `rag_query_latency_seconds{stage}`
  - `rag_llm_tokens_total{model,type}`
  - `rag_llm_cost_usd_total{model}`
  - `rag_refusals_total`
  - `rag_invalid_citations_total`
  - `rag_ingestion_jobs_total{status}`
  - `rag_retrieved_chunks`
- OpenTelemetry spans per pipeline stage (optional in M8).
- A `debug=true` flag on queries returns the retrieval breakdown, admin-only.

## 12. Security
- JWT access and refresh tokens; roles plus per-collection permissions; tenant ID enforced in every repository method.
- Upload validation: mime sniffing (not the file extension), a max size, and rejection of encrypted or zip-bomb documents. Parsing runs in the worker, never in the API process.
- Prompt injection: retrieved text is framed as untrusted data, the system prompt is never exposed, and an adversarial evaluation subset exists.
- Secrets only via environment variables; no raw prompts containing PII in info-level logs (configurable redaction).
- Rate limits per user and tenant (Redis token bucket).
- Documented limitation: LLM providers see document content. The data-residency implications go in the README.

## 13. Deployment
- Local: `docker compose up` brings up the api, worker, postgres (pgvector) and redis. `make seed` loads the demo corpus.
- Cloud (M9, via `cloud-infra-lab`): ECS/Fargate or a single EC2 instance, RDS Postgres with pgvector, ElastiCache or a Redis container, S3 for files. Scale to zero, or tear down after demos.
- Configuration via environment variables; model names and prices in `config/models.yaml`.

## 14. Evaluation / measurements to collect (all TBD until measured)
**Dataset construction:**
- A fixed corpus of 30–80 public documents (e.g. an open-source project's docs, public policy PDFs, or RFCs). Licence recorded in `evaluation/datasets/corpus/SOURCES.md`.
- **At least 60 hand-written questions**, each with `expected_sources` (document plus section or page, mapped to chunk IDs after ingestion by a matching script), `expected_answer` (a reference answer), `type`, and `difficulty`.
- Question types: factual lookup, keyword/exact-term, multi-hop (2 sources), paraphrase, **unanswerable** (at least 10%, to test refusal), and **adversarial** (an injected document).
- I write the questions by hand. An LLM may *propose* candidates, but every item gets verified and edited by hand, and the `dataset_sha` gets recorded.

**Dataset schema (`datasets/*.json`):**
```json
{"name":"handbook_v1","corpus":"corpus/handbook","items":[
 {"id":"q001","question":"...","expected_sources":[{"document":"x.pdf","page":4}],
  "expected_answer":"...","type":"factual","answerable":true}]}
```

**Metrics (`evaluation/metrics/`):**

| Metric | How computed | Value |
|---|---|---|
| precision@k, recall@k, hit-rate@k | against expected_sources | TBD |
| MRR | rank of the first relevant chunk | TBD |
| context relevance | LLM judge, per retrieved chunk, 0–1 | TBD |
| answer relevance | LLM judge versus the question | TBD |
| faithfulness / groundedness | LLM judge: claims supported by the context | TBD |
| citation correctness | exact: cited chunks ∈ context ∧ ∈ expected_sources; plus an invalid-citation rate | TBD |
| refusal accuracy | on unanswerable items (correct refusals / total) and on answerable items (false refusals) | TBD |
| latency p50/p95 per stage | from query_logs | TBD |
| tokens / query, est. cost / query | from usage × `models.yaml` prices | TBD |

**Runners and reports:**
- `evaluation/runners/` runs a config across the dataset, with concurrency limits and caching of LLM judge calls.
- `evaluation/reports/` holds JSON plus Markdown: a per-config table and the worst-10 failures with retrieved chunks for error analysis.

**Planned ablations:** chunk size (256/512/1024), overlap, BM25-only vs vector-only vs hybrid vs +rerank, k, and embedding model.

Judge caveat: LLM-judge scores are noisy. Spot-check 20 judgements by hand and report the agreement rate.

## 15. Prerequisite learning
- `learning/python/`: async, typing, dataclasses, packaging.
- `learning/backend/`: fastapi, auth, postgres (indexes, transactions), redis.
- `learning/ai/`: tokens, embeddings, vector-search (ANN/HNSW), bm25-and-hybrid, reranking, structured-outputs, rag, evaluation, prompt-injection.
- Recommended before M1: `python-backend-lab` M1–M3.

## 16. Interview talking points
- Why hybrid retrieval? Show where vector-only failed on the eval set, with measured cases.
- How RRF works, and why it doesn't need score normalisation.
- How you chose chunk size (the ablation results).
- How you prevent cross-tenant leakage in vector search.
- How you know the answers are grounded. Metrics, judge validation, refusal threshold tuning.
- The cost per query and how to cut it (smaller context, caching, cheaper model routing through `ai-gateway`).
- Versioning without downtime (the atomic pointer flip).
- LlamaIndex versus by hand: what the framework gave you and what it hid.
- What breaks at 10M chunks? Partitioning, a dedicated vector DB, async re-embedding migrations.

## 17. Resume bullet templates
- "Built a multi-tenant RAG backend (FastAPI, PostgreSQL/pgvector, Redis): hybrid BM25 + vector retrieval with RRF fusion and cross-encoder reranking, and structured answers with validated citations."
- "Designed an evaluation harness (`rag-eval`) on a [N]-question hand-labelled dataset. Hybrid + rerank improved recall@5 from [MEASURED] to [MEASURED] over vector-only, with faithfulness at [MEASURED]."
- "Implemented grounded refusal, correctly declining [MEASURED]% of unanswerable questions, with p95 latency of [MEASURED] ms and about $[MEASURED] per query."

## 18. Open questions / uncertainties
- Evaluation corpus choice: it needs a licence-clean public corpus with enough depth for multi-hop questions. To be decided at M4.
- The pgvector filtered-HNSW recall behaviour depends on the pgvector version. Verify iterative scan support at implementation time.
- Local reranker latency on CPU may be too slow for good UX. Measure it; a hosted rerank API is the fallback.
- LLM-judge metrics depend on the judge model. Record the judge model and prompt version with every run; don't compare across judges.
- Whether to use arq or Celery: arq is simpler and async-native; Celery is more widely known. Decide in M6. `distributed-job-platform` covers queues in depth anyway.

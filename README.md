# RAG Engine

A multi-tenant document question-answering backend: **hybrid retrieval (hand-written BM25 + pgvector) → RRF fusion →
reranking → grounded answers with validated citations**. It ships with an **evaluation harness (`rag-eval`)** that
measures retrieval and answer quality instead of assuming it.

Everything runs **offline by default**, with deterministic fake providers, so tests, evaluation and CI need no API
keys. Real embedding and LLM providers (OpenAI, Gemini, local sentence-transformers) plug in through the same
interfaces.

---

## Problem
Most RAG demos embed one PDF, cut it every 1,000 characters and hope for the best. With real documents and users
that breaks down:
- exact tokens (error codes, policy numbers) get lost in pure vector search
- answers have no checkable sources
- tenant A can retrieve tenant B's text
- re-uploads duplicate the index
- nobody measures quality

## Why it exists
To build the full pipeline the way a production team would, with each stage visible and each claim measurable.
Chunking, BM25, fusion and context construction are **implemented by hand** so the trade-offs are understood.
LlamaIndex is added afterwards as a comparison ([docs/llamaindex-comparison.md](docs/llamaindex-comparison.md)).

## Architecture
```
                ┌──────────────────────── FastAPI (async) ────────────────────────┐
 client ──HTTP▶ │ /auth /collections /documents /jobs /query (JSON|SSE) /conversations │
                └──────┬──────────────────────────────────────────┬───────────────┘
                upload │ 202 + job id                              │ query
                       ▼                                           ▼
              file storage ──▶ Redis (arq) ──▶ worker    ┌─ authorised SearchFilter (tenant + collections)
                                  parse → clean → chunk  ├─ condense follow-up (conversation)
                                  → embed → replace      ├─ BM25 (hand) ─┐
                                  chunks → READY →       ├─ pgvector ────┴─▶ RRF ─▶ rerank
                                  atomic version flip    ├─ refusal gate (score threshold, no LLM call)
                                                         ├─ context builder: budget, dedupe, injection screen, [S#]
                                                         ├─ LLM (JSON mode | streaming markers)
                                                         └─ citation validation → query log (tokens, cost, latency)
                       ▼                                           ▼
          PostgreSQL 16 + pgvector: tenants, users, collections, documents, versions,
          chunks (vector HNSW + tsvector GIN), conversations, messages, query_logs
```
There's a longer walk-through in [docs/architecture.md](docs/architecture.md), and a file-by-file study guide in
[docs/LEARNING_GUIDE.md](docs/LEARNING_GUIDE.md).

## Features
- **Ingestion:** PDF, DOCX, Markdown, HTML and TXT. Mime type is sniffed from bytes, not trusted from the extension.
  Heading paths and page numbers are preserved.
- **Chunking** (by hand): a recursive structural chunker that never crosses headings or pages, plus token windows.
  Both use overlap and are tested with Hypothesis.
- **Versioning:** identical re-uploads are no-ops (sha256), new versions flip atomically, a failed ingestion keeps the
  old version live, and old versions stay queryable by id.
- **Retrieval:** hand-written BM25 (k1/b, Lucene IDF) or Postgres FTS; pgvector HNSW with iterative scans for filtered
  search; RRF or weighted fusion; lexical or cross-encoder reranking.
- **Answers:** structured JSON output, `[S#]` citations validated against the context, quote verification, and a
  refusal gate.
- **Security:** JWT access/refresh tokens, scrypt password hashing, tenant roles plus per-collection permissions,
  tenant scoping in every store query, a prompt-injection screen, upload size and type limits, and per-user
  token-bucket rate limiting.
- **Streaming:** Server-Sent Events (`retrieval` (admin debug) → `token`… → `final`).
- **Conversations:** follow-up questions get rewritten into standalone ones using the history.
- **Observability:** Prometheus `/metrics` (stage latency, tokens, cost, refusals, invalid citations, injection flags,
  ingestion outcomes) and JSON logs with request ids.
- **Evaluation:** a 60-item hand-labelled dataset; retrieval metrics (hit/precision/recall@k, MRR); generation
  metrics (refusal accuracy, keyword recall, citation precision, invalid-citation rate, adversarial resistance, tokens,
  cost); judged metrics (faithfulness, answer/context relevance); `compare` with a regression gate in CI; `spot-check`
  sheets for measuring judge agreement.

## Tech stack
Python 3.12 · FastAPI · Pydantic v2 · SQLAlchemy 2 (async, Core) · asyncpg · Alembic · PostgreSQL 16 + pgvector ·
Redis · arq · PyJWT · prometheus-client · pypdf · python-docx · BeautifulSoup · numpy · pytest · Hypothesis · mypy
(strict) · ruff · Docker · GitHub Actions. Optional: LlamaIndex, sentence-transformers.

## Quick start
```bash
# 1. offline, in-memory, no Docker: search a folder from the CLI
uv sync
uv run rag search "what does E-4471 mean" --corpus evaluation/datasets/corpus/kestrel

# 2. run the evaluation
uv run rag-eval run evaluation/datasets/kestrel_v1.json --generate

# 3. full stack: API + worker + Postgres + Redis
docker compose --profile app up -d --build     # API on http://localhost:58000 (OpenAPI at /docs)
./examples/quickstart.sh                       # signup → collection → upload corpus → ask
```
Development: `make up && make migrate` starts Postgres (:55433) and Redis (:56380), then run `make check` and
`make serve`.

## API usage
```bash
curl -X POST localhost:58000/v1/query -H "authorization: Bearer $TOKEN" -H 'content-type: application/json' \
  -d '{"question":"What does E-4471 mean?","collection_ids":["<id>"]}'
```
Real response from the Docker stack with the default offline providers (trimmed; token counts are estimates
because the fake LLM has no tokenizer):
```json
{"answer": "- E-4471: the upstream payment gateway timed out [S1].",
 "citations": [{"source_label": "S1", "title": "On-call Runbook", "section": "On-call Runbook > Error Codes", "quote_verified": true}],
 "insufficient_context": false,
 "usage": {"prompt_tokens": 642, "completion_tokens": 72, "estimated": true, "est_cost_usd": 0.0},
 "latency_ms": {"bm25": 5.136, "vector": 3.697, "rerank": 0.48, "context": 0.513, "llm_total": 0.301, "total": 10.345}}
```
Send `Accept: text/event-stream` to stream. The full endpoint list is at `/docs`, and there's a Python example in
[examples/client.py](examples/client.py).

## Evaluation results

> **Offline fake-provider baseline, not a quality claim.** These numbers use the deterministic hashing embedder
> (which can't match synonyms), the lexical reranker heuristic and an extractive fake LLM. They show the harness
> works and allow regression checks. **Real-model numbers are TBD.**

Retrieval, `kestrel_v1` (53 answerable items), k = 5, current defaults (light BM25 stemming), from
[`evaluation/reports/bm25_stemming_light.md`](evaluation/reports/bm25_stemming_light.md):

| config | hit@5 | precision@5 | recall@5 | MRR |
|---|---|---|---|---|
| keyword (BM25, light stemming) | 0.981 | 0.215 | 0.972 | 0.921 |
| vector (hashing embedder) | 0.830 | 0.174 | 0.792 | 0.736 |
| hybrid RRF | 0.943 | 0.200 | 0.915 | 0.850 |
| hybrid weighted | 0.962 | 0.207 | 0.943 | 0.866 |
| hybrid RRF + lexical rerank | **0.981** | **0.207** | **0.953** | **0.893** |

BM25 stemming was measured before/after in
[`evaluation/reports/bm25_stemming.md`](evaluation/reports/bm25_stemming.md): the hand-written `light` stemmer
improved every BM25-based config with no per-item regressions (e.g. keyword recall@5 0.924 → 0.972, MRR 0.844 →
0.921; default pipeline recall@5 0.934 → 0.953), so it's the default. Snowball scored similarly but regressed one
question. The earlier M4 baseline (no stemming) is kept in
[`m4_retrieval_baseline.md`](evaluation/reports/m4_retrieval_baseline.md).

Generation, from [`evaluation/reports/m7_generation_judged.md`](evaluation/reports/m7_generation_judged.md),
hybrid RRF + rerank:
- refusal accuracy on unanswerable items: 0.429
- false-refusal rate: 0.226
- keyword recall: 0.717
- citation precision: 0.764
- invalid citations: 0
- adversarial resistance: 1.0

Faithfulness shows 1.0, but that's **trivially true** for an *extractive* fake LLM judged by a lexical heuristic, so
don't read it as a result. The judge spot-check sheet (`m7_spot_check_TODO.md`) hasn't been hand-labelled yet, so
judge agreement is **TBD**.

What the error analysis shows (worst-items section of each report): stemming fixed "password" vs "Passwords"
(q014); paraphrases like "beer" → "alcohol" (q037) are misses that only a semantic embedding model can fix. That's
the next experiment.

## Testing
```bash
make check        # ruff + ruff format --check + mypy --strict + pytest (101 tests)
make eval-check   # re-runs retrieval eval, fails if any metric drops > 0.02 vs the committed baseline
```
- **Unit tests:** parsers (real generated PDF/DOCX), chunker properties (Hypothesis), BM25 against hand-computed
  scores, RRF, context budget and dedupe, citation stripping, refusal, the API authorisation matrix, and SSE event
  order.
- **Store contract tests** run on both MemoryStore and PgStore: tenant isolation *inside vector search*, version flip,
  failed-ingestion rollback, soft delete, and no orphan rows on storage failure.
- **Integration tests** (auto-skip if Docker services are down): the API on Postgres, the arq worker in burst mode,
  the Redis rate limiter and cache, and hand BM25 vs Postgres FTS agreement.

## Deployment
- `docker compose --profile app up`: api (runs migrations on start), worker, postgres (pgvector), redis. Uploads live
  on a volume shared by the api and the worker.
- The image is multi-stage, runs as a non-root user and has a health check.
- Cloud deployment (AWS, via `cloud-infra-lab`): TBD.

## Security
- Every store method takes a tenant id or a `SearchFilter`, and there's no cross-tenant query path. Other tenants'
  collections return 404, not 403, so they can't be discovered.
- Viewers are capped at read permission. Retrieval debug output is admin-only.
- JWT: HS256 on an explicit algorithm allow-list, typed tokens (a refresh token can't be used as an access token),
  and a secret of at least 32 characters is enforced.
- Retrieved text is treated as untrusted: injection patterns are screened, sources are framed as data, and there's an
  adversarial eval subset. Pattern screens can be bypassed, and that's documented.
- Limitation: hosted LLM providers see document content, which has data-residency implications. Refresh-token
  revocation (a denylist) is not implemented.

## Performance considerations
- Hybrid search runs BM25 and vector search per query; rerank only sees the top `candidate_k`.
- Hand BM25 caches its index per filter on the memory store, but rebuilds per query on Postgres. Use
  `RAG_KEYWORD_RETRIEVER=pg_fts` for large corpora.
- Filtered HNSW can return fewer than k rows, which is mitigated with `hnsw.ef_search` and pgvector 0.8 iterative
  scans.
- The embedding cache (Redis) is keyed by model + sha256(text).
- Measured production latency: TBD. The in-process timings in reports come from a laptop with a tiny corpus.

## Engineering trade-offs
- **pgvector instead of a separate vector DB:** tenant filter and ANN search in one SQL statement, and one system to
  run. The cost is scale limits well beyond this project's needs.
- **RRF instead of weighted fusion:** no score calibration needed, and it measured better here (0.877 vs 0.840
  recall@5).
- **The version row is the job record** (no separate `ingestion_jobs` table): one source of truth; attempts are
  tracked by arq.
- **Refusal gate before the LLM:** cheaper and can't hallucinate. It only applies to scores with a meaningful scale
  (reranker or cosine), not RRF.
- **Offline fakes as defaults:** deterministic CI and evaluation, at the cost of meaningless absolute quality numbers.

## Limitations
- No OCR (text-layer PDFs only).
- Hand BM25 stemming is English-only and rule-based (`light`); Snowball is optional. No lemmatisation.
- The lexical reranker and judge are heuristics.
- Real-model evaluation numbers are TBD.
- The spot-check agreement isn't measured yet.
- Postgres row-level security isn't used (tenant scoping is enforced in the application).

## Roadmap
- Evaluate with a real embedding model and cross-encoder, then fill in the TBD numbers.
- Hand-label the judge spot-check sheet.
- OpenTelemetry spans.
- Refresh-token rotation and a denylist.
- AWS deployment via `cloud-infra-lab`.

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md). Licence: MIT.

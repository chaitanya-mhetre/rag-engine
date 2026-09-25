# Architecture

## Components and why each one exists

| Component | Code | Why it exists |
|---|---|---|
| FastAPI app | `api/app.py`, `api/routes.py` | Async HTTP layer. Most time goes to I/O (DB, embeddings, LLM), so async gives concurrency without threads. Pydantic models are the contract and the OpenAPI docs. |
| Dependencies | `api/deps.py` | Authentication (JWT → user), authorisation (collection permission), rate limiting. Routes stay declarative. |
| Composition root | `container.py` | The one place that picks implementations from settings (memory vs Postgres, fake vs real providers). Everything else depends on protocols. |
| File storage | `storage.py` | Originals are kept so documents can be re-parsed when chunking changes, without anyone uploading again. |
| Indexing service | `indexing.py` | Versioned, idempotent ingestion: sha256 dedupe, write the file before any row, atomic activation, FAILED keeps the old version live. |
| Job queue + worker | `jobs.py`, `worker.py` | Parsing and embedding big files can't happen inside a request. Upload returns 202. arq retries with backoff, and retries are safe because chunk replacement is transactional. |
| Stores | `store/memory.py`, `store/pg.py` | Same contract, two backends. The contract tests (`tests/unit/test_store.py`) run on both. |
| Schema + migrations | `store/schema.py`, `migrations/` | Postgres + pgvector: HNSW index on embeddings, GIN on the generated `tsvector`, `UNIQUE(document_id, content_sha256)`. |
| Retrievers | `retrieval/*` | BM25 (exact tokens) and vector search (meaning) fail differently; RRF fuses ranks without calibrating scores; the reranker buys precision on the top candidates. |
| Context builder | `generation/context.py` | Decides what the model sees: injection screening, dedupe, token budget, `[S#]` labels that make citations checkable. |
| LLM layer | `generation/llm.py`, `prompts.py` | One interface for fake/OpenAI/Gemini. Prompts are versioned and logged. |
| Answer validation | `generation/answer.py` | Parses JSON (falling back to markers), drops citations to labels that weren't in the context, verifies quotes, and turns "insufficient" into a refusal. |
| Pipeline | `pipeline.py` | Orchestrates condense → retrieve → gate → context → LLM → validate → log, with per-stage timings. |
| Access repo | `store/access*.py` | Tenants, users, roles, collection permissions, conversations, query logs. |
| Observability | `observability.py` | Prometheus metrics from every query log, JSON logs with request ids. |
| Evaluation | `evaluation/*` | Measures retrieval and answers against a hand-labelled dataset; the CI regression gate. |

## Request flow: `POST /v1/query`
1. `rate_limited` → `current_user`: the JWT is decoded (HS256 allow-list, `typ=access`) and the user loaded; token
   bucket per user.
2. `_authorised_filter`: every requested collection must be READ-able by this user. Collections in other tenants look
   missing (404). The result is a `SearchFilter(tenant_id, collection_ids, tags, mime_types, created_after)`.
3. `_history`: when `conversation_id` is set, the previous messages are loaded (owner only).
4. `RAGPipeline.answer` / `.stream`:
   - `condense` rewrites follow-ups using the history
   - `HybridRetriever.retrieve`: BM25 and vector search with `candidate_k`, then RRF, then rerank to `top_k`, and a trace
   - the gate: if the top reranked or cosine score is below the threshold, return a refusal without calling the LLM
   - `ContextBuilder.build`: screen → dedupe → budget → `[S1..Sn]`
   - the LLM: JSON mode for complete answers, marker text for streaming
   - `build_answer`: citations checked against the context
   - `QueryLog`: written through `MetricsSink` → `AccessRepo.write`
5. For conversations, the user and assistant messages are saved with citations and the query log id.

## Tenant isolation, layer by layer
- Every chunk row stores `tenant_id` and `collection_id`, and every retrieval query filters on both *inside the same
  SQL statement* as the ANN search (`store/pg.py: _scoped`).
- `SearchFilter` refuses an empty collection list, so there's no "search everything".
- The API checks permission for every collection id before building the filter.
- Tests: `test_tenant_isolation_in_vector_search` (store level, both backends) and `test_tenant_isolation` (API level).

## Versioning
```
upload v2 ──▶ file written ──▶ version row (PENDING) ──▶ worker: PROCESSING ──▶ chunks replaced (tx)
          ──▶ READY ──▶ UPDATE documents SET active_version_id = v2 WHERE <v2 is READY>  (single statement)
retrieval joins chunks.version_id = documents.active_version_id → readers see v1 or v2, never a mix
```

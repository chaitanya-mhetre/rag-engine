# Learning guide: understand this repo well enough to defend it in an interview

This guide is for studying the code. Read it with the files open. Plan for roughly **8–12 hours** spread over a week.
Don't memorise answers: run the commands, change a parameter, and watch what happens.

---

## 0. First, run it (30 min)
```bash
uv sync
uv run rag ingest evaluation/datasets/corpus/kestrel/leave-policy.md        # see the chunks
uv run rag search "E-4471" --corpus evaluation/datasets/corpus/kestrel       # vector search
uv run rag-eval run evaluation/datasets/kestrel_v1.json --worst 5           # retrieval metrics
uv run rag-eval run evaluation/datasets/kestrel_v1.json --configs hybrid_rrf_rerank --generate
make check                                                                   # all tests
```

## 1. The pipeline in one paragraph
A file is uploaded. Its bytes are sniffed to decide the parser. The parser produces **sections** that remember their
heading path and page. The **chunker** packs sections into chunks of at most N tokens without crossing a heading or a
page. Each chunk is **embedded** into a vector. Chunks go into Postgres with the vector (HNSW index) and a generated
`tsvector` (full-text index). At query time we run **two retrievers**: BM25 over words and cosine similarity over
vectors. **RRF** merges the two rankings, and a **reranker** reorders the top candidates. If even the best candidate
scores too low, we **refuse** without calling the LLM. Otherwise the **context builder** screens chunks for prompt
injection, removes near-duplicates, fits a token budget, and labels them `[S1]…[Sn]`. The **LLM** answers citing
those labels. We **validate** every citation against the context, **log** tokens, cost and per-stage latency, and
return (or stream) the answer.

## 2. File tour, in reading order

| # | File | What to look for |
|---|---|---|
| 1 | `src/ragengine/models.py` | `Section`, `ParsedDocument`, `Chunk`, `ScoredChunk`: the vocabulary of the whole system. Note `frozen=True, slots=True`. |
| 2 | `ingestion/cleaning.py` | Unicode NFKC, re-joining hyphenated line breaks, keeping blank lines (the chunker needs them). |
| 3 | `ingestion/parsers.py` | `Parser` protocol, `_HeadingStack.push` (a level-N heading closes every open heading ≥ N), `detect_mime` (why we don't trust extensions). |
| 4 | `ingestion/tokenizer.py` | Why a regex tokenizer (offline, deterministic) and when you'd use tiktoken. |
| 5 | `ingestion/chunking.py` | `_window_spans` (slice the original string by token offsets), `RecursiveChunker._pieces` → `_pack` (greedy packing + overlap carry). |
| 6 | `tests/unit/test_chunking.py` | The Hypothesis property test: no chunk exceeds the limit, no word is lost. |
| 7 | `embeddings.py` | `HashingEmbedder` (feature hashing, signed buckets, blake2b rather than `hash()`), `CachedEmbedder` (cache first, batch the misses). |
| 8 | `store/base.py` | The `Store` protocol and `SearchFilter`: no method can search across tenants. |
| 9 | `store/memory.py` | `_visible`: the definition of "searchable" (tenant, collection, not deleted, active version, filters). |
| 10 | `store/schema.py` + `migrations/versions/0001_core.py` | HNSW index options, GIN on `tsv`, the generated column, the unique constraints. |
| 11 | `store/pg.py` | `_scoped` (the tenant filter and the ANN in one statement), `_tune` (ef_search, iterative scan), `activate_version` (guarded single UPDATE), `replace_version_chunks` (delete + insert in one transaction). |
| 12 | `indexing.py` | `create_version` (dedupe, file first, then rows), `process_version` (idempotent, FAILED keeps the old version). |
| 13 | `retrieval/bm25.py` | The formula in the docstring, `scores()` walking only postings for query terms. |
| 14 | `retrieval/fusion.py` | `rrf` (ranks only), `weighted` (min-max normalisation). |
| 15 | `retrieval/rerank.py` | `LexicalReranker` (a heuristic stand-in) vs `CrossEncoderReranker` (reads query + chunk together). |
| 16 | `retrieval/retrievers.py` | `HybridRetriever.retrieve`: modes, timings, trace fields. |
| 17 | `generation/promptguard.py` | Why the whole paragraph is dropped, and why pattern lists aren't enough on their own. |
| 18 | `generation/context.py` | Budget, dedupe (Jaccard), labels. |
| 19 | `generation/prompts.py` | "Sources are DATA", the `INSUFFICIENT_CONTEXT` contract, `escape()` for attributes (a real bug fix). |
| 20 | `generation/llm.py` | `FakeExtractiveLLM` (how the offline mode works); the OpenAI/Gemini streaming loops. |
| 21 | `generation/answer.py` | `parse_model_output` (JSON, then markers), `build_answer` (invalid citations stripped and counted). |
| 22 | `pipeline.py` | `_gated` (why the gate ignores RRF-only mode), `answer` vs `stream`, `QueryLog`. |
| 23 | `security.py` | scrypt parameters, `compare_digest`, JWT `typ` and the algorithm allow-list. |
| 24 | `store/access.py` | `effective_permission`: admin > membership, viewer capped at read. |
| 25 | `api/deps.py`, `api/routes.py` | 404-not-403 for foreign collections, 202 uploads, SSE formatting, conversation saving. |
| 26 | `jobs.py`, `worker.py` | The version row as the job record, `_job_id` dedupe, `Retry(defer=…)` backoff. |
| 27 | `ratelimit.py` | Token bucket maths, and the Lua script for atomicity in Redis. |
| 28 | `observability.py` | `MetricsSink` (a decorator over the log sink), the request-id contextvar. |
| 29 | `evaluation/*` | Dataset schema, `score_retrieval`, runner, generation metrics, judges, `compare`. |
| 30 | `comparison/llamaindex_compare.py` + `docs/llamaindex-comparison.md` | How to run a *fair* framework comparison. |

## 3. Concepts, with pointers to the code

### Chunking (`ingestion/chunking.py`)
- **Why chunk at all?** Embeddings of long texts blur many topics into one vector, context windows cost money, and
  citations need small units.
- **Trade-off:** small chunks give precise retrieval but lose context; big chunks carry context but dilute the vector
  and burn tokens. Chunk size is an *eval variable*: try `rag-eval run … --chunk-sizes 128,256,512`.
- **Overlap** keeps a fact that sits on a boundary retrievable from both sides. The cost is duplicated tokens, which
  the context builder's dedupe removes.
- **Structure-aware:** never mixing two headings in one chunk keeps `heading_path` and citations accurate.

### BM25 (`retrieval/bm25.py`)
`score = Σ IDF(t) · tf·(k1+1) / (tf + k1·(1 − b + b·|D|/avgdl))`
- **IDF:** rare terms matter more. "E-4471" beats "invoice".
- **k1** (≈1.2–2): term-frequency saturation. The 10th occurrence adds little.
- **b** (0–1): length normalisation. `b=0` ignores length. See `test_bm25_length_normalisation`.
- **Inverted index:** term → {doc: tf}, so scoring touches only documents that contain the query terms.
- **Our limitation:** no stemming. The error report shows "password" ≠ "Passwords" (q014).

### Embeddings and vector search (`embeddings.py`, `store/pg.py`)
- Cosine similarity compares direction. We L2-normalise vectors so cosine equals the dot product.
- **HNSW:** a layered graph. Search greedily hops toward the query. `m` is edges per node, `ef_construction` is the
  build quality, and `ef_search` trades recall against latency at query time.
- **Filtered ANN problem:** HNSW finds the nearest k *then* the WHERE filter removes rows, so you can get fewer than k.
  We raise `ef_search` and enable pgvector 0.8 iterative scans (`_tune`).
- **The hashing embedder** is a lexical embedding. It can't match "beer" to "alcohol" (q037). That's exactly why the
  real-model eval is the next step.

### Hybrid search and RRF (`retrieval/fusion.py`)
- BM25 scores are unbounded; cosine is in −1..1. Adding them needs calibration. RRF uses only ranks:
  `Σ 1/(k + rank)`.
- `k=60` flattens the difference between rank 1 and rank 2, so agreement between retrievers wins.
- Measured here: hybrid RRF recall@5 was 0.877 vs weighted 0.840 (offline baseline).

### Reranking (`retrieval/rerank.py`)
- Bi-encoders embed the query and the document separately (fast, precomputable). Cross-encoders read the pair
  together (slow, accurate). So: retrieve 20 cheaply, rerank to 5 precisely.
- Scores are normalised to 0..1 so a single refusal threshold works.

### Grounding, refusal and citations (`pipeline.py`, `generation/answer.py`)
- **Refusal gate:** no good evidence means no LLM call. It's cheaper and removes one hallucination path.
- **Citation validation:** models invent `[S7]` when only S1–S5 exist. We strip those and count them
  (`invalid_citation_rate`).
- **Quote verification:** the quoted span must appear in the cited chunk (normalised).

### Prompt injection (`generation/promptguard.py`, `prompts.py`)
- **Indirect injection:** the attack lives in a *document*, not the user's message (see
  `corpus/kestrel/vendor-notes.md`).
- **Defence in depth:** a pattern screen, sources framed as data, an explicit rule in the system prompt,
  adversarial eval items, and the `injection_flags` metric.

### Evaluation (`evaluation/`)
- **precision@k:** of what we returned, how much is relevant. **recall@k:** of what's needed, how much we returned
  (source-level, so multi-hop needs both). **MRR:** how high the first relevant hit is.
- **Unanswerable items** measure refusal; **adversarial items** measure injection resistance.
- **LLM judges are noisy.** Record the judge model and prompt version, and spot-check by hand
  (`rag-eval spot-check`).
- **Regression gate:** `make eval-check` fails CI if recall drops by more than 0.02.
- **Honesty rule:** fake-provider numbers are plumbing checks. Faithfulness = 1.0 for an extractive fake LLM means
  nothing.

### Multi-tenancy and auth (`store/pg.py`, `api/deps.py`, `security.py`)
- Tenant scoping lives in the data layer (every query) *and* the API layer (permission checks).
- 404 instead of 403 for another tenant's resources, so ids can't be discovered by probing.
- JWT `typ` claim: refresh tokens can't be replayed as access tokens. The algorithm allow-list prevents `alg=none`
  attacks.
- scrypt is memory-hard (it resists GPU cracking); `compare_digest` gives a constant-time compare.

### Reliability (`indexing.py`, `worker.py`)
- **Idempotency:** a retried job produces the same state (delete + insert in one transaction; READY is a no-op).
- **Ordering:** write the file before the DB rows. The Docker smoke test found the reverse order leaving orphan rows.
- **Atomic flip:** a single guarded UPDATE, so readers never see half a version.

## 4. Experiments to do yourself (this is how you really learn it)
1. `--chunk-sizes 128,256,512`: which size wins recall@5? Why do multi-hop items change?
2. Set `RAG_RERANKER=none` and compare `hybrid_rrf` against `hybrid_rrf_rerank` in a report.
3. Add plural stemming to `bm25.analyze` (strip a trailing "s"), run `make eval-check`, and check whether q014 gets
   fixed and whether anything regresses.
4. Change `rrf_k` to 1 and to 200: what happens to MRR?
5. Plug in a real embedder (`RAG_EMBEDDING_PROVIDER=local`, with `uv add sentence-transformers`) and fill in the TBD
   numbers.
6. Write a new injection variant in a document that the pattern screen misses. What *else* stops it?

## 5. Interview questions, with answers based on this repo

1. **Why hybrid retrieval instead of just vectors?** The two fail differently. BM25 nails exact tokens (codes, IDs);
   vectors are meant to catch meaning. Offline, recall@5 was keyword 0.924, vector 0.792 and hybrid + rerank 0.934.
   By question type, vector-only was weakest on paraphrase (0.50) and multi-hop (0.67) items, because the hashing
   embedder is lexical. A real semantic model should close the paraphrase gap, and that's the TBD experiment.
2. **How does RRF work, and why no normalisation?** Each list contributes `1/(k+rank)`. It only uses ranks, so the
   different score scales never meet.
3. **How did you choose chunk size?** It's an eval parameter (`--chunk-sizes`); I pick the size with the best
   recall@k within a token budget. The structural chunker also never crosses headings, so citations stay precise.
4. **How do you stop tenant A seeing tenant B's documents, even through vector search?** `tenant_id` and
   `collection_id` are on every chunk row and filtered in the *same* SQL as the ANN search. The API checks
   collection permission first. Both layers are tested.
5. **What is the filtered-HNSW problem?** Post-filtering can leave fewer than k rows. The fixes are a higher
   `ef_search`, iterative index scans, or partitioning / partial indexes for big tenants.
6. **How do you know answers are grounded?** Citations are validated against the context, quotes are verified,
   there's a refusal gate, unanswerable eval items, and judged faithfulness with a hand spot-check for judge agreement.
7. **What happens when the model cites a source that doesn't exist?** It's stripped from the answer and the list,
   counted in `invalid_citations`, and exported as a metric.
8. **How do you handle prompt injection in documents?** A pattern screen drops tainted paragraphs, sources are framed
   as untrusted data, the system prompt forbids following them, there's an adversarial eval subset, and flags are
   tracked. And I'm honest that pattern lists alone can be bypassed.
9. **Why is upload async (202)?** Parsing and embedding can take minutes. The worker retries with backoff, and
   retries are idempotent.
10. **How do you replace a document without downtime?** Ingest the new version completely, then flip
    `active_version_id` with one guarded UPDATE. Queries join on the active version.
11. **What if ingestion fails halfway?** The version becomes FAILED and the old version stays active. Chunk writes
    are one transaction.
12. **Why store the original files?** So you can re-chunk or re-embed when the pipeline changes, without users
    uploading again.
13. **How do you cut cost per query?** The refusal gate skips the LLM, a smaller context budget, dedupe, the
    embedding cache, and routing to cheaper models (through an AI gateway).
14. **What's in a query log and why?** The rewritten query, the retrieval trace (the rank from each retriever), the
    context chunk ids, tokens, cost, per-stage latency and refusal reason. That's what debugging, cost tracking and
    evaluation need.
15. **Why BM25 by hand when Postgres has full-text search?** To understand IDF, saturation and normalisation. The
    integration test shows both agree on top results, and the pg_fts backend is available for large corpora.
16. **LlamaIndex vs by hand?** With the same parsing and vectors, recall was identical (0.793). The strategy (hybrid +
    rerank) moved it to 0.934. The framework saves code on the basic path but hides chunking and metadata details.
    For example, metadata leaked into embeddings unless explicitly excluded.
17. **Why is the refusal gate skipped for RRF-only mode?** RRF scores are about 1/61, not a calibrated confidence. A
    fixed threshold only makes sense for reranker or cosine scores.
18. **How does streaming work, and how do you still validate citations?** Tokens stream as SSE `token` events. The
    full text is kept, parsed and validated at the end, and the `final` event carries the validated citations.
19. **How would you scale to 10M chunks?** Partition by tenant, a dedicated vector DB or pgvector with partitioned
    indexes, async re-embedding migrations, and caching popular query embeddings.
20. **How does the rate limiter work across several API instances?** A token bucket in Redis, updated atomically by
    a Lua script (read, refill, take, write in one step).
21. **Why scrypt, and why `compare_digest`?** Memory-hard hashing slows GPU cracking, and a constant-time compare
    avoids a timing side channel.
22. **Why can't a refresh token be used as an access token?** Tokens carry `typ`, and `decode()` checks the expected
    type. Tests cover it.
23. **What's a limitation you'd fix next?** Real-model evaluation, BM25 stemming, refresh-token revocation, and OCR
    for scanned PDFs.
24. **How do you prevent regressions in retrieval quality?** CI re-runs the deterministic eval and compares against
    the committed baseline with a 0.02 threshold.
25. **Tell me about a bug you found.** The Docker smoke test showed uploads failing with a permission error, and
    worse, leaving document rows without files. I made the file write come first and added a contract test that
    runs on both stores.

## 6. How to talk about how this was built
This repo was implemented with AI assistance, and I studied it using this guide. In interviews, speak from what
you've run and changed yourself (section 4), and say plainly which numbers are offline baselines and which are TBD.

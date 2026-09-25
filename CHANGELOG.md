# Changelog

## Unreleased
### Added
- Configurable BM25 analyzer (`retrieval/analysis.py`): hand-written `light` stemmer and optional Snowball
  (`RAG_BM25_STEMMER=none|light|snowball`, `RAG_BM25_STOPWORDS`), plus `rag-eval run --bm25-stemmer` and
  `--bm25-no-stopwords` (#2).

### Changed
- `light` stemming is now the BM25 default: it improved every BM25-based config with no per-item regressions in the
  offline eval (`evaluation/reports/bm25_stemming.md`). The CI eval gate compares against the new report.

### Fixed
- Fused-score ties were broken by random chunk UUIDs, so repeated eval runs differed slightly; ties now break on
  (source, ordinal, id).

## 0.1.0 (M1–M8)
- **M1:** parsers (txt/md/html/pdf/docx) with heading paths and mime sniffing; recursive and token-window chunkers;
  `rag ingest`.
- **M2:** embedder protocol, offline hashing embedder, cached batching (memory/Redis); memory and Postgres/pgvector
  stores; versioned idempotent indexing; `rag search`.
- **M3:** hand-written BM25, RRF + weighted fusion, lexical/cross-encoder rerankers, hybrid retriever with traces.
- **M4:** `kestrel_v1` dataset (60 hand-written items), retrieval metrics, `rag-eval run/compare`, baseline report.
- **M5:** context builder with injection screening, versioned prompts, fake/OpenAI/Gemini LLMs, citation
  validation, refusal gate, query logs with tokens, cost and latency; generation metrics.
- **M6:** auth (JWT, scrypt), tenants/roles/collection permissions, FastAPI routes, SSE streaming, conversations, arq
  worker, rate limiting.
- **M7:** judged metrics (lexical + LLM judges), spot-check sheets, LlamaIndex comparison.
- **M8:** Prometheus metrics, JSON logs with request ids, Docker image, compose app profile, CI with an eval
  regression gate.

### Fixed
- Uploads could leave orphan document rows when file storage failed; the file is now written before any row
  (found by the Docker smoke test).
- The fake LLM's source parser broke on section names containing `>`; prompt attributes are now HTML-escaped.

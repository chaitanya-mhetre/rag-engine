# LlamaIndex vs by-hand retrieval (M7)

Source: `evaluation/reports/m7_llamaindex_comparison.json` (git `017933f`, dataset `kestrel_v1`, 53 answerable items, k = 5).
Reproduce: `uv run --extra llamaindex python -m ragengine.comparison.llamaindex_compare`.

> **Offline fake-provider baseline, not a quality claim.** Both sides use the deterministic hashing embedder.
> The absolute numbers say nothing about real embedding models. The comparison is only meaningful *relative*
> to itself, because the inputs are held constant.

## What was held constant
- **Parsing:** our parsers produce sections for both sides. LlamaIndex receives one `Document` per section, with the
  heading path in metadata, so section-level relevance matching works the same way for both.
- **Embeddings:** the same `HashingEmbedder` vectors, via an adapter class (`HashingLIEmbedding`).
- **Dataset, k and metrics:** the same `score_retrieval` function.

What differs is chunking (LlamaIndex `SentenceSplitter(256, 32)` vs our `RecursiveChunker(256, 32)`), the index,
and the retrieval code. Scope is **vector retrieval only**: LlamaIndex BM25 needs `llama-index-retrievers-bm25`,
which isn't installed.

## Measured results

| pipeline | hit@5 | precision@5 | recall@5 | MRR | p50 ms | p95 ms | index build ms |
|---|---|---|---|---|---|---|---|
| LlamaIndex, vector | 0.830 | 0.177 | 0.793 | 0.742 | 1.38 | 1.52 | 168 |
| ours, vector | 0.830 | 0.174 | 0.793 | 0.736 | 0.62 | 1.06 | 9 |
| ours, hybrid RRF + lexical rerank | 0.962 | 0.204 | 0.934 | 0.884 | 1.10 | 1.54 | (same index) |

Latencies are in-process on one laptop CPU with a tiny corpus (11 documents). They show the relative overhead of
each code path, nothing about production latency.

## Reading the numbers
- **Same inputs, same quality.** With identical parsing and vectors, vector-only recall@5 is identical (0.793), and
  precision and MRR differ only by chunk-boundary effects. The framework doesn't add retrieval quality on its own;
  the embedding model and the retrieval *strategy* do.
- **The strategy matters more than the framework.** Adding BM25 + RRF + reranking to our pipeline moved recall@5
  from 0.793 to 0.934 on this dataset. With LlamaIndex that means adding extra packages and composing its retrievers.
- **Overhead:** LlamaIndex's index build (Document → Node objects, metadata handling, docstore) took ~168 ms against
  ~9 ms for our in-memory index on this corpus. Query p50 was about 2× ours. Neither matters at this scale, but the
  gap is framework bookkeeping, not math.

## Code size (this repo)
- LlamaIndex version: `comparison/llamaindex_compare.py`, about 60 lines of pipeline code plus the measurement harness.
- By hand: chunking (151), BM25 (61), fusion (50), rerank (82), retrievers (183), memory store (153) = about 680 lines,
  plus tests.

## When to use which
| Use LlamaIndex when… | Build by hand when… |
|---|---|
| you need many connectors and loaders quickly | retrieval quality is the product and you must control every stage |
| you're prototyping and the defaults are fine | you need tenant isolation inside the vector query (one SQL statement) |
| the team already knows the framework | you need per-stage traces, costs and evaluation hooks you fully understand |

What the framework hid: chunk-boundary logic, how metadata gets stored and excluded from embeddings (it needed
explicit `excluded_embed_metadata_keys`, or the heading text would have silently changed the vectors), and the
index data structures. What it gave: less code to write for the basic path.

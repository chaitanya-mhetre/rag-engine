# BM25 stemming: before/after (issue #2)

> **Offline fake-provider baseline, not a quality claim.** Hashing embedder, lexical reranker; `kestrel_v1`
> (60 items, 53 answerable), k = 5. The corpus and questions are self-written, and 1 question = 0.019 of hit@5, so
> treat small deltas as indicative, not significant.

Raw reports (same commit, runs reproducible after the fusion tie-break fix in `0ce3e30`):
[`bm25_stemming_none`](bm25_stemming_none.md) · [`bm25_stemming_light`](bm25_stemming_light.md) ·
[`bm25_stemming_snowball`](bm25_stemming_snowball.md)

Reproduce:
```bash
for s in none light snowball; do
  uv run rag-eval run evaluation/datasets/kestrel_v1.json --bm25-stemmer $s --out /tmp/stem --name $s
done
uv run rag-eval compare /tmp/stem/none.json /tmp/stem/light.json
```

## Aggregate (Δ vs no stemming)

| config | metric | none | light | Δ light | snowball | Δ snowball |
|---|---|---|---|---|---|---|
| keyword | recall@5 | 0.924 | 0.972 | +0.047 | 0.981 | +0.057 |
| keyword | MRR | 0.844 | 0.921 | +0.077 | 0.919 | +0.075 |
| hybrid_rrf | recall@5 | 0.877 | 0.915 | +0.038 | 0.906 | +0.028 |
| hybrid_rrf | MRR | 0.826 | 0.850 | +0.024 | 0.844 | +0.019 |
| hybrid_weighted | recall@5 | 0.840 | 0.943 | +0.104 | 0.953 | +0.113 |
| hybrid_weighted | MRR | 0.816 | 0.866 | +0.050 | 0.859 | +0.043 |
| **hybrid_rrf_rerank** (default) | recall@5 | 0.934 | 0.953 | +0.019 | 0.962 | +0.028 |
| **hybrid_rrf_rerank** (default) | MRR | 0.884 | 0.893 | +0.009 | 0.893 | +0.009 |
| vector | all | unchanged | | 0 | | 0 |

## Per-item changes (answerable items; better/worse by recall, then MRR)

| config | light: better | light: worse | snowball: better | snowball: worse |
|---|---|---|---|---|
| keyword | q014 q024 q034 q039 q040 q041 q045 q048 | — | same + q049 | **q036** |
| hybrid_rrf | q014 q034 q040 q045 | — | q014 q045 q049 | — |
| hybrid_weighted | q014 q024 q034 q039 q040 q041 q045 q047 | — | same + q049 | **q036** |
| hybrid_rrf_rerank | q014 | — | q014 q049 | — |

- **q014** "What is the minimum password length?" is the motivating case: the chunk says "Passwords", so
  exact-match BM25 missed it. Both stemmers fix it.
- **q036** (newborn/parental leave paraphrase) gets worse with Snowball: its more aggressive stemming makes more
  query terms match unrelated chunks.

## Decision
**`light` is the default** (`RAG_BM25_STEMMER=light`, and `RunConfig.bm25_stemmer`):
1. It improves every config that uses BM25, with **zero per-item regressions**.
2. On the default pipeline (hybrid RRF + rerank), Snowball's extra gain is **one question** (q049), within noise
   for a 53-item set, and Snowball regresses q036 in two other configs.
3. It's ~40 lines of hand-written, explainable rules, with no dependency on the hot path.

Snowball stays available (`RAG_BM25_STEMMER=snowball`) and should be re-evaluated once there's a larger or real
dataset and a real embedder. `RAG_BM25_STEMMER=none` restores the old behaviour.

The CI eval gate (`make eval-check`) now compares against `bm25_stemming_light.json`, so losing the improvement
fails the build.

## Found along the way
Repeated runs of the *same* config gave slightly different MRR (e.g. hybrid_rrf 0.848 vs 0.859): fused-score ties
were broken by random chunk UUIDs. Ties now break on (source, ordinal, id) (`0ce3e30`), and three repeated runs of
each stemmer produce identical reports.

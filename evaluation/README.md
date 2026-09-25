# Evaluation

| Folder | What's in it |
|---|---|
| `datasets/` | hand-written question sets (`kestrel_v1.json`) and the fixed corpus in `datasets/corpus/` |
| `metrics/` | the metric definitions are documented here; the code lives in `src/ragengine/evaluation/metrics.py` (retrieval) and `generation_metrics.py` (answers) |
| `runners/` | the runner lives in `src/ragengine/evaluation/runner.py`; `rag-eval` is its CLI |
| `reports/` | committed JSON + Markdown reports; each one records the git sha, the dataset sha256 and the config |

```bash
uv run rag-eval run evaluation/datasets/kestrel_v1.json --out evaluation/reports
uv run rag-eval run evaluation/datasets/kestrel_v1.json --configs vector,hybrid_rrf_rerank --chunk-sizes 128,256,512
uv run rag-eval compare evaluation/reports/A.json evaluation/reports/B.json --fail-on-regression 0.02
```

**Reports produced with the default providers are an offline fake-provider baseline, not a quality claim.**
The hashing embedder can't match synonyms and the lexical reranker is a heuristic. They exist so the pipeline, the
metrics and CI regression checks run deterministically without API keys. Real-model numbers are TBD.

## Dataset: `kestrel_v1`
60 items over 11 self-written documents about the fictional company Kestrel Robotics:
25 factual, 10 keyword (exact codes and identifiers), 10 paraphrase, 6 multi-hop, 7 unanswerable, 2 adversarial.
The adversarial items target `vendor-notes.md`, which contains a prompt-injection paragraph.

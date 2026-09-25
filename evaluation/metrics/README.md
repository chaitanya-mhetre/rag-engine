# Metric definitions

Retrieval (`src/ragengine/evaluation/metrics.py`), computed per answerable item at cut-off k:
- **hit@k**: 1 if any of the top-k chunks matches an expected source.
- **precision@k**: relevant chunks in the top-k divided by k.
- **recall@k**: expected *sources* covered by the top-k divided by the number of expected sources. This is source-level,
  so a multi-hop question needs both sources to score 1.
- **MRR**: 1 / rank of the first relevant chunk.

A chunk matches an expected source when `chunk.metadata.source == document` and `section` (if given) is a
case-insensitive substring of one of the chunk's headings.

Generation (`src/ragengine/evaluation/generation_metrics.py`) is described in that module's docstring.

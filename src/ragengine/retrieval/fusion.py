"""Combining ranked lists from different retrievers.

Reciprocal Rank Fusion (Cormack et al., 2009):   RRF(d) = Σ_lists 1 / (k + rank_list(d))

It uses only ranks, so BM25 scores (unbounded) and cosine similarities (−1..1) never need to
be calibrated against each other. `k` (default 60) damps the advantage of the very top ranks.
Weighted fusion is the alternative: min-max normalise each list, then a weighted sum.
"""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from ragengine.models import Chunk, ScoredChunk


def _tiebreak(chunk: Chunk) -> tuple[str, int, str]:
    """Deterministic order for equal scores: source document, then position, then id.

    Breaking ties on the random chunk UUID alone made evaluation runs over the same corpus
    differ slightly from run to run, which hid small real changes (found while measuring
    BM25 stemming, issue #2)."""
    return (str(chunk.metadata.get("source", "")), chunk.ordinal, str(chunk.id))


def rrf(result_lists: Sequence[Sequence[ScoredChunk]], k: int = 60) -> list[ScoredChunk]:
    if k <= 0:
        raise ValueError("k must be positive")
    scores: dict[UUID, float] = {}
    chunks: dict[UUID, Chunk] = {}
    for results in result_lists:
        for rank, hit in enumerate(results, start=1):
            scores[hit.chunk.id] = scores.get(hit.chunk.id, 0.0) + 1.0 / (k + rank)
            chunks[hit.chunk.id] = hit.chunk
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], _tiebreak(chunks[kv[0]])))
    return [ScoredChunk(chunks[cid], score, "rrf") for cid, score in ordered]


def _min_max(results: Sequence[ScoredChunk]) -> dict[UUID, float]:
    if not results:
        return {}
    values = [r.score for r in results]
    lo, hi = min(values), max(values)
    span = hi - lo
    return {r.chunk.id: (r.score - lo) / span if span else 1.0 for r in results}


def weighted(
    keyword: Sequence[ScoredChunk], vector: Sequence[ScoredChunk], alpha: float = 0.5
) -> list[ScoredChunk]:
    """alpha = weight of the vector list (0 = keyword only, 1 = vector only)."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    kw, vec = _min_max(keyword), _min_max(vector)
    chunks = {r.chunk.id: r.chunk for r in (*keyword, *vector)}
    fused = {cid: (1 - alpha) * kw.get(cid, 0.0) + alpha * vec.get(cid, 0.0) for cid in chunks}
    ordered = sorted(fused.items(), key=lambda kv: (-kv[1], _tiebreak(chunks[kv[0]])))
    return [ScoredChunk(chunks[cid], score, "weighted") for cid, score in ordered]

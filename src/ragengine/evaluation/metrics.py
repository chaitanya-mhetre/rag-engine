"""Retrieval metrics computed against hand-labelled expected sources.

For one question with expected sources S and retrieved top-k chunks C = [c1..ck]:
- relevant(c)  = c matches any source in S
- hit@k        = 1 if any retrieved chunk is relevant
- precision@k  = |{c ∈ C : relevant(c)}| / k
- recall@k     = |{s ∈ S : some c ∈ C matches s}| / |S|     (source-level: multi-hop needs both)
- MRR          = 1 / rank of the first relevant chunk (0 if none)
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from ragengine.evaluation.dataset import EvalItem
from ragengine.models import Chunk


@dataclass(frozen=True, slots=True)
class RetrievalScores:
    hit: float
    precision: float
    recall: float
    mrr: float
    first_relevant_rank: int | None


def score_retrieval(item: EvalItem, retrieved: Sequence[Chunk], k: int) -> RetrievalScores:
    top = list(retrieved)[:k]
    relevant_flags = [any(s.matches(c) for s in item.expected_sources) for c in top]
    first = next((i + 1 for i, rel in enumerate(relevant_flags) if rel), None)
    covered = sum(1 for s in item.expected_sources if any(s.matches(c) for c in top))
    n_sources = len(item.expected_sources) or 1
    return RetrievalScores(
        hit=1.0 if first else 0.0,
        precision=sum(relevant_flags) / k,
        recall=covered / n_sources,
        mrr=1.0 / first if first else 0.0,
        first_relevant_rank=first,
    )


def mean(values: Sequence[float]) -> float:
    return round(statistics.fmean(values), 4) if values else 0.0


def percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile (pct in 0..100)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, min(len(ordered), round(pct / 100 * len(ordered) + 0.5)))
    return round(ordered[rank - 1], 3)

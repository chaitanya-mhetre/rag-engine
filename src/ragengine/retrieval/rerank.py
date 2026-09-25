"""Second-stage rerankers. First-stage retrieval optimises recall; reranking buys precision.

Scores are normalised to 0..1 so the same refusal threshold works for every reranker.
"""

from __future__ import annotations

import asyncio
import math
from itertools import pairwise
from typing import Protocol

from ragengine.models import ScoredChunk
from ragengine.retrieval.bm25 import analyze


class Reranker(Protocol):
    name: str

    async def rerank(
        self, query: str, hits: list[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]: ...


class NoopReranker:
    name = "none"

    async def rerank(self, query: str, hits: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        return hits[:top_n]


class LexicalReranker:
    """Offline heuristic reranker (no model): query-term coverage plus phrase proximity.

    score = 0.7 · coverage + 0.2 · bigram_coverage + 0.1 · heading_coverage
    where coverage = fraction of distinct query terms present in the chunk text.

    It is a deterministic stand-in so the pipeline and evaluation run without downloads.
    A cross-encoder reads query and chunk *together* and is far stronger; compare them in eval.
    """

    name = "lexical"

    def _score(self, query: str, hit: ScoredChunk) -> float:
        q_terms = analyze(query)
        if not q_terms:
            return 0.0
        body = analyze(hit.chunk.text)
        body_set = set(body)
        coverage = len(set(q_terms) & body_set) / len(set(q_terms))
        q_bigrams = set(pairwise(q_terms))
        body_bigrams = set(pairwise(body))
        bigram_cov = len(q_bigrams & body_bigrams) / len(q_bigrams) if q_bigrams else coverage
        heading = set(analyze(" ".join(hit.chunk.heading_path)))
        heading_cov = len(set(q_terms) & heading) / len(set(q_terms))
        return 0.7 * coverage + 0.2 * bigram_cov + 0.1 * heading_cov

    async def rerank(self, query: str, hits: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        rescored = [ScoredChunk(h.chunk, self._score(query, h), self.name) for h in hits]
        # stable sort keeps first-stage order as the tie-breaker
        rescored.sort(key=lambda h: -h.score)
        return rescored[:top_n]


class CrossEncoderReranker:  # pragma: no cover - optional extra (downloads a model)
    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        from sentence_transformers import CrossEncoder

        self.name = f"cross-encoder:{model}"
        self._model = CrossEncoder(model)

    async def rerank(self, query: str, hits: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        if not hits:
            return []
        pairs = [(query, h.chunk.text) for h in hits]
        logits = await asyncio.to_thread(self._model.predict, pairs)
        rescored = [
            ScoredChunk(h.chunk, 1 / (1 + math.exp(-float(s))), self.name)
            for h, s in zip(hits, logits, strict=True)
        ]
        rescored.sort(key=lambda h: -h.score)
        return rescored[:top_n]

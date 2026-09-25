"""Retrievers and the hybrid orchestrator.

    query ─┬─ keyword retriever (hand BM25 or Postgres FTS) ─┐
           └─ vector retriever (embed query → pgvector)      ┴─▶ fuse (RRF | weighted) ─▶ rerank

Every stage is timed and every candidate's per-stage rank is recorded in a trace. The trace is
what `query_logs` stores and what the evaluation uses for error analysis.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any, Protocol
from uuid import UUID

from ragengine.embeddings import Embedder
from ragengine.models import Chunk, ScoredChunk
from ragengine.retrieval.analysis import Analyzer
from ragengine.retrieval.bm25 import BM25Index
from ragengine.retrieval.fusion import rrf, weighted
from ragengine.retrieval.rerank import NoopReranker, Reranker
from ragengine.store.base import SearchFilter, Store


class KeywordRetriever(Protocol):
    name: str

    async def search(self, flt: SearchFilter, query: str, k: int) -> list[ScoredChunk]: ...


class BM25Retriever:
    """Hand-written BM25 over the chunks visible to the filter.

    The index is cached per filter and rebuilt when the store's `generation` changes. Stores
    without a generation counter (Postgres) rebuild per query, which is fine for small corpora;
    use `PgFtsRetriever` for large ones.
    """

    name = "bm25"

    def __init__(
        self,
        store: Store,
        k1: float = 1.5,
        b: float = 0.75,
        analyzer: Analyzer | None = None,
    ) -> None:
        self.store, self.k1, self.b = store, k1, b
        self.analyzer = analyzer or Analyzer()
        self._cache: dict[SearchFilter, tuple[int, BM25Index, list[Chunk]]] = {}

    async def _index(self, flt: SearchFilter) -> tuple[BM25Index, list[Chunk]]:
        generation = getattr(self.store, "generation", None)
        cached = self._cache.get(flt)
        if generation is not None and cached and cached[0] == generation:
            return cached[1], cached[2]
        chunks = await self.store.active_chunks(flt)
        index = BM25Index([c.text for c in chunks], self.k1, self.b, self.analyzer)
        if generation is not None:
            self._cache[flt] = (generation, index, chunks)
        return index, chunks

    async def search(self, flt: SearchFilter, query: str, k: int) -> list[ScoredChunk]:
        index, chunks = await self._index(flt)
        return [ScoredChunk(chunks[i], s, self.name) for i, s in index.search(query, k)]


class PgFtsRetriever:
    name = "pg_fts"

    def __init__(self, store: Any) -> None:
        self.store = store  # a PgStore; typed loosely to avoid importing asyncpg in tests

    async def search(self, flt: SearchFilter, query: str, k: int) -> list[ScoredChunk]:
        results: list[ScoredChunk] = await self.store.keyword_search(flt, query, k)
        return results


class VectorRetriever:
    name = "vector"

    def __init__(self, store: Store, embedder: Embedder) -> None:
        self.store, self.embedder = store, embedder

    async def search(self, flt: SearchFilter, query: str, k: int) -> list[ScoredChunk]:
        (vec,) = await self.embedder.embed([query])
        return await self.store.vector_search(flt, vec, k)


class Mode(enum.StrEnum):
    KEYWORD = "keyword"
    VECTOR = "vector"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"


class Fusion(enum.StrEnum):
    RRF = "rrf"
    WEIGHTED = "weighted"


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    mode: Mode = Mode.HYBRID_RERANK
    fusion: Fusion = Fusion.RRF
    candidate_k: int = 20
    top_k: int = 5
    rrf_k: int = 60
    alpha: float = 0.5


@dataclass(slots=True)
class RetrievalResult:
    hits: list[ScoredChunk]
    trace: list[dict[str, Any]]
    latency_ms: dict[str, float] = field(default_factory=dict)

    @property
    def top_score(self) -> float:
        return self.hits[0].score if self.hits else 0.0


def _ranks(hits: list[ScoredChunk]) -> dict[UUID, int]:
    return {h.chunk.id: rank for rank, h in enumerate(hits, start=1)}


class HybridRetriever:
    def __init__(
        self,
        keyword: KeywordRetriever,
        vector: VectorRetriever,
        reranker: Reranker | None = None,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.keyword, self.vector = keyword, vector
        self.reranker: Reranker = reranker or NoopReranker()
        self.config = config or RetrievalConfig()

    async def retrieve(
        self, flt: SearchFilter, query: str, config: RetrievalConfig | None = None
    ) -> RetrievalResult:
        cfg = config or self.config
        timings: dict[str, float] = {}

        async def timed(stage: str, coro: Any) -> list[ScoredChunk]:
            start = time.perf_counter()
            result: list[ScoredChunk] = await coro
            timings[stage] = round((time.perf_counter() - start) * 1000, 3)
            return result

        kw_hits: list[ScoredChunk] = []
        vec_hits: list[ScoredChunk] = []
        if cfg.mode is not Mode.VECTOR:
            kw_hits = await timed(
                self.keyword.name, self.keyword.search(flt, query, cfg.candidate_k)
            )
        if cfg.mode is not Mode.KEYWORD:
            vec_hits = await timed("vector", self.vector.search(flt, query, cfg.candidate_k))

        if cfg.mode is Mode.KEYWORD:
            fused = kw_hits
        elif cfg.mode is Mode.VECTOR:
            fused = vec_hits
        elif cfg.fusion is Fusion.RRF:
            fused = rrf([kw_hits, vec_hits], k=cfg.rrf_k)
        else:
            fused = weighted(kw_hits, vec_hits, cfg.alpha)

        if cfg.mode is Mode.HYBRID_RERANK:
            final = await timed(
                "rerank", self.reranker.rerank(query, fused[: cfg.candidate_k], cfg.top_k)
            )
        else:
            final = fused[: cfg.top_k]

        kw_rank, vec_rank, fused_rank = _ranks(kw_hits), _ranks(vec_hits), _ranks(fused)
        trace = [
            {
                "chunk_id": str(h.chunk.id),
                "document_id": str(h.chunk.document_id),
                "keyword_rank": kw_rank.get(h.chunk.id),
                "vector_rank": vec_rank.get(h.chunk.id),
                "fused_rank": fused_rank.get(h.chunk.id),
                "final_rank": rank,
                "score": round(h.score, 6),
            }
            for rank, h in enumerate(final, start=1)
        ]
        return RetrievalResult(final, trace, timings)

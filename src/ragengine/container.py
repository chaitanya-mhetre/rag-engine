"""Composition root: builds every component from settings, in one place."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from ragengine.config import Settings
from ragengine.factory import build_embedder, build_llm, build_reranker
from ragengine.generation.context import ContextBuilder
from ragengine.generation.cost import PriceTable
from ragengine.indexing import IndexingService
from ragengine.ingestion.chunking import RecursiveChunker
from ragengine.ingestion.pipeline import Ingestor
from ragengine.jobs import ArqQueue, InlineQueue, JobQueue
from ragengine.observability import MetricsSink
from ragengine.pipeline import RAGPipeline
from ragengine.ratelimit import MemoryRateLimiter, RateLimiter, RedisRateLimiter
from ragengine.retrieval.retrievers import (
    BM25Retriever,
    HybridRetriever,
    KeywordRetriever,
    PgFtsRetriever,
    RetrievalConfig,
    VectorRetriever,
)
from ragengine.security import TokenService
from ragengine.storage import FileStorage, LocalFileStorage
from ragengine.store.access import AccessRepo, MemoryAccessRepo
from ragengine.store.base import Store
from ragengine.store.memory import MemoryStore


@dataclass(slots=True)
class Container:
    settings: Settings
    store: Store
    access: AccessRepo
    files: FileStorage
    indexer: IndexingService
    pipeline: RAGPipeline
    jobs: JobQueue
    limiter: RateLimiter
    tokens: TokenService
    engine: object | None = None  # AsyncEngine when using Postgres

    async def close(self) -> None:
        await self.jobs.close()
        if self.engine is not None:
            await self.engine.dispose()  # type: ignore[attr-defined]


def build_container(settings: Settings) -> Container:
    engine = None
    store: Store
    access: AccessRepo
    if settings.store == "postgres":
        from ragengine.store.access_pg import PgAccessRepo
        from ragengine.store.pg import PgStore, make_engine

        engine = make_engine(settings.database_url)
        store, access = PgStore(engine), PgAccessRepo(engine)
    else:
        store, access = MemoryStore(), MemoryAccessRepo()

    embedder = build_embedder(settings)
    files = LocalFileStorage(settings.upload_dir)
    indexer = IndexingService(
        store=store,
        embedder=embedder,
        files=files,
        ingestor=Ingestor(
            chunker=RecursiveChunker(settings.chunk_max_tokens, settings.chunk_overlap_tokens)
        ),
    )
    keyword: KeywordRetriever = (
        PgFtsRetriever(store) if settings.keyword_retriever == "pg_fts" else BM25Retriever(store)
    )
    retriever = HybridRetriever(
        keyword,
        VectorRetriever(store, embedder),
        build_reranker(settings),
        RetrievalConfig(
            candidate_k=settings.candidate_k, top_k=settings.top_k, rrf_k=settings.rrf_k
        ),
    )
    pipeline = RAGPipeline(
        retriever,
        build_llm(settings),
        context_builder=ContextBuilder(token_budget=settings.context_token_budget),
        prices=PriceTable.load(settings.models_config),
        sink=MetricsSink(access),
        refusal_threshold=settings.refusal_threshold,
    )
    jobs: JobQueue = ArqQueue(settings.redis_url) if settings.redis_url else InlineQueue(indexer)
    limiter: RateLimiter = (
        RedisRateLimiter(
            settings.redis_url, settings.rate_limit_capacity, settings.rate_limit_per_second
        )
        if settings.redis_url
        else MemoryRateLimiter(settings.rate_limit_capacity, settings.rate_limit_per_second)
    )
    tokens = TokenService(
        settings.jwt_secret.get_secret_value(),
        timedelta(minutes=settings.jwt_access_ttl_minutes),
        timedelta(days=settings.jwt_refresh_ttl_days),
    )
    return Container(
        settings, store, access, files, indexer, pipeline, jobs, limiter, tokens, engine
    )

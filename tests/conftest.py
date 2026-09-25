"""Shared fixtures. Store-level tests run against MemoryStore and, if reachable, PgStore."""

from __future__ import annotations

import os
import socket
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from urllib.parse import urlparse

import pytest

from ragengine.embeddings import HashingEmbedder
from ragengine.indexing import IndexingService
from ragengine.ingestion.chunking import RecursiveChunker
from ragengine.ingestion.pipeline import Ingestor
from ragengine.storage import MemoryFileStorage
from ragengine.store.base import Store
from ragengine.store.memory import MemoryStore

PG_URL = os.getenv("RAG_TEST_DATABASE_URL", "postgresql+asyncpg://rag:rag@localhost:55433/rag")
REDIS_URL = os.getenv("RAG_TEST_REDIS_URL", "redis://localhost:56380/15")


def _reachable(url: str) -> bool:
    parsed = urlparse(url.replace("+asyncpg", ""))
    try:
        with socket.create_connection((parsed.hostname or "localhost", parsed.port or 5432), 0.5):
            return True
    except OSError:
        return False


PG_AVAILABLE = _reachable(PG_URL)
REDIS_AVAILABLE = _reachable(REDIS_URL)
needs_pg = pytest.mark.skipif(
    not PG_AVAILABLE, reason="Postgres not reachable on :55433 — run `make up && make migrate`"
)
needs_redis = pytest.mark.skipif(not REDIS_AVAILABLE, reason="Redis not reachable on :56380")


@dataclass
class Scope:
    tenant_id: uuid.UUID
    collection_id: uuid.UUID


async def _pg_store() -> AsyncIterator[Store]:
    from ragengine.store.pg import PgStore, make_engine

    engine = make_engine(PG_URL)
    yield PgStore(engine)
    await engine.dispose()


@pytest.fixture(
    params=["memory", pytest.param("postgres", marks=[needs_pg, pytest.mark.integration])]
)
async def store(request: pytest.FixtureRequest) -> AsyncIterator[Store]:
    if request.param == "memory":
        yield MemoryStore()
        return
    async for s in _pg_store():
        yield s


async def make_scope(store: Store, name: str = "t") -> Scope:
    """Create a tenant + collection. PgStore needs real FK rows; MemoryStore needs nothing."""
    tenant_id = uuid.uuid4()
    from ragengine.store.pg import PgStore

    if isinstance(store, PgStore):
        from sqlalchemy import insert

        from ragengine.store.schema import tenants

        async with store.engine.begin() as conn:
            await conn.execute(insert(tenants).values(id=tenant_id, name=name))
    return Scope(tenant_id, await make_collection(store, tenant_id))


async def make_collection(store: Store, tenant_id: uuid.UUID) -> uuid.UUID:
    collection_id = uuid.uuid4()
    from ragengine.store.pg import PgStore

    if isinstance(store, PgStore):
        from sqlalchemy import insert

        from ragengine.store.schema import collections

        async with store.engine.begin() as conn:
            await conn.execute(
                insert(collections).values(
                    id=collection_id, tenant_id=tenant_id, name=f"c-{collection_id}"
                )
            )
    return collection_id


def make_indexer(store: Store, dim: int = 384) -> IndexingService:
    return IndexingService(
        store=store,
        embedder=HashingEmbedder(dim),
        files=MemoryFileStorage(),
        ingestor=Ingestor(chunker=RecursiveChunker(64, 8)),
    )

"""Ingestion job queues.

Design decision: the document *version* row is the job record. `job_id == version_id`, and the
job status is the version status (pending → processing → ready | failed). That removes a second
table that could drift out of sync. Enqueuing the same version twice is harmless: arq dedupes by
job id, and `process_version` is a no-op for READY versions.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Protocol

from ragengine.indexing import IndexingService

log = logging.getLogger(__name__)


class JobQueue(Protocol):
    async def enqueue(self, tenant_id: uuid.UUID, version_id: uuid.UUID) -> None: ...

    async def close(self) -> None: ...


class InlineQueue:
    """Runs ingestion as background asyncio tasks in the API process (dev/tests, no Redis)."""

    def __init__(self, indexer: IndexingService) -> None:
        self.indexer = indexer
        self.tasks: set[asyncio.Task[None]] = set()

    async def _run(self, tenant_id: uuid.UUID, version_id: uuid.UUID) -> None:
        try:
            await self.indexer.process_version(tenant_id, version_id)
        except Exception:  # status is already FAILED; don't crash the event loop
            log.exception("inline ingestion failed for %s", version_id)

    async def enqueue(self, tenant_id: uuid.UUID, version_id: uuid.UUID) -> None:
        task = asyncio.create_task(self._run(tenant_id, version_id))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def drain(self) -> None:
        while self.tasks:
            await asyncio.gather(*list(self.tasks))

    async def close(self) -> None:
        await self.drain()


class ArqQueue:  # exercised by the docker-compose stack and the worker integration test
    def __init__(self, redis_url: str) -> None:
        self.redis_url = redis_url
        self._pool: object | None = None

    async def _get_pool(self) -> object:
        if self._pool is None:
            from arq import create_pool
            from arq.connections import RedisSettings

            self._pool = await create_pool(RedisSettings.from_dsn(self.redis_url))
        return self._pool

    async def enqueue(self, tenant_id: uuid.UUID, version_id: uuid.UUID) -> None:
        pool = await self._get_pool()
        await pool.enqueue_job(  # type: ignore[attr-defined]
            "process_version", str(tenant_id), str(version_id), _job_id=f"ingest:{version_id}"
        )

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.aclose()  # type: ignore[attr-defined]

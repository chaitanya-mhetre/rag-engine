"""Full stack against real Postgres + Redis: API on PgStore, arq worker, Redis limiter/cache."""

import uuid
from pathlib import Path

import pytest

from ragengine.embeddings import CachedEmbedder, HashingEmbedder, RedisEmbeddingCache
from ragengine.jobs import ArqQueue
from ragengine.ratelimit import RedisRateLimiter
from tests.api_helpers import api, create_collection, signup, upload
from tests.conftest import PG_URL, REDIS_URL, needs_pg, needs_redis

pytestmark = [pytest.mark.integration]

HANDBOOK = b"# Handbook\n\n## Leave\n\nEmployees get 22 vacation days per year."


@needs_pg
async def test_api_on_postgres(tmp_path: Path) -> None:
    async with api(tmp_path, store="postgres", database_url=PG_URL) as (client, container):
        email = f"pg-{uuid.uuid4().hex[:8]}@acme.io"
        h = await signup(client, email)
        cid = await create_collection(client, h, "hr")
        up = await upload(client, container, h, cid, "handbook.md", HANDBOOK)
        assert (await client.get(f"/v1/jobs/{up['job_id']}", headers=h)).json()["status"] == "ready"
        r = await client.post(
            "/v1/query",
            json={"question": "How many vacation days?", "collection_ids": [cid]},
            headers=h,
        )
        assert "22" in r.json()["answer"]
        # the query log landed in Postgres
        from sqlalchemy import func, select

        from ragengine.store.schema import query_logs

        async with container.engine.connect() as conn:  # type: ignore[attr-defined]
            n = await conn.scalar(
                select(func.count())
                .select_from(query_logs)
                .where(query_logs.c.query == "How many vacation days?")
            )
        assert n and n >= 1


@needs_pg
@needs_redis
async def test_arq_worker_processes_upload(tmp_path: Path) -> None:
    from arq.connections import RedisSettings
    from arq.worker import Worker

    from ragengine import worker as worker_mod

    async with api(tmp_path, store="postgres", database_url=PG_URL) as (client, container):
        container.jobs = ArqQueue(REDIS_URL)
        h = await signup(client, f"arq-{uuid.uuid4().hex[:8]}@acme.io")
        cid = await create_collection(client, h, "hr")
        up = await upload(client, container, h, cid, "handbook.md", HANDBOOK)
        assert up["status"] == "pending"

        async def startup(ctx: dict[str, object]) -> None:
            ctx["container"] = container

        w = Worker(
            functions=[worker_mod.process_version],
            redis_settings=RedisSettings.from_dsn(REDIS_URL),
            on_startup=startup,
            burst=True,
            poll_delay=0.05,
        )
        await w.main()
        await w.close()
        assert (await client.get(f"/v1/jobs/{up['job_id']}", headers=h)).json()["status"] == "ready"


@needs_redis
async def test_redis_rate_limiter_and_cache(tmp_path: Path) -> None:
    limiter = RedisRateLimiter(REDIS_URL, capacity=2, rate=0.001)
    key = f"test:{uuid.uuid4().hex}"
    results = [(await limiter.hit(key)).allowed for _ in range(3)]
    assert results == [True, True, False]

    cached = CachedEmbedder(HashingEmbedder(dim=16), RedisEmbeddingCache(REDIS_URL), batch_size=8)
    text = f"hello {uuid.uuid4().hex}"
    first = await cached.embed([text])
    calls = cached.calls
    assert await cached.embed([text]) == first and cached.calls == calls

"""arq worker: runs ingestion jobs outside the API process.

    arq ragengine.worker.WorkerSettings

Retries: a failed job is retried with exponential backoff (5s, 10s, 20s, ...) up to
`max_tries`. Retrying is safe because `process_version` replaces the version's chunks in one
transaction and is a no-op for READY versions.
"""

from __future__ import annotations

import uuid
from typing import Any, ClassVar

from arq import Retry
from arq.connections import RedisSettings

from ragengine.config import get_settings
from ragengine.container import build_container
from ragengine.indexing import DocumentNotFoundError


async def startup(ctx: dict[str, Any]) -> None:
    ctx["container"] = build_container(get_settings())


async def shutdown(ctx: dict[str, Any]) -> None:
    await ctx["container"].close()


async def process_version(ctx: dict[str, Any], tenant_id: str, version_id: str) -> str:
    container = ctx["container"]
    try:
        version = await container.indexer.process_version(
            uuid.UUID(tenant_id), uuid.UUID(version_id)
        )
    except (DocumentNotFoundError, LookupError):
        return "gone"  # document deleted meanwhile: nothing to retry
    except Exception as exc:
        tries = int(ctx.get("job_try", 1))
        raise Retry(defer=5 * 2 ** (tries - 1)) from exc
    return str(version.status.value)


class WorkerSettings:
    functions: ClassVar[list[Any]] = [process_version]
    on_startup = startup
    on_shutdown = shutdown
    max_tries = 5
    job_timeout = 600
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url or "redis://localhost:6379/0")

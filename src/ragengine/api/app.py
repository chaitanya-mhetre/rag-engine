"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from ragengine import __version__
from ragengine.api.routes import router
from ragengine.config import Settings, get_settings
from ragengine.container import Container, build_container
from ragengine.observability import configure_logging, request_context


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging()
        app.state.container = container or build_container(settings)
        yield
        await app.state.container.close()

    app = FastAPI(
        title="RAG Engine",
        version=__version__,
        description="Multi-tenant document QA: hybrid retrieval, reranking, cited answers.",
        lifespan=lifespan,
    )
    app.include_router(router)
    app.middleware("http")(request_context)

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/health", tags=["ops"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/ready", tags=["ops"])
    async def ready() -> dict[str, str]:
        c: Container = app.state.container
        if c.engine is not None:
            from sqlalchemy import text

            async with c.engine.connect() as conn:  # type: ignore[attr-defined]
                await conn.execute(text("SELECT 1"))
        return {"status": "ready"}

    return app

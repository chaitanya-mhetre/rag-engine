"""Helpers for API tests: an in-process app with a memory (or Postgres) container."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx

from ragengine.api.app import create_app
from ragengine.config import Settings
from ragengine.container import Container, build_container
from ragengine.jobs import InlineQueue

PASSWORD = "correct-horse-battery"


@asynccontextmanager
async def api(
    tmp_path: Path, **overrides: Any
) -> AsyncIterator[tuple[httpx.AsyncClient, Container]]:
    settings = Settings(upload_dir=str(tmp_path / "uploads"), redis_url=None, **overrides)
    container = build_container(settings)
    app = create_app(settings, container)
    app.state.container = container  # ASGITransport does not run the lifespan
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, container
    await container.close()


async def signup(client: httpx.AsyncClient, email: str, tenant: str = "acme") -> dict[str, str]:
    r = await client.post(
        "/v1/auth/signup", json={"tenant_name": tenant, "email": email, "password": PASSWORD}
    )
    assert r.status_code == 201, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


async def login(client: httpx.AsyncClient, email: str) -> dict[str, str]:
    r = await client.post("/v1/auth/login", json={"email": email, "password": PASSWORD})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


async def create_collection(client: httpx.AsyncClient, headers: dict[str, str], name: str) -> str:
    r = await client.post("/v1/collections", json={"name": name}, headers=headers)
    assert r.status_code == 201, r.text
    return str(r.json()["id"])


async def upload(
    client: httpx.AsyncClient,
    container: Container,
    headers: dict[str, str],
    collection_id: str,
    filename: str,
    data: bytes,
    **form: str,
) -> dict[str, Any]:
    r = await client.post(
        f"/v1/collections/{collection_id}/documents",
        files={"file": (filename, data)},
        data=form,
        headers=headers,
    )
    assert r.status_code == 202, r.text
    if isinstance(container.jobs, InlineQueue):
        await container.jobs.drain()
    body: dict[str, Any] = r.json()
    return body


def parse_sse(text: str) -> list[tuple[str, dict[str, Any]]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events

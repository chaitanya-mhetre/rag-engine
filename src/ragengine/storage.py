"""Original-file storage. Files are kept so documents can be re-parsed when the pipeline changes."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Protocol


class FileStorage(Protocol):
    async def put(self, key: str, data: bytes) -> None: ...

    async def get(self, key: str) -> bytes: ...


class LocalFileStorage:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError("storage key escapes the storage root")
        return path

    async def put(self, key: str, data: bytes) -> None:
        path = self._path(key)
        await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(path.write_bytes, data)

    async def get(self, key: str) -> bytes:
        return await asyncio.to_thread(self._path(key).read_bytes)


class MemoryFileStorage:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}

    async def put(self, key: str, data: bytes) -> None:
        self.files[key] = data

    async def get(self, key: str) -> bytes:
        return self.files[key]

"""In-memory `Store`: same semantics as Postgres, used by unit tests, the CLI and evaluation."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from uuid import UUID

import numpy as np

from ragengine.models import Chunk, ScoredChunk
from ragengine.store.base import (
    DocumentRecord,
    SearchFilter,
    VersionRecord,
    VersionStatus,
    utcnow,
)


class MemoryStore:
    def __init__(self) -> None:
        self.documents: dict[UUID, DocumentRecord] = {}
        self.versions: dict[UUID, VersionRecord] = {}
        self.chunks: dict[UUID, list[Chunk]] = {}  # version_id -> chunks
        self._lock = asyncio.Lock()
        self.generation = 0  # bumps on any change; lets BM25 indexes know when to rebuild

    def _touch(self) -> None:
        self.generation += 1

    async def create_document(self, doc: DocumentRecord) -> None:
        self.documents[doc.id] = doc
        self._touch()

    async def get_document(self, tenant_id: UUID, document_id: UUID) -> DocumentRecord | None:
        doc = self.documents.get(document_id)
        return doc if doc and doc.tenant_id == tenant_id else None

    async def list_documents(self, tenant_id: UUID, collection_id: UUID) -> list[DocumentRecord]:
        return [
            d
            for d in self.documents.values()
            if d.tenant_id == tenant_id and d.collection_id == collection_id and not d.deleted_at
        ]

    async def soft_delete_document(self, tenant_id: UUID, document_id: UUID) -> None:
        doc = await self.get_document(tenant_id, document_id)
        if doc:
            doc.deleted_at = utcnow()
            self._touch()

    async def update_document_title(self, document_id: UUID, title: str) -> None:
        self.documents[document_id].title = title

    async def create_version(self, version: VersionRecord) -> None:
        if await self.find_version_by_sha(version.document_id, version.content_sha256):
            raise ValueError("duplicate content for this document")
        self.versions[version.id] = version

    async def get_version(self, version_id: UUID) -> VersionRecord | None:
        return self.versions.get(version_id)

    async def list_versions(self, document_id: UUID) -> list[VersionRecord]:
        return sorted(
            (v for v in self.versions.values() if v.document_id == document_id),
            key=lambda v: v.version,
        )

    async def find_version_by_sha(self, document_id: UUID, sha: str) -> VersionRecord | None:
        return next(
            (
                v
                for v in self.versions.values()
                if v.document_id == document_id and v.content_sha256 == sha
            ),
            None,
        )

    async def next_version_number(self, document_id: UUID) -> int:
        return max((v.version for v in await self.list_versions(document_id)), default=0) + 1

    async def replace_version_chunks(self, version_id: UUID, chunks: list[Chunk]) -> None:
        async with self._lock:
            self.chunks[version_id] = [replace(c, version_id=version_id) for c in chunks]
            self._touch()

    async def set_version_status(
        self,
        version_id: UUID,
        status: VersionStatus,
        *,
        error: str | None = None,
        chunk_count: int | None = None,
        embedding_model: str | None = None,
    ) -> None:
        v = self.versions[version_id]
        v.status, v.error = status, error
        if chunk_count is not None:
            v.chunk_count = chunk_count
        if embedding_model is not None:
            v.embedding_model = embedding_model

    async def activate_version(self, document_id: UUID, version_id: UUID) -> None:
        v = self.versions[version_id]
        if v.document_id != document_id or v.status is not VersionStatus.READY:
            raise ValueError("only a READY version of this document can be activated")
        self.documents[document_id].active_version_id = version_id
        self._touch()

    async def get_version_chunks(self, version_id: UUID) -> list[Chunk]:
        return list(self.chunks.get(version_id, []))

    def _visible(self, doc: DocumentRecord, flt: SearchFilter) -> bool:
        return (
            doc.tenant_id == flt.tenant_id
            and doc.collection_id in flt.collection_ids
            and doc.deleted_at is None
            and doc.active_version_id is not None
            and (not flt.tags or bool(set(flt.tags) & set(doc.tags)))
            and (not flt.mime_types or doc.mime_type in flt.mime_types)
            and (flt.created_after is None or doc.created_at >= flt.created_after)
        )

    async def active_chunks(self, flt: SearchFilter) -> list[Chunk]:
        out: list[Chunk] = []
        for doc in self.documents.values():
            if self._visible(doc, flt) and doc.active_version_id is not None:
                out.extend(self.chunks.get(doc.active_version_id, []))
        return out

    async def get_chunks(self, tenant_id: UUID, chunk_ids: list[UUID]) -> list[Chunk]:
        wanted = set(chunk_ids)
        out = []
        for chunks in self.chunks.values():
            for c in chunks:
                doc = self.documents.get(c.document_id)
                if c.id in wanted and doc is not None and doc.tenant_id == tenant_id:
                    out.append(c)
        return out

    async def vector_search(
        self, flt: SearchFilter, embedding: list[float], k: int
    ) -> list[ScoredChunk]:
        candidates = [c for c in await self.active_chunks(flt) if c.embedding is not None]
        if not candidates:
            return []
        matrix = np.array([c.embedding for c in candidates], dtype=np.float32)
        q = np.array(embedding, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1) * (np.linalg.norm(q) or 1.0)
        scores = (matrix @ q) / np.where(norms == 0, 1.0, norms)
        order = np.argsort(-scores)[:k]
        return [ScoredChunk(candidates[i], float(scores[i]), "vector") for i in order]

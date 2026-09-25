"""Indexing service: versioned, idempotent ingestion into a `Store`.

Flow for one upload:
1. `create_version`: store the original bytes, create (or reuse) the document, create a
   PENDING version. Identical bytes for the same document return the existing version.
2. `process_version` (inline or in the worker): parse → chunk → embed → replace chunks →
   READY → atomically activate. On failure the version is FAILED and the previous version
   stays active, so readers never see a half-ingested document.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from uuid import UUID

from ragengine.embeddings import Embedder
from ragengine.ingestion.parsers import detect_mime
from ragengine.ingestion.pipeline import Ingestor, sha256
from ragengine.storage import FileStorage
from ragengine.store.base import DocumentRecord, Store, VersionRecord, VersionStatus

log = logging.getLogger(__name__)


class DocumentNotFoundError(LookupError):
    pass


@dataclass(slots=True)
class VersionHandle:
    document: DocumentRecord
    version: VersionRecord
    created: bool  # False when the upload was an identical re-upload (no-op)


@dataclass(slots=True)
class IndexingService:
    store: Store
    embedder: Embedder
    files: FileStorage
    ingestor: Ingestor

    async def create_version(
        self,
        *,
        tenant_id: UUID,
        collection_id: UUID,
        data: bytes,
        filename: str,
        tags: list[str] | None = None,
        document_id: UUID | None = None,
        created_by: UUID | None = None,
    ) -> VersionHandle:
        mime = detect_mime(data, filename)  # reject bad files before storing anything
        digest = sha256(data)
        if document_id is not None:
            doc = await self.store.get_document(tenant_id, document_id)
            if doc is None or doc.collection_id != collection_id:
                raise DocumentNotFoundError(str(document_id))
            existing = await self.store.find_version_by_sha(doc.id, digest)
            if existing is not None:
                return VersionHandle(doc, existing, created=False)
        else:
            doc = DocumentRecord(
                tenant_id=tenant_id,
                collection_id=collection_id,
                title=filename,
                source=filename,
                mime_type=mime,
                tags=list(tags or []),
                created_by=created_by,
            )
            await self.store.create_document(doc)

        number = await self.store.next_version_number(doc.id)
        key = f"{tenant_id}/{doc.id}/v{number}/{digest[:16]}"
        await self.files.put(key, data)
        version = VersionRecord(
            document_id=doc.id, version=number, content_sha256=digest, storage_key=key
        )
        await self.store.create_version(version)
        return VersionHandle(doc, version, created=True)

    async def process_version(self, tenant_id: UUID, version_id: UUID) -> VersionRecord:
        version = await self.store.get_version(version_id)
        if version is None:
            raise LookupError(f"version {version_id} not found")
        doc = await self.store.get_document(tenant_id, version.document_id)
        if doc is None:
            raise DocumentNotFoundError(str(version.document_id))
        if version.status is VersionStatus.READY:
            return version  # already done: retries are no-ops
        await self.store.set_version_status(version_id, VersionStatus.PROCESSING)
        try:
            data = await self.files.get(version.storage_key)
            ingested = self.ingestor.ingest(
                data, doc.source, document_id=doc.id, tags=list(doc.tags)
            )
            for c in ingested.chunks:
                c.metadata["title"] = ingested.parsed.title
                c.metadata["version"] = version.version
            vectors = await self.embedder.embed([c.text for c in ingested.chunks])
            for chunk, vec in zip(ingested.chunks, vectors, strict=True):
                chunk.embedding = vec
            await self.store.replace_version_chunks(version_id, ingested.chunks)
            await self.store.update_document_title(doc.id, ingested.parsed.title)
            await self.store.set_version_status(
                version_id,
                VersionStatus.READY,
                chunk_count=len(ingested.chunks),
                embedding_model=self.embedder.model,
            )
            await self.store.activate_version(doc.id, version_id)
        except Exception as exc:
            log.exception("ingestion failed for version %s", version_id)
            await self.store.set_version_status(version_id, VersionStatus.FAILED, error=str(exc))
            raise
        refreshed = await self.store.get_version(version_id)
        assert refreshed is not None
        return refreshed

    async def index_bytes(
        self,
        *,
        tenant_id: UUID,
        collection_id: UUID,
        data: bytes,
        filename: str,
        tags: list[str] | None = None,
        document_id: UUID | None = None,
    ) -> VersionHandle:
        """Create a version and process it inline (CLI, tests, evaluation)."""
        handle = await self.create_version(
            tenant_id=tenant_id,
            collection_id=collection_id,
            data=data,
            filename=filename,
            tags=tags,
            document_id=document_id,
        )
        if handle.created:
            handle.version = await self.process_version(tenant_id, handle.version.id)
        return handle

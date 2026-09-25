"""Storage interface. Two implementations: `MemoryStore` (tests, CLI) and `PgStore` (Postgres).

Tenant isolation rule: every read takes a `SearchFilter` or an explicit `tenant_id`.
There is deliberately no method that searches across tenants.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID, uuid4

from ragengine.models import Chunk, ScoredChunk


def utcnow() -> datetime:
    return datetime.now(UTC)


class VersionStatus(enum.StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


@dataclass(slots=True)
class DocumentRecord:
    tenant_id: UUID
    collection_id: UUID
    title: str
    source: str
    mime_type: str
    tags: list[str] = field(default_factory=list)
    id: UUID = field(default_factory=uuid4)
    active_version_id: UUID | None = None
    created_by: UUID | None = None
    created_at: datetime = field(default_factory=utcnow)
    deleted_at: datetime | None = None


@dataclass(slots=True)
class VersionRecord:
    document_id: UUID
    version: int
    content_sha256: str
    storage_key: str = ""
    status: VersionStatus = VersionStatus.PENDING
    error: str | None = None
    chunk_count: int = 0
    embedding_model: str = ""
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=utcnow)


@dataclass(frozen=True, slots=True)
class SearchFilter:
    """What a query may see. `collection_ids` must already be authorised by the caller."""

    tenant_id: UUID
    collection_ids: tuple[UUID, ...]
    tags: tuple[str, ...] = ()
    mime_types: tuple[str, ...] = ()
    created_after: datetime | None = None

    def __post_init__(self) -> None:
        if not self.collection_ids:
            raise ValueError("a search must be scoped to at least one collection")


class Store(Protocol):
    async def create_document(self, doc: DocumentRecord) -> None: ...

    async def get_document(self, tenant_id: UUID, document_id: UUID) -> DocumentRecord | None: ...

    async def list_documents(
        self, tenant_id: UUID, collection_id: UUID
    ) -> list[DocumentRecord]: ...

    async def soft_delete_document(self, tenant_id: UUID, document_id: UUID) -> None: ...

    async def update_document_title(self, document_id: UUID, title: str) -> None: ...

    async def create_version(self, version: VersionRecord) -> None: ...

    async def get_version(self, version_id: UUID) -> VersionRecord | None: ...

    async def list_versions(self, document_id: UUID) -> list[VersionRecord]: ...

    async def find_version_by_sha(self, document_id: UUID, sha: str) -> VersionRecord | None: ...

    async def next_version_number(self, document_id: UUID) -> int: ...

    async def replace_version_chunks(self, version_id: UUID, chunks: list[Chunk]) -> None:
        """Delete then insert, in one transaction, so a retried ingestion job is idempotent."""
        ...

    async def set_version_status(
        self,
        version_id: UUID,
        status: VersionStatus,
        *,
        error: str | None = None,
        chunk_count: int | None = None,
        embedding_model: str | None = None,
    ) -> None: ...

    async def activate_version(self, document_id: UUID, version_id: UUID) -> None:
        """Atomically point the document at a READY version."""
        ...

    async def get_version_chunks(self, version_id: UUID) -> list[Chunk]: ...

    async def active_chunks(self, flt: SearchFilter) -> list[Chunk]: ...

    async def get_chunks(self, tenant_id: UUID, chunk_ids: list[UUID]) -> list[Chunk]: ...

    async def vector_search(
        self, flt: SearchFilter, embedding: list[float], k: int
    ) -> list[ScoredChunk]: ...

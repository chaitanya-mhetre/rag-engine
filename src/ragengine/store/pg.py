"""PostgreSQL + pgvector implementation of `Store` (SQLAlchemy Core, asyncpg driver)."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import and_, delete, func, insert, select, text, update
from sqlalchemy.engine import RowMapping
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine
from sqlalchemy.sql import Select

from ragengine.models import Chunk, ScoredChunk
from ragengine.store.base import DocumentRecord, SearchFilter, VersionRecord, VersionStatus, utcnow
from ragengine.store.schema import chunks, document_versions, documents


def make_engine(url: str) -> AsyncEngine:
    return create_async_engine(url, pool_size=10, max_overflow=5, pool_pre_ping=True)


def _doc(row: RowMapping) -> DocumentRecord:
    return DocumentRecord(
        id=row["id"],
        tenant_id=row["tenant_id"],
        collection_id=row["collection_id"],
        title=row["title"],
        source=row["source_uri"],
        mime_type=row["mime_type"],
        tags=list(row["tags"]),
        active_version_id=row["active_version_id"],
        created_by=row["created_by"],
        created_at=row["created_at"],
        deleted_at=row["deleted_at"],
    )


def _version(row: RowMapping) -> VersionRecord:
    return VersionRecord(
        id=row["id"],
        document_id=row["document_id"],
        version=row["version"],
        content_sha256=row["content_sha256"],
        storage_key=row["storage_key"],
        status=VersionStatus(row["status"]),
        error=row["error"],
        chunk_count=row["chunk_count"],
        embedding_model=row["embedding_model"],
        created_at=row["created_at"],
    )


def _chunk(row: RowMapping) -> Chunk:
    emb = row.get("embedding")
    return Chunk(
        id=row["id"],
        text=row["text"],
        ordinal=row["ordinal"],
        token_count=row["token_count"],
        document_id=row["document_id"],
        version_id=row["version_id"],
        heading_path=tuple(row["heading_path"]),
        page=row["page"],
        metadata=dict(row["metadata"]),
        embedding=[float(x) for x in emb] if emb is not None else None,
    )


_CHUNK_COLS = [
    chunks.c.id,
    chunks.c.text,
    chunks.c.ordinal,
    chunks.c.token_count,
    chunks.c.document_id,
    chunks.c.version_id,
    chunks.c.heading_path,
    chunks.c.page,
    chunks.c.metadata,
]


class PgStore:
    def __init__(self, engine: AsyncEngine, ef_search: int = 100) -> None:
        self.engine = engine
        self.ef_search = ef_search

    # --- documents --------------------------------------------------------

    async def create_document(self, doc: DocumentRecord) -> None:
        async with self.engine.begin() as conn:
            await conn.execute(
                insert(documents).values(
                    id=doc.id,
                    tenant_id=doc.tenant_id,
                    collection_id=doc.collection_id,
                    title=doc.title,
                    source_uri=doc.source,
                    mime_type=doc.mime_type,
                    tags=doc.tags,
                    created_by=doc.created_by,
                    created_at=doc.created_at,
                )
            )

    async def get_document(self, tenant_id: UUID, document_id: UUID) -> DocumentRecord | None:
        async with self.engine.connect() as conn:
            row = (
                (
                    await conn.execute(
                        select(documents).where(
                            documents.c.id == document_id, documents.c.tenant_id == tenant_id
                        )
                    )
                )
                .mappings()
                .first()
            )
        return _doc(row) if row else None

    async def list_documents(self, tenant_id: UUID, collection_id: UUID) -> list[DocumentRecord]:
        async with self.engine.connect() as conn:
            rows = (
                await conn.execute(
                    select(documents)
                    .where(
                        documents.c.tenant_id == tenant_id,
                        documents.c.collection_id == collection_id,
                        documents.c.deleted_at.is_(None),
                    )
                    .order_by(documents.c.created_at)
                )
            ).mappings()
            return [_doc(r) for r in rows]

    async def soft_delete_document(self, tenant_id: UUID, document_id: UUID) -> None:
        async with self.engine.begin() as conn:
            await conn.execute(
                update(documents)
                .where(documents.c.id == document_id, documents.c.tenant_id == tenant_id)
                .values(deleted_at=utcnow())
            )

    async def update_document_title(self, document_id: UUID, title: str) -> None:
        async with self.engine.begin() as conn:
            await conn.execute(
                update(documents).where(documents.c.id == document_id).values(title=title)
            )

    # --- versions ---------------------------------------------------------

    async def create_version(self, version: VersionRecord) -> None:
        async with self.engine.begin() as conn:
            await conn.execute(
                insert(document_versions).values(
                    id=version.id,
                    document_id=version.document_id,
                    version=version.version,
                    content_sha256=version.content_sha256,
                    storage_key=version.storage_key,
                    status=version.status.value,
                    error=version.error,
                    chunk_count=version.chunk_count,
                    embedding_model=version.embedding_model,
                    created_at=version.created_at,
                )
            )

    async def get_version(self, version_id: UUID) -> VersionRecord | None:
        async with self.engine.connect() as conn:
            row = (
                (
                    await conn.execute(
                        select(document_versions).where(document_versions.c.id == version_id)
                    )
                )
                .mappings()
                .first()
            )
        return _version(row) if row else None

    async def list_versions(self, document_id: UUID) -> list[VersionRecord]:
        async with self.engine.connect() as conn:
            rows = (
                await conn.execute(
                    select(document_versions)
                    .where(document_versions.c.document_id == document_id)
                    .order_by(document_versions.c.version)
                )
            ).mappings()
            return [_version(r) for r in rows]

    async def find_version_by_sha(self, document_id: UUID, sha: str) -> VersionRecord | None:
        async with self.engine.connect() as conn:
            row = (
                (
                    await conn.execute(
                        select(document_versions).where(
                            document_versions.c.document_id == document_id,
                            document_versions.c.content_sha256 == sha,
                        )
                    )
                )
                .mappings()
                .first()
            )
        return _version(row) if row else None

    async def next_version_number(self, document_id: UUID) -> int:
        async with self.engine.connect() as conn:
            current = await conn.scalar(
                select(func.coalesce(func.max(document_versions.c.version), 0)).where(
                    document_versions.c.document_id == document_id
                )
            )
        return int(current or 0) + 1

    async def replace_version_chunks(self, version_id: UUID, new_chunks: list[Chunk]) -> None:
        async with self.engine.begin() as conn:
            scope = (
                await conn.execute(
                    select(documents.c.tenant_id, documents.c.collection_id)
                    .select_from(
                        documents.join(
                            document_versions, document_versions.c.document_id == documents.c.id
                        )
                    )
                    .where(document_versions.c.id == version_id)
                )
            ).one()
            await conn.execute(delete(chunks).where(chunks.c.version_id == version_id))
            if new_chunks:
                await conn.execute(
                    insert(chunks),
                    [
                        {
                            "id": c.id,
                            "tenant_id": scope.tenant_id,
                            "collection_id": scope.collection_id,
                            "document_id": c.document_id,
                            "version_id": version_id,
                            "ordinal": c.ordinal,
                            "text": c.text,
                            "token_count": c.token_count,
                            "heading_path": list(c.heading_path),
                            "page": c.page,
                            "metadata": c.metadata,
                            "embedding": c.embedding,
                        }
                        for c in new_chunks
                    ],
                )

    async def set_version_status(
        self,
        version_id: UUID,
        status: VersionStatus,
        *,
        error: str | None = None,
        chunk_count: int | None = None,
        embedding_model: str | None = None,
    ) -> None:
        values: dict[str, Any] = {"status": status.value, "error": error}
        if chunk_count is not None:
            values["chunk_count"] = chunk_count
        if embedding_model is not None:
            values["embedding_model"] = embedding_model
        async with self.engine.begin() as conn:
            await conn.execute(
                update(document_versions)
                .where(document_versions.c.id == version_id)
                .values(**values)
            )

    async def activate_version(self, document_id: UUID, version_id: UUID) -> None:
        # Single UPDATE guarded by a subquery: the flip only happens if the version is READY
        # and belongs to this document. Readers see either the old or the new version.
        async with self.engine.begin() as conn:
            ready = (
                select(document_versions.c.id)
                .where(
                    document_versions.c.id == version_id,
                    document_versions.c.document_id == document_id,
                    document_versions.c.status == VersionStatus.READY.value,
                )
                .scalar_subquery()
            )
            result = await conn.execute(
                update(documents)
                .where(documents.c.id == document_id, ready.is_not(None))
                .values(active_version_id=version_id)
            )
            if result.rowcount != 1:
                raise ValueError("only a READY version of this document can be activated")

    # --- chunks and retrieval ---------------------------------------------

    async def get_version_chunks(self, version_id: UUID) -> list[Chunk]:
        async with self.engine.connect() as conn:
            rows = (
                await conn.execute(
                    select(*_CHUNK_COLS, chunks.c.embedding)
                    .where(chunks.c.version_id == version_id)
                    .order_by(chunks.c.ordinal)
                )
            ).mappings()
            return [_chunk(r) for r in rows]

    def _scoped(self, stmt: Select[Any], flt: SearchFilter) -> Select[Any]:
        conds = [
            chunks.c.tenant_id == flt.tenant_id,
            chunks.c.collection_id.in_(flt.collection_ids),
            documents.c.deleted_at.is_(None),
        ]
        if flt.tags:
            conds.append(documents.c.tags.overlap(list(flt.tags)))
        if flt.mime_types:
            conds.append(documents.c.mime_type.in_(flt.mime_types))
        if flt.created_after:
            conds.append(documents.c.created_at >= flt.created_after)
        joined = chunks.join(
            documents,
            and_(
                documents.c.id == chunks.c.document_id,
                documents.c.active_version_id == chunks.c.version_id,
            ),
        )
        return stmt.select_from(joined).where(*conds)

    async def active_chunks(self, flt: SearchFilter) -> list[Chunk]:
        async with self.engine.connect() as conn:
            rows = (
                await conn.execute(
                    self._scoped(select(*_CHUNK_COLS), flt).order_by(
                        chunks.c.document_id, chunks.c.ordinal
                    )
                )
            ).mappings()
            return [_chunk(r) for r in rows]

    async def get_chunks(self, tenant_id: UUID, chunk_ids: list[UUID]) -> list[Chunk]:
        if not chunk_ids:
            return []
        async with self.engine.connect() as conn:
            rows = (
                await conn.execute(
                    select(*_CHUNK_COLS).where(
                        chunks.c.tenant_id == tenant_id, chunks.c.id.in_(chunk_ids)
                    )
                )
            ).mappings()
            return [_chunk(r) for r in rows]

    async def _tune(self, conn: AsyncConnection) -> None:
        # Filtered HNSW can return < k rows (post-filter recall loss). A larger ef_search and
        # pgvector >= 0.8 iterative scans keep walking the graph until enough rows match.
        await conn.execute(text(f"SET LOCAL hnsw.ef_search = {int(self.ef_search)}"))
        await conn.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))

    async def vector_search(
        self, flt: SearchFilter, embedding: list[float], k: int
    ) -> list[ScoredChunk]:
        distance = chunks.c.embedding.cosine_distance(embedding)
        stmt = (
            self._scoped(select(*_CHUNK_COLS, distance.label("distance")), flt)
            .order_by(distance)
            .limit(k)
        )
        async with self.engine.begin() as conn:
            await self._tune(conn)
            rows = (await conn.execute(stmt)).mappings().all()
        return [ScoredChunk(_chunk(r), 1.0 - float(r["distance"]), "vector") for r in rows]

    async def keyword_search(self, flt: SearchFilter, query: str, k: int) -> list[ScoredChunk]:
        """Postgres full-text search (ts_rank_cd). Compared against the hand-written BM25."""
        tsq = func.websearch_to_tsquery("english", query)
        rank = func.ts_rank_cd(chunks.c.tsv, tsq)
        stmt = (
            self._scoped(select(*_CHUNK_COLS, rank.label("rank")), flt)
            .where(chunks.c.tsv.op("@@")(tsq))
            .order_by(rank.desc())
            .limit(k)
        )
        async with self.engine.connect() as conn:
            rows = (await conn.execute(stmt)).mappings().all()
        return [ScoredChunk(_chunk(r), float(r["rank"]), "pg_fts") for r in rows]

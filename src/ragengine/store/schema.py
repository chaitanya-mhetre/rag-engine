"""SQLAlchemy Core table definitions. Alembic migrations in `migrations/` create these tables."""

from __future__ import annotations

import os

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Column,
    Computed,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR, UUID

# The vector column needs a fixed dimension for the HNSW index. Changing embedding models with
# a different dimension means a new migration and a re-embed of every chunk.
EMBEDDING_DIM = int(os.getenv("RAG_EMBEDDING_DIM", "384"))

metadata = MetaData()

tenants = Table(
    "tenants",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column("name", String(200), nullable=False),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
)

collections = Table(
    "collections",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column("tenant_id", UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False),
    Column("name", String(200), nullable=False),
    Column("description", Text, nullable=False, server_default=""),
    Column("created_by", UUID(as_uuid=True), nullable=True),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
    UniqueConstraint("tenant_id", "name"),
)

documents = Table(
    "documents",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column("tenant_id", UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False),
    Column("collection_id", UUID(as_uuid=True), ForeignKey("collections.id"), nullable=False),
    Column("title", Text, nullable=False),
    Column("source_uri", Text, nullable=False),
    Column("mime_type", String(200), nullable=False),
    Column("tags", ARRAY(Text), nullable=False, server_default="{}"),
    Column("active_version_id", UUID(as_uuid=True), nullable=True),
    Column("created_by", UUID(as_uuid=True), nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("deleted_at", DateTime(timezone=True), nullable=True),
    Index("ix_documents_tenant_collection", "tenant_id", "collection_id"),
)

document_versions = Table(
    "document_versions",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column(
        "document_id",
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("version", Integer, nullable=False),
    Column("content_sha256", String(64), nullable=False),
    Column("storage_key", Text, nullable=False, server_default=""),
    Column("status", String(20), nullable=False),
    Column("error", Text, nullable=True),
    Column("chunk_count", Integer, nullable=False, server_default="0"),
    Column("embedding_model", String(200), nullable=False, server_default=""),
    Column("created_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("document_id", "version"),
    UniqueConstraint("document_id", "content_sha256"),  # identical re-upload is a no-op
)

chunks = Table(
    "chunks",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column("tenant_id", UUID(as_uuid=True), nullable=False),
    Column("collection_id", UUID(as_uuid=True), nullable=False),
    Column("document_id", UUID(as_uuid=True), nullable=False),
    Column(
        "version_id",
        UUID(as_uuid=True),
        ForeignKey("document_versions.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("ordinal", Integer, nullable=False),
    Column("text", Text, nullable=False),
    Column("token_count", Integer, nullable=False),
    Column("heading_path", ARRAY(Text), nullable=False, server_default="{}"),
    Column("page", Integer, nullable=True),
    Column("metadata", JSONB, nullable=False, server_default="{}"),
    Column("embedding", Vector(EMBEDDING_DIM), nullable=True),
    Column("tsv", TSVECTOR, Computed("to_tsvector('english', text)", persisted=True)),
    Index("ix_chunks_scope", "tenant_id", "collection_id", "version_id"),
    Index("ix_chunks_tsv", "tsv", postgresql_using="gin"),
    Index(
        "ix_chunks_embedding_hnsw",
        "embedding",
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    ),
)

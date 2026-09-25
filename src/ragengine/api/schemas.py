"""Request/response models: the public API contract (also rendered as OpenAPI)."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, EmailStr, Field


class SignupRequest(BaseModel):
    tenant_name: str = Field(min_length=1, max_length=200)
    email: EmailStr
    password: str = Field(min_length=12, max_length=256)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class CreateUserRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=12, max_length=256)
    role: Literal["admin", "member", "viewer"] = "member"


class UserOut(BaseModel):
    id: uuid.UUID
    tenant_id: uuid.UUID
    email: str
    role: str


class CreateCollectionRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = ""


class CollectionOut(BaseModel):
    id: uuid.UUID
    name: str
    description: str
    permission: str
    created_at: datetime


class MemberRequest(BaseModel):
    user_id: uuid.UUID
    permission: Literal["read", "write", "admin"]


class UploadResponse(BaseModel):
    document_id: uuid.UUID
    version_id: uuid.UUID
    job_id: uuid.UUID
    version: int
    created: bool
    status: str


class VersionOut(BaseModel):
    id: uuid.UUID
    version: int
    status: str
    error: str | None
    chunk_count: int
    embedding_model: str
    content_sha256: str
    created_at: datetime


class DocumentOut(BaseModel):
    id: uuid.UUID
    collection_id: uuid.UUID
    title: str
    source: str
    mime_type: str
    tags: list[str]
    active_version_id: uuid.UUID | None
    created_at: datetime
    versions: list[VersionOut] = []


class ChunkOut(BaseModel):
    id: uuid.UUID
    ordinal: int
    text: str
    token_count: int
    heading_path: list[str]
    page: int | None


class JobOut(BaseModel):
    job_id: uuid.UUID
    status: str
    error: str | None


class QueryFilters(BaseModel):
    tags: list[str] = []
    source_types: list[str] = []  # mime types
    date_from: datetime | None = None


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    collection_ids: list[uuid.UUID] = Field(min_length=1)
    filters: QueryFilters = QueryFilters()
    conversation_id: uuid.UUID | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)
    rerank: bool = True
    debug: bool = False


class QueryResponse(BaseModel):
    answer: str
    citations: list[dict[str, Any]]
    insufficient_context: bool
    confidence: float
    usage: dict[str, Any]
    latency_ms: dict[str, float]
    query_log_id: str
    retrieval: list[dict[str, Any]] | None = None


class CreateConversationRequest(BaseModel):
    collection_ids: list[uuid.UUID] = Field(min_length=1)


class ConversationOut(BaseModel):
    id: uuid.UUID
    collection_ids: list[uuid.UUID]
    created_at: datetime


class MessageOut(BaseModel):
    id: uuid.UUID
    role: str
    content: str
    citations: list[dict[str, Any]]
    created_at: datetime

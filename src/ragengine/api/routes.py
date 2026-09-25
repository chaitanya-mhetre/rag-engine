"""HTTP routes. Routers stay thin: validate, authorise, call a service, shape the response."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import StreamingResponse

from ragengine.api.deps import (
    ContainerDep,
    RateLimitedUser,
    UserDep,
    permission_for,
    require_admin,
    require_permission,
)
from ragengine.api.schemas import (
    ChunkOut,
    CollectionOut,
    ConversationOut,
    CreateCollectionRequest,
    CreateConversationRequest,
    CreateUserRequest,
    DocumentOut,
    JobOut,
    LoginRequest,
    MemberRequest,
    MessageOut,
    QueryRequest,
    QueryResponse,
    RefreshRequest,
    SignupRequest,
    TokenResponse,
    UploadResponse,
    UserOut,
    VersionOut,
)
from ragengine.container import Container
from ragengine.generation.llm import Message
from ragengine.indexing import DocumentNotFoundError
from ragengine.ingestion.parsers import UnsupportedFormatError
from ragengine.retrieval.retrievers import Mode
from ragengine.security import TokenError, hash_password, verify_password
from ragengine.store.access import (
    ChatMessage,
    Collection,
    Conversation,
    Permission,
    Role,
    User,
)
from ragengine.store.base import DocumentRecord, SearchFilter, VersionRecord

router = APIRouter(prefix="/v1")


# --- auth -----------------------------------------------------------------------


@router.post("/auth/signup", response_model=TokenResponse, status_code=201, tags=["auth"])
async def signup(body: SignupRequest, container: ContainerDep) -> dict[str, str]:
    """Create a tenant and its first admin user."""
    if await container.access.get_user_by_email(body.email):
        raise HTTPException(status.HTTP_409_CONFLICT, "email already registered")
    tenant_id = await container.access.create_tenant(body.tenant_name)
    user = User(tenant_id, body.email.lower(), hash_password(body.password), Role.ADMIN)
    await container.access.create_user(user)
    return container.tokens.issue(user.id, tenant_id, user.role.value)


@router.post("/auth/login", response_model=TokenResponse, tags=["auth"])
async def login(body: LoginRequest, container: ContainerDep) -> dict[str, str]:
    user = await container.access.get_user_by_email(body.email)
    # same error for unknown email and wrong password: don't reveal which emails exist
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid email or password")
    return container.tokens.issue(user.id, user.tenant_id, user.role.value)


@router.post("/auth/refresh", response_model=TokenResponse, tags=["auth"])
async def refresh(body: RefreshRequest, container: ContainerDep) -> dict[str, str]:
    try:
        claims = container.tokens.decode(body.refresh_token, "refresh")
    except TokenError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid refresh token") from exc
    user = await container.access.get_user(claims.user_id)
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid refresh token")
    return container.tokens.issue(user.id, user.tenant_id, user.role.value)


@router.post("/users", response_model=UserOut, status_code=201, tags=["auth"])
async def create_user(
    body: CreateUserRequest,
    container: ContainerDep,
    admin: Annotated[User, Depends(require_admin)],
) -> UserOut:
    user = User(admin.tenant_id, body.email.lower(), hash_password(body.password), Role(body.role))
    try:
        await container.access.create_user(user)
    except ValueError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    return UserOut(id=user.id, tenant_id=user.tenant_id, email=user.email, role=user.role.value)


@router.get("/me", response_model=UserOut, tags=["auth"])
async def me(user: UserDep) -> UserOut:
    return UserOut(id=user.id, tenant_id=user.tenant_id, email=user.email, role=user.role.value)


# --- collections ------------------------------------------------------------------


@router.post("/collections", response_model=CollectionOut, status_code=201, tags=["collections"])
async def create_collection(
    body: CreateCollectionRequest, container: ContainerDep, user: UserDep
) -> CollectionOut:
    if user.role is Role.VIEWER:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "viewers cannot create collections")
    collection = Collection(user.tenant_id, body.name, body.description, user.id)
    try:
        await container.access.create_collection(collection)
    except ValueError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    await container.access.set_member(collection.id, user.id, Permission.ADMIN)
    return CollectionOut(
        id=collection.id,
        name=collection.name,
        description=collection.description,
        permission="admin",
        created_at=collection.created_at,
    )


@router.get("/collections", response_model=list[CollectionOut], tags=["collections"])
async def list_collections(container: ContainerDep, user: UserDep) -> list[CollectionOut]:
    out = []
    for c in await container.access.list_collections(user.tenant_id):
        perm = await permission_for(container, user, c.id)
        if perm is not None:
            out.append(
                CollectionOut(
                    id=c.id,
                    name=c.name,
                    description=c.description,
                    permission=perm.name.lower(),
                    created_at=c.created_at,
                )
            )
    return out


@router.post("/collections/{collection_id}/members", status_code=204, tags=["collections"])
async def add_member(
    collection_id: uuid.UUID, body: MemberRequest, container: ContainerDep, user: UserDep
) -> None:
    await require_permission(container, user, collection_id, Permission.ADMIN)
    member = await container.access.get_user(body.user_id)
    if member is None or member.tenant_id != user.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "user not found")
    await container.access.set_member(
        collection_id, body.user_id, Permission.parse(body.permission)
    )


# --- documents --------------------------------------------------------------------


def _version_out(v: VersionRecord) -> VersionOut:
    return VersionOut(
        id=v.id,
        version=v.version,
        status=v.status.value,
        error=v.error,
        chunk_count=v.chunk_count,
        embedding_model=v.embedding_model,
        content_sha256=v.content_sha256,
        created_at=v.created_at,
    )


def _doc_out(d: DocumentRecord, versions: list[VersionRecord]) -> DocumentOut:
    return DocumentOut(
        id=d.id,
        collection_id=d.collection_id,
        title=d.title,
        source=d.source,
        mime_type=d.mime_type,
        tags=d.tags,
        active_version_id=d.active_version_id,
        created_at=d.created_at,
        versions=[_version_out(v) for v in versions],
    )


@router.post(
    "/collections/{collection_id}/documents",
    response_model=UploadResponse,
    status_code=202,
    tags=["documents"],
)
async def upload_document(
    collection_id: uuid.UUID,
    container: ContainerDep,
    user: UserDep,
    file: Annotated[UploadFile, File()],
    tags: Annotated[str, Form()] = "",
    document_id: Annotated[uuid.UUID | None, Form()] = None,
) -> UploadResponse:
    """Upload a file (a new document, or a new version when `document_id` is given).

    Returns 202: parsing and embedding run in the worker. Poll `/v1/jobs/{job_id}`.
    """
    await require_permission(container, user, collection_id, Permission.WRITE)
    limit = container.settings.max_upload_bytes
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, f"max {limit} bytes")
    if not data:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "empty file")
    try:
        handle = await container.indexer.create_version(
            tenant_id=user.tenant_id,
            collection_id=collection_id,
            data=data,
            filename=file.filename or "upload",
            tags=[t.strip() for t in tags.split(",") if t.strip()],
            document_id=document_id,
            created_by=user.id,
        )
    except UnsupportedFormatError as exc:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(exc)) from exc
    except DocumentNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found") from exc
    if handle.created:
        await container.jobs.enqueue(user.tenant_id, handle.version.id)
    return UploadResponse(
        document_id=handle.document.id,
        version_id=handle.version.id,
        job_id=handle.version.id,
        version=handle.version.version,
        created=handle.created,
        status=handle.version.status.value,
    )


async def _readable_document(
    container: Container, user: User, document_id: uuid.UUID
) -> DocumentRecord:
    doc = await container.store.get_document(user.tenant_id, document_id)
    if doc is None or doc.deleted_at is not None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found")
    await require_permission(container, user, doc.collection_id, Permission.READ)
    return doc


@router.get(
    "/collections/{collection_id}/documents", response_model=list[DocumentOut], tags=["documents"]
)
async def list_documents(
    collection_id: uuid.UUID, container: ContainerDep, user: UserDep
) -> list[DocumentOut]:
    await require_permission(container, user, collection_id, Permission.READ)
    docs = await container.store.list_documents(user.tenant_id, collection_id)
    return [_doc_out(d, []) for d in docs]


@router.get("/documents/{document_id}", response_model=DocumentOut, tags=["documents"])
async def get_document(
    document_id: uuid.UUID, container: ContainerDep, user: UserDep
) -> DocumentOut:
    doc = await _readable_document(container, user, document_id)
    return _doc_out(doc, await container.store.list_versions(doc.id))


@router.get(
    "/documents/{document_id}/versions/{version}/chunks",
    response_model=list[ChunkOut],
    tags=["documents"],
)
async def get_chunks(
    document_id: uuid.UUID,
    version: int,
    container: ContainerDep,
    user: UserDep,
    offset: int = 0,
    limit: int = 50,
) -> list[ChunkOut]:
    doc = await _readable_document(container, user, document_id)
    match = next(
        (v for v in await container.store.list_versions(doc.id) if v.version == version), None
    )
    if match is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "version not found")
    chunks = await container.store.get_version_chunks(match.id)
    return [
        ChunkOut(
            id=c.id,
            ordinal=c.ordinal,
            text=c.text,
            token_count=c.token_count,
            heading_path=list(c.heading_path),
            page=c.page,
        )
        for c in chunks[offset : offset + min(limit, 200)]
    ]


@router.delete("/documents/{document_id}", status_code=204, tags=["documents"])
async def delete_document(document_id: uuid.UUID, container: ContainerDep, user: UserDep) -> None:
    doc = await _readable_document(container, user, document_id)
    await require_permission(container, user, doc.collection_id, Permission.WRITE)
    await container.store.soft_delete_document(user.tenant_id, doc.id)


@router.get("/jobs/{job_id}", response_model=JobOut, tags=["documents"])
async def get_job(job_id: uuid.UUID, container: ContainerDep, user: UserDep) -> JobOut:
    version = await container.store.get_version(job_id)
    doc = (
        await container.store.get_document(user.tenant_id, version.document_id) if version else None
    )
    if version is None or doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "job not found")
    await require_permission(container, user, doc.collection_id, Permission.READ)
    return JobOut(job_id=version.id, status=version.status.value, error=version.error)


# --- query --------------------------------------------------------------------------


async def _authorised_filter(container: Container, user: User, body: QueryRequest) -> SearchFilter:
    for cid in body.collection_ids:
        await require_permission(container, user, cid, Permission.READ)
    return SearchFilter(
        tenant_id=user.tenant_id,
        collection_ids=tuple(dict.fromkeys(body.collection_ids)),
        tags=tuple(body.filters.tags),
        mime_types=tuple(body.filters.source_types),
        created_after=body.filters.date_from,
    )


async def _history(
    container: Container, user: User, body: QueryRequest
) -> tuple[Conversation | None, list[Message]]:
    if body.conversation_id is None:
        return None, []
    convo = await container.access.get_conversation(user.tenant_id, body.conversation_id)
    if convo is None or convo.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "conversation not found")
    msgs = await container.access.list_messages(convo.id)
    return convo, [{"role": m.role, "content": m.content} for m in msgs]


def _retrieval_config(container: Container, body: QueryRequest) -> Any:
    from dataclasses import replace

    cfg = container.pipeline.retriever.config
    if body.top_k:
        cfg = replace(cfg, top_k=body.top_k)
    if not body.rerank and cfg.mode is Mode.HYBRID_RERANK:
        cfg = replace(cfg, mode=Mode.HYBRID)
    return cfg


async def _save_turn(
    container: Container, convo: Conversation | None, question: str, final: dict[str, Any]
) -> None:
    if convo is None:
        return
    await container.access.add_message(ChatMessage(convo.id, "user", question))
    await container.access.add_message(
        ChatMessage(
            convo.id,
            "assistant",
            final["answer"],
            citations=final["citations"],
            query_log_id=uuid.UUID(final["query_log_id"]),
        )
    )


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@router.post("/query", response_model=QueryResponse, tags=["query"])
async def query(
    body: QueryRequest, request: Request, container: ContainerDep, user: RateLimitedUser
) -> Any:
    """Ask a question. Send `Accept: text/event-stream` to stream tokens as Server-Sent Events."""
    flt = await _authorised_filter(container, user, body)
    convo, history = await _history(container, user, body)
    cfg = _retrieval_config(container, body)
    debug = body.debug and user.role is Role.ADMIN  # retrieval internals are admin-only

    if "text/event-stream" in request.headers.get("accept", ""):

        async def events() -> AsyncIterator[str]:
            async for name, data in container.pipeline.stream(
                flt, body.question, history=history, user_id=user.id, config=cfg, debug=debug
            ):
                if name == "final":
                    await _save_turn(container, convo, body.question, data)
                yield _sse(name, data)

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    result = await container.pipeline.answer(
        flt, body.question, history=history, user_id=user.id, config=cfg
    )
    final = result.final_event()
    await _save_turn(container, convo, body.question, final)
    return QueryResponse(**final, retrieval=result.retrieval.trace if debug else None)


# --- conversations --------------------------------------------------------------------


@router.post(
    "/conversations", response_model=ConversationOut, status_code=201, tags=["conversations"]
)
async def create_conversation(
    body: CreateConversationRequest, container: ContainerDep, user: UserDep
) -> ConversationOut:
    for cid in body.collection_ids:
        await require_permission(container, user, cid, Permission.READ)
    convo = Conversation(user.tenant_id, user.id, list(body.collection_ids))
    await container.access.create_conversation(convo)
    return ConversationOut(
        id=convo.id, collection_ids=convo.collection_ids, created_at=convo.created_at
    )


@router.get(
    "/conversations/{conversation_id}/messages",
    response_model=list[MessageOut],
    tags=["conversations"],
)
async def list_messages(
    conversation_id: uuid.UUID, container: ContainerDep, user: UserDep
) -> list[MessageOut]:
    convo = await container.access.get_conversation(user.tenant_id, conversation_id)
    if convo is None or convo.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "conversation not found")
    return [
        MessageOut(
            id=m.id, role=m.role, content=m.content, citations=m.citations, created_at=m.created_at
        )
        for m in await container.access.list_messages(convo.id)
    ]

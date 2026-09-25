"""Postgres implementation of `AccessRepo`."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import insert, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import RowMapping
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine

from ragengine.pipeline import QueryLog
from ragengine.store.access import (
    ChatMessage,
    Collection,
    Conversation,
    Permission,
    Role,
    User,
)
from ragengine.store.schema import (
    collection_members,
    collections,
    conversations,
    messages,
    query_logs,
    tenants,
    users,
)


def _user(r: RowMapping) -> User:
    return User(
        id=r["id"],
        tenant_id=r["tenant_id"],
        email=r["email"],
        password_hash=r["password_hash"],
        role=Role(r["role"]),
        created_at=r["created_at"],
    )


def _collection(r: RowMapping) -> Collection:
    return Collection(
        id=r["id"],
        tenant_id=r["tenant_id"],
        name=r["name"],
        description=r["description"],
        created_by=r["created_by"],
        created_at=r["created_at"],
    )


class PgAccessRepo:
    def __init__(self, engine: AsyncEngine) -> None:
        self.engine = engine

    async def _one(self, stmt: Any) -> RowMapping | None:
        async with self.engine.connect() as conn:
            row: RowMapping | None = (await conn.execute(stmt)).mappings().first()
            return row

    async def create_tenant(self, name: str) -> uuid.UUID:
        tenant_id = uuid.uuid4()
        async with self.engine.begin() as conn:
            await conn.execute(insert(tenants).values(id=tenant_id, name=name))
        return tenant_id

    async def create_user(self, user: User) -> None:
        try:
            async with self.engine.begin() as conn:
                await conn.execute(
                    insert(users).values(
                        id=user.id,
                        tenant_id=user.tenant_id,
                        email=user.email.lower(),
                        password_hash=user.password_hash,
                        role=user.role.value,
                        created_at=user.created_at,
                    )
                )
        except IntegrityError as exc:
            raise ValueError("email already registered") from exc

    async def get_user(self, user_id: uuid.UUID) -> User | None:
        row = await self._one(select(users).where(users.c.id == user_id))
        return _user(row) if row else None

    async def get_user_by_email(self, email: str) -> User | None:
        row = await self._one(select(users).where(users.c.email == email.lower()))
        return _user(row) if row else None

    async def create_collection(self, collection: Collection) -> None:
        try:
            async with self.engine.begin() as conn:
                await conn.execute(
                    insert(collections).values(
                        id=collection.id,
                        tenant_id=collection.tenant_id,
                        name=collection.name,
                        description=collection.description,
                        created_by=collection.created_by,
                        created_at=collection.created_at,
                    )
                )
        except IntegrityError as exc:
            raise ValueError("collection name already exists in this tenant") from exc

    async def get_collection(
        self, tenant_id: uuid.UUID, collection_id: uuid.UUID
    ) -> Collection | None:
        row = await self._one(
            select(collections).where(
                collections.c.id == collection_id, collections.c.tenant_id == tenant_id
            )
        )
        return _collection(row) if row else None

    async def list_collections(self, tenant_id: uuid.UUID) -> list[Collection]:
        async with self.engine.connect() as conn:
            rows = (
                await conn.execute(
                    select(collections)
                    .where(collections.c.tenant_id == tenant_id)
                    .order_by(collections.c.created_at)
                )
            ).mappings()
            return [_collection(r) for r in rows]

    async def set_member(
        self, collection_id: uuid.UUID, user_id: uuid.UUID, permission: Permission
    ) -> None:
        stmt = pg_insert(collection_members).values(
            collection_id=collection_id, user_id=user_id, permission=int(permission)
        )
        async with self.engine.begin() as conn:
            await conn.execute(
                stmt.on_conflict_do_update(
                    index_elements=["collection_id", "user_id"],
                    set_={"permission": stmt.excluded.permission},
                )
            )

    async def member_permission(
        self, collection_id: uuid.UUID, user_id: uuid.UUID
    ) -> Permission | None:
        row = await self._one(
            select(collection_members.c.permission).where(
                collection_members.c.collection_id == collection_id,
                collection_members.c.user_id == user_id,
            )
        )
        return Permission(row["permission"]) if row else None

    async def create_conversation(self, conversation: Conversation) -> None:
        async with self.engine.begin() as conn:
            await conn.execute(
                insert(conversations).values(
                    id=conversation.id,
                    tenant_id=conversation.tenant_id,
                    user_id=conversation.user_id,
                    collection_ids=conversation.collection_ids,
                    created_at=conversation.created_at,
                )
            )

    async def get_conversation(
        self, tenant_id: uuid.UUID, conversation_id: uuid.UUID
    ) -> Conversation | None:
        row = await self._one(
            select(conversations).where(
                conversations.c.id == conversation_id, conversations.c.tenant_id == tenant_id
            )
        )
        if row is None:
            return None
        return Conversation(
            id=row["id"],
            tenant_id=row["tenant_id"],
            user_id=row["user_id"],
            collection_ids=list(row["collection_ids"]),
            created_at=row["created_at"],
        )

    async def add_message(self, message: ChatMessage) -> None:
        async with self.engine.begin() as conn:
            await conn.execute(
                insert(messages).values(
                    id=message.id,
                    conversation_id=message.conversation_id,
                    role=message.role,
                    content=message.content,
                    citations=message.citations,
                    query_log_id=message.query_log_id,
                    created_at=message.created_at,
                )
            )

    async def list_messages(self, conversation_id: uuid.UUID) -> list[ChatMessage]:
        async with self.engine.connect() as conn:
            rows = (
                await conn.execute(
                    select(messages)
                    .where(messages.c.conversation_id == conversation_id)
                    .order_by(messages.c.created_at)
                )
            ).mappings()
            return [
                ChatMessage(
                    id=r["id"],
                    conversation_id=r["conversation_id"],
                    role=r["role"],
                    content=r["content"],
                    citations=list(r["citations"]),
                    query_log_id=r["query_log_id"],
                    created_at=r["created_at"],
                )
                for r in rows
            ]

    async def write(self, log: QueryLog) -> None:
        async with self.engine.begin() as conn:
            await conn.execute(
                insert(query_logs).values(
                    id=log.id,
                    tenant_id=log.tenant_id,
                    user_id=log.user_id,
                    query=log.query,
                    rewritten_query=log.rewritten_query,
                    filters=log.filters,
                    retrieved=log.retrieved,
                    context_chunk_ids=[uuid.UUID(c) for c in log.context_chunk_ids],
                    model=log.model,
                    prompt_version=log.prompt_version,
                    prompt_tokens=log.prompt_tokens,
                    completion_tokens=log.completion_tokens,
                    usage_estimated=log.usage_estimated,
                    est_cost_usd=log.est_cost_usd,
                    latency_ms=log.latency_ms,
                    refused=log.refused,
                    refusal_reason=log.refusal_reason,
                    invalid_citations=log.invalid_citations,
                    injection_flags=log.injection_flags,
                    created_at=log.created_at,
                )
            )

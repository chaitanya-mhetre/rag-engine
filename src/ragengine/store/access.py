"""Tenants, users, collections, permissions, conversations, and query logs.

Permission model:
- tenant role `admin` → admin on every collection in the tenant
- otherwise a per-collection permission from `collection_members`: read < write < admin
- `viewer` users can never get more than read, whatever the membership row says
"""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from ragengine.pipeline import QueryLog


def _now() -> datetime:
    return datetime.now(UTC)


class Role(enum.StrEnum):
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"


class Permission(enum.IntEnum):
    READ = 1
    WRITE = 2
    ADMIN = 3

    @classmethod
    def parse(cls, name: str) -> Permission:
        return cls[name.upper()]


@dataclass(slots=True)
class User:
    tenant_id: uuid.UUID
    email: str
    password_hash: str
    role: Role = Role.MEMBER
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    created_at: datetime = field(default_factory=_now)


@dataclass(slots=True)
class Collection:
    tenant_id: uuid.UUID
    name: str
    description: str = ""
    created_by: uuid.UUID | None = None
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    created_at: datetime = field(default_factory=_now)


@dataclass(slots=True)
class Conversation:
    tenant_id: uuid.UUID
    user_id: uuid.UUID
    collection_ids: list[uuid.UUID]
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    created_at: datetime = field(default_factory=_now)


@dataclass(slots=True)
class ChatMessage:
    conversation_id: uuid.UUID
    role: str  # "user" | "assistant"
    content: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    query_log_id: uuid.UUID | None = None
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    created_at: datetime = field(default_factory=_now)


def effective_permission(user: User, member_perm: Permission | None) -> Permission | None:
    if user.role is Role.ADMIN:
        return Permission.ADMIN
    if member_perm is None:
        return None
    if user.role is Role.VIEWER:
        return Permission.READ
    return member_perm


class AccessRepo(Protocol):
    async def create_tenant(self, name: str) -> uuid.UUID: ...

    async def create_user(self, user: User) -> None: ...

    async def get_user(self, user_id: uuid.UUID) -> User | None: ...

    async def get_user_by_email(self, email: str) -> User | None: ...

    async def create_collection(self, collection: Collection) -> None: ...

    async def get_collection(
        self, tenant_id: uuid.UUID, collection_id: uuid.UUID
    ) -> Collection | None: ...

    async def list_collections(self, tenant_id: uuid.UUID) -> list[Collection]: ...

    async def set_member(
        self, collection_id: uuid.UUID, user_id: uuid.UUID, permission: Permission
    ) -> None: ...

    async def member_permission(
        self, collection_id: uuid.UUID, user_id: uuid.UUID
    ) -> Permission | None: ...

    async def create_conversation(self, conversation: Conversation) -> None: ...

    async def get_conversation(
        self, tenant_id: uuid.UUID, conversation_id: uuid.UUID
    ) -> Conversation | None: ...

    async def add_message(self, message: ChatMessage) -> None: ...

    async def list_messages(self, conversation_id: uuid.UUID) -> list[ChatMessage]: ...

    async def write(self, log: QueryLog) -> None:
        """QueryLogSink: persist a query log."""
        ...


class MemoryAccessRepo:
    def __init__(self) -> None:
        self.tenants: dict[uuid.UUID, str] = {}
        self.users: dict[uuid.UUID, User] = {}
        self.collections: dict[uuid.UUID, Collection] = {}
        self.members: dict[tuple[uuid.UUID, uuid.UUID], Permission] = {}
        self.conversations: dict[uuid.UUID, Conversation] = {}
        self.messages: list[ChatMessage] = []
        self.query_logs: list[QueryLog] = []

    async def create_tenant(self, name: str) -> uuid.UUID:
        tenant_id = uuid.uuid4()
        self.tenants[tenant_id] = name
        return tenant_id

    async def create_user(self, user: User) -> None:
        if await self.get_user_by_email(user.email):
            raise ValueError("email already registered")
        self.users[user.id] = user

    async def get_user(self, user_id: uuid.UUID) -> User | None:
        return self.users.get(user_id)

    async def get_user_by_email(self, email: str) -> User | None:
        email = email.lower()
        return next((u for u in self.users.values() if u.email == email), None)

    async def create_collection(self, collection: Collection) -> None:
        if any(
            c.tenant_id == collection.tenant_id and c.name == collection.name
            for c in self.collections.values()
        ):
            raise ValueError("collection name already exists in this tenant")
        self.collections[collection.id] = collection

    async def get_collection(
        self, tenant_id: uuid.UUID, collection_id: uuid.UUID
    ) -> Collection | None:
        c = self.collections.get(collection_id)
        return c if c and c.tenant_id == tenant_id else None

    async def list_collections(self, tenant_id: uuid.UUID) -> list[Collection]:
        return [c for c in self.collections.values() if c.tenant_id == tenant_id]

    async def set_member(
        self, collection_id: uuid.UUID, user_id: uuid.UUID, permission: Permission
    ) -> None:
        self.members[(collection_id, user_id)] = permission

    async def member_permission(
        self, collection_id: uuid.UUID, user_id: uuid.UUID
    ) -> Permission | None:
        return self.members.get((collection_id, user_id))

    async def create_conversation(self, conversation: Conversation) -> None:
        self.conversations[conversation.id] = conversation

    async def get_conversation(
        self, tenant_id: uuid.UUID, conversation_id: uuid.UUID
    ) -> Conversation | None:
        c = self.conversations.get(conversation_id)
        return c if c and c.tenant_id == tenant_id else None

    async def add_message(self, message: ChatMessage) -> None:
        self.messages.append(message)

    async def list_messages(self, conversation_id: uuid.UUID) -> list[ChatMessage]:
        return [m for m in self.messages if m.conversation_id == conversation_id]

    async def write(self, log: QueryLog) -> None:
        self.query_logs.append(log)

"""FastAPI dependencies: container access, authentication, authorisation, rate limiting."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ragengine.container import Container
from ragengine.security import TokenError
from ragengine.store.access import Permission, Role, User, effective_permission

bearer = HTTPBearer(auto_error=False)


def get_container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


ContainerDep = Annotated[Container, Depends(get_container)]


async def current_user(
    container: ContainerDep,
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
) -> User:
    unauthorized = HTTPException(
        status.HTTP_401_UNAUTHORIZED,
        "invalid or missing access token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if creds is None:
        raise unauthorized
    try:
        claims = container.tokens.decode(creds.credentials, "access")
    except TokenError as exc:
        raise unauthorized from exc
    user = await container.access.get_user(claims.user_id)
    if user is None or user.tenant_id != claims.tenant_id:
        raise unauthorized
    return user


UserDep = Annotated[User, Depends(current_user)]


def require_admin(user: UserDep) -> User:
    if user.role is not Role.ADMIN:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "tenant admin role required")
    return user


async def permission_for(
    container: Container, user: User, collection_id: uuid.UUID
) -> Permission | None:
    collection = await container.access.get_collection(user.tenant_id, collection_id)
    if collection is None:
        return None  # other tenants' collections look exactly like missing ones
    member = await container.access.member_permission(collection_id, user.id)
    return effective_permission(user, member)


async def require_permission(
    container: Container, user: User, collection_id: uuid.UUID, needed: Permission
) -> None:
    perm = await permission_for(container, user, collection_id)
    if perm is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "collection not found")
    if perm < needed:
        raise HTTPException(status.HTTP_403_FORBIDDEN, f"{needed.name.lower()} permission required")


async def rate_limited(container: ContainerDep, user: UserDep) -> User:
    decision = await container.limiter.hit(f"user:{user.id}")
    if not decision.allowed:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "rate limit exceeded",
            headers={"Retry-After": str(max(1, round(decision.retry_after)))},
        )
    return user


RateLimitedUser = Annotated[User, Depends(rate_limited)]

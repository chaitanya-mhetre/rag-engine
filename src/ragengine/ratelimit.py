"""Token-bucket rate limiting per key (user or tenant).

A bucket holds up to `capacity` tokens and refills at `rate` tokens/second. Each request takes
one token; an empty bucket means HTTP 429 with a Retry-After hint. The Redis version runs the
same algorithm atomically in a Lua script so several API processes share one limit.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class Decision:
    allowed: bool
    retry_after: float  # seconds until one token is available (0 when allowed)


class RateLimiter(Protocol):
    async def hit(self, key: str) -> Decision: ...


class MemoryRateLimiter:
    def __init__(self, capacity: int, rate: float) -> None:
        self.capacity, self.rate = capacity, rate
        self._buckets: dict[str, tuple[float, float]] = {}  # key -> (tokens, last_ts)

    async def hit(self, key: str, now: float | None = None) -> Decision:
        now = time.monotonic() if now is None else now
        tokens, last = self._buckets.get(key, (float(self.capacity), now))
        tokens = min(self.capacity, tokens + (now - last) * self.rate)
        if tokens >= 1:
            self._buckets[key] = (tokens - 1, now)
            return Decision(True, 0.0)
        self._buckets[key] = (tokens, now)
        return Decision(False, round((1 - tokens) / self.rate, 3))


_LUA = """
local key, capacity, rate, now = KEYS[1], tonumber(ARGV[1]), tonumber(ARGV[2]), tonumber(ARGV[3])
local state = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(state[1]) or capacity
local ts = tonumber(state[2]) or now
tokens = math.min(capacity, tokens + (now - ts) * rate)
local allowed = 0
if tokens >= 1 then tokens = tokens - 1; allowed = 1 end
redis.call('HSET', key, 'tokens', tokens, 'ts', now)
redis.call('EXPIRE', key, math.ceil(capacity / rate) + 1)
return {allowed, tostring(tokens)}
"""


class RedisRateLimiter:  # pragma: no cover - covered by the integration test when Redis is up
    def __init__(self, url: str, capacity: int, rate: float) -> None:
        from redis.asyncio import Redis

        self._redis = Redis.from_url(url)
        self.capacity, self.rate = capacity, rate
        self._script = self._redis.register_script(_LUA)

    async def hit(self, key: str) -> Decision:
        allowed, tokens = await self._script(
            keys=[f"rl:{key}"], args=[self.capacity, self.rate, time.time()]
        )
        if int(allowed):
            return Decision(True, 0.0)
        return Decision(False, round((1 - float(tokens)) / self.rate, 3))

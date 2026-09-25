"""Embedding providers behind one async interface, plus caching and batching.

`HashingEmbedder` is the offline default. It uses feature hashing over word unigrams and
bigrams, so texts that share words get similar vectors. It is deterministic and free, which
makes tests and CI evaluation reproducible. It is NOT a semantic model: it cannot match
synonyms ("car" vs "automobile"). Real models plug in through the same `Embedder` protocol.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from itertools import pairwise
from typing import Protocol

import httpx

from ragengine.ingestion.tokenizer import words

STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "there",
        "these",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "do",
        "does",
        "did",
        "can",
        "you",
        "your",
        "we",
        "our",
        "they",
        "them",
        "he",
        "she",
        "his",
        "her",
        "not",
        "no",
    }
)


class Embedder(Protocol):
    model: str
    dim: int

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


def l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else vec


def _stable_hash(token: str) -> int:
    # Python's hash() is randomised per process; blake2b makes vectors identical across runs.
    return int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "big")


class HashingEmbedder:
    def __init__(self, dim: int = 384, model: str = "hashing-v1") -> None:
        self.dim = dim
        self.model = model

    def _features(self, text: str) -> Counter[str]:
        toks = [w for w in words(text) if w not in STOPWORDS]
        feats: Counter[str] = Counter(toks)
        feats.update(f"{a}_{b}" for a, b in pairwise(toks))
        return feats

    def embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for feat, tf in self._features(text).items():
            h = _stable_hash(feat)
            sign = 1.0 if (h >> 63) & 1 else -1.0  # signed hashing reduces collision bias
            vec[h % self.dim] += sign * (1.0 + math.log(tf))
        return l2_normalize(vec)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]


class OpenAIEmbedder:  # pragma: no cover - network provider, exercised only in live tests
    def __init__(self, api_key: str, model: str = "text-embedding-3-small", dim: int = 384) -> None:
        self.model, self.dim = model, dim
        self._client = httpx.AsyncClient(
            base_url="https://api.openai.com/v1",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        resp = await self._client.post(
            "/embeddings", json={"model": self.model, "input": texts, "dimensions": self.dim}
        )
        resp.raise_for_status()
        return [item["embedding"] for item in resp.json()["data"]]


class GeminiEmbedder:  # pragma: no cover - network provider
    def __init__(self, api_key: str, model: str = "text-embedding-004", dim: int = 384) -> None:
        self.model, self.dim = model, dim
        self._client = httpx.AsyncClient(
            base_url="https://generativelanguage.googleapis.com/v1beta",
            headers={"x-goog-api-key": api_key},
            timeout=30,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        body = {
            "requests": [
                {
                    "model": f"models/{self.model}",
                    "content": {"parts": [{"text": t}]},
                    "outputDimensionality": self.dim,
                }
                for t in texts
            ]
        }
        resp = await self._client.post(f"/models/{self.model}:batchEmbedContents", json=body)
        resp.raise_for_status()
        return [l2_normalize(e["values"]) for e in resp.json()["embeddings"]]


class LocalEmbedder:  # pragma: no cover - optional extra: `uv sync --extra local`
    def __init__(self, model: str = "BAAI/bge-small-en-v1.5") -> None:
        from sentence_transformers import SentenceTransformer

        self.model = model
        self._st = SentenceTransformer(model)
        self.dim = int(self._st.get_sentence_embedding_dimension())

    async def embed(self, texts: list[str]) -> list[list[float]]:
        import asyncio

        vectors = await asyncio.to_thread(self._st.encode, texts, normalize_embeddings=True)
        return [list(map(float, v)) for v in vectors]


class EmbeddingCache(Protocol):
    async def get_many(self, keys: list[str]) -> list[list[float] | None]: ...

    async def set_many(self, items: dict[str, list[float]]) -> None: ...


class MemoryEmbeddingCache:
    def __init__(self) -> None:
        self._data: dict[str, list[float]] = {}

    async def get_many(self, keys: list[str]) -> list[list[float] | None]:
        return [self._data.get(k) for k in keys]

    async def set_many(self, items: dict[str, list[float]]) -> None:
        self._data.update(items)


class RedisEmbeddingCache:
    """Stores vectors as JSON strings with a TTL. Key: emb:{model}:{sha256(text)}."""

    def __init__(self, url: str, ttl_seconds: int = 7 * 24 * 3600) -> None:
        from redis.asyncio import Redis

        self._redis = Redis.from_url(url)
        self._ttl = ttl_seconds

    async def get_many(self, keys: list[str]) -> list[list[float] | None]:
        import json

        raw = await self._redis.mget(keys)
        return [json.loads(r) if r is not None else None for r in raw]

    async def set_many(self, items: dict[str, list[float]]) -> None:
        import json

        async with self._redis.pipeline(transaction=False) as pipe:
            for k, v in items.items():
                pipe.set(k, json.dumps(v), ex=self._ttl)
            await pipe.execute()


class CachedEmbedder:
    """Wraps any embedder: cache lookup first, then batched calls for the misses only."""

    def __init__(self, inner: Embedder, cache: EmbeddingCache, batch_size: int = 64) -> None:
        self.inner, self.cache, self.batch_size = inner, cache, batch_size
        self.model, self.dim = inner.model, inner.dim
        self.calls = 0  # number of provider batches sent (for tests and metrics)

    def _key(self, text: str) -> str:
        return f"emb:{self.model}:{hashlib.sha256(text.encode()).hexdigest()}"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        keys = [self._key(t) for t in texts]
        cached = await self.cache.get_many(keys)
        result: list[list[float] | None] = list(cached)
        missing = [i for i, v in enumerate(cached) if v is None]
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start : start + self.batch_size]
            vectors = await self.inner.embed([texts[i] for i in batch])
            self.calls += 1
            await self.cache.set_many({keys[i]: v for i, v in zip(batch, vectors, strict=True)})
            for i, v in zip(batch, vectors, strict=True):
                result[i] = v
        return [v for v in result if v is not None]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0

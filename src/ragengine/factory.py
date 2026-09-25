"""Build providers from settings. Offline fakes unless a real provider is configured."""

from __future__ import annotations

from ragengine.config import Settings, get_settings
from ragengine.embeddings import (
    CachedEmbedder,
    Embedder,
    GeminiEmbedder,
    HashingEmbedder,
    LocalEmbedder,
    MemoryEmbeddingCache,
    OpenAIEmbedder,
    RedisEmbeddingCache,
)
from ragengine.generation.llm import LLM, FakeExtractiveLLM, GeminiChat, OpenAIChat
from ragengine.retrieval.rerank import LexicalReranker, NoopReranker, Reranker


def _key(value: object, name: str) -> str:
    secret = getattr(value, "get_secret_value", None)
    if secret is None:
        raise RuntimeError(f"{name} must be set to use this provider")
    key: str = secret()
    return key


def build_embedder(settings: Settings | None = None) -> Embedder:
    s = settings or get_settings()
    inner: Embedder
    if s.embedding_provider == "hashing":
        inner = HashingEmbedder(s.embedding_dim, s.embedding_model)
    elif s.embedding_provider == "openai":  # pragma: no cover - network
        inner = OpenAIEmbedder(
            _key(s.openai_api_key, "RAG_OPENAI_API_KEY"), s.embedding_model, s.embedding_dim
        )
    elif s.embedding_provider == "gemini":  # pragma: no cover - network
        inner = GeminiEmbedder(
            _key(s.gemini_api_key, "RAG_GEMINI_API_KEY"), s.embedding_model, s.embedding_dim
        )
    else:  # pragma: no cover - optional extra
        inner = LocalEmbedder(s.embedding_model)
    cache = RedisEmbeddingCache(s.redis_url) if s.redis_url else MemoryEmbeddingCache()
    return CachedEmbedder(inner, cache)


def build_llm(settings: Settings | None = None) -> LLM:
    s = settings or get_settings()
    if s.llm_provider == "fake":
        return FakeExtractiveLLM(
            s.llm_model if s.llm_model.startswith("fake") else "fake-extractive-v1"
        )
    if s.llm_provider == "openai":  # pragma: no cover - network
        return OpenAIChat(_key(s.openai_api_key, "RAG_OPENAI_API_KEY"), s.llm_model)
    return GeminiChat(_key(s.gemini_api_key, "RAG_GEMINI_API_KEY"), s.llm_model)  # pragma: no cover


def build_reranker(settings: Settings | None = None) -> Reranker:
    s = settings or get_settings()
    if s.reranker == "none":
        return NoopReranker()
    if s.reranker == "lexical":
        return LexicalReranker()
    from ragengine.retrieval.rerank import CrossEncoderReranker  # pragma: no cover

    return CrossEncoderReranker(s.reranker_model)  # pragma: no cover

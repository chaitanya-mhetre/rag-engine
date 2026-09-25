from ragengine.embeddings import (
    CachedEmbedder,
    HashingEmbedder,
    MemoryEmbeddingCache,
    cosine,
)


async def test_hashing_embedder_is_deterministic_and_normalised() -> None:
    e = HashingEmbedder(dim=128)
    a1, a2 = await e.embed(["refund policy for annual plans", "refund policy for annual plans"])
    assert a1 == a2
    assert abs(sum(v * v for v in a1) - 1.0) < 1e-9
    assert len(a1) == 128


async def test_shared_words_are_closer_than_unrelated_text() -> None:
    e = HashingEmbedder()
    q, related, unrelated = await e.embed(
        ["how many vacation days", "employees get 20 vacation days per year", "the api uses jwt"]
    )
    assert cosine(q, related) > cosine(q, unrelated)


async def test_stopwords_only_text_gives_zero_vector() -> None:
    (vec,) = await HashingEmbedder(dim=16).embed(["the of and"])
    assert vec == [0.0] * 16


async def test_cache_skips_provider_for_known_texts_and_batches_misses() -> None:
    cached = CachedEmbedder(HashingEmbedder(dim=32), MemoryEmbeddingCache(), batch_size=2)
    first = await cached.embed(["a1", "b2", "c3"])
    assert cached.calls == 2  # 3 misses in batches of 2
    second = await cached.embed(["c3", "a1", "d4"])
    assert cached.calls == 3  # only "d4" was sent
    assert second[0] == first[2] and second[1] == first[0]

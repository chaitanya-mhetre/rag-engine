import math
import uuid

import pytest

from ragengine.embeddings import HashingEmbedder
from ragengine.models import Chunk, ScoredChunk
from ragengine.retrieval.bm25 import BM25Index, analyze
from ragengine.retrieval.fusion import rrf, weighted
from ragengine.retrieval.rerank import LexicalReranker
from ragengine.retrieval.retrievers import (
    BM25Retriever,
    HybridRetriever,
    Mode,
    RetrievalConfig,
    VectorRetriever,
)
from ragengine.store.memory import MemoryStore
from tests.conftest import make_indexer, make_scope


def chunk(text: str) -> Chunk:
    return Chunk(text=text, ordinal=0, token_count=0, document_id=uuid.uuid4())


def test_analyze_drops_stopwords_and_punctuation() -> None:
    assert analyze("What is the Refund-policy?") == ["refund", "policy"]


def test_bm25_matches_hand_computed_score() -> None:
    docs = ["apple banana", "apple apple cherry", "cherry date"]
    index = BM25Index(docs, k1=1.5, b=0.75)
    # term "apple": N=3, df=2 → idf = ln((3-2+0.5)/(2+0.5) + 1) = ln(1.6)
    idf = math.log(1.6)
    avgdl = (2 + 3 + 2) / 3
    # doc 1: tf=2, |D|=3
    expected = idf * 2 * 2.5 / (2 + 1.5 * (1 - 0.75 + 0.75 * 3 / avgdl))
    assert index.scores("apple")[1] == pytest.approx(expected)
    assert index.idf("apple") == pytest.approx(idf)
    assert index.idf("zebra") == pytest.approx(math.log(3.5 / 0.5 + 1))


def test_bm25_rare_terms_outweigh_common_ones() -> None:
    docs = ["the invoice number INV-4471 was paid", "invoice paid", "invoice sent", "invoice due"]
    index = BM25Index(docs)
    (top, _), *_ = index.search("invoice 4471", 1)
    assert top == 0  # the exact-token match wins; this is why BM25 is in the hybrid


def test_bm25_length_normalisation() -> None:
    short = "refund policy"
    long = "refund policy " + " ".join(f"filler{i}" for i in range(50))
    index = BM25Index([short, long], b=0.75)
    s = index.scores("refund")
    assert s[0] > s[1]
    assert BM25Index([short, long], b=0.0).scores("refund")[0] == pytest.approx(
        BM25Index([short, long], b=0.0).scores("refund")[1]
    )


def test_bm25_rejects_bad_params() -> None:
    with pytest.raises(ValueError):
        BM25Index(["x"], b=1.5)


def test_rrf_orders_by_reciprocal_rank_sum() -> None:
    a, b, c = chunk("a"), chunk("b"), chunk("c")
    list1 = [ScoredChunk(a, 9), ScoredChunk(b, 5)]
    list2 = [ScoredChunk(b, 0.9), ScoredChunk(c, 0.8)]
    fused = rrf([list1, list2], k=60)
    assert [h.chunk.text for h in fused] == ["b", "a", "c"]
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert fused[1].score == pytest.approx(1 / 61)


def test_weighted_fusion_alpha_extremes() -> None:
    a, b = chunk("a"), chunk("b")
    kw = [ScoredChunk(a, 10), ScoredChunk(b, 1)]
    vec = [ScoredChunk(b, 0.9), ScoredChunk(a, 0.1)]
    assert weighted(kw, vec, alpha=0.0)[0].chunk is a
    assert weighted(kw, vec, alpha=1.0)[0].chunk is b


async def test_lexical_reranker_prefers_full_coverage() -> None:
    hits = [
        ScoredChunk(chunk("refunds are processed weekly"), 0.9),
        ScoredChunk(chunk("annual plan refund policy: full refund within 30 days"), 0.1),
    ]
    out = await LexicalReranker().rerank("annual plan refund", hits, top_n=2)
    assert out[0].chunk.text.startswith("annual plan")
    assert 0 <= out[1].score <= out[0].score <= 1


CORPUS = {
    "leave.md": b"# Leave\n\nEmployees receive 20 vacation days each calendar year.",
    "errors.md": b"# Errors\n\nError code E-4471 means the upstream payment gateway timed out.",
    "pay.md": b"# Payroll\n\nSalaries are paid on the last working day of each month.",
}


async def build() -> tuple[HybridRetriever, object]:
    store = MemoryStore()
    scope = await make_scope(store)
    idx = make_indexer(store)
    for name, data in CORPUS.items():
        await idx.index_bytes(
            tenant_id=scope.tenant_id, collection_id=scope.collection_id, data=data, filename=name
        )
    from ragengine.store.base import SearchFilter

    retriever = HybridRetriever(
        BM25Retriever(store), VectorRetriever(store, HashingEmbedder()), LexicalReranker()
    )
    return retriever, SearchFilter(scope.tenant_id, (scope.collection_id,))


@pytest.mark.parametrize("mode", list(Mode))
async def test_every_mode_finds_the_exact_error_code(mode: Mode) -> None:
    retriever, flt = await build()
    result = await retriever.retrieve(flt, "what does E-4471 mean", RetrievalConfig(mode=mode))  # type: ignore[arg-type]
    assert result.hits[0].chunk.metadata["source"] == "errors.md"
    assert result.trace[0]["final_rank"] == 1
    if mode in (Mode.HYBRID, Mode.HYBRID_RERANK):
        assert set(result.latency_ms) >= {"bm25", "vector"}
        assert result.trace[0]["keyword_rank"] is not None


async def test_bm25_index_rebuilds_after_new_document() -> None:
    store = MemoryStore()
    scope = await make_scope(store)
    idx = make_indexer(store)
    from ragengine.store.base import SearchFilter

    flt = SearchFilter(scope.tenant_id, (scope.collection_id,))
    bm25 = BM25Retriever(store)
    assert await bm25.search(flt, "vacation", 3) == []
    await idx.index_bytes(
        tenant_id=scope.tenant_id,
        collection_id=scope.collection_id,
        data=CORPUS["leave.md"],
        filename="leave.md",
    )
    assert len(await bm25.search(flt, "vacation", 3)) == 1


def test_fusion_ties_break_deterministically_not_by_random_id() -> None:
    doc = uuid.uuid4()

    def chunk(source: str, ordinal: int) -> Chunk:
        return Chunk(f"{source}-{ordinal}", ordinal, 1, doc, metadata={"source": source})

    for _ in range(5):  # fresh random UUIDs each time; order must not change
        a, b, c = chunk("b.md", 0), chunk("a.md", 1), chunk("a.md", 0)
        tied = [ScoredChunk(x, 1.0, "bm25") for x in (a, b, c)]
        assert [h.chunk.text for h in rrf([tied[:1], tied[1:2], tied[2:]])] == [
            "a.md-0",
            "a.md-1",
            "b.md-0",
        ]
        assert [h.chunk.text for h in weighted(tied, [], alpha=0.0)] == [
            "a.md-0",
            "a.md-1",
            "b.md-0",
        ]

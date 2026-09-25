"""BM25 analysis: stopwords, the hand-written light stemmer, and Snowball."""

from __future__ import annotations

import pytest

from ragengine.retrieval.analysis import Analyzer, LightStemmer, SnowballStemmer, make_stemmer
from ragengine.retrieval.bm25 import BM25Index, analyze


def test_default_analyzer_matches_original_behaviour() -> None:
    text = "What is the Refund-policy for booked flights?"
    assert Analyzer()(text) == analyze(text)


def test_stopwords_can_be_kept() -> None:
    assert "the" in Analyzer(stopwords=False)("the refund")
    assert "the" not in Analyzer()("the refund")


@pytest.mark.parametrize(
    ("word", "stem"),
    [
        ("refunds", "refund"),
        ("policies", "policy"),
        ("classes", "class"),
        ("matches", "match"),
        ("boxes", "box"),
        ("expenses", "expense"),
        ("booking", "book"),
        ("booked", "book"),
        ("running", "run"),
        ("billing", "bill"),
        # things that must NOT change
        ("business", "business"),
        ("status", "status"),
        ("analysis", "analysis"),
        ("speed", "speed"),
        ("used", "used"),
        ("red", "red"),
        ("bus", "bus"),
        ("string", "string"),
        ("e4471", "e4471"),
    ],
)
def test_light_stemmer(word: str, stem: str) -> None:
    assert LightStemmer().stem(word) == stem


def test_snowball_stemmer_conflates_derivations() -> None:
    s = SnowballStemmer()
    assert s.stem("reimbursement") == s.stem("reimbursed") == s.stem("reimburse")


def test_unknown_stemmer_rejected() -> None:
    with pytest.raises(ValueError):
        make_stemmer("porter")  # type: ignore[arg-type]


def test_stemming_lets_bm25_match_inflected_forms() -> None:
    docs = ["Refunds are processed within ten days.", "Laptops are replaced every three years."]
    assert BM25Index(docs).search("refund", k=2) == []  # exact-match BM25 misses "refunds"
    hits = BM25Index(docs, analyzer=Analyzer(stemmer="light")).search("refund", k=2)
    assert [doc for doc, _ in hits] == [0]


def test_stem_cache_returns_same_result() -> None:
    analyzer = Analyzer(stemmer="snowball")
    assert analyzer("travelling travelling") == ["travel", "travel"]

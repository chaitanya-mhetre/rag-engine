"""Text analysis for keyword retrieval: tokenise → (drop stopwords) → (stem).

Why it matters for BM25: BM25 only matches identical terms. Without stemming, a query for
"refund" never matches a chunk that only says "refunds" or "refunded", so recall depends on
the user guessing the exact word form. Stemming maps inflected forms to a shared stem.

Three stemmers, from simplest to most aggressive:

- ``none``     identity; the original behaviour.
- ``light``    a hand-written plural/tense stripper (in the spirit of Harman's S-stemmer,
               1991): refunds → refund, policies → policy, booked → book. Low risk of
               conflating unrelated words.
- ``snowball`` the Snowball English ("Porter2") stemmer via the ``snowballstemmer`` package:
               reimbursement → reimburs, travelling → travel. Higher recall, more
               over-stemming (e.g. "university" and "universe" both → "univers").

The index and the query MUST use the same analyzer, otherwise stems never line up. This module
only affects BM25; the reranker, fake LLM and lexical judge keep the plain ``analyze`` in
``bm25.py`` so evaluation measures one change at a time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

from ragengine.embeddings import STOPWORDS
from ragengine.ingestion.tokenizer import words

StemmerName = Literal["none", "light", "snowball"]

_VOWELS = frozenset("aeiou")


class Stemmer(Protocol):
    name: str

    def stem(self, word: str) -> str: ...


class NoStemmer:
    name = "none"

    def stem(self, word: str) -> str:
        return word


class LightStemmer:
    """Strips common English plural and tense suffixes; never touches words of <= 3 letters.

    Rules are tried in order and the first match wins. Each rule keeps a stem of at least three
    letters containing a vowel, which stops "red" → "r" and "bus" → "bu".
    """

    name = "light"

    def stem(self, word: str) -> str:
        if len(word) <= 3 or not word.isalpha():
            return word
        if word.endswith("eed"):
            return word  # speed, agreed, need: stripping "ed" would mangle them
        for suffix, replacement in (
            ("ies", "y"),  # policies → policy (but "series" → "sery"; acceptable for recall)
            ("sses", "ss"),  # classes → class
            ("ing", ""),  # booking → book
            ("ed", ""),  # booked → book
            ("es", "e"),  # expenses → expense, invoices → invoice (sibilants handled below)
            ("s", ""),  # refunds → refund
        ):
            if not word.endswith(suffix):
                continue
            if suffix == "s" and word.endswith(("ss", "us", "is")):
                return word  # business, status, analysis are not plurals
            stem = word[: -len(suffix)] + replacement
            if suffix == "es" and word[:-2].endswith(("ch", "sh", "x", "z")):
                stem = word[:-2]  # matches → match, boxes → box
            if len(stem) >= 3 and any(c in _VOWELS for c in stem):
                return _undouble(stem) if suffix in ("ing", "ed") else stem
            return word
        return word


def _undouble(stem: str) -> str:
    """running → runn → run; but keep "ll/ss/zz" (billing → bill, not bil)."""
    if len(stem) >= 4 and stem[-1] == stem[-2] and stem[-1] not in "lsz" and stem[-1].isalpha():
        return stem[:-1]
    return stem


class SnowballStemmer:
    name = "snowball"

    def __init__(self) -> None:
        import snowballstemmer  # type: ignore[import-untyped]

        self._stemmer = snowballstemmer.stemmer("english")

    def stem(self, word: str) -> str:
        return str(self._stemmer.stemWord(word))


def make_stemmer(name: StemmerName) -> Stemmer:
    if name == "none":
        return NoStemmer()
    if name == "light":
        return LightStemmer()
    if name == "snowball":
        return SnowballStemmer()
    raise ValueError(f"unknown stemmer {name!r} (expected none | light | snowball)")


@dataclass(frozen=True, slots=True)
class Analyzer:
    """Configurable BM25 analyzer. The default reproduces the original behaviour exactly:
    lower-cased word tokens, stopwords removed, no stemming."""

    stopwords: bool = True
    stemmer: StemmerName = "none"
    _stem: Stemmer = field(init=False, repr=False, compare=False)
    # Corpora repeat the same words constantly, so caching stems makes stemming nearly free.
    _cache: dict[str, str] = field(init=False, repr=False, compare=False, default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_stem", make_stemmer(self.stemmer))

    def __call__(self, text: str) -> list[str]:
        tokens = words(text)
        if self.stopwords:
            tokens = [t for t in tokens if t not in STOPWORDS]
        if self.stemmer == "none":
            return tokens
        return [self._stemmed(t) for t in tokens]

    def _stemmed(self, word: str) -> str:
        stem = self._cache.get(word)
        if stem is None:
            stem = self._stem.stem(word)
            if len(self._cache) < _CACHE_LIMIT:
                self._cache[word] = stem
        return stem


_CACHE_LIMIT = 100_000  # bounded so a huge vocabulary can't grow memory without limit

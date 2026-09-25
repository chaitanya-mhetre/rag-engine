"""BM25 implemented by hand: an inverted index, IDF, and the k1/b saturation terms.

    score(D, Q) = Σ_{t ∈ Q} IDF(t) · tf(t,D) · (k1 + 1) / (tf(t,D) + k1 · (1 − b + b · |D| / avgdl))
    IDF(t)      = ln( (N − df(t) + 0.5) / (df(t) + 0.5) + 1 )     # Lucene's variant, never negative

- k1 controls term-frequency saturation (the 10th "refund" adds little over the 3rd).
- b controls length normalisation (a long chunk is not rewarded just for being long).
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence

from ragengine.embeddings import STOPWORDS
from ragengine.ingestion.tokenizer import words
from ragengine.retrieval.analysis import Analyzer


def analyze(text: str) -> list[str]:
    """Lower-case word tokens without stopwords, no stemming. Shared by the lexical reranker,
    the fake LLM and the lexical judge. BM25 itself uses a configurable `Analyzer` (whose
    default is identical to this function), so stemming can be evaluated in isolation."""
    return [w for w in words(text) if w not in STOPWORDS]


class BM25Index:
    def __init__(
        self,
        texts: Sequence[str],
        k1: float = 1.5,
        b: float = 0.75,
        analyzer: Analyzer | None = None,
    ) -> None:
        if k1 < 0 or not 0 <= b <= 1:
            raise ValueError("k1 must be >= 0 and 0 <= b <= 1")
        self.k1, self.b = k1, b
        # The same analyzer must process documents and queries, or stems never match.
        self.analyzer = analyzer or Analyzer()
        self.postings: dict[str, dict[int, int]] = defaultdict(dict)
        self.doc_len: list[int] = []
        for i, text in enumerate(texts):
            tokens = self.analyzer(text)
            self.doc_len.append(len(tokens))
            for term, tf in Counter(tokens).items():
                self.postings[term][i] = tf
        self.n_docs = len(self.doc_len)
        self.avgdl = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0

    def idf(self, term: str) -> float:
        df = len(self.postings.get(term, {}))
        return math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)

    def scores(self, query: str) -> dict[int, float]:
        """Only documents containing at least one query term get a score (that's the point of
        an inverted index: we never touch documents that can't match)."""
        out: dict[int, float] = defaultdict(float)
        for term in set(self.analyzer(query)):
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf(term)
            for doc, tf in postings.items():
                norm = 1 - self.b + self.b * (self.doc_len[doc] / self.avgdl if self.avgdl else 0)
                out[doc] += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        return dict(out)

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        ranked = sorted(self.scores(query).items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:k]

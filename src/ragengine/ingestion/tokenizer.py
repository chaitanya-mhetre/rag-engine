"""Tokenizers used for chunk sizing and context budgets.

The default `RegexTokenizer` is deterministic and dependency-free, so chunking works offline.
Its counts approximate BPE tokenizers (roughly 1 token per word or punctuation mark); swap in
`TiktokenTokenizer` when exact counts for an OpenAI model matter.
"""

from __future__ import annotations

import re
from typing import Protocol

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class Tokenizer(Protocol):
    def count(self, text: str) -> int: ...

    def split(self, text: str) -> list[str]:
        """Split text into token-sized pieces whose concatenation (with spaces) is readable."""
        ...


class RegexTokenizer:
    """Words and individual punctuation marks count as one token each."""

    def count(self, text: str) -> int:
        return len(_TOKEN_RE.findall(text))

    def split(self, text: str) -> list[str]:
        return _TOKEN_RE.findall(text)


class TiktokenTokenizer:  # pragma: no cover - optional dependency, needs network on first use
    def __init__(self, encoding: str = "cl100k_base") -> None:
        import tiktoken  # type: ignore[import-not-found]

        self._enc = tiktoken.get_encoding(encoding)

    def count(self, text: str) -> int:
        return len(self._enc.encode(text))

    def split(self, text: str) -> list[str]:
        return [self._enc.decode([t]) for t in self._enc.encode(text)]


def words(text: str) -> list[str]:
    """Lower-cased word tokens (no punctuation), used by BM25 and the hashing embedder."""
    return [t.lower() for t in re.findall(r"\w+", text, re.UNICODE)]

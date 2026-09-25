"""Chunkers, implemented by hand so their trade-offs are visible.

- `TokenWindowChunker`: fixed windows of N tokens with overlap. Simple and predictable, but it
  cuts sentences and ignores document structure.
- `RecursiveChunker`: respects structure. It never crosses a heading or page boundary, prefers
  paragraph breaks, falls back to sentence breaks, and only as a last resort to token windows.
  Adjacent pieces are packed greedily up to `max_tokens`, with overlap carried between chunks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import groupby
from typing import Protocol
from uuid import UUID

from ragengine.ingestion.tokenizer import RegexTokenizer, Tokenizer
from ragengine.models import Chunk, ParsedDocument, Section

_TOKEN_SPAN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


class Chunker(Protocol):
    def chunk(self, doc: ParsedDocument, document_id: UUID) -> list[Chunk]: ...


def _window_spans(text: str, size: int, overlap: int) -> list[str]:
    """Slice `text` into windows of `size` tokens, stepping by `size - overlap`.

    Slicing the original string between token offsets keeps the original spacing and punctuation.
    """
    spans = [m.span() for m in _TOKEN_SPAN.finditer(text)]
    if not spans:
        return []
    step = size - overlap
    out: list[str] = []
    for start in range(0, len(spans), step):
        window = spans[start : start + size]
        out.append(text[window[0][0] : window[-1][1]])
        if start + size >= len(spans):
            break
    return out


def _validate(size: int, overlap: int) -> None:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and smaller than the chunk size")


def _base_metadata(doc: ParsedDocument) -> dict[str, object]:
    return {"title": doc.title, "mime_type": doc.mime_type, **doc.metadata}


@dataclass(frozen=True, slots=True)
class TokenWindowChunker:
    size: int = 256
    overlap: int = 32
    tokenizer: Tokenizer = field(default_factory=RegexTokenizer)

    def __post_init__(self) -> None:
        _validate(self.size, self.overlap)

    def chunk(self, doc: ParsedDocument, document_id: UUID) -> list[Chunk]:
        texts = _window_spans(doc.full_text, self.size, self.overlap)
        base = _base_metadata(doc)
        return [
            Chunk(
                text=t,
                ordinal=i,
                token_count=self.tokenizer.count(t),
                document_id=document_id,
                metadata=dict(base),
            )
            for i, t in enumerate(texts)
        ]


@dataclass(frozen=True, slots=True)
class RecursiveChunker:
    max_tokens: int = 256
    overlap_tokens: int = 32
    tokenizer: Tokenizer = field(default_factory=RegexTokenizer)

    def __post_init__(self) -> None:
        _validate(self.max_tokens, self.overlap_tokens)

    def _pieces(self, text: str) -> list[str]:
        """Break one paragraph into pieces that each fit in `max_tokens`."""
        if self.tokenizer.count(text) <= self.max_tokens:
            return [text]
        pieces: list[str] = []
        for sentence in _SENTENCE_END.split(text):
            if self.tokenizer.count(sentence) <= self.max_tokens:
                pieces.append(sentence)
            else:
                pieces.extend(_window_spans(sentence, self.max_tokens, 0))
        return pieces

    def _pack(self, pieces: list[str]) -> list[str]:
        """Greedy packing with overlap: carry trailing pieces worth <= overlap tokens forward."""
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0
        for piece in pieces:
            n = self.tokenizer.count(piece)
            if current and current_tokens + n > self.max_tokens:
                chunks.append("\n\n".join(current))
                carried: list[str] = []
                carried_tokens = 0
                for prev in reversed(current):
                    t = self.tokenizer.count(prev)
                    if (
                        carried_tokens + t > self.overlap_tokens
                        or carried_tokens + t + n > self.max_tokens
                    ):
                        break
                    carried.insert(0, prev)
                    carried_tokens += t
                current, current_tokens = carried, carried_tokens
            current.append(piece)
            current_tokens += n
        if current:
            chunks.append("\n\n".join(current))
        return chunks

    def chunk(self, doc: ParsedDocument, document_id: UUID) -> list[Chunk]:
        base = _base_metadata(doc)
        out: list[Chunk] = []

        def key(s: Section) -> tuple[tuple[str, ...], int | None]:
            return (s.heading_path, s.page)

        for (heading_path, page), group in groupby(doc.sections, key=key):
            pieces = [p for section in group for p in self._pieces(section.text)]
            for text in self._pack(pieces):
                out.append(
                    Chunk(
                        text=text,
                        ordinal=len(out),
                        token_count=self.tokenizer.count(text),
                        document_id=document_id,
                        heading_path=heading_path,
                        page=page,
                        metadata={**base, "heading_path": list(heading_path), "page": page},
                    )
                )
        return out

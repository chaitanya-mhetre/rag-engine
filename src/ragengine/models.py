"""Core domain objects shared by ingestion, retrieval and generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4


@dataclass(frozen=True, slots=True)
class Section:
    """A block of text produced by a parser, with its structural position."""

    text: str
    heading_path: tuple[str, ...] = ()
    page: int | None = None


@dataclass(slots=True)
class ParsedDocument:
    """Parser output: ordered sections plus document-level metadata."""

    title: str
    mime_type: str
    sections: list[Section]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(s.text for s in self.sections)


@dataclass(slots=True)
class Chunk:
    """A retrievable unit of text. `embedding` is filled in by the embedder."""

    text: str
    ordinal: int
    token_count: int
    document_id: UUID
    heading_path: tuple[str, ...] = ()
    page: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    id: UUID = field(default_factory=uuid4)
    embedding: list[float] | None = None


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """A chunk with a retrieval score and the stage that produced it."""

    chunk: Chunk
    score: float
    source: str = ""

"""Ingestion: bytes → parsed document → cleaned sections → chunks (→ embeddings, see M2)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from uuid import UUID, uuid4

from ragengine.ingestion.chunking import Chunker, RecursiveChunker
from ragengine.ingestion.parsers import ParserRegistry
from ragengine.models import Chunk, ParsedDocument


@dataclass(slots=True)
class IngestedDocument:
    document_id: UUID
    filename: str
    content_sha256: str
    parsed: ParsedDocument
    chunks: list[Chunk]
    tags: list[str] = field(default_factory=list)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(slots=True)
class Ingestor:
    parsers: ParserRegistry = field(default_factory=ParserRegistry)
    chunker: Chunker = field(default_factory=RecursiveChunker)

    def ingest(
        self,
        data: bytes,
        filename: str,
        *,
        document_id: UUID | None = None,
        tags: list[str] | None = None,
    ) -> IngestedDocument:
        parsed = self.parsers.parse(data, filename)
        parsed.metadata.setdefault("source", filename)
        doc_id = document_id or uuid4()
        chunks = self.chunker.chunk(parsed, doc_id)
        for c in chunks:
            c.metadata["source"] = filename
            c.metadata["tags"] = list(tags or [])
        return IngestedDocument(
            document_id=doc_id,
            filename=filename,
            content_sha256=sha256(data),
            parsed=parsed,
            chunks=chunks,
            tags=list(tags or []),
        )

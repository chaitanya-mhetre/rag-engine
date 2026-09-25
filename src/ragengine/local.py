"""Helpers to index a folder of files into an in-memory store (CLI and evaluation)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

from ragengine.embeddings import Embedder, HashingEmbedder
from ragengine.indexing import IndexingService
from ragengine.ingestion.chunking import RecursiveChunker
from ragengine.ingestion.parsers import UnsupportedFormatError
from ragengine.ingestion.pipeline import Ingestor
from ragengine.storage import MemoryFileStorage
from ragengine.store.base import SearchFilter
from ragengine.store.memory import MemoryStore

SUPPORTED = {".txt", ".md", ".markdown", ".html", ".htm", ".pdf", ".docx"}


@dataclass(slots=True)
class LocalCorpus:
    store: MemoryStore
    indexer: IndexingService
    tenant_id: uuid.UUID
    collection_id: uuid.UUID
    document_ids: dict[str, uuid.UUID]  # filename -> document id

    @property
    def filter(self) -> SearchFilter:
        return SearchFilter(self.tenant_id, (self.collection_id,))


async def index_folder(
    folder: str | Path,
    *,
    embedder: Embedder | None = None,
    max_tokens: int = 256,
    overlap_tokens: int = 32,
) -> LocalCorpus:
    store = MemoryStore()
    indexer = IndexingService(
        store=store,
        embedder=embedder or HashingEmbedder(),
        files=MemoryFileStorage(),
        ingestor=Ingestor(chunker=RecursiveChunker(max_tokens, overlap_tokens)),
    )
    corpus = LocalCorpus(store, indexer, uuid.uuid4(), uuid.uuid4(), {})
    for path in sorted(Path(folder).rglob("*")):
        if path.suffix.lower() not in SUPPORTED or not path.is_file():
            continue
        try:
            handle = await indexer.index_bytes(
                tenant_id=corpus.tenant_id,
                collection_id=corpus.collection_id,
                data=path.read_bytes(),
                filename=path.name,
            )
        except UnsupportedFormatError:
            continue
        corpus.document_ids[path.name] = handle.document.id
    return corpus

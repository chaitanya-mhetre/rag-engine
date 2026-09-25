"""Evaluation dataset schema (JSON). Every item is hand-written and hand-checked."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ragengine.models import Chunk

ItemType = Literal["factual", "keyword", "paraphrase", "multi_hop", "unanswerable", "adversarial"]


class ExpectedSource(BaseModel):
    document: str
    section: str | None = None  # matched case-insensitively against the chunk's heading path
    page: int | None = None

    def matches(self, chunk: Chunk) -> bool:
        if chunk.metadata.get("source") != self.document:
            return False
        if self.page is not None and chunk.page != self.page:
            return False
        if self.section is None:
            return True
        needle = self.section.lower()
        return any(needle in heading.lower() for heading in chunk.heading_path)


class EvalItem(BaseModel):
    id: str
    question: str
    expected_sources: list[ExpectedSource] = Field(default_factory=list)
    expected_answer: str = ""
    answer_keywords: list[str] = Field(default_factory=list)
    forbidden_keywords: list[str] = Field(default_factory=list)
    type: ItemType
    answerable: bool = True
    difficulty: Literal["easy", "medium", "hard"] = "easy"


class Dataset(BaseModel):
    name: str
    corpus: str  # path relative to the dataset file
    description: str = ""
    items: list[EvalItem]

    @property
    def answerable(self) -> list[EvalItem]:
        return [i for i in self.items if i.answerable]


def load_dataset(path: str | Path) -> tuple[Dataset, Path, str]:
    """Returns the dataset, the resolved corpus folder, and the sha256 of the dataset file."""
    path = Path(path)
    raw = path.read_bytes()
    dataset = Dataset.model_validate_json(raw)
    corpus = (path.parent / dataset.corpus).resolve()
    return dataset, corpus, hashlib.sha256(raw).hexdigest()

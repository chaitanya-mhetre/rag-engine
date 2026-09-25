"""Structured answers: parsing model output, validating citations, and refusal."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, ValidationError

from ragengine.generation.context import Context
from ragengine.generation.prompts import INSUFFICIENT
from ragengine.ingestion.tokenizer import words

REFUSAL_TEXT = "I don't have enough information in the provided documents to answer that."
_MARKER = re.compile(r"\[(S\d+)\]")


class RawCitation(BaseModel):
    source: str
    quote: str = ""


class RawAnswer(BaseModel):
    """What we ask the model for in JSON mode."""

    answer: str
    citations: list[RawCitation] = Field(default_factory=list)
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    insufficient_context: bool = False


@dataclass(frozen=True, slots=True)
class Citation:
    source_label: str
    chunk_id: UUID
    document_id: UUID
    title: str
    section: str
    page: int | None
    quote: str
    quote_verified: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_label": self.source_label,
            "chunk_id": str(self.chunk_id),
            "document_id": str(self.document_id),
            "title": self.title,
            "section": self.section,
            "page": self.page,
            "quote": self.quote,
            "quote_verified": self.quote_verified,
        }


@dataclass(slots=True)
class Answer:
    answer: str
    citations: list[Citation]
    insufficient_context: bool
    confidence: float
    invalid_citations: list[str] = field(default_factory=list)
    refusal_reason: str | None = None  # "low_retrieval_score" | "model_insufficient" | None
    parse_mode: str = "json"


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    return text


def parse_model_output(text: str) -> tuple[RawAnswer, str]:
    """JSON first; if the model ignored the format, fall back to [S#] markers in plain text."""
    try:
        return RawAnswer.model_validate(json.loads(_strip_fences(text))), "json"
    except (json.JSONDecodeError, ValidationError, TypeError):
        pass
    plain = text.strip()
    insufficient = plain.startswith(INSUFFICIENT)
    labels = list(dict.fromkeys(_MARKER.findall(plain)))
    return (
        RawAnswer(
            answer=plain,
            citations=[RawCitation(source=label) for label in labels],
            insufficient_context=insufficient,
        ),
        "markers",
    )


def _quote_in(quote: str, text: str) -> bool:
    if not quote:
        return False
    q, t = " ".join(words(quote)), " ".join(words(text))
    return bool(q) and q in t


def refusal(reason: str) -> Answer:
    return Answer(REFUSAL_TEXT, [], True, 0.0, refusal_reason=reason)


def build_answer(raw: RawAnswer, context: Context, parse_mode: str) -> Answer:
    """Keep only citations that point at blocks actually in the context; count the rest.

    Labels used as [S#] markers in the answer text count as citations even when the model
    forgot to list them, so the answer text and citation list can't disagree.
    """
    if raw.insufficient_context or raw.answer.strip().startswith(INSUFFICIENT):
        return refusal("model_insufficient")
    quotes: dict[str, str] = {}
    order: list[str] = []
    for c in raw.citations:
        quotes.setdefault(c.source, c.quote)
        order.append(c.source)
    order.extend(_MARKER.findall(raw.answer))
    citations: list[Citation] = []
    invalid: list[str] = []
    for label in dict.fromkeys(order):
        block = context.by_label(label)
        if block is None:
            invalid.append(label)
            continue
        quote = quotes.get(label, "")
        citations.append(
            Citation(
                source_label=label,
                chunk_id=block.chunk.id,
                document_id=block.chunk.document_id,
                title=block.title,
                section=block.section,
                page=block.chunk.page,
                quote=quote,
                quote_verified=_quote_in(quote, block.text),
            )
        )
    text = raw.answer
    for label in invalid:
        text = text.replace(f"[{label}]", "")
    return Answer(
        answer=re.sub(r"\s{2,}", " ", text).strip(),
        citations=citations,
        insufficient_context=False,
        confidence=raw.confidence,
        invalid_citations=invalid,
        parse_mode=parse_mode,
    )

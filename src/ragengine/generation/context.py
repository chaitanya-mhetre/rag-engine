"""Context builder: decides exactly what the model sees.

- labels each chunk [S1]..[Sn] so citations can be checked against the context,
- screens chunks for prompt injection,
- drops near-duplicates (overlapping chunks repeat text),
- enforces a token budget (highest-ranked chunks first; never splits a chunk mid-way).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ragengine.generation.promptguard import screen
from ragengine.ingestion.tokenizer import RegexTokenizer, Tokenizer, words
from ragengine.models import Chunk, ScoredChunk


@dataclass(frozen=True, slots=True)
class ContextBlock:
    label: str
    chunk: Chunk
    text: str  # screened text actually shown to the model
    score: float

    @property
    def title(self) -> str:
        return str(self.chunk.metadata.get("title") or self.chunk.metadata.get("source") or "")

    @property
    def section(self) -> str:
        return " > ".join(self.chunk.heading_path)


@dataclass(slots=True)
class Context:
    blocks: list[ContextBlock]
    token_count: int
    dropped_duplicates: int = 0
    dropped_budget: int = 0
    injection_flags: int = 0
    removed_text: list[str] = field(default_factory=list)

    def by_label(self, label: str) -> ContextBlock | None:
        return next((b for b in self.blocks if b.label == label), None)


def jaccard(a: str, b: str) -> float:
    sa, sb = set(words(a)), set(words(b))
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


@dataclass(slots=True)
class ContextBuilder:
    token_budget: int = 1500
    dedupe_threshold: float = 0.8
    tokenizer: Tokenizer = field(default_factory=RegexTokenizer)

    def build(self, hits: list[ScoredChunk]) -> Context:
        blocks: list[ContextBlock] = []
        used = 0
        ctx = Context(blocks=blocks, token_count=0)
        for hit in hits:
            screened = screen(hit.chunk.text)
            if screened.flagged:
                ctx.injection_flags += len(screened.removed)
                ctx.removed_text.extend(screened.removed)
            text = screened.text.strip()
            if not text:
                continue
            if any(jaccard(text, b.text) >= self.dedupe_threshold for b in blocks):
                ctx.dropped_duplicates += 1
                continue
            cost = self.tokenizer.count(text)
            if used + cost > self.token_budget:
                ctx.dropped_budget += 1
                continue
            blocks.append(ContextBlock(f"S{len(blocks) + 1}", hit.chunk, text, hit.score))
            used += cost
        ctx.token_count = used
        return ctx

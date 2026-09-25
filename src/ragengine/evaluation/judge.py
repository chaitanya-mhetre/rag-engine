"""Judged generation metrics: faithfulness, answer relevance, context relevance.

Two judges share one interface:
- `LLMJudge`: asks a model with versioned prompts (JUDGE_PROMPT_VERSION). Scores depend on the
  judge model, so reports record the model and prompt version, and runs judged by different
  models must not be compared.
- `LexicalJudge`: an offline heuristic based on term overlap. It is deterministic and free, but
  it only approximates the real metrics (it can't recognise paraphrase or reasoning). Reports
  label it clearly.

LLM judges are noisy: `rag-eval spot-check` samples judgements for hand-labelling so the
judge's agreement with a human can be measured before its numbers are trusted.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol

from ragengine.generation.answer import _strip_fences
from ragengine.generation.llm import LLM
from ragengine.retrieval.bm25 import analyze

JUDGE_PROMPT_VERSION = "judge-v1"
_MARKERS = re.compile(r"\[S\d+\]")
_SENTENCES = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True, slots=True)
class Judgement:
    faithfulness: float | None  # share of answer claims supported by the context
    answer_relevance: float | None  # does the answer address the question (0..1)
    context_relevance: float  # mean relevance of retrieved chunks to the question (0..1)
    unsupported_claims: tuple[str, ...] = ()


class Judge(Protocol):
    name: str

    async def judge(self, question: str, answer: str | None, contexts: list[str]) -> Judgement: ...


def claims(answer: str) -> list[str]:
    text = _MARKERS.sub("", answer)
    return [s.strip() for s in _SENTENCES.split(text) if len(analyze(s)) >= 2]


class LexicalJudge:
    name = "lexical-heuristic-v1"

    def __init__(self, support_threshold: float = 0.6) -> None:
        self.support_threshold = support_threshold

    @staticmethod
    def _coverage(needle: str, haystack_terms: set[str]) -> float:
        terms = set(analyze(needle))
        return len(terms & haystack_terms) / len(terms) if terms else 0.0

    async def judge(self, question: str, answer: str | None, contexts: list[str]) -> Judgement:
        context_relevance = (
            sum(self._coverage(question, set(analyze(c))) for c in contexts) / len(contexts)
            if contexts
            else 0.0
        )
        if answer is None:
            return Judgement(None, None, round(context_relevance, 4))
        ctx_terms = set(analyze(" ".join(contexts)))
        answer_claims = claims(answer)
        unsupported = tuple(
            c for c in answer_claims if self._coverage(c, ctx_terms) < self.support_threshold
        )
        faithfulness = 1 - len(unsupported) / len(answer_claims) if answer_claims else None
        answer_relevance = self._coverage(question, set(analyze(answer)))
        return Judgement(
            round(faithfulness, 4) if faithfulness is not None else None,
            round(answer_relevance, 4),
            round(context_relevance, 4),
            unsupported,
        )


FAITHFULNESS_PROMPT = """You check whether an answer is supported by source passages.
For each claim in the ANSWER decide if the PASSAGES support it. Output JSON only:
{"claims": [{"claim": "<claim>", "supported": true|false}]}"""

RELEVANCE_PROMPT = """Rate how well the ANSWER addresses the QUESTION, from 0 (not at all) to 1
(fully and directly). Ignore whether it is correct. Output JSON only: {"score": <0..1>}"""

CONTEXT_PROMPT = """Rate how useful the PASSAGE is for answering the QUESTION, from 0 (irrelevant)
to 1 (contains the answer). Output JSON only: {"score": <0..1>}"""


class LLMJudge:  # pragma: no cover - needs a real model; exercised in live runs
    def __init__(self, llm: LLM) -> None:
        self.llm = llm
        self.name = f"llm:{llm.model}:{JUDGE_PROMPT_VERSION}"

    async def _json(self, system: str, user: str) -> dict[str, object]:
        resp = await self.llm.complete(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            json_mode=True,
        )
        try:
            data = json.loads(_strip_fences(resp.text))
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    async def _score(self, system: str, user: str) -> float:
        value = (await self._json(system, user)).get("score", 0.0)
        try:
            return max(0.0, min(1.0, float(value)))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0.0

    async def judge(self, question: str, answer: str | None, contexts: list[str]) -> Judgement:
        ctx_scores = [
            await self._score(CONTEXT_PROMPT, f"QUESTION: {question}\n\nPASSAGE:\n{c}")
            for c in contexts
        ]
        context_relevance = sum(ctx_scores) / len(ctx_scores) if ctx_scores else 0.0
        if answer is None:
            return Judgement(None, None, round(context_relevance, 4))
        passages = "\n\n".join(contexts)
        data = await self._json(FAITHFULNESS_PROMPT, f"PASSAGES:\n{passages}\n\nANSWER:\n{answer}")
        raw_claims = data.get("claims", [])
        items = (
            [c for c in raw_claims if isinstance(c, dict)] if isinstance(raw_claims, list) else []
        )
        unsupported = tuple(str(c.get("claim", "")) for c in items if not c.get("supported"))
        faithfulness = 1 - len(unsupported) / len(items) if items else None
        relevance = await self._score(RELEVANCE_PROMPT, f"QUESTION: {question}\n\nANSWER: {answer}")
        return Judgement(
            round(faithfulness, 4) if faithfulness is not None else None,
            round(relevance, 4),
            round(context_relevance, 4),
            unsupported,
        )

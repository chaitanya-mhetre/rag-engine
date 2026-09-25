"""The query pipeline: authorised filter → (condense) → retrieve → gate → context → LLM →
validate citations → log.

    1 caller passes a SearchFilter built from the collections the user may read
    2 follow-up questions are rewritten into standalone questions using the conversation
    3 hybrid retrieval (+ rerank)
    4 refusal gate: if the best reranked score is below the threshold we answer "not enough
      information" without calling the LLM (cheaper, and no chance to hallucinate)
    5 context builder (budget, dedupe, injection screening, [S#] labels)
    6 LLM (JSON mode for complete answers, marker mode for streaming)
    7 citation validation (drop citations to labels that weren't in the context)
    8 query log: retrieval trace, tokens, estimated cost, per-stage latency
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from ragengine.generation.answer import Answer, build_answer, parse_model_output, refusal
from ragengine.generation.context import Context, ContextBuilder
from ragengine.generation.cost import PriceTable
from ragengine.generation.llm import LLM, LLMResponse, Message
from ragengine.generation.prompts import CONDENSE, PROMPT_VERSION, SYSTEM, user_message
from ragengine.retrieval.retrievers import HybridRetriever, Mode, RetrievalConfig, RetrievalResult
from ragengine.store.base import SearchFilter


@dataclass(slots=True)
class QueryLog:
    tenant_id: uuid.UUID
    query: str
    rewritten_query: str
    filters: dict[str, Any]
    retrieved: list[dict[str, Any]]
    context_chunk_ids: list[str]
    model: str
    prompt_version: str
    prompt_tokens: int
    completion_tokens: int
    usage_estimated: bool
    est_cost_usd: float | None
    latency_ms: dict[str, float]
    refused: bool
    refusal_reason: str | None
    invalid_citations: list[str]
    injection_flags: int
    user_id: uuid.UUID | None = None
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for key in ("tenant_id", "user_id", "id"):
            d[key] = str(d[key]) if d[key] else None
        d["created_at"] = self.created_at.isoformat()
        return d


class QueryLogSink(Protocol):
    async def write(self, log: QueryLog) -> None: ...


class MemoryQueryLogSink:
    def __init__(self) -> None:
        self.logs: list[QueryLog] = []

    async def write(self, log: QueryLog) -> None:
        self.logs.append(log)


@dataclass(slots=True)
class QueryResult:
    answer: Answer
    log: QueryLog
    retrieval: RetrievalResult
    context: Context

    def final_event(self) -> dict[str, Any]:
        return {
            "answer": self.answer.answer,
            "citations": [c.as_dict() for c in self.answer.citations],
            "insufficient_context": self.answer.insufficient_context,
            "confidence": self.answer.confidence,
            "usage": {
                "prompt_tokens": self.log.prompt_tokens,
                "completion_tokens": self.log.completion_tokens,
                "estimated": self.log.usage_estimated,
                "est_cost_usd": self.log.est_cost_usd,
            },
            "latency_ms": self.log.latency_ms,
            "query_log_id": str(self.log.id),
        }


def _filters(flt: SearchFilter) -> dict[str, Any]:
    return {
        "collection_ids": [str(c) for c in flt.collection_ids],
        "tags": list(flt.tags),
        "mime_types": list(flt.mime_types),
        "created_after": flt.created_after.isoformat() if flt.created_after else None,
    }


class RAGPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        llm: LLM,
        *,
        context_builder: ContextBuilder | None = None,
        prices: PriceTable | None = None,
        sink: QueryLogSink | None = None,
        refusal_threshold: float = 0.2,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.context_builder = context_builder or ContextBuilder()
        self.prices = prices or PriceTable({})
        self.sink: QueryLogSink = sink or MemoryQueryLogSink()
        self.refusal_threshold = refusal_threshold

    async def condense(self, question: str, history: list[Message]) -> tuple[str, float]:
        """Rewrite a follow-up into a standalone question. Returns (question, latency_ms)."""
        if not history:
            return question, 0.0
        convo = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
        response = await self.llm.complete(
            [
                {"role": "system", "content": CONDENSE},
                {"role": "user", "content": f"{convo}\n\nFollow-up: {question}"},
            ]
        )
        return response.text.strip() or question, response.latency_ms

    def _gated(self, retrieval: RetrievalResult, cfg: RetrievalConfig) -> bool:
        """The gate only uses scores with a meaningful scale (reranker or cosine similarity).
        RRF scores (~1/61) and raw BM25 scores are not comparable to a fixed threshold."""
        if not retrieval.hits:
            return True
        if cfg.mode in (Mode.HYBRID_RERANK, Mode.VECTOR):
            return retrieval.top_score < self.refusal_threshold
        return False

    def _log(
        self,
        flt: SearchFilter,
        question: str,
        rewritten: str,
        retrieval: RetrievalResult,
        context: Context,
        answer: Answer,
        usage: LLMResponse | None,
        latency: dict[str, float],
        user_id: uuid.UUID | None,
    ) -> QueryLog:
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        model = usage.model if usage else self.llm.model
        return QueryLog(
            tenant_id=flt.tenant_id,
            user_id=user_id,
            query=question,
            rewritten_query=rewritten,
            filters=_filters(flt),
            retrieved=retrieval.trace,
            context_chunk_ids=[str(b.chunk.id) for b in context.blocks],
            model=model,
            prompt_version=PROMPT_VERSION,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            usage_estimated=bool(usage and usage.usage_estimated),
            est_cost_usd=(
                self.prices.cost(model, prompt_tokens, completion_tokens) if usage else 0.0
            ),
            latency_ms=latency,
            refused=answer.insufficient_context,
            refusal_reason=answer.refusal_reason,
            invalid_citations=answer.invalid_citations,
            injection_flags=context.injection_flags,
        )

    async def _prepare(
        self,
        flt: SearchFilter,
        question: str,
        history: list[Message] | None,
        config: RetrievalConfig | None,
    ) -> tuple[str, RetrievalResult, Context, dict[str, float], RetrievalConfig]:
        cfg = config or self.retriever.config
        latency: dict[str, float] = {}
        rewritten, condense_ms = await self.condense(question, history or [])
        if condense_ms:
            latency["condense"] = condense_ms
        retrieval = await self.retriever.retrieve(flt, rewritten, cfg)
        latency.update(retrieval.latency_ms)
        start = time.perf_counter()
        context = self.context_builder.build(retrieval.hits)
        latency["context"] = round((time.perf_counter() - start) * 1000, 3)
        return rewritten, retrieval, context, latency, cfg

    def _messages(self, question: str, context: Context, *, json_mode: bool) -> list[Message]:
        return [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_message(question, context, json_mode=json_mode)},
        ]

    async def answer(
        self,
        flt: SearchFilter,
        question: str,
        *,
        history: list[Message] | None = None,
        user_id: uuid.UUID | None = None,
        config: RetrievalConfig | None = None,
    ) -> QueryResult:
        total = time.perf_counter()
        rewritten, retrieval, context, latency, cfg = await self._prepare(
            flt, question, history, config
        )
        usage: LLMResponse | None = None
        if self._gated(retrieval, cfg) or not context.blocks:
            answer = refusal("low_retrieval_score")
        else:
            usage = await self.llm.complete(
                self._messages(rewritten, context, json_mode=True), json_mode=True
            )
            latency["llm_total"] = usage.latency_ms
            raw, mode = parse_model_output(usage.text)
            answer = build_answer(raw, context, mode)
        latency["total"] = round((time.perf_counter() - total) * 1000, 3)
        log = self._log(
            flt, question, rewritten, retrieval, context, answer, usage, latency, user_id
        )
        await self.sink.write(log)
        return QueryResult(answer, log, retrieval, context)

    async def stream(
        self,
        flt: SearchFilter,
        question: str,
        *,
        history: list[Message] | None = None,
        user_id: uuid.UUID | None = None,
        config: RetrievalConfig | None = None,
        debug: bool = False,
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        """Yields (event, data): optional "retrieval", then "token"s, then "final"."""
        total = time.perf_counter()
        rewritten, retrieval, context, latency, cfg = await self._prepare(
            flt, question, history, config
        )
        if debug:
            yield "retrieval", {"rewritten_query": rewritten, "chunks": retrieval.trace}
        usage: LLMResponse | None = None
        if self._gated(retrieval, cfg) or not context.blocks:
            answer = refusal("low_retrieval_score")
            yield "token", {"text": answer.answer}
        else:
            parts: list[str] = []
            start = time.perf_counter()
            first_token_ms: float | None = None
            async for delta in self.llm.stream(self._messages(rewritten, context, json_mode=False)):
                if first_token_ms is None:
                    first_token_ms = round((time.perf_counter() - start) * 1000, 3)
                parts.append(delta)
                yield "token", {"text": delta}
            usage = self.llm.last_usage
            latency["llm_first_token"] = first_token_ms or 0.0
            latency["llm_total"] = round((time.perf_counter() - start) * 1000, 3)
            raw, mode = parse_model_output("".join(parts))
            answer = build_answer(raw, context, mode)
        latency["total"] = round((time.perf_counter() - total) * 1000, 3)
        log = self._log(
            flt, question, rewritten, retrieval, context, answer, usage, latency, user_id
        )
        await self.sink.write(log)
        yield "final", QueryResult(answer, log, retrieval, context).final_event()

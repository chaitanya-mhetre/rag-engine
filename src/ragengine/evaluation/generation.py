"""Generation hook for the evaluation runner: run the real answer pipeline per item."""

from __future__ import annotations

from typing import Any

from ragengine.evaluation.dataset import EvalItem
from ragengine.evaluation.judge import Judge, LexicalJudge
from ragengine.evaluation.runner import GenerationHook, RunConfig, make_reranker
from ragengine.factory import build_llm
from ragengine.generation.context import ContextBuilder
from ragengine.generation.cost import PriceTable
from ragengine.generation.llm import LLM
from ragengine.local import LocalCorpus
from ragengine.pipeline import RAGPipeline
from ragengine.retrieval.retrievers import BM25Retriever, HybridRetriever, VectorRetriever


def make_generation_hook(
    llm: LLM | None = None,
    *,
    refusal_threshold: float = 0.2,
    prices: PriceTable | None = None,
    token_budget: int = 1500,
    judge: Judge | None = None,
) -> GenerationHook:
    model = llm or build_llm()
    judge_impl: Judge = judge or LexicalJudge()
    price_table = prices or PriceTable.load("config/models.json")

    async def hook(corpus: LocalCorpus, cfg: RunConfig, item: EvalItem, k: int) -> dict[str, Any]:
        retriever = HybridRetriever(
            BM25Retriever(corpus.store, analyzer=cfg.analyzer()),
            VectorRetriever(corpus.store, corpus.indexer.embedder),
            make_reranker(cfg.reranker),
        )
        pipeline = RAGPipeline(
            retriever,
            model,
            context_builder=ContextBuilder(token_budget=token_budget),
            prices=price_table,
            refusal_threshold=refusal_threshold,
        )
        result = await pipeline.answer(corpus.filter, item.question, config=cfg.retrieval(k))
        chunks = {b.chunk.id: b.chunk for b in result.context.blocks}
        cited = [chunks[c.chunk_id] for c in result.answer.citations if c.chunk_id in chunks]
        text = result.answer.answer.lower()
        verdict = await judge_impl.judge(
            item.question,
            None if result.answer.insufficient_context else result.answer.answer,
            [b.text for b in result.context.blocks],
        )
        return {
            "faithfulness": verdict.faithfulness,
            "answer_relevance": verdict.answer_relevance,
            "context_relevance": verdict.context_relevance,
            "unsupported_claims": list(verdict.unsupported_claims),
            "judge": judge_impl.name,
            "answer": result.answer.answer,
            "refused": result.answer.insufficient_context,
            "refusal_reason": result.answer.refusal_reason,
            "citations": [
                {"source": c.metadata.get("source"), "heading_path": list(c.heading_path)}
                for c in cited
            ],
            "citation_matches": [any(s.matches(c) for s in item.expected_sources) for c in cited],
            "invalid_citations": len(result.answer.invalid_citations),
            "quote_verified": [c.quote_verified for c in result.answer.citations],
            "keyword_hits": [kw.lower() in text for kw in item.answer_keywords],
            "forbidden_hits": [kw.lower() in text for kw in item.forbidden_keywords],
            "prompt_tokens": result.log.prompt_tokens,
            "completion_tokens": result.log.completion_tokens,
            "est_cost_usd": result.log.est_cost_usd,
            "llm_ms": result.log.latency_ms.get("llm_total"),
            "total_ms": result.log.latency_ms.get("total"),
            "injection_flags": result.log.injection_flags,
            "model": result.log.model,
        }

    return hook

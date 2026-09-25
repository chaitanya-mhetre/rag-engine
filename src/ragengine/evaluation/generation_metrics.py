"""Generation metrics (exact, no LLM judge):

- refusal_accuracy: share of UNANSWERABLE items the system declined to answer
- false_refusal_rate: share of ANSWERABLE items it wrongly declined
- keyword_recall: share of hand-picked answer keywords present in the answer (answerable items;
  a refusal scores 0). A cheap, exact proxy for correctness; it can't judge paraphrases.
- citation_precision: share of citations that point at an expected source (answered items)
- invalid_citation_rate: citations to labels that were not in the context / all citations
- quote_verified_rate: citation quotes found verbatim (normalised) in the cited chunk
- adversarial_resistance: adversarial items whose answer contains none of the forbidden phrases
- tokens / cost / latency per query

Judged metrics (see `judge.py`; the judge name is recorded with them):
- faithfulness: mean share of answer claims supported by the context (answered items)
- answer_relevance: mean judged relevance of the answer to the question (answered items)
- context_relevance: mean judged relevance of the retrieved context (all items)
"""

from __future__ import annotations

from typing import Any

from ragengine.evaluation.dataset import Dataset
from ragengine.evaluation.metrics import mean, percentile


def _ratio(num: float, den: float) -> float:
    return round(num / den, 4) if den else 0.0


def summarise_generation(results: list[Any], dataset: Dataset) -> dict[str, Any]:
    typed = [(r.type, r.generation) for r in results if r.generation]
    gens: list[dict[str, Any]] = [g for _, g in typed]
    unanswerable = [g for t, g in typed if t == "unanswerable"]
    answerable = [g for t, g in typed if t != "unanswerable"]
    answered = [g for g in answerable if not g["refused"]]
    adversarial = [g for t, g in typed if t == "adversarial"]

    citations = [m for g in answered for m in g["citation_matches"]]
    invalid = sum(g["invalid_citations"] for g in gens)
    quotes = [q for g in answered for q in g["quote_verified"]]
    costs = [g["est_cost_usd"] for g in gens]
    tokens = [g["prompt_tokens"] + g["completion_tokens"] for g in gens]
    llm_ms = [g["llm_ms"] for g in gens if g["llm_ms"] is not None]
    return {
        "refusal_accuracy": _ratio(sum(g["refused"] for g in unanswerable), len(unanswerable)),
        "false_refusal_rate": _ratio(sum(g["refused"] for g in answerable), len(answerable)),
        "keyword_recall": mean(
            [
                sum(g["keyword_hits"]) / len(g["keyword_hits"])
                for g in answerable
                if g["keyword_hits"]
            ]
        ),
        "citation_precision": _ratio(sum(citations), len(citations)),
        "invalid_citation_rate": _ratio(invalid, len(citations) + invalid),
        "quote_verified_rate": _ratio(sum(quotes), len(quotes)),
        "adversarial_resistance": _ratio(
            sum(not any(g["forbidden_hits"]) for g in adversarial), len(adversarial)
        ),
        "tokens_per_query": mean([float(t) for t in tokens]),
        "est_cost_per_query_usd": (
            None if any(c is None for c in costs) else round(sum(costs) / max(len(costs), 1), 8)
        ),
        "llm_p50_ms": percentile(llm_ms, 50),
        "llm_p95_ms": percentile(llm_ms, 95),
        "faithfulness": mean(
            [g["faithfulness"] for g in answered if g.get("faithfulness") is not None]
        ),
        "answer_relevance": mean(
            [g["answer_relevance"] for g in answered if g.get("answer_relevance") is not None]
        ),
        "context_relevance": mean(
            [g["context_relevance"] for g in gens if "context_relevance" in g]
        ),
        "judge": gens[0].get("judge") if gens else None,
        "model": gens[0]["model"] if gens else None,
    }

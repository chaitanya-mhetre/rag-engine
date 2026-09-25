"""Evaluation runner: index the fixed corpus, run each config over the dataset, collect metrics.

The runner uses the same `HybridRetriever` (and, when enabled, the same answer pipeline) as
the API, so the numbers describe the real system rather than a separate copy of it.
"""

from __future__ import annotations

import subprocess
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ragengine.embeddings import Embedder, HashingEmbedder
from ragengine.evaluation.dataset import EvalItem, load_dataset
from ragengine.evaluation.metrics import mean, percentile, score_retrieval
from ragengine.local import LocalCorpus, index_folder
from ragengine.retrieval.rerank import LexicalReranker, NoopReranker, Reranker
from ragengine.retrieval.retrievers import (
    BM25Retriever,
    Fusion,
    HybridRetriever,
    Mode,
    RetrievalConfig,
    VectorRetriever,
)

DISCLAIMER = (
    "Offline fake-provider baseline — not a quality claim. Embeddings come from the "
    "deterministic hashing embedder and reranking from the lexical heuristic; real-model "
    "numbers are TBD."
)


@dataclass(frozen=True, slots=True)
class RunConfig:
    name: str
    mode: Mode = Mode.HYBRID_RERANK
    fusion: Fusion = Fusion.RRF
    chunk_max_tokens: int = 256
    chunk_overlap_tokens: int = 32
    candidate_k: int = 20
    rrf_k: int = 60
    alpha: float = 0.5
    reranker: str = "lexical"
    generate: bool = False  # run the answer pipeline too (generation metrics)

    def retrieval(self, k: int) -> RetrievalConfig:
        return RetrievalConfig(
            mode=self.mode,
            fusion=self.fusion,
            candidate_k=max(self.candidate_k, k),
            top_k=k,
            rrf_k=self.rrf_k,
            alpha=self.alpha,
        )


PRESETS: dict[str, RunConfig] = {
    "keyword": RunConfig("keyword", Mode.KEYWORD),
    "vector": RunConfig("vector", Mode.VECTOR),
    "hybrid_rrf": RunConfig("hybrid_rrf", Mode.HYBRID, Fusion.RRF),
    "hybrid_weighted": RunConfig("hybrid_weighted", Mode.HYBRID, Fusion.WEIGHTED),
    "hybrid_rrf_rerank": RunConfig("hybrid_rrf_rerank", Mode.HYBRID_RERANK, Fusion.RRF),
}


def git_sha() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return f"{sha}{'-dirty' if dirty else ''}"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def make_reranker(name: str) -> Reranker:
    if name == "none":
        return NoopReranker()
    if name == "lexical":
        return LexicalReranker()
    if name.startswith("cross-encoder"):  # pragma: no cover - optional model download
        from ragengine.retrieval.rerank import CrossEncoderReranker

        model = name.split(":", 1)[1] if ":" in name else "cross-encoder/ms-marco-MiniLM-L-6-v2"
        return CrossEncoderReranker(model)
    raise ValueError(f"unknown reranker {name!r}")


@dataclass(slots=True)
class ItemResult:
    id: str
    type: str
    question: str
    retrieved: list[dict[str, Any]]
    hit: float
    precision: float
    recall: float
    mrr: float
    first_relevant_rank: int | None
    latency_ms: dict[str, float]
    generation: dict[str, Any] = field(default_factory=dict)


GenerationHook = Callable[[LocalCorpus, RunConfig, EvalItem, int], Any]


class EvalRunner:
    def __init__(
        self,
        dataset_path: str | Path,
        *,
        k: int = 5,
        embedder_factory: Callable[[], Embedder] = HashingEmbedder,
        generation_hook: GenerationHook | None = None,
    ) -> None:
        self.dataset, self.corpus_dir, self.dataset_sha = load_dataset(dataset_path)
        self.k = k
        self.embedder_factory = embedder_factory
        self.generation_hook = generation_hook
        self._corpora: dict[tuple[int, int], LocalCorpus] = {}

    async def corpus(self, cfg: RunConfig) -> LocalCorpus:
        key = (cfg.chunk_max_tokens, cfg.chunk_overlap_tokens)
        if key not in self._corpora:
            self._corpora[key] = await index_folder(
                self.corpus_dir,
                embedder=self.embedder_factory(),
                max_tokens=cfg.chunk_max_tokens,
                overlap_tokens=cfg.chunk_overlap_tokens,
            )
        return self._corpora[key]

    async def run_config(self, cfg: RunConfig) -> dict[str, Any]:
        corpus = await self.corpus(cfg)
        retriever = HybridRetriever(
            BM25Retriever(corpus.store),
            VectorRetriever(corpus.store, corpus.indexer.embedder),
            make_reranker(cfg.reranker),
        )
        results: list[ItemResult] = []
        items = self.dataset.items if cfg.generate else self.dataset.answerable
        for item in items:
            start = time.perf_counter()
            result = await retriever.retrieve(corpus.filter, item.question, cfg.retrieval(self.k))
            total_ms = (time.perf_counter() - start) * 1000
            chunks = [h.chunk for h in result.hits]
            scores = score_retrieval(item, chunks, self.k)
            item_result = ItemResult(
                id=item.id,
                type=item.type,
                question=item.question,
                retrieved=[
                    {
                        "source": c.metadata.get("source"),
                        "heading_path": list(c.heading_path),
                        "score": round(h.score, 4),
                        "text": c.text[:200],
                    }
                    for h, c in zip(result.hits, chunks, strict=True)
                ],
                hit=scores.hit,
                precision=scores.precision,
                recall=scores.recall,
                mrr=scores.mrr,
                first_relevant_rank=scores.first_relevant_rank,
                latency_ms={**result.latency_ms, "retrieval_total": round(total_ms, 3)},
            )
            if cfg.generate and self.generation_hook is not None:
                item_result.generation = await self.generation_hook(corpus, cfg, item, self.k)
            results.append(item_result)
        return self._summarise(cfg, results)

    def _summarise(self, cfg: RunConfig, results: list[ItemResult]) -> dict[str, Any]:
        answerable = [r for r in results if r.type != "unanswerable"]

        def block(rs: list[ItemResult]) -> dict[str, float]:
            return {
                f"hit@{self.k}": mean([r.hit for r in rs]),
                f"precision@{self.k}": mean([r.precision for r in rs]),
                f"recall@{self.k}": mean([r.recall for r in rs]),
                "mrr": mean([r.mrr for r in rs]),
                "n": float(len(rs)),
            }

        by_type: dict[str, list[ItemResult]] = defaultdict(list)
        for r in answerable:
            by_type[r.type].append(r)
        stages: dict[str, list[float]] = defaultdict(list)
        for r in results:
            for stage, ms in r.latency_ms.items():
                stages[stage].append(ms)
        summary: dict[str, Any] = {
            "name": cfg.name,
            "config": {k: (v.value if hasattr(v, "value") else v) for k, v in asdict(cfg).items()},
            "retrieval": block(answerable),
            "by_type": {t: block(rs) for t, rs in sorted(by_type.items())},
            "latency_ms": {
                s: {"p50": percentile(v, 50), "p95": percentile(v, 95)}
                for s, v in sorted(stages.items())
            },
            "items": [asdict(r) for r in results],
        }
        if cfg.generate and self.generation_hook is not None:
            from ragengine.evaluation.generation_metrics import summarise_generation

            summary["generation"] = summarise_generation(results, self.dataset)
        return summary

    async def run(self, configs: list[RunConfig], label: str = "") -> dict[str, Any]:
        runs = [await self.run_config(cfg) for cfg in configs]
        return {
            "dataset": self.dataset.name,
            "dataset_sha256": self.dataset_sha,
            "n_items": len(self.dataset.items),
            "n_answerable": len(self.dataset.answerable),
            "k": self.k,
            "git_sha": git_sha(),
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "label": label,
            "embedder": self.embedder_factory().model,
            "disclaimer": DISCLAIMER if self.embedder_factory is HashingEmbedder else "",
            "runs": runs,
        }


def expand_chunk_ablation(configs: list[RunConfig], sizes: list[int]) -> list[RunConfig]:
    if not sizes:
        return configs
    return [
        replace(
            c,
            name=f"{c.name}@{size}",
            chunk_max_tokens=size,
            chunk_overlap_tokens=min(c.chunk_overlap_tokens, size // 4),
        )
        for c in configs
        for size in sizes
    ]

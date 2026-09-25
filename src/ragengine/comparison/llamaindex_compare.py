"""M7 comparison: the same retrieval task implemented with LlamaIndex vs by hand.

Held constant so the comparison measures the framework, not the inputs:
- parsing: our parsers produce the sections for both (LlamaIndex gets one Document per section,
  with the heading path in metadata)
- embeddings: both use the same deterministic HashingEmbedder vectors
- dataset, k, and relevance matching

LlamaIndex does: chunking (SentenceSplitter), node/metadata handling, the vector index, and
retrieval. Scope: vector retrieval only, since BM25 in LlamaIndex lives in a separate package
(`llama-index-retrievers-bm25`) that isn't installed here.

    uv run --extra llamaindex python -m ragengine.comparison.llamaindex_compare
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.node_parser import SentenceSplitter

from ragengine.embeddings import HashingEmbedder
from ragengine.evaluation.dataset import load_dataset
from ragengine.evaluation.metrics import mean, percentile, score_retrieval
from ragengine.evaluation.runner import PRESETS, EvalRunner, git_sha
from ragengine.ingestion.parsers import ParserRegistry
from ragengine.local import SUPPORTED
from ragengine.models import Chunk


class HashingLIEmbedding(BaseEmbedding):
    """Adapter so LlamaIndex uses exactly the same vectors as our pipeline."""

    def _vec(self, text: str) -> list[float]:
        return HashingEmbedder().embed_one(text)

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._vec(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._vec(text)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._vec(query)


def build_index(corpus: Path, chunk_size: int = 256, overlap: int = 32) -> VectorStoreIndex:
    Settings.embed_model = HashingLIEmbedding(model_name="hashing-v1")
    Settings.llm = None  # retrieval only; stops LlamaIndex from reaching for an LLM
    parsers = ParserRegistry()
    documents: list[Document] = []
    for path in sorted(corpus.rglob("*")):
        if path.suffix.lower() not in SUPPORTED:
            continue
        parsed = parsers.parse(path.read_bytes(), path.name)
        for section in parsed.sections:
            documents.append(
                Document(
                    text=section.text,
                    metadata={
                        "source": path.name,
                        "heading_path": "\x1f".join(section.heading_path),
                    },
                    excluded_embed_metadata_keys=["source", "heading_path"],
                    excluded_llm_metadata_keys=["source", "heading_path"],
                )
            )
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return VectorStoreIndex.from_documents(documents, transformations=[splitter])


def to_chunk(node: Any) -> Chunk:
    meta = node.metadata
    heading = tuple(h for h in str(meta.get("heading_path", "")).split("\x1f") if h)
    return Chunk(
        text=node.get_content(),
        ordinal=0,
        token_count=0,
        document_id=uuid.uuid4(),
        heading_path=heading,
        metadata={"source": meta.get("source")},
    )


async def run(dataset_path: str, k: int = 5) -> dict[str, Any]:
    dataset, corpus, sha = load_dataset(dataset_path)
    start = time.perf_counter()
    index = build_index(corpus)
    li_build_ms = (time.perf_counter() - start) * 1000
    retriever = index.as_retriever(similarity_top_k=k)

    scores, latencies = [], []
    for item in dataset.answerable:
        t0 = time.perf_counter()
        nodes = await retriever.aretrieve(item.question)
        latencies.append((time.perf_counter() - t0) * 1000)
        scores.append(score_retrieval(item, [to_chunk(n.node) for n in nodes], k))
    li = {
        f"hit@{k}": mean([s.hit for s in scores]),
        f"precision@{k}": mean([s.precision for s in scores]),
        f"recall@{k}": mean([s.recall for s in scores]),
        "mrr": mean([s.mrr for s in scores]),
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
        "index_build_ms": round(li_build_ms, 1),
    }

    runner = EvalRunner(dataset_path, k=k)
    t0 = time.perf_counter()
    await runner.corpus(PRESETS["vector"])
    ours_build_ms = (time.perf_counter() - t0) * 1000
    ours = {}
    for name in ("vector", "hybrid_rrf_rerank"):
        run_ = await runner.run_config(PRESETS[name])
        lat = run_["latency_ms"]["retrieval_total"]
        ours[name] = {**run_["retrieval"], "p50_ms": lat["p50"], "p95_ms": lat["p95"]}
        ours[name].pop("n", None)
    return {
        "dataset": dataset.name,
        "dataset_sha256": sha,
        "git_sha": git_sha(),
        "k": k,
        "embedder": "hashing-v1 (both)",
        "llamaindex_vector": li,
        "ours_index_build_ms": round(ours_build_ms, 1),
        "ours": ours,
        "disclaimer": "Offline fake-provider baseline — not a quality claim.",
    }


def main() -> int:
    dataset = sys.argv[1] if len(sys.argv) > 1 else "evaluation/datasets/kestrel_v1.json"
    result = asyncio.run(run(dataset))
    out = Path("evaluation/reports/m7_llamaindex_comparison.json")
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

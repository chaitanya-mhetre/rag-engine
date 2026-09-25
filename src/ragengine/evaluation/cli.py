"""`rag-eval`: run and compare evaluations.

    rag-eval run evaluation/datasets/kestrel_v1.json --out evaluation/reports
    rag-eval run evaluation/datasets/kestrel_v1.json --configs vector,hybrid_rrf_rerank \
        --chunk-sizes 128,256,512
    rag-eval compare reports/A.json reports/B.json --fail-on-regression 0.02
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import replace
from pathlib import Path

from ragengine.evaluation.report import compare, to_markdown, write_report
from ragengine.evaluation.runner import PRESETS, EvalRunner, expand_chunk_ablation


def _run(args: argparse.Namespace) -> int:
    names = [n.strip() for n in args.configs.split(",") if n.strip()]
    unknown = [n for n in names if n not in PRESETS]
    if unknown:
        print(f"unknown config(s): {', '.join(unknown)}. Presets: {', '.join(PRESETS)}")
        return 2
    configs = [PRESETS[n] for n in names]
    if args.reranker:
        configs = [replace(c, reranker=args.reranker) for c in configs]
    if args.generate:
        configs = [replace(c, generate=True) for c in configs]
    if args.bm25_stemmer:
        configs = [replace(c, bm25_stemmer=args.bm25_stemmer) for c in configs]
    if args.bm25_no_stopwords:
        configs = [replace(c, bm25_stopwords=False) for c in configs]
    sizes = [int(s) for s in args.chunk_sizes.split(",")] if args.chunk_sizes else []
    configs = expand_chunk_ablation(configs, sizes)

    hook = None
    if args.generate:
        from ragengine.evaluation.generation import make_generation_hook

        hook = make_generation_hook()
    runner = EvalRunner(args.dataset, k=args.k, generation_hook=hook)
    report = asyncio.run(runner.run(configs, label=args.label))
    if args.out:
        path = write_report(report, args.out, args.name)
        print(f"wrote {path} and {path.with_suffix('.md')}")
    print(to_markdown(report, worst=args.worst))
    return 0


def _compare(args: argparse.Namespace) -> int:
    a = json.loads(Path(args.a).read_text())
    b = json.loads(Path(args.b).read_text())
    table, regressed = compare(a, b, threshold=args.fail_on_regression or 0.0)
    print(table)
    if args.fail_on_regression is not None and regressed:
        print(f"REGRESSION: a metric dropped by more than {args.fail_on_regression}")
        return 1
    return 0


def _spot_check(args: argparse.Namespace) -> int:
    """Sample judged items into a Markdown checklist for hand-labelling (judge agreement)."""
    import random

    report = json.loads(Path(args.report).read_text())
    rows = [
        (run["name"], item)
        for run in report["runs"]
        for item in run["items"]
        if item.get("generation", {}).get("faithfulness") is not None
    ]
    rng = random.Random(args.seed)
    sample = rng.sample(rows, min(args.n, len(rows)))
    lines = [
        f"# Judge spot-check ({len(sample)} items, seed {args.seed})",
        "",
        "For each item, tick whether YOU agree with the judge. Agreement rate = ticks / items.",
        "",
    ]
    for name, item in sample:
        gen = item["generation"]
        lines += [
            f"## {item['id']} ({name}): {item['question']}",
            f"- answer: {gen['answer']}",
            f"- judge `{gen.get('judge')}`: faithfulness={gen['faithfulness']}, "
            f"answer_relevance={gen['answer_relevance']}, unsupported={gen['unsupported_claims']}",
            "- [ ] I agree with the faithfulness score",
            "- [ ] I agree with the relevance score",
            "",
        ]
    output = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(output)
        print(f"wrote {args.out}")
    else:
        print(output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-eval", description="RAG evaluation harness")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run configs over a dataset")
    run.add_argument("dataset")
    run.add_argument("--configs", default=",".join(PRESETS))
    run.add_argument("--k", type=int, default=5)
    run.add_argument("--chunk-sizes", default="", help="comma list, e.g. 128,256,512")
    run.add_argument("--reranker", default="", help="none | lexical | cross-encoder[:model]")
    run.add_argument("--generate", action="store_true", help="also run answer generation")
    run.add_argument(
        "--bm25-stemmer",
        choices=["none", "light", "snowball"],
        default=None,
        help="override the BM25 stemmer for every config (default: the config's own, 'light')",
    )
    run.add_argument(
        "--bm25-no-stopwords", action="store_true", help="keep stopwords in the BM25 index"
    )
    run.add_argument("--out", default="", help="directory for JSON + Markdown reports")
    run.add_argument("--name", default=None, help="report file stem")
    run.add_argument("--label", default="")
    run.add_argument("--worst", type=int, default=10)
    run.set_defaults(func=_run)

    cmp_ = sub.add_parser("compare", help="diff two reports")
    cmp_.add_argument("a")
    cmp_.add_argument("b")
    cmp_.add_argument("--fail-on-regression", type=float, default=None)
    cmp_.set_defaults(func=_compare)

    spot = sub.add_parser("spot-check", help="sample judged items for hand-labelling")
    spot.add_argument("report")
    spot.add_argument("--n", type=int, default=20)
    spot.add_argument("--seed", type=int, default=7)
    spot.add_argument("--out", default="")
    spot.set_defaults(func=_spot_check)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    code: int = args.func(args)
    return code


if __name__ == "__main__":
    sys.exit(main())

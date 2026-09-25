"""Write evaluation reports as JSON (machine-readable) and Markdown (for humans)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def to_markdown(report: dict[str, Any], worst: int = 10) -> str:
    k = report["k"]
    lines = [
        f"# Evaluation report: {report['dataset']}",
        "",
        f"- created: {report['created_at']}  ·  git: `{report['git_sha']}`  ·  "
        f"dataset sha256: `{report['dataset_sha256'][:12]}`",
        f"- items: {report['n_items']} ({report['n_answerable']} answerable)  ·  k = {k}  ·  "
        f"embedder: `{report['embedder']}`",
    ]
    if report.get("label"):
        lines.append(f"- label: {report['label']}")
    if report.get("disclaimer"):
        lines += ["", f"> **{report['disclaimer']}**"]

    metric_keys = [f"hit@{k}", f"precision@{k}", f"recall@{k}", "mrr"]
    lines += ["", "## Retrieval (answerable items)", ""]
    lines.append("| config | " + " | ".join(metric_keys) + " | p50 ms | p95 ms |")
    lines.append("|---" * (len(metric_keys) + 3) + "|")
    for run in report["runs"]:
        lat = run["latency_ms"].get("retrieval_total", {"p50": 0, "p95": 0})
        cells = [_fmt(run["retrieval"][m]) for m in metric_keys]
        lines.append(
            f"| {run['name']} | " + " | ".join(cells) + f" | {lat['p50']} | {lat['p95']} |"
        )

    lines += ["", f"## Recall@{k} by question type", ""]
    types = sorted({t for run in report["runs"] for t in run["by_type"]})
    lines.append("| config | " + " | ".join(types) + " |")
    lines.append("|---" * (len(types) + 1) + "|")
    for run in report["runs"]:
        cells = [_fmt(run["by_type"].get(t, {}).get(f"recall@{k}", "")) for t in types]
        lines.append(f"| {run['name']} | " + " | ".join(cells) + " |")

    gen_runs = [r for r in report["runs"] if "generation" in r]
    if gen_runs:
        keys = list(gen_runs[0]["generation"].keys())
        lines += ["", "## Generation", ""]
        lines.append("| config | " + " | ".join(keys) + " |")
        lines.append("|---" * (len(keys) + 1) + "|")
        for run in gen_runs:
            lines.append(
                f"| {run['name']} | "
                + " | ".join(_fmt(run["generation"][key]) for key in keys)
                + " |"
            )

    last = report["runs"][-1]
    failures = sorted(
        (i for i in last["items"] if i["type"] != "unanswerable"),
        key=lambda i: (i["recall"], i["mrr"]),
    )[:worst]
    lines += ["", f"## Worst {len(failures)} items for `{last['name']}` (error analysis)", ""]
    for item in failures:
        got = ", ".join(
            f"{r['source']}>{'/'.join(r['heading_path'][-1:])}" for r in item["retrieved"][:3]
        )
        lines.append(
            f"- **{item['id']}** ({item['type']}) recall={item['recall']:.2f} "
            f"mrr={item['mrr']:.2f}: {item['question']}  \n  top-3: {got}"
        )
    return "\n".join(lines) + "\n"


def write_report(report: dict[str, Any], out_dir: str | Path, name: str | None = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = name or f"{report['dataset']}_{report['created_at'].replace(':', '')[:15]}"
    json_path = out / f"{stem}.json"
    json_path.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    (out / f"{stem}.md").write_text(to_markdown(report))
    return json_path


def compare(a: dict[str, Any], b: dict[str, Any], threshold: float = 0.0) -> tuple[str, bool]:
    """Per-config, per-metric diff of two reports. Returns (markdown, regressed?)."""
    k = a["k"]
    metric_keys = [f"hit@{k}", f"precision@{k}", f"recall@{k}", "mrr"]
    a_runs = {r["name"]: r for r in a["runs"]}
    lines = ["| config | metric | A | B | Δ |", "|---|---|---|---|---|"]
    regressed = False
    for run in b["runs"]:
        base = a_runs.get(run["name"])
        if base is None:
            continue
        for m in metric_keys:
            va, vb = base["retrieval"][m], run["retrieval"][m]
            delta = vb - va
            flag = ""
            if delta < -threshold:
                flag, regressed = " ⚠", True
            lines.append(f"| {run['name']} | {m} | {va:.3f} | {vb:.3f} | {delta:+.3f}{flag} |")
    return "\n".join(lines) + "\n", regressed

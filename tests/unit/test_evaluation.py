import json
import uuid
from pathlib import Path

import pytest

from ragengine.evaluation.cli import main as eval_main
from ragengine.evaluation.dataset import EvalItem, ExpectedSource
from ragengine.evaluation.metrics import percentile, score_retrieval
from ragengine.evaluation.report import compare
from ragengine.evaluation.runner import PRESETS, EvalRunner, expand_chunk_ablation
from ragengine.models import Chunk


def chunk(source: str, heading: tuple[str, ...] = ()) -> Chunk:
    return Chunk(
        text="x",
        ordinal=0,
        token_count=1,
        document_id=uuid.uuid4(),
        heading_path=heading,
        metadata={"source": source},
    )


def item(*sources: ExpectedSource) -> EvalItem:
    return EvalItem(id="q", question="?", expected_sources=list(sources), type="factual")


def test_section_matching_is_case_insensitive_substring() -> None:
    src = ExpectedSource(document="a.md", section="sick")
    assert src.matches(chunk("a.md", ("Leave", "Sick Leave")))
    assert not src.matches(chunk("a.md", ("Leave", "Vacation")))
    assert not src.matches(chunk("b.md", ("Sick Leave",)))


def test_retrieval_scores() -> None:
    it = item(ExpectedSource(document="a.md"), ExpectedSource(document="b.md"))
    retrieved = [chunk("z.md"), chunk("a.md"), chunk("a.md"), chunk("y.md"), chunk("x.md")]
    s = score_retrieval(it, retrieved, k=5)
    assert s.hit == 1.0
    assert s.first_relevant_rank == 2 and s.mrr == 0.5
    assert s.precision == pytest.approx(2 / 5)
    assert s.recall == 0.5  # only one of the two expected sources was found


def test_retrieval_scores_miss() -> None:
    s = score_retrieval(item(ExpectedSource(document="a.md")), [chunk("b.md")], k=5)
    assert (s.hit, s.mrr, s.recall, s.first_relevant_rank) == (0.0, 0.0, 0.0, None)


def test_percentile_nearest_rank() -> None:
    assert percentile([1, 2, 3, 4], 50) == 2
    assert percentile([1, 2, 3, 4], 95) == 4
    assert percentile([], 50) == 0.0


def write_dataset(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "leave.md").write_text("# Leave\n\n## Vacation\n\nStaff get 22 vacation days.")
    (corpus / "api.md").write_text(
        "# API\n\n## Limits\n\nEach key may send 600 requests per minute."
    )
    data = {
        "name": "tiny",
        "corpus": "corpus",
        "items": [
            {
                "id": "q1",
                "question": "how many vacation days",
                "type": "factual",
                "expected_sources": [{"document": "leave.md", "section": "Vacation"}],
            },
            {
                "id": "q2",
                "question": "requests per minute limit",
                "type": "keyword",
                "expected_sources": [{"document": "api.md"}],
            },
            {"id": "q3", "question": "who is the ceo", "type": "unanswerable", "answerable": False},
        ],
    }
    path = tmp_path / "tiny.json"
    path.write_text(json.dumps(data))
    return path


async def test_runner_end_to_end(tmp_path: Path) -> None:
    runner = EvalRunner(write_dataset(tmp_path), k=2)
    report = await runner.run([PRESETS["keyword"], PRESETS["hybrid_rrf_rerank"]])
    assert report["n_answerable"] == 2
    for run in report["runs"]:
        assert run["retrieval"]["recall@2"] == 1.0
        assert len(run["items"]) == 2  # unanswerable items skipped for retrieval metrics
    assert "Offline fake-provider baseline" in report["disclaimer"]
    assert len(report["dataset_sha256"]) == 64


def test_chunk_ablation_expands_configs() -> None:
    out = expand_chunk_ablation([PRESETS["vector"]], [128, 512])
    assert [c.name for c in out] == ["vector@128", "vector@512"]
    assert out[0].chunk_overlap_tokens == 32 and out[0].chunk_max_tokens == 128


def test_compare_flags_regressions() -> None:
    def rep(recall: float) -> dict[str, object]:
        return {
            "k": 5,
            "runs": [
                {
                    "name": "c",
                    "retrieval": {"hit@5": 1.0, "precision@5": 0.2, "recall@5": recall, "mrr": 0.9},
                }
            ],
        }

    _, regressed = compare(rep(0.9), rep(0.85), threshold=0.02)
    assert regressed
    _, ok = compare(rep(0.9), rep(0.89), threshold=0.02)
    assert not ok


def test_cli_run_and_compare(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ds = write_dataset(tmp_path)
    out = tmp_path / "reports"
    assert (
        eval_main(["run", str(ds), "--configs", "keyword", "--out", str(out), "--name", "a"]) == 0
    )
    assert (out / "a.json").exists() and (out / "a.md").exists()
    assert (
        eval_main(
            ["compare", str(out / "a.json"), str(out / "a.json"), "--fail-on-regression", "0.01"]
        )
        == 0
    )
    assert eval_main(["run", str(ds), "--configs", "nope"]) == 2
    assert "recall@5" in capsys.readouterr().out


async def test_generation_eval_end_to_end(tmp_path: Path) -> None:
    from dataclasses import replace

    from ragengine.evaluation.generation import make_generation_hook
    from ragengine.generation.llm import FakeExtractiveLLM

    runner = EvalRunner(
        write_dataset(tmp_path), k=2, generation_hook=make_generation_hook(FakeExtractiveLLM())
    )
    report = await runner.run([replace(PRESETS["hybrid_rrf_rerank"], generate=True)])
    run = report["runs"][0]
    assert len(run["items"]) == 3  # unanswerable items are included when generating
    gen = run["generation"]
    assert gen["false_refusal_rate"] == 0.0
    assert gen["refusal_accuracy"] == 1.0  # "who is the ceo" is declined
    assert gen["invalid_citation_rate"] == 0.0
    assert gen["model"] == "fake-extractive-v1"


def test_cli_generate_flag(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ds = write_dataset(tmp_path)
    assert eval_main(["run", str(ds), "--configs", "hybrid_rrf_rerank", "--generate"]) == 0
    assert "## Generation" in capsys.readouterr().out

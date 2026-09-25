from ragengine.cli import main
from ragengine.ingestion.pipeline import Ingestor


def test_ingest_sets_source_tags_and_hash() -> None:
    doc = Ingestor().ingest(b"# H\n\nSome body text.", "a.md", tags=["hr"])
    assert len(doc.content_sha256) == 64
    assert doc.chunks and all(c.metadata["source"] == "a.md" for c in doc.chunks)
    assert doc.chunks[0].metadata["tags"] == ["hr"]
    assert doc.chunks[0].metadata["title"] == "H"


def test_same_bytes_same_hash() -> None:
    a = Ingestor().ingest(b"same", "a.txt")
    b = Ingestor().ingest(b"same", "b.txt")
    assert a.content_sha256 == b.content_sha256


def test_cli_ingest(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    f = tmp_path / "doc.md"
    f.write_text("# Title\n\nHello world.")
    assert main(["ingest", str(f)]) == 0
    assert "1 chunks" in capsys.readouterr().out


def test_cli_search(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    (tmp_path / "a.md").write_text("# Leave\n\nEmployees get 20 vacation days.")
    (tmp_path / "b.md").write_text("# API\n\nTokens are JWTs.")
    assert main(["search", "vacation days", "--corpus", str(tmp_path), "-k", "1"]) == 0
    out = capsys.readouterr().out
    assert "a.md" in out and "b.md" not in out

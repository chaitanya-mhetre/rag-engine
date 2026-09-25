import pytest

from ragengine.ingestion.cleaning import clean_text
from ragengine.ingestion.parsers import ParserRegistry, UnsupportedFormatError, detect_mime
from tests.helpers import make_docx, make_pdf

registry = ParserRegistry()


def test_clean_text_joins_hyphen_breaks_and_collapses_space() -> None:
    raw = "Retrieval-\naugmented   generation\r\n\n\n\nNext  para"
    assert clean_text(raw) == "Retrievalaugmented generation\n\nNext para"


def test_text_parser_splits_paragraphs() -> None:
    doc = registry.parse(b"First para.\n\nSecond para.", "notes_v1.txt")
    assert doc.title == "notes v1"
    assert [s.text for s in doc.sections] == ["First para.", "Second para."]


def test_markdown_heading_path() -> None:
    md = (
        b"# Guide\n\nIntro.\n\n## Install\n\nRun it.\n\n"
        b"### Linux\n\nUse apt.\n\n## Usage\n\nCall it."
    )
    doc = registry.parse(md, "guide.md")
    assert doc.title == "Guide"
    paths = {s.text: s.heading_path for s in doc.sections}
    assert paths["Intro."] == ("Guide",)
    assert paths["Use apt."] == ("Guide", "Install", "Linux")
    assert paths["Call it."] == ("Guide", "Usage")  # ## closes the open ###


def test_markdown_ignores_hash_inside_code_fence() -> None:
    md = b"# Title\n\n```\n# not a heading\n```\n\nAfter."
    doc = registry.parse(md, "x.md")
    assert all(s.heading_path == ("Title",) for s in doc.sections)


def test_html_parser_drops_scripts_and_tracks_headings() -> None:
    html = (
        b"<html><head><title>Policy</title><script>alert(1)</script></head><body>"
        b"<h1>Leave</h1><p>Staff get 20 days.</p><h2>Sick</h2><ul><li>10 days sick.</li></ul>"
        b"</body></html>"
    )
    doc = registry.parse(html, "policy.html")
    assert doc.title == "Policy"
    assert [(s.text, s.heading_path) for s in doc.sections] == [
        ("Staff get 20 days.", ("Leave",)),
        ("10 days sick.", ("Leave", "Sick")),
    ]
    assert "alert" not in doc.full_text


def test_pdf_parser_keeps_page_numbers() -> None:
    data = make_pdf(["Page one text.", "Page two text."])
    doc = registry.parse(data, "report.pdf")
    assert doc.mime_type == "application/pdf"
    assert [(s.page, s.text) for s in doc.sections] == [
        (1, "Page one text."),
        (2, "Page two text."),
    ]
    assert doc.metadata["pages"] == 2


def test_docx_parser_uses_heading_styles() -> None:
    data = make_docx([("Title", "Handbook"), ("Heading 1", "Leave"), ("Normal", "20 days.")])
    doc = registry.parse(data, "handbook.docx")
    assert doc.title == "Handbook"
    assert [(s.text, s.heading_path) for s in doc.sections] == [("20 days.", ("Leave",))]


def test_mime_is_sniffed_not_trusted_from_extension() -> None:
    assert detect_mime(make_pdf(["x"]), "renamed.txt") == "application/pdf"
    with pytest.raises(UnsupportedFormatError):
        detect_mime(b"just text", "fake.pdf")
    with pytest.raises(UnsupportedFormatError):
        detect_mime(b"PK\x03\x04garbage", "archive.zip")
    with pytest.raises(UnsupportedFormatError):
        detect_mime(b"\x00\x01", "binary.exe")

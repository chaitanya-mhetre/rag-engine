from itertools import pairwise
from uuid import uuid4

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ragengine.ingestion.chunking import RecursiveChunker, TokenWindowChunker
from ragengine.ingestion.tokenizer import RegexTokenizer, words
from ragengine.models import ParsedDocument, Section

tok = RegexTokenizer()


def doc_of(*sections: Section) -> ParsedDocument:
    return ParsedDocument(title="T", mime_type="text/plain", sections=list(sections))


def test_token_window_sizes_and_overlap() -> None:
    text = " ".join(f"w{i}" for i in range(100))
    chunks = TokenWindowChunker(size=30, overlap=10).chunk(doc_of(Section(text)), uuid4())
    assert [c.token_count for c in chunks] == [30, 30, 30, 30, 20]
    first, second = chunks[0].text.split(), chunks[1].text.split()
    assert first[-10:] == second[:10]  # overlap is exactly 10 tokens


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (10, 10), (10, -1)])
def test_invalid_config_rejected(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        TokenWindowChunker(size=size, overlap=overlap)


def test_recursive_never_crosses_heading_or_page() -> None:
    d = doc_of(
        Section("Alpha one.", ("A",)),
        Section("Alpha two.", ("A",)),
        Section("Beta one.", ("B",)),
        Section("Page two.", ("B",), page=2),
    )
    chunks = RecursiveChunker(max_tokens=100, overlap_tokens=0).chunk(d, uuid4())
    assert [(c.heading_path, c.page, c.text) for c in chunks] == [
        (("A",), None, "Alpha one.\n\nAlpha two."),
        (("B",), None, "Beta one."),
        (("B",), 2, "Page two."),
    ]
    assert chunks[0].metadata["heading_path"] == ["A"]


def test_recursive_splits_long_paragraph_on_sentences() -> None:
    para = " ".join(f"Sentence number {i} is here." for i in range(20))  # 6 tokens each
    chunks = RecursiveChunker(max_tokens=20, overlap_tokens=0).chunk(doc_of(Section(para)), uuid4())
    assert all(c.token_count <= 20 for c in chunks)
    assert all(c.text.rstrip().endswith(".") for c in chunks)  # no mid-sentence cuts


def test_recursive_overlap_carries_previous_piece() -> None:
    paras = [Section(f"Paragraph {i} has five.") for i in range(6)]  # 5 tokens each
    chunks = RecursiveChunker(max_tokens=12, overlap_tokens=5).chunk(doc_of(*paras), uuid4())
    for prev, nxt in pairwise(chunks):
        assert nxt.text.startswith(prev.text.split("\n\n")[-1])


sentences = st.lists(
    st.text(alphabet="abcdefghij ", min_size=1, max_size=60).map(lambda s: s.strip() + "."),
    min_size=1,
    max_size=30,
)


@settings(max_examples=60, deadline=None)
@given(paragraphs=sentences, max_tokens=st.integers(5, 60), data=st.data())
def test_recursive_properties(paragraphs: list[str], max_tokens: int, data: st.DataObject) -> None:
    overlap = data.draw(st.integers(0, max_tokens - 1))
    d = doc_of(*(Section(p) for p in paragraphs))
    chunks = RecursiveChunker(max_tokens=max_tokens, overlap_tokens=overlap).chunk(d, uuid4())
    # 1. no chunk exceeds the ceiling
    assert all(tok.count(c.text) <= max_tokens for c in chunks)
    # 2. every word of the input survives somewhere (coverage)
    chunk_words = set(words(" ".join(c.text for c in chunks)))
    assert set(words(d.full_text)) <= chunk_words
    # 3. ordinals are dense and ordered
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))

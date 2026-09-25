"""Hand-written BM25 vs Postgres full-text search on the same small corpus."""

import pytest

from ragengine.retrieval.retrievers import BM25Retriever, PgFtsRetriever
from ragengine.store.base import SearchFilter
from tests.conftest import PG_URL, make_indexer, make_scope, needs_pg

pytestmark = [needs_pg, pytest.mark.integration]

DOCS = {
    "leave.md": b"# Leave\n\nEmployees receive 20 vacation days each calendar year.",
    "errors.md": b"# Errors\n\nError code E4471 means the upstream payment gateway timed out.",
    "pay.md": b"# Payroll\n\nSalaries are paid on the last working day of each month.",
    "travel.md": b"# Travel\n\nBook flights through the travel portal; economy class only.",
}


async def test_bm25_and_pg_fts_agree_on_top_result() -> None:
    from ragengine.store.pg import PgStore, make_engine

    engine = make_engine(PG_URL)
    try:
        store = PgStore(engine)
        scope = await make_scope(store)
        idx = make_indexer(store)
        for name, data in DOCS.items():
            await idx.index_bytes(
                tenant_id=scope.tenant_id,
                collection_id=scope.collection_id,
                data=data,
                filename=name,
            )
        flt = SearchFilter(scope.tenant_id, (scope.collection_id,))
        bm25, fts = BM25Retriever(store), PgFtsRetriever(store)
        for query, expected in [
            ("vacation days", "leave.md"),
            ("E4471", "errors.md"),
            ("salaries paid", "pay.md"),
            ("book flights", "travel.md"),
        ]:
            ours = await bm25.search(flt, query, 1)
            theirs = await fts.search(flt, query, 1)
            assert ours[0].chunk.metadata["source"] == expected
            assert theirs[0].chunk.metadata["source"] == expected
    finally:
        await engine.dispose()

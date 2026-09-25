"""Store contract tests: run on MemoryStore always, and on PgStore when Postgres is up."""

import pytest

from ragengine.embeddings import HashingEmbedder
from ragengine.store.base import SearchFilter, Store, VersionStatus
from tests.conftest import make_collection, make_indexer, make_scope

HANDBOOK = (
    b"# Handbook\n\n## Leave\n\nEmployees get 20 vacation days per year.\n\n"
    b"## Pay\n\nSalaries are paid monthly."
)


async def test_index_and_vector_search(store: Store) -> None:
    scope = await make_scope(store)
    idx = make_indexer(store)
    handle = await idx.index_bytes(
        tenant_id=scope.tenant_id,
        collection_id=scope.collection_id,
        data=HANDBOOK,
        filename="hb.md",
    )
    assert handle.version.status is VersionStatus.READY
    assert handle.version.chunk_count == 2
    doc = await store.get_document(scope.tenant_id, handle.document.id)
    assert doc is not None and doc.title == "Handbook"

    (q,) = await HashingEmbedder().embed(["vacation days"])
    flt = SearchFilter(scope.tenant_id, (scope.collection_id,))
    hits = await store.vector_search(flt, q, k=1)
    assert "vacation" in hits[0].chunk.text
    assert hits[0].chunk.heading_path == ("Handbook", "Leave")


async def test_tenant_isolation_in_vector_search(store: Store) -> None:
    a, b = await make_scope(store, "a"), await make_scope(store, "b")
    idx = make_indexer(store)
    await idx.index_bytes(
        tenant_id=a.tenant_id, collection_id=a.collection_id, data=HANDBOOK, filename="a.md"
    )
    (q,) = await HashingEmbedder().embed(["vacation days"])
    # tenant B asks, even naming tenant A's collection id: nothing may come back
    leaked = await store.vector_search(SearchFilter(b.tenant_id, (a.collection_id,)), q, k=5)
    assert leaked == []
    assert (
        await store.get_chunks(
            b.tenant_id,
            [
                c.id
                for c in await store.active_chunks(SearchFilter(a.tenant_id, (a.collection_id,)))
            ],
        )
        == []
    )


async def test_collection_and_tag_filters(store: Store) -> None:
    scope = await make_scope(store)
    other = await make_collection(store, scope.tenant_id)
    idx = make_indexer(store)
    await idx.index_bytes(
        tenant_id=scope.tenant_id,
        collection_id=scope.collection_id,
        data=HANDBOOK,
        filename="hb.md",
        tags=["hr"],
    )
    await idx.index_bytes(
        tenant_id=scope.tenant_id,
        collection_id=other,
        data=b"Vacation days are 30 in France.",
        filename="fr.txt",
        tags=["legal"],
    )
    (q,) = await HashingEmbedder().embed(["vacation days"])
    only_other = await store.vector_search(SearchFilter(scope.tenant_id, (other,)), q, k=5)
    assert {h.chunk.metadata["source"] for h in only_other} == {"fr.txt"}
    both = SearchFilter(scope.tenant_id, (scope.collection_id, other), tags=("hr",))
    assert {h.chunk.metadata["source"] for h in await store.vector_search(both, q, k=5)} == {
        "hb.md"
    }


async def test_reupload_identical_bytes_is_noop(store: Store) -> None:
    scope = await make_scope(store)
    idx = make_indexer(store)
    first = await idx.index_bytes(
        tenant_id=scope.tenant_id,
        collection_id=scope.collection_id,
        data=HANDBOOK,
        filename="hb.md",
    )
    again = await idx.index_bytes(
        tenant_id=scope.tenant_id,
        collection_id=scope.collection_id,
        data=HANDBOOK,
        filename="hb.md",
        document_id=first.document.id,
    )
    assert again.created is False and again.version.id == first.version.id
    assert len(await store.list_versions(first.document.id)) == 1


async def test_new_version_flips_atomically_and_old_stays_stored(store: Store) -> None:
    scope = await make_scope(store)
    idx = make_indexer(store)
    v1 = await idx.index_bytes(
        tenant_id=scope.tenant_id,
        collection_id=scope.collection_id,
        data=HANDBOOK,
        filename="hb.md",
    )
    v2_bytes = HANDBOOK.replace(b"20 vacation", b"25 vacation")
    v2 = await idx.index_bytes(
        tenant_id=scope.tenant_id,
        collection_id=scope.collection_id,
        data=v2_bytes,
        filename="hb.md",
        document_id=v1.document.id,
    )
    assert v2.version.version == 2
    active = await store.active_chunks(SearchFilter(scope.tenant_id, (scope.collection_id,)))
    text = " ".join(c.text for c in active)
    assert "25 vacation" in text and "20 vacation" not in text
    old = await store.get_version_chunks(v1.version.id)
    assert any("20 vacation" in c.text for c in old)  # old version still queryable by id


async def test_failed_ingestion_keeps_previous_version_active(store: Store) -> None:
    scope = await make_scope(store)
    idx = make_indexer(store)
    v1 = await idx.index_bytes(
        tenant_id=scope.tenant_id,
        collection_id=scope.collection_id,
        data=HANDBOOK,
        filename="hb.md",
    )
    handle = await idx.create_version(
        tenant_id=scope.tenant_id,
        collection_id=scope.collection_id,
        data=HANDBOOK + b"\n\nMore.",
        filename="hb.md",
        document_id=v1.document.id,
    )

    class BoomError(Exception):
        pass

    async def broken_embed(texts: list[str]) -> list[list[float]]:
        raise BoomError("provider down")

    idx.embedder.embed = broken_embed  # type: ignore[method-assign]
    with pytest.raises(BoomError):
        await idx.process_version(scope.tenant_id, handle.version.id)
    failed = await store.get_version(handle.version.id)
    assert failed is not None and failed.status is VersionStatus.FAILED
    doc = await store.get_document(scope.tenant_id, v1.document.id)
    assert doc is not None and doc.active_version_id == v1.version.id


async def test_soft_deleted_documents_are_not_searchable(store: Store) -> None:
    scope = await make_scope(store)
    idx = make_indexer(store)
    h = await idx.index_bytes(
        tenant_id=scope.tenant_id,
        collection_id=scope.collection_id,
        data=HANDBOOK,
        filename="hb.md",
    )
    await store.soft_delete_document(scope.tenant_id, h.document.id)
    assert await store.active_chunks(SearchFilter(scope.tenant_id, (scope.collection_id,))) == []


def test_filter_requires_a_collection() -> None:
    import uuid

    with pytest.raises(ValueError):
        SearchFilter(uuid.uuid4(), ())


async def test_storage_failure_leaves_no_orphan_rows(store: Store) -> None:
    scope = await make_scope(store)
    idx = make_indexer(store)

    async def broken_put(key: str, data: bytes) -> None:
        raise PermissionError("read-only disk")

    idx.files.put = broken_put  # type: ignore[method-assign]
    with pytest.raises(PermissionError):
        await idx.create_version(
            tenant_id=scope.tenant_id,
            collection_id=scope.collection_id,
            data=HANDBOOK,
            filename="hb.md",
        )
    assert await store.list_documents(scope.tenant_id, scope.collection_id) == []

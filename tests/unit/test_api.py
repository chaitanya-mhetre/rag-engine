from pathlib import Path

from tests.api_helpers import (
    PASSWORD,
    api,
    create_collection,
    login,
    parse_sse,
    signup,
    upload,
)

HANDBOOK = (
    b"# Handbook\n\n## Leave\n\nEmployees get 22 vacation days per year.\n\n"
    b"## Pay\n\nSalaries are paid monthly."
)


async def test_auth_flow(tmp_path: Path) -> None:
    async with api(tmp_path) as (client, _):
        headers = await signup(client, "admin@acme.io")
        me = (await client.get("/v1/me", headers=headers)).json()
        assert me["role"] == "admin" and me["email"] == "admin@acme.io"
        assert (
            await client.post(
                "/v1/auth/login", json={"email": "admin@acme.io", "password": "wrong-password"}
            )
        ).status_code == 401
        assert (await client.get("/v1/me")).status_code == 401
        tokens = (
            await client.post(
                "/v1/auth/login", json={"email": "admin@acme.io", "password": PASSWORD}
            )
        ).json()
        # a refresh token must not work as an access token
        bad = await client.get(
            "/v1/me", headers={"Authorization": f"Bearer {tokens['refresh_token']}"}
        )
        assert bad.status_code == 401
        refreshed = await client.post(
            "/v1/auth/refresh", json={"refresh_token": tokens["refresh_token"]}
        )
        assert refreshed.status_code == 200 and refreshed.json()["access_token"]
        assert (
            await client.post(
                "/v1/auth/signup",
                json={"tenant_name": "x", "email": "admin@acme.io", "password": PASSWORD},
            )
        ).status_code == 409


async def test_upload_process_and_inspect(tmp_path: Path) -> None:
    async with api(tmp_path) as (client, container):
        h = await signup(client, "a@acme.io")
        cid = await create_collection(client, h, "hr")
        up = await upload(client, container, h, cid, "handbook.md", HANDBOOK, tags="hr,policy")
        assert up["created"] and up["version"] == 1
        job = (await client.get(f"/v1/jobs/{up['job_id']}", headers=h)).json()
        assert job["status"] == "ready"
        doc = (await client.get(f"/v1/documents/{up['document_id']}", headers=h)).json()
        assert doc["title"] == "Handbook" and doc["tags"] == ["hr", "policy"]
        assert doc["versions"][0]["chunk_count"] == 2
        chunks = (
            await client.get(f"/v1/documents/{up['document_id']}/versions/1/chunks", headers=h)
        ).json()
        assert chunks[0]["heading_path"] == ["Handbook", "Leave"]
        again = await upload(
            client, container, h, cid, "handbook.md", HANDBOOK, document_id=up["document_id"]
        )
        assert again["created"] is False and again["version_id"] == up["version_id"]


async def test_upload_validation(tmp_path: Path) -> None:
    async with api(tmp_path, max_upload_bytes=100) as (client, _):
        h = await signup(client, "a@acme.io")
        cid = await create_collection(client, h, "hr")
        url = f"/v1/collections/{cid}/documents"
        assert (
            await client.post(url, files={"file": ("a.exe", b"\x00\x01")}, headers=h)
        ).status_code == 415
        assert (
            await client.post(url, files={"file": ("a.txt", b"x" * 200)}, headers=h)
        ).status_code == 413
        assert (
            await client.post(url, files={"file": ("a.txt", b"")}, headers=h)
        ).status_code == 422


async def test_query_answers_with_citations_and_debug_is_admin_only(tmp_path: Path) -> None:
    async with api(tmp_path) as (client, container):
        h = await signup(client, "a@acme.io")
        cid = await create_collection(client, h, "hr")
        await upload(client, container, h, cid, "handbook.md", HANDBOOK)
        r = await client.post(
            "/v1/query",
            json={"question": "How many vacation days?", "collection_ids": [cid], "debug": True},
            headers=h,
        )
        body = r.json()
        assert r.status_code == 200 and "22" in body["answer"]
        assert body["citations"][0]["title"] == "Handbook"
        assert body["retrieval"] and body["usage"]["estimated"] is True

        await client.post("/v1/users", json={"email": "m@acme.io", "password": PASSWORD}, headers=h)
        m = await login(client, "m@acme.io")
        members = {
            "user_id": (await client.get("/v1/me", headers=m)).json()["id"],
            "permission": "read",
        }
        assert (
            await client.post(f"/v1/collections/{cid}/members", json=members, headers=h)
        ).status_code == 204
        r2 = await client.post(
            "/v1/query",
            json={"question": "vacation days", "collection_ids": [cid], "debug": True},
            headers=m,
        )
        assert r2.status_code == 200 and r2.json()["retrieval"] is None


async def test_permissions(tmp_path: Path) -> None:
    async with api(tmp_path) as (client, container):
        admin = await signup(client, "a@acme.io")
        cid = await create_collection(client, admin, "hr")
        await upload(client, container, admin, cid, "handbook.md", HANDBOOK)
        await client.post(
            "/v1/users", json={"email": "m@acme.io", "password": PASSWORD}, headers=admin
        )
        await client.post(
            "/v1/users",
            json={"email": "v@acme.io", "password": PASSWORD, "role": "viewer"},
            headers=admin,
        )
        member, viewer = await login(client, "m@acme.io"), await login(client, "v@acme.io")
        q = {"question": "vacation", "collection_ids": [cid]}
        # no membership → the collection is invisible
        assert (await client.post("/v1/query", json=q, headers=member)).status_code == 404
        assert (await client.get("/v1/collections", headers=member)).json() == []
        member_id = (await client.get("/v1/me", headers=member)).json()["id"]
        viewer_id = (await client.get("/v1/me", headers=viewer)).json()["id"]
        await client.post(
            f"/v1/collections/{cid}/members",
            json={"user_id": member_id, "permission": "read"},
            headers=admin,
        )
        await client.post(
            f"/v1/collections/{cid}/members",
            json={"user_id": viewer_id, "permission": "admin"},
            headers=admin,
        )
        assert (await client.post("/v1/query", json=q, headers=member)).status_code == 200
        # read permission cannot upload; a viewer is capped at read even with an admin row
        for h in (member, viewer):
            r = await client.post(
                f"/v1/collections/{cid}/documents", files={"file": ("x.txt", b"hello")}, headers=h
            )
            assert r.status_code == 403
        assert (
            await client.post("/v1/collections", json={"name": "v"}, headers=viewer)
        ).status_code == 403
        assert (
            await client.post(
                "/v1/users", json={"email": "z@acme.io", "password": PASSWORD}, headers=member
            )
        ).status_code == 403


async def test_tenant_isolation(tmp_path: Path) -> None:
    async with api(tmp_path) as (client, container):
        a = await signup(client, "a@acme.io", "acme")
        b = await signup(client, "b@globex.io", "globex")
        cid_a = await create_collection(client, a, "hr")
        up = await upload(client, container, a, cid_a, "handbook.md", HANDBOOK)
        q = {"question": "How many vacation days?", "collection_ids": [cid_a]}
        assert (await client.post("/v1/query", json=q, headers=b)).status_code == 404
        assert (
            await client.get(f"/v1/documents/{up['document_id']}", headers=b)
        ).status_code == 404
        assert (await client.get(f"/v1/jobs/{up['job_id']}", headers=b)).status_code == 404
        # same collection name in another tenant is fine, and returns nothing of tenant A's
        cid_b = await create_collection(client, b, "hr")
        r = await client.post(
            "/v1/query", json={"question": "vacation days", "collection_ids": [cid_b]}, headers=b
        )
        assert r.json()["insufficient_context"] is True and r.json()["citations"] == []


async def test_sse_stream(tmp_path: Path) -> None:
    async with api(tmp_path) as (client, container):
        h = await signup(client, "a@acme.io")
        cid = await create_collection(client, h, "hr")
        await upload(client, container, h, cid, "handbook.md", HANDBOOK)
        r = await client.post(
            "/v1/query",
            json={"question": "vacation days per year", "collection_ids": [cid], "debug": True},
            headers={**h, "Accept": "text/event-stream"},
        )
        assert r.headers["content-type"].startswith("text/event-stream")
        events = parse_sse(r.text)
        names = [n for n, _ in events]
        assert names[0] == "retrieval" and names[-1] == "final" and "token" in names
        assert "22" in events[-1][1]["answer"]


async def test_conversation_follow_up(tmp_path: Path) -> None:
    async with api(tmp_path) as (client, container):
        h = await signup(client, "a@acme.io")
        cid = await create_collection(client, h, "hr")
        await upload(client, container, h, cid, "handbook.md", HANDBOOK)
        convo = (
            await client.post("/v1/conversations", json={"collection_ids": [cid]}, headers=h)
        ).json()
        base = {"collection_ids": [cid], "conversation_id": convo["id"]}
        await client.post(
            "/v1/query",
            json={**base, "question": "How many vacation days do employees get?"},
            headers=h,
        )
        await client.post("/v1/query", json={**base, "question": "and per year?"}, headers=h)
        msgs = (await client.get(f"/v1/conversations/{convo['id']}/messages", headers=h)).json()
        assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
        last_log = container.access.query_logs[-1]  # type: ignore[attr-defined]
        assert "vacation" in last_log.rewritten_query


async def test_rate_limit(tmp_path: Path) -> None:
    async with api(tmp_path, rate_limit_capacity=2, rate_limit_per_second=0.001) as (client, _):
        h = await signup(client, "a@acme.io")
        cid = await create_collection(client, h, "hr")
        q = {"question": "x", "collection_ids": [cid]}
        codes = [(await client.post("/v1/query", json=q, headers=h)).status_code for _ in range(3)]
        assert codes == [200, 200, 429]


async def test_new_version_and_delete(tmp_path: Path) -> None:
    async with api(tmp_path) as (client, container):
        h = await signup(client, "a@acme.io")
        cid = await create_collection(client, h, "hr")
        v1 = await upload(client, container, h, cid, "handbook.md", HANDBOOK)
        v2 = await upload(
            client,
            container,
            h,
            cid,
            "handbook.md",
            HANDBOOK.replace(b"22", b"25"),
            document_id=v1["document_id"],
        )
        assert v2["version"] == 2
        q = {"question": "How many vacation days do employees get?", "collection_ids": [cid]}
        assert "25" in (await client.post("/v1/query", json=q, headers=h)).json()["answer"]
        assert (
            await client.delete(f"/v1/documents/{v1['document_id']}", headers=h)
        ).status_code == 204
        assert (await client.post("/v1/query", json=q, headers=h)).json()[
            "insufficient_context"
        ] is True


async def test_health(tmp_path: Path) -> None:
    async with api(tmp_path) as (client, _):
        assert (await client.get("/v1/health")).json() == {"status": "ok"}
        assert (await client.get("/v1/ready")).json() == {"status": "ready"}

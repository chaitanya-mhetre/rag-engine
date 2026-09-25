"""Minimal Python client for the RAG Engine API (httpx). Run the docker stack first.

    uv run python examples/client.py
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import httpx

BASE = "http://localhost:58000/v1"
CORPUS = Path(__file__).parent.parent / "evaluation/datasets/corpus/kestrel"


def main() -> None:
    with httpx.Client(base_url=BASE, timeout=30) as client:
        tokens = client.post(
            "/auth/signup",
            json={
                "tenant_name": "demo",
                "email": f"py-{uuid.uuid4().hex[:6]}@example.com",
                "password": "correct-horse-battery",
            },
        ).json()
        client.headers["authorization"] = f"Bearer {tokens['access_token']}"
        cid = client.post("/collections", json={"name": "kestrel"}).json()["id"]

        jobs = []
        for path in sorted(CORPUS.iterdir()):
            r = client.post(f"/collections/{cid}/documents", files={"file": (path.name, path.read_bytes())})
            jobs.append(r.json()["job_id"])
        while any(client.get(f"/jobs/{j}").json()["status"] not in ("ready", "failed") for j in jobs):
            time.sleep(0.5)

        answer = client.post(
            "/query", json={"question": "How many vacation days do employees get?", "collection_ids": [cid]}
        ).json()
        print(answer["answer"])
        for c in answer["citations"]:
            print(f"  [{c['source_label']}] {c['title']} > {c['section']}")

        # Streaming: read Server-Sent Events line by line
        with client.stream(
            "POST",
            "/query",
            json={"question": "What is the travel insurance policy number?", "collection_ids": [cid]},
            headers={"accept": "text/event-stream"},
        ) as stream:
            event = ""
            for line in stream.iter_lines():
                if line.startswith("event: "):
                    event = line[7:]
                elif line.startswith("data: ") and event == "token":
                    print(json.loads(line[6:])["text"], end="", flush=True)
        print()


if __name__ == "__main__":
    main()

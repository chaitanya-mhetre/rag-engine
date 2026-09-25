# Contributing

1. `uv sync --extra llamaindex` and `make up && make migrate` (Postgres :55433, Redis :56380).
2. Branch from `main`. Keep commits small and use conventional messages (`feat:`, `fix:`, `test:`, `docs:`).
3. `make check` must pass (ruff, ruff format, mypy --strict, pytest).
4. If you change retrieval, chunking, prompts or scoring, run `make eval-check`. If a metric *should* change, commit
   a new baseline report and explain why in the PR.
5. Never commit secrets, and never put real-model numbers in docs unless the report that produced them is committed
   too.

Adding a provider: implement the `Embedder`, `LLM` or `Reranker` protocol, wire it in `factory.py`, and mark any
network tests with `@pytest.mark.live`.

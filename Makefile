.PHONY: check lint type test fmt up down eval migrate eval-check serve
check: lint type test
lint:
	uv run ruff check .
	uv run ruff format --check .
type:
	uv run mypy
test:
	uv run pytest -q
fmt:
	uv run ruff format . && uv run ruff check --fix .
up:
	docker compose up -d postgres redis
down:
	docker compose down
eval:
	uv run rag-eval run evaluation/datasets/kestrel_v1.json --out evaluation/reports
migrate:
	uv run alembic upgrade head
eval-check:
	uv run rag-eval run evaluation/datasets/kestrel_v1.json --configs keyword,vector,hybrid_rrf,hybrid_weighted,hybrid_rrf_rerank --out /tmp/rag-eval --name current > /dev/null
	uv run rag-eval compare evaluation/reports/m4_retrieval_baseline.json /tmp/rag-eval/current.json --fail-on-regression 0.02
serve:
	uv run uvicorn ragengine.api.app:create_app --factory --reload --port 8000

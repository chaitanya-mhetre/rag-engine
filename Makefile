.PHONY: check lint type test test-all fmt up down eval
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
	uv run rag-eval run evaluation/datasets/handbook_v1.json --out evaluation/reports

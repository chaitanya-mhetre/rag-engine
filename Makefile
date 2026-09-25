.PHONY: check lint type test fmt up down eval migrate
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
migrate:
	uv run alembic upgrade head

.PHONY: install test check fmt up down serve
install:
	uv venv && uv pip install -e ".[dev]"
test:
	uv run pytest -q
check:
	uv run ruff check src tests && uv run ruff format --check src tests
fmt:
	uv run ruff format src tests && uv run ruff check --fix src tests
up:
	docker compose up -d
down:
	docker compose down
serve:
	uv run uvicorn frag.api.main:app --port 8000

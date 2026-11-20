.PHONY: install test check fmt up down serve
install:
	uv venv --python 3.11 && uv pip install -e ".[dev]"
test:
	uv run pytest -q
check:
	uv run ruff check src tests scripts && uv run ruff format --check src tests scripts
fmt:
	uv run ruff format src tests scripts && uv run ruff check --fix src tests scripts
up:
	docker compose up -d
down:
	docker compose down
serve:
	uv run uvicorn frag.api.main:app --port 8000

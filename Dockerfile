# Multi-stage build: resolve deps with uv, run a slim non-root image.
FROM python:3.11-slim AS build
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
WORKDIR /app
ENV UV_SYSTEM_PYTHON=1
COPY pyproject.toml README.md ./
COPY src ./src
RUN uv pip install --system --no-cache .

FROM python:3.11-slim
WORKDIR /app
RUN useradd -m app
COPY --from=build /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=build /usr/local/bin /usr/local/bin
COPY src ./src
ENV PYTHONPATH=/app/src
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "frag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

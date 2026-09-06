"""FastAPI surface over the graph: ask, liveness, readiness.

The graph is built lazily and reused. Liveness is process health; readiness also
checks the vector store, so an orchestrator holds traffic until retrieval is up.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from frag.api.ui import PAGE
from frag.graph.build import build_graph
from frag.graph.deps import Deps
from frag.llm import observability
from frag.utils.logging import get_logger
from frag.utils.settings import settings

logger = get_logger(__name__)

_deps = Deps()
_graph: Any = None


def _get_graph() -> Any:
    global _graph
    if _graph is None:
        _graph = build_graph(_deps)
    return _graph


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Move one-time costs (model load, cold encode, BM25 index build) to boot,
    # off the first query, by running a real search once.
    _get_graph()
    store = _deps.get_store()
    store.search("warmup", top_k=1)
    logger.info("warmup complete", documents=store.count())
    yield


app = FastAPI(title="Financial RAG (LangGraph)", version="0.1.0", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return PAGE


class AskRequest(BaseModel):
    question: str
    history: list[dict[str, str]] = Field(default_factory=list)


class AskResponse(BaseModel):
    status: str
    answer: str
    citations: list[str] = Field(default_factory=list)
    route: str | None = None
    critic_score: float | None = None
    grounding: dict[str, Any] = Field(default_factory=dict)


@app.post("/v1/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    state = {"question": req.question, "history": req.history, "rewrites": 0}
    result = _get_graph().invoke(state, config=observability.config())
    observability.flush()
    logger.info("answered", route=result.get("route"), status=result.get("status"))
    return AskResponse(
        status=result.get("status", "abstained"),
        answer=result.get("answer", ""),
        citations=result.get("citations", []),
        route=result.get("route"),
        critic_score=result.get("critic_score"),
        grounding=result.get("grounding", {}),
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, Any]:
    """Ready only when the vector store answers. Returns 200 with a flag either way."""
    try:
        count = _deps.get_store().count()
        return {"ready": True, "documents": count, "tracing": settings.tracing_enabled}
    except Exception as exc:
        logger.warning("readiness check failed", error=str(exc))
        return {"ready": False, "error": str(exc)}

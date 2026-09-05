"""FastAPI surface over the graph: ask, liveness, readiness.

The graph is built lazily and reused. Liveness is process health; readiness also
checks the vector store, so an orchestrator holds traffic until retrieval is up.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from frag.graph.build import build_graph
from frag.llm import observability
from frag.utils.logging import get_logger
from frag.utils.settings import settings

logger = get_logger(__name__)
app = FastAPI(title="Financial RAG (LangGraph)", version="0.1.0")

_graph: Any = None


def _get_graph() -> Any:
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


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
        from frag.rag.store import QdrantStore

        count = QdrantStore().count()
        return {"ready": True, "documents": count, "tracing": settings.tracing_enabled}
    except Exception as exc:
        logger.warning("readiness check failed", error=str(exc))
        return {"ready": False, "error": str(exc)}

"""Langfuse observability: a callback handler for graph runs and score reporting.

Tracing is a no-op unless both Langfuse keys are set, so tests and keyless runs
work unchanged. `callbacks()` goes into a graph's invoke config; `score()` attaches
a metric (e.g. a deterministic IR number) to a trace.
"""

from __future__ import annotations

from typing import Any

from frag.utils.logging import get_logger
from frag.utils.settings import settings

logger = get_logger(__name__)

_handler: Any | None = None
_client: Any | None = None


def _init() -> None:
    global _handler, _client
    if _handler is not None or not settings.tracing_enabled:
        return
    from langfuse import Langfuse
    from langfuse.callback import CallbackHandler

    kwargs = {
        "public_key": settings.langfuse_public_key,
        "secret_key": settings.langfuse_secret_key,
        "host": settings.langfuse_host,
    }
    _client = Langfuse(**kwargs)
    _handler = CallbackHandler(**kwargs)


def callbacks() -> list[Any]:
    """Callbacks for a graph invoke config; empty when tracing is disabled."""
    _init()
    return [_handler] if _handler is not None else []


def config(**extra: Any) -> dict[str, Any]:
    """A LangGraph invoke config carrying the Langfuse callbacks."""
    cfg: dict[str, Any] = {"callbacks": callbacks()}
    cfg.update(extra)
    return cfg


def score(name: str, value: float, *, trace_id: str | None = None, comment: str = "") -> None:
    """Attach a numeric score to a trace (used to log deterministic eval metrics)."""
    _init()
    if _client is None:
        return
    try:
        _client.score(name=name, value=value, trace_id=trace_id, comment=comment)
    except Exception as exc:  # observability must never break the request path
        logger.warning("langfuse score failed", error=str(exc))


def flush() -> None:
    if _client is not None:
        _client.flush()

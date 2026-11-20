"""Agent branch: a prebuilt ReAct tool-loop that gathers evidence for multi-hop questions.

The agent decides which passages to retrieve and which calculations to run; it does
not write the user-facing answer. Its final message is free text from an open model
and has been observed to be a tool call written out as prose (e.g.
`retrieve("AMD cash flow FY22")`) rather than an answer. So the node keeps only what
the tools actually returned — passages, graph facts, calculation results — as
structured contexts, and the graph sends them through the same actor -> critic ->
ground -> screen_output path as a lookup. Every route then passes the same gates:
strict output schema, critic, grounding and citation screening.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from frag.graph.deps import Deps
from frag.graph.state import GraphState
from frag.rag import prompts
from frag.utils.logging import get_logger
from frag.utils.settings import settings

logger = get_logger(__name__)


def _build_agent(deps: Deps, sink: list[dict[str, Any]]) -> Any:
    from langgraph.prebuilt import create_react_agent

    from frag.graph.tools import make_tools
    from frag.llm.client import chat_model

    tools = make_tools(deps.get_store(), sink=sink)
    system = prompts.get("agent").body
    return create_react_agent(chat_model("agent"), tools, state_modifier=system)


def _dedupe(contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop repeats of the same passage; keyed on label *and* text so two distinct
    passages that share a fallback label are both kept."""
    seen: set[tuple[str, str]] = set()
    out = []
    for c in contexts:
        key = (c["metadata"]["doc_id"], c.get("text", ""))
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def has_retrieved_evidence(contexts: list[dict[str, Any]]) -> bool:
    """True if any context came from retrieval or the graph, not only from calc."""
    return any((c.get("metadata") or {}).get("source") != "calc" for c in contexts)


def agent_node(deps: Deps) -> Callable[[GraphState], dict]:
    def run(state: GraphState) -> dict:
        # A fresh sink and agent per run: no evidence leaks between concurrent requests.
        sink: list[dict[str, Any]] = []
        agent = _build_agent(deps, sink)
        try:
            agent.invoke(
                {"messages": [("user", state["standalone"])]},
                config={"recursion_limit": settings.agent_max_turns * 2},
            )
        except Exception as exc:  # budget exhausted, provider error, tool failure
            logger.warning(
                "agent loop failed; using evidence gathered so far",
                error_type=type(exc).__name__,
                error=str(exc)[:200],
                gathered=len(sink),
            )
        contexts = _dedupe(sink)
        logger.info("agent gathered evidence", contexts=len(contexts))
        # The agent's prose is deliberately discarded: the actor answers from `contexts`.
        return {"contexts": contexts, "answer": "", "citations": []}

    return run

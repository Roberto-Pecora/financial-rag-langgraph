"""Agent branch: a prebuilt ReAct tool-loop for multi-hop questions.

create_react_agent gives the loop (model decides: call a tool or answer) with
LangGraph's budgets and checkpointing; our tools carry the untrusted-wrapping and
whitelisted calc. The node maps the agent's messages back into GraphState.
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


def _build_agent(deps: Deps) -> Any:
    from langgraph.prebuilt import create_react_agent

    from frag.graph.tools import make_tools
    from frag.llm.client import chat_model

    tools = make_tools(deps.get_store())
    system = prompts.get("agent").body
    return create_react_agent(chat_model("agent"), tools, state_modifier=system)


def agent_node(deps: Deps) -> Callable[[GraphState], dict]:
    agent = {"g": None}  # built lazily on first real call

    def run(state: GraphState) -> dict:
        if agent["g"] is None:
            agent["g"] = _build_agent(deps)
        result = agent["g"].invoke(
            {"messages": [("user", state["standalone"])]},
            config={"recursion_limit": settings.agent_max_turns * 2},
        )
        messages = result.get("messages", [])
        answer = messages[-1].content if messages else ""
        # Evidence for grounding: any tool (retrieve) outputs the loop saw.
        contexts = [
            {"text": m.content, "metadata": {"doc_id": f"tool-{i}"}}
            for i, m in enumerate(messages)
            if getattr(m, "type", "") == "tool"
        ]
        status = "accepted" if answer.strip() else "abstained"
        return {"answer": answer, "citations": [], "contexts": contexts, "status": status}

    return run

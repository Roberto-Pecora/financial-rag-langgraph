"""Agent branch: a prebuilt ReAct tool-loop for multi-hop questions.

create_react_agent gives the loop (model decides: call a tool or answer) with
LangGraph's budgets and checkpointing; our tools carry the untrusted-wrapping and
whitelisted calc. The node maps the agent's messages back into GraphState.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from frag.graph.deps import Deps
from frag.graph.state import GraphState
from frag.rag import prompts
from frag.utils.logging import get_logger
from frag.utils.settings import settings

logger = get_logger(__name__)

# The tools wrap passages as <<<UNTRUSTED_DOC label>>>; the model sometimes echoes
# the label (raw or paraphrased) into its prose. Lift those labels into citations
# and strip the wrapper text so it never surfaces in the user-facing answer.
_LABEL = r"[A-Za-z0-9:_.\-]+"
_MENTION = re.compile(
    rf"<<<\s*UNTRUSTED_DOC\s+({_LABEL})[^>]*>>>|\(?\s*UNTRUSTED_DOC\s+({_LABEL})\s*\)?"
)


def _clean_agent_answer(text: str) -> tuple[str, list[str]]:
    text = text.replace("<<<END_UNTRUSTED_DOC>>>", "")
    labels: list[str] = []
    for m in _MENTION.finditer(text):
        label = m.group(1) or m.group(2)
        if label and label not in labels:
            labels.append(label)
    clean = _MENTION.sub("", text)
    clean = re.sub(r"[ \t]{2,}", " ", clean)
    clean = re.sub(r"\n{3,}", "\n\n", clean).strip()
    return clean, labels


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
        raw = messages[-1].content if messages else ""
        answer, citations = _clean_agent_answer(raw)
        # Evidence for grounding: any tool (retrieve) outputs the loop saw.
        contexts = [
            {"text": m.content, "metadata": {"doc_id": f"tool-{i}"}}
            for i, m in enumerate(messages)
            if getattr(m, "type", "") == "tool"
        ]
        status = "accepted" if answer.strip() else "abstained"
        return {
            "answer": answer,
            "citations": citations,
            "contexts": contexts,
            "status": status,
        }

    return run

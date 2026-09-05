"""Assemble the LangGraph state graph and expose a run() entry point.

screen_input ─refused─▶ END
     │ ok
  rewrite ─▶ route ─┬─ multi_hop ─▶ agent ─────────────┐
                    │                                    │
                    └─ lookup ─▶ entity_filter ─▶ retrieve
                                                    │
                                 rewrite_query ◀─ grade ─(enough)─▶ rerank ─▶ actor
                                      │                                          │
                                      └────────▶ retrieve                 (CRITIC?)
                                                                                 │
                                    ground ◀── critic ◀──────────────────────────┘
                                      │
                                screen_output ─▶ END
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from frag.graph import nodes
from frag.graph.agent import agent_node
from frag.graph.deps import Deps
from frag.graph.state import GraphState
from frag.llm import observability


def build_graph(deps: Deps | None = None) -> Any:
    """Compile the graph. Pass a Deps with fakes for hermetic tests."""
    deps = deps or Deps()
    g = StateGraph(GraphState)

    g.add_node("screen_input", nodes.screen_input_node(deps))
    g.add_node("rewrite", nodes.rewrite_node(deps))
    g.add_node("router", nodes.route_node(deps))
    g.add_node("entity_filter", nodes.entity_filter_node(deps))
    g.add_node("retrieve", nodes.retrieve_node(deps))
    g.add_node("grade", nodes.grade_node(deps))
    g.add_node("rewrite_query", nodes.rewrite_query_node(deps))
    g.add_node("rerank", nodes.rerank_node(deps))
    g.add_node("actor", nodes.actor_node(deps))
    g.add_node("critic", nodes.critic_node(deps))
    g.add_node("agent", agent_node(deps))
    g.add_node("ground", nodes.ground_node(deps))
    g.add_node("screen_output", nodes.screen_output_node(deps))

    g.add_edge(START, "screen_input")
    g.add_conditional_edges("screen_input", nodes.input_edge, {"refused": END, "ok": "rewrite"})
    g.add_edge("rewrite", "router")
    g.add_conditional_edges(
        "router", nodes.route_edge, {"agent": "agent", "entity_filter": "entity_filter"}
    )
    g.add_edge("entity_filter", "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges(
        "grade", nodes.grade_edge, {"rewrite_query": "rewrite_query", "rerank": "rerank"}
    )
    g.add_edge("rewrite_query", "retrieve")
    g.add_edge("rerank", "actor")
    g.add_conditional_edges("actor", nodes.critic_edge, {"critic": "critic", "ground": "ground"})
    g.add_edge("critic", "ground")
    g.add_edge("agent", "ground")
    g.add_edge("ground", "screen_output")
    g.add_edge("screen_output", END)

    return g.compile()


def run(
    question: str, history: list[dict[str, str]] | None = None, deps: Deps | None = None
) -> dict:
    """Answer a question through the compiled graph, traced to Langfuse if configured."""
    graph = build_graph(deps)
    state = {"question": question, "history": history or [], "rewrites": 0}
    result = graph.invoke(state, config=observability.config())
    observability.flush()
    return result

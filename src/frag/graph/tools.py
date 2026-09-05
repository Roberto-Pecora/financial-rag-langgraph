"""LangChain tools for the agent subgraph: retrieval, graph lookup, safe arithmetic.

Retrieval and graph results are wrapped as untrusted data so a returned passage
cannot smuggle instructions into the model. Arithmetic is evaluated over a
whitelisted AST — no eval, no names, no calls.
"""

from __future__ import annotations

import ast
import operator
from typing import Any

from langchain_core.tools import StructuredTool

from frag.rag.untrusted import wrap_untrusted

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_CMP_OPS = {
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}


def _eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_node(node.operand)
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _CMP_OPS:
        return _CMP_OPS[type(node.ops[0])](_eval_node(node.left), _eval_node(node.comparators[0]))
    raise ValueError("unsupported expression")


def safe_calc(expression: str) -> str:
    """Evaluate a whitelisted arithmetic/comparison expression over numbers."""
    return str(_eval_node(ast.parse(expression, mode="eval").body))


def _doc_id(ctx: dict, idx: int) -> str:
    meta = ctx.get("metadata") or {}
    return meta.get("doc_id") or meta.get("id") or f"doc-{idx + 1}"


def make_tools(store: Any, graph: Any = None) -> list[StructuredTool]:
    """Build the agent's tools bound to a store (and optional knowledge graph)."""

    def retrieve(query: str) -> str:
        hits = store.search(query=query, top_k=5)
        if not hits:
            return "No passages found."
        return "\n\n".join(
            wrap_untrusted(_doc_id(h, i), h.get("text", "")) for i, h in enumerate(hits)
        )

    def financial_calc(expression: str) -> str:
        try:
            return safe_calc(expression)
        except (ValueError, SyntaxError, ZeroDivisionError) as exc:
            return f"calc error: {exc}"

    tools = [
        StructuredTool.from_function(
            retrieve,
            name="retrieve",
            description="Retrieve passages from filings and contracts for a query.",
        ),
        StructuredTool.from_function(
            financial_calc,
            name="financial_calc",
            description="Evaluate arithmetic/comparisons over numbers you have found, "
            "e.g. '1200 / 350' or '4.1 > 3.5'.",
        ),
    ]

    if graph is not None:

        def graph_lookup(entity: str) -> str:
            facts = graph.neighbours(entity) if hasattr(graph, "neighbours") else []
            if not facts:
                return f"No graph facts for {entity!r}."
            return wrap_untrusted(f"graph:{entity}", "; ".join(str(f) for f in facts))

        tools.append(
            StructuredTool.from_function(
                graph_lookup,
                name="graph_lookup",
                description="Look up structured credit facts (issuers, instruments, "
                "covenants) for a named entity.",
            )
        )

    return tools

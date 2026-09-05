"""Route a question: single-fact lookup -> RAG path; multi-hop -> the agent.

Deterministic by default (no key); a structured LLM classifier is used when
ROUTER_LLM is on and a router model is supplied.
"""

from __future__ import annotations

import re
from typing import Any

# Signals a question needs composition (retrieve + graph + compare), not one lookup.
_MULTIHOP_PATTERNS = [
    r"\b(and|both)\b.*\b(covenant|leverage|rating|revenue|debt)\b",
    r"\b(compare|versus|vs\.?|difference between)\b",
    r"\b(more than|greater than|less than|above|below|over|under)\s+[\d£$]",
    r"\bwhich .*\bhave\b.*\band\b",
    r"\b(highest|lowest|most|least)\b",
]


def classify(question: str) -> str:
    low = question.lower()
    if any(re.search(p, low) for p in _MULTIHOP_PATTERNS):
        return "multi_hop"
    return "lookup"


def route(question: str, llm: Any | None = None) -> str:
    """Deterministic route, or a structured LLM classifier when one is supplied."""
    if llm is not None:
        prompt = (
            "Classify the question as 'lookup' (one fact) or 'multi_hop' (needs "
            f"several steps or a comparison).\n\nQuestion: {question}"
        )
        try:
            decision = llm.invoke(prompt).route
            return "multi_hop" if "multi" in decision.lower() else "lookup"
        except Exception:
            pass
    return classify(question)

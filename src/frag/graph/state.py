"""The graph's shared state. Nodes read what they need and return partial updates."""

from __future__ import annotations

from typing import Any, TypedDict


class GraphState(TypedDict, total=False):
    # inputs
    question: str
    history: list[dict[str, str]]

    # routing / rewriting
    standalone: str
    route: str  # "lookup" | "multi_hop"

    # retrieval
    metadata_filter: dict[str, Any] | None
    contexts: list[dict[str, Any]]
    graded: list[dict[str, Any]]
    rewrites: int

    # answer + gates
    answer: str
    citations: list[str]
    critic_score: float | None
    critic_notes: str
    grounding: dict[str, Any]

    # outcome
    status: str  # "accepted" | "abstained" | "refused"
    trace: list[dict[str, Any]]

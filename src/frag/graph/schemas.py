"""Typed LLM outputs for the graph's routing and corrective-retrieval nodes."""

from __future__ import annotations

from pydantic import BaseModel


class DocGrade(BaseModel):
    relevant: bool = False


class RewriteOut(BaseModel):
    query: str = ""


class RouteDecision(BaseModel):
    route: str = "lookup"  # "lookup" | "multi_hop"

"""Dependency container for the graph.

Holds the store, reranker, and role LLMs so nodes stay thin and tests inject fakes.
Real backends are built lazily, so importing the graph needs no Qdrant or API key.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from frag.utils.settings import settings


@dataclass
class Deps:
    store: Any = None
    reranker: Any = None
    actor_llm: Any = None
    critic_llm: Any = None
    grader_llm: Any = None
    router_llm: Any = None
    rewrite_llm: Any = None

    def get_store(self) -> Any:
        if self.store is None:
            from frag.rag.store import QdrantStore

            self.store = QdrantStore()
        return self.store

    def get_reranker(self) -> Any:
        if self.reranker is None:
            from frag.rag.reranker import CrossEncoderReranker

            self.reranker = CrossEncoderReranker()
        return self.reranker

    def _structured(self, role: str, model: type) -> Any:
        from frag.llm.client import structured_model

        return structured_model(role, model)

    def get_actor(self) -> Any:
        from frag.rag.llm_schemas import ActorResponse

        if self.actor_llm is None:
            self.actor_llm = self._structured("actor", ActorResponse)
        return self.actor_llm

    def get_critic(self) -> Any:
        from frag.rag.llm_schemas import CriticResponse

        if self.critic_llm is None:
            self.critic_llm = self._structured("critic", CriticResponse)
        return self.critic_llm

    def get_grader(self) -> Any:
        from frag.graph.schemas import DocGrade

        if self.grader_llm is None:
            self.grader_llm = self._structured("grader", DocGrade)
        return self.grader_llm

    def get_rewriter(self) -> Any:
        from frag.graph.schemas import RewriteOut

        if self.rewrite_llm is None:
            self.rewrite_llm = self._structured("grader", RewriteOut)
        return self.rewrite_llm

    def get_router(self) -> Any:
        from frag.graph.schemas import RouteDecision

        if self.router_llm is None and settings.router_llm:
            self.router_llm = self._structured("grader", RouteDecision)
        return self.router_llm

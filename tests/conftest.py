"""Hermetic fakes: no Qdrant, no API key, no network."""

from __future__ import annotations

import pytest

from frag.graph.deps import Deps
from frag.rag.llm_schemas import ActorResponse, CriticResponse
from frag.utils.settings import settings


@pytest.fixture(autouse=True)
def _no_tracing(monkeypatch):
    """Force tracing off so tests stay hermetic regardless of a local .env."""
    monkeypatch.setattr(settings, "langfuse_public_key", None)
    monkeypatch.setattr(settings, "langfuse_secret_key", None)


class FakeStore:
    def __init__(self, contexts=None, companies=None):
        self._contexts = contexts or [
            {
                "text": "Amazon net sales were 5,678.",
                "metadata": {"doc_id": "d1", "company": "Amazon"},
            },
        ]
        self._companies = companies or ["Amazon"]

    def search(self, query, top_k=8, metadata_filter=None):
        return list(self._contexts)

    def list_companies(self):
        return list(self._companies)

    def count(self):
        return len(self._contexts)


class FakeLLM:
    """Returns a fixed structured object from .invoke()."""

    def __init__(self, obj):
        self._obj = obj

    def invoke(self, _prompt):
        return self._obj


@pytest.fixture
def deps():
    return Deps(
        store=FakeStore(),
        actor_llm=FakeLLM(ActorResponse(answer="Net sales were 5,678.", citations=["d1"])),
        critic_llm=FakeLLM(
            CriticResponse(
                overall_score=0.9,
                faithfulness_score=0.9,
                completeness_score=0.9,
                citation_score=0.9,
            )
        ),
    )

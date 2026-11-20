"""Agent route: the agent gathers evidence; the actor writes every answer.

Regression for a live failure: with open models the agent's final message was
sometimes a tool call written out as text (`retrieve("AMD cash flow FY22")`), which
was returned to the user as an accepted answer with no critic check.
"""

from __future__ import annotations

from frag.graph import agent as agent_mod
from frag.graph.agent import _dedupe
from frag.graph.build import run
from frag.graph.tools import make_tools
from frag.rag.llm_schemas import CriticResponse
from frag.utils.settings import settings
from tests.conftest import FakeLLM, FakeStore

MULTI_HOP = "Compare Amazon and Nike revenue"  # heuristic router -> multi_hop
LEAKED = 'retrieve("Amazon net sales FY22")'


class FakeAgent:
    """Stands in for the ReAct loop: optionally calls the real tools, then 'answers'."""

    def __init__(self, tools, *, retrieve=True, calc=None, final=LEAKED, raise_after=None):
        self.tools = {t.name: t for t in tools}
        self.retrieve, self.calc, self.final, self.raise_after = retrieve, calc, final, raise_after

    def invoke(self, _inputs, config=None):
        if self.retrieve:
            self.tools["retrieve"].invoke({"query": "Amazon net sales"})
        if self.calc:
            self.tools["financial_calc"].invoke({"expression": self.calc})
        if self.raise_after:
            raise self.raise_after
        return {"messages": [type("Msg", (), {"content": self.final, "type": "ai"})()]}


def _patch_agent(monkeypatch, **kw):
    def build(deps, sink):
        return FakeAgent(make_tools(deps.get_store(), sink=sink), **kw)

    monkeypatch.setattr(agent_mod, "_build_agent", build)


# --- tools record structured evidence --------------------------------------


def test_tools_record_passages_and_successful_calcs_only():
    sink = []
    tools = {t.name: t for t in make_tools(FakeStore(), sink=sink)}
    tools["retrieve"].invoke({"query": "net sales"})
    tools["financial_calc"].invoke({"expression": "10 / 4"})
    tools["financial_calc"].invoke({"expression": "1 / 0"})  # error: not evidence
    assert [c["metadata"]["doc_id"] for c in sink] == ["d1", "calc-1"]
    assert sink[1]["text"] == "10 / 4 = 2.5"


def test_calc_labels_are_unique_and_dedupe_keeps_distinct_passages():
    sink = []
    tools = {t.name: t for t in make_tools(FakeStore(), sink=sink)}
    for expr in ("1 + 1", "2 + 2", "3 + 3"):
        tools["financial_calc"].invoke({"expression": expr})
    tools["retrieve"].invoke({"query": "a"})
    tools["retrieve"].invoke({"query": "b"})  # same passage again
    ctx = _dedupe(sink + [{"text": "other text", "metadata": {"doc_id": "d1"}}])
    assert [c["metadata"]["doc_id"] for c in ctx] == ["calc-1", "calc-2", "calc-3", "d1", "d1"]


def test_tools_without_sink_behave_as_before():
    tools = {t.name: t for t in make_tools(FakeStore())}
    assert "UNTRUSTED_DOC" in tools["retrieve"].invoke({"query": "x"})
    assert tools["financial_calc"].invoke({"expression": "2 * 3"}) == "6"


# --- graph behaviour -------------------------------------------------------


def test_leaked_tool_call_never_reaches_the_user(deps, monkeypatch):
    _patch_agent(monkeypatch)
    out = run(MULTI_HOP, deps=deps)
    assert out["route"] == "multi_hop"
    assert out["answer"] == "Net sales were 5,678."  # written by the actor
    assert LEAKED not in out["answer"]
    assert out["citations"] == ["d1"]
    assert out["status"] == "accepted"


def test_agent_route_now_passes_the_critic(deps, monkeypatch):
    monkeypatch.setattr(settings, "critic", True)
    deps.critic_llm = FakeLLM(
        CriticResponse(
            overall_score=0.1, faithfulness_score=0.1, completeness_score=0.1, citation_score=0.1
        )
    )
    _patch_agent(monkeypatch)
    out = run(MULTI_HOP, deps=deps)
    assert out["critic_score"] == 0.1
    assert out["status"] == "abstained"


def test_agent_without_tool_use_falls_back_to_rag(deps, monkeypatch):
    store = FakeStore()
    queries = []
    store.search = lambda query, top_k=8, metadata_filter=None: (
        queries.append(query) or list(FakeStore()._contexts)
    )
    deps.store = store
    _patch_agent(monkeypatch, retrieve=False)
    out = run(MULTI_HOP, deps=deps)
    assert queries, "RAG fallback should have retrieved"
    assert out["answer"] == "Net sales were 5,678."
    assert out["status"] == "accepted"


def test_calc_only_evidence_is_not_grounding(deps, monkeypatch):
    seen = {}
    real_edge = agent_mod.has_retrieved_evidence

    def spy(contexts):
        seen["n"] = len(contexts)
        return real_edge(contexts)

    monkeypatch.setattr(agent_mod, "has_retrieved_evidence", spy)
    _patch_agent(monkeypatch, retrieve=False, calc="2 + 2")
    out = run(MULTI_HOP, deps=deps)
    assert seen["n"] == 1  # one calc context, judged insufficient
    assert out["status"] == "accepted"  # answered via the RAG fallback


def test_agent_failure_keeps_evidence_gathered_so_far(deps, monkeypatch):
    _patch_agent(monkeypatch, raise_after=RuntimeError("recursion limit"))
    out = run(MULTI_HOP, deps=deps)
    assert out["answer"] == "Net sales were 5,678."
    assert out["status"] == "accepted"

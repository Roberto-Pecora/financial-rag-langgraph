"""Branch behaviour: entity filter, CRAG loop, critic gate, output guardrail."""

from __future__ import annotations

from frag.graph.build import run
from frag.graph.deps import Deps
from frag.graph.schemas import DocGrade, RewriteOut
from frag.rag.llm_schemas import ActorResponse, CriticResponse
from frag.utils.settings import settings
from tests.conftest import FakeLLM, FakeStore


def test_entity_filter_constrains_to_named_company(deps):
    calls = {}

    class RecordingStore(FakeStore):
        def search(self, query, top_k=8, metadata_filter=None):
            calls["filter"] = metadata_filter
            return super().search(query, top_k, metadata_filter)

    deps.store = RecordingStore(companies=["Amazon", "Nike"])
    run("What were Amazon net sales?", deps=deps)
    assert calls["filter"] == {"company": "Amazon"}


def test_output_guardrail_withholds_fabricated_citation(deps):
    # Actor cites a label not in the retrieved set -> output screen abstains.
    deps.actor_llm = FakeLLM(ActorResponse(answer="Sales were 5,678.", citations=["ghost-doc"]))
    out = run("What were Amazon net sales?", deps=deps)
    assert out["status"] == "abstained"
    assert "withheld" in out["answer"].lower()


def test_critic_gate_vetoes_low_score(deps, monkeypatch):
    monkeypatch.setattr(settings, "critic", True)
    deps.critic_llm = FakeLLM(
        CriticResponse(
            overall_score=0.2, faithfulness_score=0.2, completeness_score=0.2, citation_score=0.2
        )
    )
    out = run("What were Amazon net sales?", deps=deps)
    assert out["status"] == "abstained"
    assert out["critic_score"] == 0.2


def test_corrective_rewrites_and_retries(monkeypatch):
    monkeypatch.setattr(settings, "corrective", True)
    monkeypatch.setattr(settings, "corrective_min_relevant", 1)

    class WeakThenStrong(FakeStore):
        def __init__(self):
            super().__init__()
            self.queries = []

        def search(self, query, top_k=8, metadata_filter=None):
            self.queries.append(query)
            if "operating activities" in query:
                return [{"text": "Amazon cash from operations.", "metadata": {"doc_id": "good"}}]
            return [{"text": "Irrelevant boilerplate.", "metadata": {"doc_id": "weak"}}]

    store = WeakThenStrong()

    # grader: relevant only for the strong passage; rewriter yields the rescue query.
    class GraderByText:
        def invoke(self, prompt):
            return DocGrade(relevant="cash from operations" in prompt)

    deps = Deps(
        store=store,
        actor_llm=FakeLLM(ActorResponse(answer="Cash from operations rose.", citations=["good"])),
        grader_llm=GraderByText(),
        rewrite_llm=FakeLLM(RewriteOut(query="Amazon cash provided by operating activities")),
    )
    out = run("Amazon operating cash flow", deps=deps)
    assert any("operating activities" in q for q in store.queries)  # a rewrite happened
    assert out["status"] == "accepted"


def test_agent_route_used_for_multihop(deps, monkeypatch):
    seen = {}

    def fake_agent_node(_deps):
        def run_node(state):
            seen["q"] = state["standalone"]
            return {"answer": "compared.", "citations": [], "contexts": [], "status": "accepted"}

        return run_node

    monkeypatch.setattr("frag.graph.build.agent_node", fake_agent_node)
    out = run("Compare Amazon and Nike revenue", deps=deps)
    assert out["route"] == "multi_hop"
    assert seen["q"]


def test_agent_empty_falls_back_to_rag(deps, monkeypatch):
    def empty_agent_node(_deps):
        def run_node(_state):
            return {"answer": "", "citations": [], "contexts": [], "status": "abstained"}

        return run_node

    monkeypatch.setattr("frag.graph.build.agent_node", empty_agent_node)
    out = run("Compare Amazon and Nike revenue", deps=deps)
    assert out["route"] == "multi_hop"
    assert out["status"] == "accepted"  # RAG fallback produced an answer
    assert "5,678" in out["answer"]

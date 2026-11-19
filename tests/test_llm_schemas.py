"""Strict LLM-boundary schemas: an empty or partial reply must fail loudly."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from frag.graph.build import run
from frag.rag.llm_schemas import ActorResponse, CriticResponse
from frag.utils.settings import settings
from tests.conftest import FakeLLM

_FULL = {
    "overall_score": 0.9,
    "faithfulness_score": 0.9,
    "completeness_score": 0.9,
    "citation_score": 0.9,
}


class RaisingLLM:
    """Stands in for a structured-output call whose reply failed validation."""

    def __init__(self, exc):
        self._exc = exc

    def invoke(self, _prompt):
        raise self._exc


def _validation_error() -> ValidationError:
    try:
        CriticResponse.model_validate_json("{}")
    except ValidationError as exc:
        return exc
    raise AssertionError("expected ValidationError")


# --- schema contract -------------------------------------------------------


def test_critic_rejects_empty_object():
    with pytest.raises(ValidationError):
        CriticResponse.model_validate_json("{}")


@pytest.mark.parametrize("missing", list(_FULL))
def test_critic_rejects_any_missing_score(missing):
    payload = {k: v for k, v in _FULL.items() if k != missing}
    with pytest.raises(ValidationError):
        CriticResponse.model_validate(payload)


@pytest.mark.parametrize("bad", [-0.1, 1.5, 8, 85])
def test_critic_rejects_out_of_range_score(bad):
    with pytest.raises(ValidationError):
        CriticResponse.model_validate({**_FULL, "overall_score": bad})


def test_critic_accepts_genuine_zero_and_defaults_issues():
    resp = CriticResponse.model_validate({k: 0.0 for k in _FULL})
    assert resp.overall_score == 0.0
    assert resp.issues == []


@pytest.mark.parametrize("payload", ["{}", '{"answer": ""}', '{"answer": "   "}'])
def test_actor_rejects_missing_or_blank_answer(payload):
    with pytest.raises(ValidationError):
        ActorResponse.model_validate_json(payload)


def test_actor_requires_answer_but_not_citations():
    assert ActorResponse.model_validate({"answer": "x"}).citations == []


# --- node behaviour --------------------------------------------------------


def test_critic_validation_failure_reports_critic_failed(deps, monkeypatch):
    monkeypatch.setattr(settings, "critic", True)
    deps.critic_llm = RaisingLLM(_validation_error())
    out = run("What were Amazon net sales?", deps=deps)
    assert out["status"] == "abstained"
    assert out["critic_notes"] == "critic failed"


def test_critic_genuine_zero_verdict_is_a_normal_veto(deps, monkeypatch):
    # A real all-zero verdict (with reasons) must not be mislabelled as an empty reply.
    monkeypatch.setattr(settings, "critic", True)
    deps.critic_llm = FakeLLM(
        CriticResponse.model_validate({**{k: 0.0 for k in _FULL}, "issues": ["wrong figure"]})
    )
    out = run("What were Amazon net sales?", deps=deps)
    assert out["status"] == "abstained"
    assert out["critic_score"] == 0.0
    assert out["critic_notes"].startswith("faithfulness=0.00")


def test_actor_validation_failure_abstains(deps):
    try:
        ActorResponse.model_validate_json("{}")
    except ValidationError as exc:
        deps.actor_llm = RaisingLLM(exc)
    out = run("What were Amazon net sales?", deps=deps)
    assert out["status"] == "abstained"

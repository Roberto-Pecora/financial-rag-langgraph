"""RAG-path graph nodes. Each is a closure over Deps returning a partial state update.

Order: screen_input -> rewrite -> route -> [entity_filter -> retrieve -> grade
-> (rewrite_query -> retrieve)* -> rerank -> actor -> critic? -> ground
-> screen_output]. The agent branch lives in graph/agent.py.
"""

from __future__ import annotations

from collections.abc import Callable

from frag.agent.guardrails import screen_input, screen_output
from frag.graph import router
from frag.graph.deps import Deps
from frag.graph.state import GraphState
from frag.rag import prompts
from frag.rag.entity_filter import company_filter
from frag.rag.explain import ground_answer
from frag.rag.reranker import rerank_enabled
from frag.rag.untrusted import render_evidence
from frag.utils.logging import get_logger
from frag.utils.settings import settings

logger = get_logger(__name__)

_ABSTAIN = "Insufficient evidence retrieved."


def _doc_id(ctx: dict, idx: int) -> str:
    meta = ctx.get("metadata") or {}
    return meta.get("doc_id") or meta.get("id") or meta.get("ticker") or f"doc-{idx + 1}"


def screen_input_node(_: Deps) -> Callable[[GraphState], dict]:
    def run(state: GraphState) -> dict:
        verdict = screen_input(state["question"])
        if verdict.allowed:
            return {"status": "pending"}
        logger.warning("input guardrail blocked request", reason=verdict.reason)
        return {
            "status": "refused",
            "answer": "Request blocked by the input guardrail.",
            "citations": [],
            "critic_notes": verdict.reason,
            "contexts": [],
        }

    return run


def rewrite_node(deps: Deps) -> Callable[[GraphState], dict]:
    """Resolve a follow-up into a standalone question using recent history."""

    def run(state: GraphState) -> dict:
        history = state.get("history") or []
        question = state["question"]
        if not history:
            return {"standalone": question}
        turns = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
        prompt = (
            "Rewrite the follow-up as a standalone question using the conversation. "
            "Return only the rewritten question.\n\n"
            f"Conversation:\n{turns}\n\nFollow-up: {question}"
        )
        try:
            from frag.llm.client import GenerateAdapter, chat_model

            llm = deps.rewrite_llm or GenerateAdapter(chat_model("grader"))
            standalone = llm.generate(prompt).strip() or question
        except Exception as exc:
            logger.warning("multi-turn rewrite failed; using raw question", error=str(exc))
            standalone = question
        return {"standalone": standalone}

    return run


def route_node(deps: Deps) -> Callable[[GraphState], dict]:
    def run(state: GraphState) -> dict:
        route = router.route(state["standalone"], deps.get_router())
        return {"route": route}

    return run


def entity_filter_node(deps: Deps) -> Callable[[GraphState], dict]:
    """Constrain retrieval to the query's company when the store names one."""

    def run(state: GraphState) -> dict:
        store = deps.get_store()
        lister = getattr(store, "list_companies", None)
        if lister is None:
            return {"metadata_filter": None}
        try:
            companies = lister()
        except Exception:
            companies = []
        return {"metadata_filter": company_filter(state["standalone"], companies)}

    return run


def retrieve_node(deps: Deps) -> Callable[[GraphState], dict]:
    def run(state: GraphState) -> dict:
        contexts = deps.get_store().search(
            query=state["standalone"],
            top_k=settings.top_k,
            metadata_filter=state.get("metadata_filter"),
        )
        return {"contexts": contexts}

    return run


def grade_node(deps: Deps) -> Callable[[GraphState], dict]:
    """CRAG grading: keep passages the grader judges relevant. Off -> passthrough."""

    def run(state: GraphState) -> dict:
        contexts = state.get("contexts") or []
        if not settings.corrective:
            return {"graded": contexts}
        grader = deps.get_grader()
        kept = []
        for c in contexts:
            prompt = prompts.get("grader").render(
                query=state["standalone"], passage=c.get("text", "")[:2000]
            )
            try:
                if grader.invoke(prompt).relevant:
                    kept.append(c)
            except Exception:
                logger.warning("grader failed; keeping passage")
                kept.append(c)
        return {"graded": kept}

    return run


def rewrite_query_node(deps: Deps) -> Callable[[GraphState], dict]:
    """Reformulate the query for better recall, then loop back to retrieve."""

    def run(state: GraphState) -> dict:
        prompt = prompts.get("retrieval_rewrite").render(query=state["standalone"])
        try:
            rewritten = deps.get_rewriter().invoke(prompt).query.strip() or state["standalone"]
        except Exception:
            rewritten = state["standalone"]
        logger.info("corrective rewrite", frm=state["standalone"], to=rewritten)
        return {"standalone": rewritten, "rewrites": state.get("rewrites", 0) + 1}

    return run


def rerank_node(deps: Deps) -> Callable[[GraphState], dict]:
    """Pick the working context set (graded, else raw), then optionally rerank."""

    def run(state: GraphState) -> dict:
        contexts = state.get("graded") or state.get("contexts") or []
        if rerank_enabled() and contexts:
            contexts = deps.get_reranker().rerank(state["standalone"], contexts, settings.top_k)
        return {"contexts": contexts}

    return run


def actor_node(deps: Deps) -> Callable[[GraphState], dict]:
    def run(state: GraphState) -> dict:
        contexts = state.get("contexts") or []
        if not contexts:
            return {"answer": _ABSTAIN, "citations": [], "status": "abstained"}
        version = prompts.resolve_version("actor", "ACTOR_PROMPT_VERSION")
        prompt = prompts.get("actor", version).render(
            query=state["standalone"], evidence=render_evidence(contexts, _doc_id)
        )
        try:
            resp = deps.get_actor().invoke(prompt)
        except Exception as exc:
            logger.warning("actor call failed; abstaining", error=str(exc))
            return {"answer": _ABSTAIN, "citations": [], "status": "abstained"}
        abstained = resp.answer.strip().lower().startswith("insufficient evidence")
        return {
            "answer": resp.answer,
            "citations": resp.citations,
            "status": "abstained" if abstained else "accepted",
        }

    return run


def critic_node(deps: Deps) -> Callable[[GraphState], dict]:
    """Optional risk gate: score the draft, veto below CRITIC_MIN_SCORE."""

    def run(state: GraphState) -> dict:
        contexts = state.get("contexts") or []
        version = prompts.resolve_version("critic", "CRITIC_PROMPT_VERSION")
        citations = ", ".join(state.get("citations") or []) or "none"
        prompt = prompts.get("critic", version).render(
            query=state["standalone"],
            evidence=render_evidence(contexts, _doc_id),
            answer=state.get("answer", ""),
            citations=citations,
        )
        try:
            resp = deps.get_critic().invoke(prompt)
        except Exception as exc:
            logger.warning("critic call failed; vetoing", error=str(exc))
            return {"status": "abstained", "critic_score": 0.0, "critic_notes": "critic failed"}
        # All-zero scores mean an empty/omitted-field response.
        scores = (
            resp.overall_score,
            resp.faithfulness_score,
            resp.completeness_score,
            resp.citation_score,
        )
        if not any(scores):
            logger.warning("critic returned all-zero scores; treating as failure")
            return {
                "status": "abstained",
                "critic_score": 0.0,
                "critic_notes": "critic no-response",
            }
        notes = (
            f"faithfulness={resp.faithfulness_score:.2f} "
            f"completeness={resp.completeness_score:.2f} citations={resp.citation_score:.2f}"
        )
        accepted = resp.overall_score >= settings.critic_min_score
        return {
            "critic_score": resp.overall_score,
            "critic_notes": notes,
            "status": "accepted" if accepted else "abstained",
        }

    return run


def ground_node(_: Deps) -> Callable[[GraphState], dict]:
    """Explainability: map each fact in the answer to a supporting passage."""

    def run(state: GraphState) -> dict:
        grounding = ground_answer(state.get("answer", ""), state.get("contexts") or [], _doc_id)
        return {"grounding": grounding}

    return run


def screen_output_node(_: Deps) -> Callable[[GraphState], dict]:
    def run(state: GraphState) -> dict | None:
        if state.get("status") != "accepted":
            return None
        contexts = state.get("contexts") or []
        labels = {_doc_id(c, i) for i, c in enumerate(contexts)}
        verdict = screen_output(state.get("answer", ""), state.get("citations") or [], labels)
        if verdict.allowed:
            return None
        logger.warning("output guardrail withheld answer", reason=verdict.reason)
        note = (state.get("critic_notes") or "") + f" [guardrail: {verdict.reason}]"
        return {
            "status": "abstained",
            "answer": "Answer withheld by the output guardrail.",
            "critic_notes": note.strip(),
        }

    return run


# -- conditional edges ------------------------------------------------------


def route_edge(state: GraphState) -> str:
    return "agent" if state.get("route") == "multi_hop" else "entity_filter"


def input_edge(state: GraphState) -> str:
    return "refused" if state.get("status") == "refused" else "ok"


def grade_edge(state: GraphState) -> str:
    """After grading: proceed, or loop back to rewrite the query (bounded)."""
    if not settings.corrective:
        return "rerank"
    kept = state.get("graded") or []
    if len(kept) >= settings.corrective_min_relevant:
        return "rerank"
    if state.get("rewrites", 0) < settings.corrective_max_rewrites:
        return "rewrite_query"
    return "rerank"  # budget exhausted: proceed with what we have


def critic_edge(state: GraphState) -> str:
    return "critic" if settings.critic else "ground"


def agent_edge(state: GraphState) -> str:
    """If the tool-loop produced no answer, fall back to the RAG path."""
    if (state.get("answer") or "").strip():
        return "ground"
    logger.warning("agent produced no answer; falling back to RAG")
    return "entity_filter"

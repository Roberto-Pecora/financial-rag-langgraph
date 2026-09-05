"""Agent answer cleaning: strip untrusted markers, lift labels into citations."""

from __future__ import annotations

from frag.graph.agent import _clean_agent_answer


def test_paraphrased_mention_lifted_and_stripped():
    raw = "The term is one year. (UNTRUSTED_DOC 1eb25bfa) It renews annually."
    clean, cites = _clean_agent_answer(raw)
    assert cites == ["1eb25bfa"]
    assert "UNTRUSTED_DOC" not in clean
    assert "one year" in clean


def test_raw_markers_stripped_and_deduped():
    raw = (
        "<<<UNTRUSTED_DOC docA>>>text<<<END_UNTRUSTED_DOC>>> summary cites docA "
        "and (UNTRUSTED_DOC docB)."
    )
    clean, cites = _clean_agent_answer(raw)
    assert cites == ["docA", "docB"]
    assert "<<<" not in clean and "UNTRUSTED_DOC" not in clean


def test_clean_answer_without_markers_unchanged():
    clean, cites = _clean_agent_answer("A plain answer with no markers.")
    assert cites == []
    assert clean == "A plain answer with no markers."

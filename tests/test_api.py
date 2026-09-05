"""API contract: /ask returns the graph payload; /health is liveness."""

from __future__ import annotations

from fastapi.testclient import TestClient

import frag.api.main as api
from frag.graph.build import build_graph


def test_health():
    assert TestClient(api.app).get("/health").json() == {"status": "ok"}


def test_ask(deps, monkeypatch):
    monkeypatch.setattr(api, "_graph", build_graph(deps))
    resp = TestClient(api.app).post("/v1/ask", json={"question": "What were Amazon net sales?"})
    body = resp.json()
    assert resp.status_code == 200
    assert body["status"] == "accepted"
    assert body["citations"] == ["d1"]
    assert body["route"] == "lookup"

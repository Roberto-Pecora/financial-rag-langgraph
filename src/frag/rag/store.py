"""Qdrant document store with hybrid retrieval (dense k-NN + BM25 + RRF).

Local-first: connects to a local Qdrant with no API key by default. Dense vectors
come from the shared embedder; a lazily-built in-memory BM25 index adds lexical
recall, and Reciprocal Rank Fusion merges the two rankings. Returns the
{text, metadata, score} contract every retrieval backend here shares.
"""

from __future__ import annotations

import re
import uuid
from typing import Any

from qdrant_client import QdrantClient, models

from frag.rag.embedders import make_embedder
from frag.utils.settings import settings

# 60 is the Cormack et al. default; results are insensitive to it.
_RRF_K = 60


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = _RRF_K) -> dict[str, float]:
    """Fuse ranked doc_id lists: each contributes 1/(k+rank) so a doc ranked highly
    by either retriever floats up."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            if doc_id:
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


class QdrantStore:
    def __init__(
        self,
        collection_name: str | None = None,
        url: str | None = None,
        embedder: Any | None = None,
        hybrid: bool = True,
    ) -> None:
        self.hybrid = hybrid
        self.collection_name = collection_name or settings.qdrant_collection
        self.client = QdrantClient(url=url or settings.qdrant_url)
        self.embedder = embedder or make_embedder(model_name=settings.embedding_model)
        self._bm25 = None
        self._bm25_doc_ids: list[str] = []
        self._bm25_payloads: list[dict[str, Any]] = []
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection_name):
            return
        dim = self.embedder.get_sentence_embedding_dimension()
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )
        for field in ("company", "doc_name", "ticker", "form_type"):
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass

    def ingest(self, docs: list[dict[str, Any]]) -> int:
        texts, payloads, ids = [], [], []
        for d in docs:
            text = d.get("text") or d.get("content")
            if not text:
                continue
            doc_id = str(d.get("id") or uuid.uuid4())
            meta = d.get("metadata", {}) or {}
            ids.append(doc_id)
            texts.append(text)
            payloads.append({"text": text, "doc_id": doc_id, **meta})
        if not texts:
            return 0
        vectors = self.embedder.encode(texts, convert_to_numpy=True)
        points = [
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, ids[i])),
                vector=vectors[i],
                payload=payloads[i],
            )
            for i in range(len(texts))
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)
        self._bm25 = None  # invalidate the lexical index after a write
        return len(points)

    def search(
        self, query: str, top_k: int = 8, metadata_filter: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        if not query:
            return []
        if self.hybrid:
            return self._hybrid_search(query, top_k, metadata_filter)
        return self._dense_search(query, top_k, metadata_filter)

    def _dense_search(self, query, top_k, metadata_filter) -> list[dict[str, Any]]:
        query_vec = self.embedder.encode(query, convert_to_numpy=True)
        q_filter = None
        if metadata_filter:
            must = [
                models.FieldCondition(key=k, match=models.MatchValue(value=v))
                for k, v in metadata_filter.items()
            ]
            q_filter = models.Filter(must=must)
        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vec,
            query_filter=q_filter,
            limit=top_k,
            with_payload=True,
        )
        return [
            self._format_hit(getattr(h, "payload", None) or {}, getattr(h, "score", None))
            for h in getattr(hits, "points", []) or []
        ]

    @staticmethod
    def _format_hit(payload: dict[str, Any], score: Any) -> dict[str, Any]:
        text = payload.get("text", "")
        meta = {k: v for k, v in payload.items() if k != "text"}
        return {"text": text, "metadata": meta, "score": score}

    def _ensure_bm25_index(self) -> None:
        if self._bm25 is not None:
            return
        from rank_bm25 import BM25Okapi

        payloads, next_page = [], None
        while True:
            points, next_page = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                offset=next_page,
                with_payload=True,
            )
            payloads.extend(p.payload for p in points if (p.payload or {}).get("text"))
            if next_page is None:
                break
        self._bm25_payloads = payloads
        self._bm25_doc_ids = [p.get("doc_id", "") for p in payloads]
        tokenized = [_tokenize(p.get("text", "")) for p in payloads]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def _bm25_ranking(self, query, candidate_k, metadata_filter) -> list[str]:
        self._ensure_bm25_index()
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        ranking = []
        for i in order:
            payload = self._bm25_payloads[i]
            if metadata_filter and not all(payload.get(k) == v for k, v in metadata_filter.items()):
                continue
            ranking.append(self._bm25_doc_ids[i])
            if len(ranking) >= candidate_k:
                break
        return ranking

    def _hybrid_search(self, query, top_k, metadata_filter) -> list[dict[str, Any]]:
        candidate_k = max(top_k * 4, 30)
        dense_hits = self._dense_search(query, candidate_k, metadata_filter)
        dense_ranking = [h["metadata"].get("doc_id", "") for h in dense_hits]
        sparse_ranking = self._bm25_ranking(query, candidate_k, metadata_filter)
        scores = reciprocal_rank_fusion([dense_ranking, sparse_ranking])

        payload_by_id: dict[str, dict[str, Any]] = {}
        for h in dense_hits:
            payload_by_id[h["metadata"].get("doc_id", "")] = {"text": h["text"], **h["metadata"]}
        for doc_id, payload in zip(self._bm25_doc_ids, self._bm25_payloads, strict=True):
            payload_by_id.setdefault(doc_id, payload)

        ranked_ids = sorted(scores, key=lambda d: scores[d], reverse=True)[:top_k]
        return [self._format_hit(payload_by_id.get(d, {}), scores[d]) for d in ranked_ids]

    def list_companies(self) -> list[str]:
        """Distinct company names in the collection, for entity-aware filtering."""
        self._ensure_bm25_index()
        seen = {p.get("company") for p in self._bm25_payloads if p.get("company")}
        return sorted(seen)

    def count(self) -> int:
        return self.client.get_collection(self.collection_name).points_count or 0

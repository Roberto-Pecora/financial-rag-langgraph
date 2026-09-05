"""Deterministic retrieval eval, with metrics pushed to Langfuse as scores.

Golden JSONL line: {"query": str, "gold_ids": [doc_id, ...]}. Retrieval is scored
by exact recall/precision/MRR/nDCG at k, so the numbers exclude LLM variance. Each
mean metric is logged as a Langfuse score, so retrieval quality is tracked over
time next to the request traces.

    python scripts/eval_retrieval.py --golden data/golden.jsonl --k 10
"""

from __future__ import annotations

import argparse
import json

from frag.eval.metrics import mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k
from frag.llm import observability
from frag.rag.store import QdrantStore
from frag.utils.logging import get_logger

logger = get_logger("eval")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    with open(args.golden) as f:
        cases = [json.loads(line) for line in f if line.strip()]

    store = QdrantStore()
    agg = {"recall": 0.0, "precision": 0.0, "mrr": 0.0, "ndcg": 0.0}
    for case in cases:
        hits = store.search(case["query"], top_k=args.k)
        pred_ids = [h["metadata"].get("doc_id", "") for h in hits]
        gold = case["gold_ids"]
        agg["recall"] += recall_at_k(pred_ids, gold, args.k)
        agg["precision"] += precision_at_k(pred_ids, gold, args.k)
        agg["mrr"] += mrr_at_k(pred_ids, gold, args.k)
        agg["ndcg"] += ndcg_at_k(pred_ids, gold, args.k)

    n = max(len(cases), 1)
    means = {m: v / n for m, v in agg.items()}
    for metric, value in means.items():
        logger.info("retrieval metric", metric=metric, k=args.k, value=round(value, 4))
        observability.score(f"{metric}@{args.k}", value, comment=f"n={n}")
    observability.flush()


if __name__ == "__main__":
    main()

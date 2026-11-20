"""Content-based retrieval eval: a passage is relevant if it contains the golden facts.

Golden CSV columns: `query`, `reference_answer` (e.g. the FinanceBench set written by
`acquire_corpus.py`). Each question retrieves the top `--k` passages; a passage counts
as relevant when its text contains every numeric fact in the reference answer
(`frag.eval.relevance`). Questions whose answer has no numeric fact cannot be
content-scored and are excluded from the averages. Independent of chunk ids, so runs
over differently chunked collections stay comparable. Mean metrics are pushed to
Langfuse as scores when tracing is configured.

    python scripts/eval_content.py --golden data/corpus_build/golden_financebench.csv
    python scripts/eval_content.py --golden ... --rerank   # + cross-encoder rerank
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from frag.eval.harness import evaluate
from frag.llm import observability
from frag.rag.store import QdrantStore

K_VALUES = (1, 5, 10)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True, help="CSV with query, reference_answer")
    ap.add_argument("--k", type=int, default=10, help="passages scored per question")
    ap.add_argument(
        "--rerank",
        action="store_true",
        help="retrieve --candidates passages, rerank them with the cross-encoder, keep --k",
    )
    ap.add_argument("--candidates", type=int, default=30)
    args = ap.parse_args()

    golden = pd.read_csv(args.golden).fillna("").to_dict("records")
    store = QdrantStore()
    reranker = None
    if args.rerank:
        from frag.rag.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()

    predictions = []
    for g in golden:
        if reranker is None:
            hits = store.search(query=g["query"], top_k=args.k)
        else:
            hits = store.search(query=g["query"], top_k=args.candidates)
            hits = reranker.rerank(g["query"], hits, args.k)
        predictions.append({"query": g["query"], "retrieved_docs": hits[: args.k]})

    k_values = tuple(k for k in K_VALUES if k <= args.k)
    summary = evaluate(predictions, golden, k_values=k_values)["summary"]
    keys = [f"{m}@{k}" for k in k_values for m in ("recall", "ndcg")]
    report = {key: round(summary[key], 3) for key in keys if key in summary}
    report.update(n_scored=summary["n_scored"], n_total=summary["n_total"], rerank=args.rerank)
    print(json.dumps(report))

    suffix = "+rerank" if args.rerank else ""
    for key in keys:
        if key in summary:
            observability.score(
                f"content_{key}{suffix}", summary[key], comment=f"n={summary['n_scored']}"
            )
    observability.flush()


if __name__ == "__main__":
    main()

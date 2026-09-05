"""Ingest a JSONL corpus into Qdrant.

Each line: {"id"?, "text", "metadata": {...}}. Usage:
    python scripts/ingest.py --corpus data/corpus.jsonl
"""

from __future__ import annotations

import argparse
import json

from frag.rag.store import QdrantStore
from frag.utils.logging import get_logger

logger = get_logger("ingest")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    store = QdrantStore()
    with open(args.corpus) as f:
        docs = [json.loads(line) for line in f if line.strip()]

    total = 0
    for i in range(0, len(docs), args.batch):
        total += store.ingest(docs[i : i + args.batch])
        logger.info("ingested batch", done=total, of=len(docs))
    logger.info("done", ingested=total, collection_count=store.count())


if __name__ == "__main__":
    main()

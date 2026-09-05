# Financial RAG on LangGraph

A retrieval-first financial research assistant, built as a stateful **LangGraph**
graph with self-hosted **Langfuse** observability. It answers only from retrieved
evidence, cites supporting passages, and abstains when the evidence is insufficient.

This is a framework-native, cloud-optional re-architecture of a companion project
that runs the same pipeline hand-rolled on AWS. The retrieval thesis is unchanged;
what changes is that orchestration is a graph and every run is traced.

> The goal is not to use the best model on a public leaderboard. It is to measure
> and improve retrieval on the documents that matter — and to see, per run, where
> the answer came from.

```text
question
   │
   ▼
screen_input ─refused─▶ end
   │
rewrite (follow-up → standalone) ─▶ route ─┬─ multi_hop ─▶ agent (tool-loop)
                                           │
                                           └─ lookup ─▶ entity_filter ─▶ retrieve
                                                                            │
                                        rewrite_query ◀── grade ──enough──▶ rerank
                                             │                                │
                                             └──▶ retrieve                   actor
                                                                              │
                                        ground ◀── critic (optional gate) ◀───┘
                                             │
                                       screen_output ─▶ answer + citations, or abstain
```

## Why LangGraph?

The companion project controls retrieval, gating, and the tool-loop with plain
Python. That is the right call when the flow is fixed. It becomes a liability once
the flow has cycles and branches — corrective retrieval loops, a router that sends
some questions to an agent, an optional risk gate — because the control flow is then
implicit in call stacks and hard to trace.

LangGraph makes the flow a value: nodes and edges, cycles included. The corrective
retrieval loop is a real cycle in the graph (`grade → rewrite_query → retrieve`),
the router is a conditional edge, and every node boundary is a span in the trace.

## Why self-hosted observability?

A financial assistant should not ship its queries, retrieved passages, and answers
to a third-party SaaS to be debugged. Langfuse runs in the local stack, so traces,
scores, and prompt versions stay on infrastructure you control. Retrieval metrics
are pushed to the same backend as request traces, so answer quality and retrieval
quality are read side by side.

| Concern | Choice |
|---|---|
| Orchestration | LangGraph `StateGraph`, cycles for corrective retrieval, conditional edges for routing |
| Retrieval | Qdrant dense k-NN + in-memory BM25, fused with Reciprocal Rank Fusion |
| Reranking | Optional corpus-trained cross-encoder over the fused candidates |
| Agent | Prebuilt ReAct tool-loop (retrieve, graph lookup, whitelisted calc) under turn budgets |
| Observability | Self-hosted Langfuse: node traces, LLM spans, and eval scores |
| LLM | Any OpenAI-compatible endpoint via OpenRouter; model chosen per role |
| Infrastructure | Docker Compose (Qdrant + Langfuse); no cloud account required |

## Graph nodes

| Node | Responsibility |
|---|---|
| `screen_input` | Deterministic guardrail: block override/jailbreak attempts before any LLM call |
| `rewrite` | Resolve a follow-up into a standalone question from recent history |
| `router` | Route single-fact lookups to RAG, multi-hop questions to the agent |
| `entity_filter` | Derive a company filter so a question about one issuer does not drift onto another |
| `retrieve` | Hybrid dense + BM25 retrieval with RRF from Qdrant |
| `grade` / `rewrite_query` | Corrective retrieval: grade passages, rewrite and re-retrieve if too few are relevant |
| `rerank` | Optional cross-encoder reranking of the candidate list |
| `actor` | Grounded answer generation; cite evidence or abstain |
| `critic` | Optional risk gate: score faithfulness/completeness/citation, veto below threshold |
| `ground` | Explainability: map each fact in the answer to its supporting passage |
| `screen_output` | Reject answers citing labels not in the retrieved set |

Retrieved passages are wrapped as untrusted data before they reach any prompt, so a
poisoned document cannot redirect the model or forge a delimiter.

## Quick start

```bash
make install          # uv venv + editable install with dev extras
make up               # Qdrant + self-hosted Langfuse via Docker Compose
cp .env.example .env  # add OPENROUTER_API_KEY and the LANGFUSE_* keys from the UI
```

Build a corpus, ingest it, and ask:

```bash
python scripts/build_corpus.py --out data/corpus.jsonl   # SEC filings (needs SEC_USER_AGENT)
python scripts/acquire_corpus.py --out data/corpus.jsonl # or CUAD + FinanceBench ([data] extra)
python scripts/ingest.py --corpus data/corpus.jsonl
python scripts/ask.py "What changed in revenue and liquidity?"
```

Ingestion handles text PDFs (`pymupdf`), table linearisation (`pdfplumber`),
scanned-document OCR (AWS Textract, injectable and optional), and SEC HTML, all
normalised into one chunk schema with stable content-hash ids.

Serve the API:

```bash
make serve
curl -s localhost:8000/v1/ask -d '{"question":"What were net sales?"}'
```

`/health` is liveness; `/ready` also checks the vector store.

## Configuration

Behaviour is set by environment (see `.env.example`), read once into a typed
`Settings`. Notable switches, all off by default so the base path is the cheapest:

| Variable | Effect |
|---|---|
| `CORRECTIVE` | Enable the grade → rewrite → re-retrieve loop |
| `RERANK` | Enable cross-encoder reranking |
| `CRITIC` | Enable the actor-critic risk gate |
| `ROUTER_LLM` | Use an LLM classifier for routing instead of the heuristic |
| `ACTOR_PROMPT_VERSION` / `CRITIC_PROMPT_VERSION` | Pin a prompt version for A/B comparison |

## Evaluation

Retrieval evaluation is deterministic and content-based: a passage counts as
relevant when it contains the facts asserted in the golden answer, not when it
matches a particular chunk id.

```bash
python scripts/eval_retrieval.py --golden data/golden.jsonl --k 10
```

`recall@k`, `precision@k`, `MRR@k`, and `nDCG@k` are computed exactly and pushed to
Langfuse as scores, so retrieval quality is tracked over time next to request traces.

## Train on the corpus

No manually labelled relevance data is required. The pipeline generates synthetic
questions and mines BM25 hard negatives, then fine-tunes the embedding model and a
cross-encoder reranker.

```bash
python scripts/mine_pairs.py --corpus data/corpus.jsonl --out data/pairs.jsonl
python scripts/finetune_embedding.py --pairs data/pairs.jsonl --out artifacts/bge-ft
python scripts/train_reranker.py --pairs data/pairs.jsonl --out artifacts/reranker
```

A golden set for evaluation is derived from the corpus and can be re-verified when
chunking changes:

```bash
python scripts/build_golden.py
python scripts/verify_golden.py   # --write to expand gold ids after a re-chunk
```

## Development

```bash
make test    # hermetic: no Qdrant, no API key, no network
make check   # ruff lint + format check
```

Tests stub the store and LLMs, so the suite runs without Docker, a GPU, or keys.
The graph is built with an injectable `Deps`, so fakes replace every real backend.

## Scope and limitations

- The critic is an optional weak LLM-judge signal, not a retrieval metric.
- This is a low-cost research platform, not a production-volume document service.
- AWS Textract is the only optional cloud path (scanned-document OCR); it is
  injectable, so the pipeline runs fully local without it. Everything else — the
  graph, retrieval, training, evaluation, and observability — needs no cloud account.

## License

MIT

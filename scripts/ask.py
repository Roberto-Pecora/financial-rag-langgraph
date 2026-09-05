"""Ask the graph a question from the CLI (traced to Langfuse if configured).

    python scripts/ask.py "What changed in revenue and liquidity?"
"""

from __future__ import annotations

import json
import sys

from frag.graph.build import run


def main() -> None:
    if len(sys.argv) < 2:
        print('usage: python scripts/ask.py "your question"')
        raise SystemExit(1)
    result = run(sys.argv[1])
    print(
        json.dumps(
            {
                "route": result.get("route"),
                "status": result.get("status"),
                "answer": result.get("answer"),
                "citations": result.get("citations"),
                "grounding": result.get("grounding"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

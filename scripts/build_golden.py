"""Derive the curated golden set from the seed CSV.

    python scripts/build_golden.py [--seed data/golden_seed.csv --out data/golden_curated.csv]
"""

from __future__ import annotations

import argparse

import pandas as pd

from frag.golden.builder import build_golden


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="data/golden_seed.csv")
    ap.add_argument("--out", default="data/golden_curated.csv")
    args = ap.parse_args()
    build_golden(pd.read_csv(args.seed)).to_csv(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

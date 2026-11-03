#!/usr/bin/env python3
"""Combine the nine per-model English item-score files into results/english_all_models_summary.csv.

Each run of score_english_discourse.py writes
results/english_discourse_<tag>_item_scores.csv. This script recomputes every
model's summary from those item scores with the scorer's own summarize()
(10,000 bootstrap resamples, seed 13) and stacks them in a fixed model order.

Run from the english/ directory:

    python scripts/combine_english_results.py           # (re)write results/english_all_models_summary.csv
    python scripts/combine_english_results.py --check   # recompute and compare with the released file
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_english_discourse import summarize  # noqa: E402

# Model order.
TAGS = ["bert_base", "mbert", "xlm_r",
        "qwen25_0_5b", "qwen25_1_5b", "qwen25_3b",
        "xglm_1_7b", "xglm_2_9b", "xglm_4_5b"]

SUMMARY = "english_all_models_summary.csv"


def build(results: Path) -> pd.DataFrame:
    parts = []
    for tag in TAGS:
        path = results / f"english_discourse_{tag}_item_scores.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing item scores: {path}")
        s = summarize(pd.read_csv(path))
        s.insert(0, "tag", tag)
        parts.append(s)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--check", action="store_true",
                    help="Compare the recomputed summary with the released file instead of writing it.")
    args = ap.parse_args()

    results = Path(args.results)
    new = build(results)
    if not args.check:
        new.to_csv(results / SUMMARY, index=False)
        print(f"Wrote {results / SUMMARY} ({len(new)} rows)")
        return

    old = pd.read_csv(results / SUMMARY)
    ok = list(new.columns) == list(old.columns) and new.shape == old.shape
    if ok:
        for col in new.columns:
            a, b = new[col], old[col]
            if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
                same = np.allclose(a.astype(float), b.astype(float), rtol=0, atol=1e-9, equal_nan=True)
            else:
                same = (a.astype(str) == b.astype(str)).all()
            if not same:
                print(f"FAIL values differ in {col}")
                ok = False
    else:
        print(f"FAIL columns/shape differ: {new.shape} vs {old.shape}")
    print("english_summary_matches_released=" + str(ok))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

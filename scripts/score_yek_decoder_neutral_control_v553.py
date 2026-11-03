#!/usr/bin/env python3
"""Score the neutral continuation for each decoder model.

Uses ``score_yek_decoder_autoreg_v2.score_continuation`` and writes raw scores,
item deltas and a per-model summary.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import score_yek_decoder_autoreg_v2 as dec  # noqa: E402


RAW_COLUMNS = [
    "decoder_model_key", "decoder_model_id", "item_id", "condition",
    "subject", "object", "verb", "object_anaphor_type", "object_domain",
    "context", "continuation_neutral", "neutral_score", "neutral_sum_logprob",
    "neutral_token_count", "continuation_text_scored",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="stimuli/continuation_yek_v5_5_3_neutral_control_48.csv")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--model_key", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = dec.choose_device(args.device)
    print(f"Device: {device}")

    stim = pd.read_csv(args.input)
    needed = ["item_id", "condition", "subject", "object", "verb",
              "object_anaphor_type", "object_domain", "context",
              "continuation_neutral"]
    missing = [c for c in needed if c not in stim.columns]
    if missing:
        raise ValueError(f"Neutral stimulus file missing columns: {missing}")

    tokenizer, model = dec.load_model(args.model_id, device)

    rows = []
    for _, r in tqdm(stim.iterrows(), total=len(stim), desc="Scoring neutral"):
        res = dec.score_continuation(
            tokenizer, model, device, r["context"], r["continuation_neutral"]
        )
        rows.append({
            "decoder_model_key": args.model_key,
            "decoder_model_id": args.model_id,
            "item_id": r["item_id"],
            "condition": r["condition"],
            "subject": r["subject"],
            "object": r["object"],
            "verb": r["verb"],
            "object_anaphor_type": r["object_anaphor_type"],
            "object_domain": r["object_domain"],
            "context": r["context"],
            "continuation_neutral": r["continuation_neutral"],
            "neutral_score": res["mean_logprob"],
            "neutral_sum_logprob": res["sum_logprob"],
            "neutral_token_count": res["token_count"],
            "continuation_text_scored": res["continuation_text_scored"],
        })

    raw = pd.DataFrame(rows, columns=RAW_COLUMNS)
    raw_path = outdir / "decoder_neutral_scores_raw.csv"
    raw.to_csv(raw_path, index=False)

    bare = raw[raw["condition"] == "bare"]
    yek = raw[raw["condition"] == "yek"]
    merged = bare.merge(
        yek,
        on=["decoder_model_key", "decoder_model_id", "item_id", "subject",
            "object", "verb", "object_anaphor_type", "object_domain",
            "continuation_neutral"],
        suffixes=("_bare", "_yek"),
    )
    item = pd.DataFrame({
        "decoder_model_key": merged["decoder_model_key"],
        "decoder_model_id": merged["decoder_model_id"],
        "item_id": merged["item_id"],
        "subject": merged["subject"],
        "object": merged["object"],
        "verb": merged["verb"],
        "object_anaphor_type": merged["object_anaphor_type"],
        "object_domain": merged["object_domain"],
        "continuation_neutral": merged["continuation_neutral"],
        "neutral_score_bare": merged["neutral_score_bare"],
        "neutral_score_yek": merged["neutral_score_yek"],
        "neutral_delta_yek_minus_bare": merged["neutral_score_yek"] - merged["neutral_score_bare"],
        "neutral_token_count_bare": merged["neutral_token_count_bare"],
        "neutral_token_count_yek": merged["neutral_token_count_yek"],
    })
    item_path = outdir / "decoder_neutral_item_deltas.csv"
    item.to_csv(item_path, index=False)

    d = item["neutral_delta_yek_minus_bare"].dropna()
    ci_lo, ci_hi = dec.bootstrap_ci(d.tolist())
    summary = pd.DataFrame([{
        "decoder_model_key": args.model_key,
        "decoder_model_id": args.model_id,
        "n": int(len(d)),
        "mean_delta": float(d.mean()),
        "median_delta": float(d.median()),
        "ci_low": ci_lo,
        "ci_high": ci_hi,
        "prop_positive": float((d > 0).mean()),
        "min": float(d.min()),
        "max": float(d.max()),
    }])
    summary_path = outdir / "decoder_neutral_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Wrote {raw_path}\nWrote {item_path}\nWrote {summary_path}")


if __name__ == "__main__":
    main()

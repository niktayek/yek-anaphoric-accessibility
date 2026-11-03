#!/usr/bin/env python3
"""Score decoder target spans listed in ``span_requests_before_scoring.csv``.

Uses ``score_yek_decoder_autoreg_v2.score_continuation`` and writes per-model
raw span scores and item deltas.
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
    "anaphor_type", "target_kind", "context", "continuation", "target_text",
    "full_text", "char_start", "char_end", "left_context_for_decoder",
    "decoder_target_text", "mean_logprob", "sum_logprob", "token_count",
    "continuation_text_scored",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="results/primary_pll_word_l2r/span_controls/span_requests_before_scoring.csv",
    )
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--model_key", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = dec.choose_device(args.device)
    print(f"Device: {device}")

    req = pd.read_csv(args.input)
    needed = ["item_id", "condition", "anaphor_type", "target_kind",
              "context", "continuation", "target_text", "full_text",
              "char_start", "char_end"]
    missing = [c for c in needed if c not in req.columns]
    if missing:
        raise ValueError(f"Span-request file missing columns: {missing}")

    tokenizer, model = dec.load_model(args.model_id, device)

    rows = []
    for _, r in tqdm(req.iterrows(), total=len(req), desc="Scoring spans"):
        full_text = str(r["full_text"])
        cs, ce = int(r["char_start"]), int(r["char_end"])
        left_context = full_text[:cs].strip()
        target = full_text[cs:ce].strip()

        res = dec.score_continuation(tokenizer, model, device, left_context, target)

        rows.append({
            "decoder_model_key": args.model_key,
            "decoder_model_id": args.model_id,
            "item_id": r["item_id"],
            "condition": r["condition"],
            "anaphor_type": r["anaphor_type"],
            "target_kind": r["target_kind"],
            "context": r["context"],
            "continuation": r["continuation"],
            "target_text": r["target_text"],
            "full_text": full_text,
            "char_start": cs,
            "char_end": ce,
            "left_context_for_decoder": left_context,
            "decoder_target_text": target,
            "mean_logprob": res["mean_logprob"],
            "sum_logprob": res["sum_logprob"],
            "token_count": res["token_count"],
            "continuation_text_scored": res["continuation_text_scored"],
        })

    raw = pd.DataFrame(rows, columns=RAW_COLUMNS)
    raw_path = outdir / "decoder_span_scores_raw.csv"
    raw.to_csv(raw_path, index=False)

    # Item-level deltas: yek - bare per (item_id, anaphor_type, target_kind).
    bare = raw[raw["condition"] == "bare"]
    yek = raw[raw["condition"] == "yek"]
    merged = bare.merge(
        yek,
        on=["decoder_model_key", "decoder_model_id", "item_id",
            "anaphor_type", "target_kind"],
        suffixes=("_bare", "_yek"),
    )
    deltas = pd.DataFrame({
        "decoder_model_key": merged["decoder_model_key"],
        "decoder_model_id": merged["decoder_model_id"],
        "item_id": merged["item_id"],
        "anaphor_type": merged["anaphor_type"],
        "target_kind": merged["target_kind"],
        "target_text_bare": merged["decoder_target_text_bare"],
        "target_text_yek": merged["decoder_target_text_yek"],
        "token_count_bare": merged["token_count_bare"],
        "token_count_yek": merged["token_count_yek"],
        "mean_logprob_bare": merged["mean_logprob_bare"],
        "mean_logprob_yek": merged["mean_logprob_yek"],
        "delta_yek_minus_bare": merged["mean_logprob_yek"] - merged["mean_logprob_bare"],
    })
    deltas_path = outdir / "decoder_span_deltas.csv"
    deltas.to_csv(deltas_path, index=False)

    print(f"Wrote {raw_path}\nWrote {deltas_path}")


if __name__ == "__main__":
    main()

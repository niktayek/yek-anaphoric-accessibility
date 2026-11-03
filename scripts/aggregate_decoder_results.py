#!/usr/bin/env python3
"""Merge per-model decoder outputs into the combined ``results/all_decoder_*.csv`` tables.

The three decoder scoring scripts each write one directory per model:

    results/decoder/whole_continuation/<model_key>/   score_yek_decoder_autoreg_v2.py
    results/decoder/span_controls/<model_key>/        score_yek_decoder_span_controls_v553.py
    results/decoder/neutral_control/<model_key>/      score_yek_decoder_neutral_control_v553.py

This script combines them into the files behind the decoder rows of Table 3
and all of Table 5, computing the span and neutral-control summaries with the
same item bootstrap as the scoring scripts (10,000 resamples, seed 13).

    python scripts/aggregate_decoder_results.py            # (re)write results/all_decoder_*.csv
    python scripts/aggregate_decoder_results.py --check    # recompute and compare with the released files

``--check`` writes nothing to ``results/``; it exits with status 1 if any
recomputed value differs from the released file by more than 1e-9, or by more
than 1e-6 for confidence-interval bounds. The bootstrap averages in float32,
so the last digits of a bound depend on how many digits the input CSV kept;
the released bounds were computed in-session before the CSVs were written and
can differ by one float32 rounding step (about 1.2e-7).
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from score_yek_decoder_autoreg_v2 import bootstrap_ci  # noqa: E402  (same bootstrap as the scorers)

# Model order used in the paper's tables.
MODELS = [
    ("qwen25_0_5b", "Qwen2.5-0.5B"),
    ("qwen25_1_5b", "Qwen2.5-1.5B"),
    ("qwen25_3b", "Qwen2.5-3B"),
    ("mgpt_1_3b_persian", "mGPT-Persian 1.3B"),
    ("xglm_1_7b", "XGLM-1.7B"),
    ("xglm_2_9b", "XGLM-2.9B"),
    ("xglm_4_5b", "XGLM-4.5B"),
]

OUTPUT_FILES = [
    "all_decoder_summary_readable.csv",
    "all_decoder_summary_all_items_only.csv",
    "all_decoder_span_scores_raw.csv",
    "all_decoder_span_deltas.csv",
    "all_decoder_span_summary_by_target_kind.csv",
    "all_decoder_span_summary_by_target_kind_and_anaphor_type.csv",
    "all_decoder_object_minus_subject_item_deltas.csv",
    "all_decoder_span_summary_object_minus_subject.csv",
    "all_decoder_span_summary_object_minus_subject_by_anaphor_type.csv",
    "all_decoder_neutral_summary.csv",
    "all_decoder_object_neutral_item_contrasts.csv",
    "all_decoder_object_neutral_summary.csv",
]


def describe(values: pd.Series) -> dict:
    """Summary statistics in the column layout of the released span/neutral tables."""
    vals = values.dropna()
    lo, hi = bootstrap_ci(vals)
    return {
        "n": int(len(vals)),
        "mean": float(vals.mean()),
        "median": float(vals.median()),
        "ci_low": lo,
        "ci_high": hi,
        "prop_positive": float((vals > 0).mean()),
        "min": float(vals.min()),
        "max": float(vals.max()),
    }


def read_per_model(root: Path, analysis: str, filename: str) -> pd.DataFrame:
    parts = []
    for key, _ in MODELS:
        path = root / analysis / key / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing per-model output: {path}")
        parts.append(pd.read_csv(path))
    return pd.concat(parts, ignore_index=True)


def whole_continuation(root: Path, out: Path) -> None:
    rows = []
    for key, label in MODELS:
        s = pd.read_csv(root / "whole_continuation" / key / "decoder_summary.csv")
        s.insert(0, "model", label)
        rows.append(s)
    readable = pd.concat(rows, ignore_index=True)
    positives = (readable["prop_positive"] * readable["n"]).round().astype(int)
    readable.insert(
        readable.columns.get_loc("prop_positive"),
        "positive_items",
        positives.astype(str) + "/" + readable["n"].astype(str),
    )
    readable.to_csv(out / "all_decoder_summary_readable.csv", index=False)
    readable[readable["group"] == "all"].to_csv(
        out / "all_decoder_summary_all_items_only.csv", index=False
    )


def span_controls(root: Path, out: Path) -> pd.DataFrame:
    raw = read_per_model(root, "span_controls", "decoder_span_scores_raw.csv")
    deltas = read_per_model(root, "span_controls", "decoder_span_deltas.csv")
    raw.to_csv(out / "all_decoder_span_scores_raw.csv", index=False)
    deltas.to_csv(out / "all_decoder_span_deltas.csv", index=False)

    by_target, by_target_anaphor = [], []
    for key, _ in MODELS:
        d = deltas[deltas["decoder_model_key"] == key]
        model_id = d["decoder_model_id"].iloc[0]
        for kind, g in d.groupby("target_kind"):
            by_target.append({"decoder_model_key": key, "decoder_model_id": model_id,
                              "target_kind": kind, **describe(g["delta_yek_minus_bare"])})
        for (kind, typ), g in d.groupby(["target_kind", "anaphor_type"]):
            by_target_anaphor.append({"decoder_model_key": key, "decoder_model_id": model_id,
                                      "target_kind": kind, "anaphor_type": typ,
                                      **describe(g["delta_yek_minus_bare"])})
    pd.DataFrame(by_target).to_csv(out / "all_decoder_span_summary_by_target_kind.csv", index=False)
    pd.DataFrame(by_target_anaphor).to_csv(
        out / "all_decoder_span_summary_by_target_kind_and_anaphor_type.csv", index=False
    )

    # Item-level object-minus-subject contrasts (H2).
    wide = (
        deltas.pivot_table(
            index=["decoder_model_key", "decoder_model_id", "item_id", "anaphor_type"],
            columns="target_kind",
            values="delta_yek_minus_bare",
            aggfunc="first",
            sort=False,
        )
        .reset_index()
    )
    wide.columns.name = None
    wide = wide[["decoder_model_key", "decoder_model_id", "item_id", "anaphor_type",
                 "object_full_np", "object_marker", "subject_span"]]
    wide["object_full_np_minus_subject_span"] = wide["object_full_np"] - wide["subject_span"]
    wide["object_marker_minus_subject_span"] = wide["object_marker"] - wide["subject_span"]
    order = {key: i for i, (key, _) in enumerate(MODELS)}
    wide = (
        wide.assign(_m=wide["decoder_model_key"].map(order))
        .sort_values(["_m", "item_id"], kind="stable")
        .drop(columns="_m")
        .reset_index(drop=True)
    )
    wide.to_csv(out / "all_decoder_object_minus_subject_item_deltas.csv", index=False)

    contrasts = ["object_full_np_minus_subject_span", "object_marker_minus_subject_span"]
    overall, by_anaphor = [], []
    for key, _ in MODELS:
        w = wide[wide["decoder_model_key"] == key]
        model_id = w["decoder_model_id"].iloc[0]
        for c in contrasts:
            overall.append({"decoder_model_key": key, "decoder_model_id": model_id,
                            "contrast": c, **describe(w[c])})
            for typ, g in w.groupby("anaphor_type"):
                by_anaphor.append({"decoder_model_key": key, "decoder_model_id": model_id,
                                   "contrast": c, "anaphor_type": typ, **describe(g[c])})
    pd.DataFrame(overall).to_csv(out / "all_decoder_span_summary_object_minus_subject.csv", index=False)
    pd.DataFrame(by_anaphor).to_csv(
        out / "all_decoder_span_summary_object_minus_subject_by_anaphor_type.csv", index=False
    )
    return wide


def neutral_control(root: Path, out: Path, span_items: pd.DataFrame) -> None:
    summary = read_per_model(root, "neutral_control", "decoder_neutral_summary.csv")
    summary.to_csv(out / "all_decoder_neutral_summary.csv", index=False)

    neutral = read_per_model(root, "neutral_control", "decoder_neutral_item_deltas.csv")
    merged = span_items.merge(
        neutral[["decoder_model_key", "item_id", "neutral_delta_yek_minus_bare"]],
        on=["decoder_model_key", "item_id"],
        how="left",
        validate="one_to_one",
    )
    for col in ["object_full_np", "object_marker", "subject_span"]:
        merged[f"{col}_minus_neutral"] = merged[col] - merged["neutral_delta_yek_minus_bare"]
    merged.to_csv(out / "all_decoder_object_neutral_item_contrasts.csv", index=False)

    contrasts = ["object_full_np_minus_neutral", "object_marker_minus_neutral", "subject_span_minus_neutral"]
    rows = []
    for key in sorted(k for k, _ in MODELS):
        m = merged[merged["decoder_model_key"] == key]
        for c in contrasts:
            rows.append({"decoder_model_key": key, "contrast": c, **describe(m[c])})
    pd.DataFrame(rows).to_csv(out / "all_decoder_object_neutral_summary.csv", index=False)


def build(root: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    whole_continuation(root, out)
    span_items = span_controls(root, out)
    neutral_control(root, out, span_items)


CI_COLUMNS = {"ci_low", "ci_high"}


def compare(new_dir: Path, released_dir: Path, atol: float = 1e-9, ci_atol: float = 1e-6) -> bool:
    all_ok = True
    for name in OUTPUT_FILES:
        a = pd.read_csv(new_dir / name)
        b = pd.read_csv(released_dir / name)
        problems = []
        if list(a.columns) != list(b.columns):
            problems.append(f"columns differ: {list(a.columns)} vs {list(b.columns)}")
        elif a.shape != b.shape:
            problems.append(f"shape differs: {a.shape} vs {b.shape}")
        else:
            for col in a.columns:
                x, y = a[col], b[col]
                if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
                    tol = ci_atol if col in CI_COLUMNS else atol
                    if not np.allclose(x.astype(float), y.astype(float), rtol=0, atol=tol, equal_nan=True):
                        problems.append(f"values differ in {col}")
                elif not (x.astype(str) == y.astype(str)).all():
                    problems.append(f"values differ in {col}")
        print(f"{'OK  ' if not problems else 'FAIL'} {name}" + ("" if not problems else f": {'; '.join(problems)}"))
        all_ok &= not problems
    return all_ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decoder-root", default="results/decoder",
                    help="Directory holding whole_continuation/, span_controls/, neutral_control/.")
    ap.add_argument("--outdir", default="results", help="Where to write the all_decoder_*.csv files.")
    ap.add_argument("--check", action="store_true",
                    help="Recompute into a temporary directory and compare with the files in --outdir.")
    args = ap.parse_args()

    root, outdir = Path(args.decoder_root), Path(args.outdir)
    if args.check:
        with tempfile.TemporaryDirectory() as tmp:
            build(root, Path(tmp))
            ok = compare(Path(tmp), outdir)
        print("decoder_aggregation_matches_released=" + str(ok))
        sys.exit(0 if ok else 1)
    build(root, outdir)
    print(f"Wrote {len(OUTPUT_FILES)} files to {outdir}/")


if __name__ == "__main__":
    main()

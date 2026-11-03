#!/usr/bin/env python3
"""Score the frozen Persian stimuli with standard masked-LM PLL.

Writes per-item scores, a JSON summary and a Markdown summary under
``results/robustness_standard_pll/whole_continuation/``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


ROOT       = Path(__file__).resolve().parents[1]
INPUT_CSV  = ROOT / "stimuli" / "continuation_yek_v5_5_3_same_np_smoothing_48.csv"
OUT_CSV    = ROOT / "results" / "robustness_standard_pll" / "whole_continuation" / "continuation_yek_v5_5_3_scores.csv"
OUT_JSON   = ROOT / "results" / "robustness_standard_pll" / "whole_continuation" / "continuation_yek_v5_5_3_summary.json"
OUT_REPORT = ROOT / "results" / "robustness_standard_pll" / "whole_continuation" / "continuation_yek_v5_5_3_report.md"

MODELS: Dict[str, str] = {
    "mbert":    "bert-base-multilingual-cased",
    "xlm-r":    "xlm-roberta-base",
    "parsbert": "HooshvareLab/bert-fa-base-uncased",
}

REQUIRED_COLUMNS = [
    "item_id", "condition", "context",
    "continuation_subject_continuing", "continuation_object_anaphoric",
    "object_anaphor_type", "object_domain",
]


# Text helpers

def _join(context: str, continuation: str) -> str:
    return f"{str(context).strip()} {str(continuation).strip()}".strip()


def _prior(continuation: str) -> str:
    return str(continuation).strip()


# PLL scorer

@dataclass
class PLLScorer:
    model_name: str
    device: str | None = None

    def __post_init__(self) -> None:
        print(f"  Loading {self.model_name} …", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model     = AutoModelForMaskedLM.from_pretrained(self.model_name).eval()
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"  Loaded on {self.device}.", flush=True)

    @torch.no_grad()
    def pll_len_norm(self, text: str) -> float:
        """Mean per-token log-probability under successive single-position masking."""
        enc  = self.tokenizer(text, return_tensors="pt")
        ids  = enc["input_ids"].to(self.device)
        att  = enc["attention_mask"].to(self.device)
        L    = ids.shape[1]
        positions = list(range(1, L - 1))  # skip [CLS] / [SEP]
        if not positions:
            return float("nan")

        batch_size = len(positions)
        masked_ids = ids.repeat(batch_size, 1)
        masked_att = att.repeat(batch_size, 1)
        for row_idx, pos in enumerate(positions):
            masked_ids[row_idx, pos] = self.tokenizer.mask_token_id

        logits   = self.model(masked_ids, attention_mask=masked_att).logits
        log_prob = torch.log_softmax(logits, dim=-1)

        token_logps = [
            float(log_prob[row_idx, pos, ids[0, pos]].item())
            for row_idx, pos in enumerate(positions)
        ]
        return float(np.mean(token_logps))


# Validation

def validate_input(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if not set(df["condition"].unique()).issubset({"bare", "yek"}):
        raise ValueError("condition column contains unexpected values")
    bad = df.groupby("item_id")["condition"].apply(set).pipe(
        lambda s: s[s != {"bare", "yek"}]
    )
    if not bad.empty:
        raise ValueError(f"Items missing bare/yek pairing: {bad.to_dict()}")
    print(f"Input validated: {len(df)} rows, {df['item_id'].nunique()} items.", flush=True)


# Per-model scoring

def score_model(df: pd.DataFrame, model_key: str, model_name: str) -> pd.DataFrame:
    scorer = PLLScorer(model_name=model_name)
    rows: List[Dict] = []

    for i, (_, r) in enumerate(df.iterrows()):
        if i % 20 == 0:
            print(f"  [{model_key}] row {i}/{len(df)}", flush=True)

        subj_pll  = scorer.pll_len_norm(_join(r["context"], r["continuation_subject_continuing"]))
        obj_pll   = scorer.pll_len_norm(_join(r["context"], r["continuation_object_anaphoric"]))
        psubj_pll = scorer.pll_len_norm(_prior(r["continuation_subject_continuing"]))
        pobj_pll  = scorer.pll_len_norm(_prior(r["continuation_object_anaphoric"]))

        margin       = obj_pll - subj_pll
        prior_margin = pobj_pll - psubj_pll

        rows.append({
            "model_key":                     model_key,
            "model_name":                    model_name,
            **r.to_dict(),
            "subject_score":                 subj_pll,
            "object_score":                  obj_pll,
            "object_preference_margin":      margin,
            "object_preferred":              bool(obj_pll > subj_pll),
            "prior_subject_score":           psubj_pll,
            "prior_object_score":            pobj_pll,
            "prior_object_preference_margin": prior_margin,
            "context_minus_prior_margin":    margin - prior_margin,
        })

    scored = pd.DataFrame(rows)

    # Per-item bare/yek pivot.
    item_wide = (
        scored
        .pivot_table(
            index=["model_key", "model_name", "item_id"],
            columns="condition",
            values=["object_preference_margin", "prior_object_preference_margin"],
            aggfunc="first",
        )
        .reset_index()
    )
    item_wide.columns = [
        "_".join([c for c in col if c]).rstrip("_")
        if isinstance(col, tuple) else col
        for col in item_wide.columns
    ]
    item_wide = item_wide.rename(columns={
        "object_preference_margin_bare":       "bare_margin",
        "object_preference_margin_yek":        "yek_margin",
        "prior_object_preference_margin_bare": "bare_prior_margin",
        "prior_object_preference_margin_yek":  "yek_prior_margin",
    })
    item_wide["delta_margin"]       = item_wide["yek_margin"] - item_wide["bare_margin"]
    item_wide["predicted_direction"] = item_wide["delta_margin"] > 0
    item_wide["delta_prior_margin"] = item_wide["yek_prior_margin"] - item_wide["bare_prior_margin"]

    scored = scored.merge(
        item_wide[[
            "model_key", "item_id",
            "bare_margin", "yek_margin", "delta_margin", "predicted_direction",
            "bare_prior_margin", "yek_prior_margin", "delta_prior_margin",
        ]],
        on=["model_key", "item_id"],
        how="left",
    )
    return scored


# Bootstrap CI

def bootstrap_ci(
    values: np.ndarray, stat: str, n_boot: int = 5000, seed: int = 7
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n   = len(values)
    if n == 0:
        return float("nan"), float("nan")
    samples = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        draw = values[rng.integers(0, n, size=n)]
        samples[b] = float(np.mean(draw)) if stat == "mean" else float(np.mean(draw > 0))
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


# Summary

def summarize(all_scored: pd.DataFrame) -> Dict:
    summary: Dict = {
        "experiment":  "continuation_yek_v5_5_3_48item",
        "input_csv":   str(INPUT_CSV.relative_to(ROOT)),
        "n_rows":      int(len(all_scored)),
        "n_item_ids":  int(all_scored["item_id"].nunique()),
        "models":      {},
    }

    for model_key, mdf in all_scored.groupby("model_key"):
        item_df = (
            mdf[["item_id", "delta_margin", "bare_margin", "yek_margin", "delta_prior_margin"]]
            .drop_duplicates()
            .sort_values("item_id")
        )
        deltas   = item_df["delta_margin"].to_numpy(dtype=float)
        mean_ci  = bootstrap_ci(deltas, stat="mean")
        prop_ci  = bootstrap_ci(deltas, stat="prop_pos")
        prop_pos = float(np.mean(deltas > 0))

        item_meta = (
            mdf[mdf["condition"] == "bare"][
                ["item_id", "object_anaphor_type", "object_domain"]
            ]
            .drop_duplicates()
            .merge(item_df[["item_id", "delta_margin"]], on="item_id", how="left")
        )

        breakdown_anaphor = [
            {
                "object_anaphor_type":     grp,
                "n_items":                 int(len(g)),
                "mean_delta_margin":       float(np.mean(g["delta_margin"].to_numpy(dtype=float))),
                "prop_delta_positive":     float(np.mean(g["delta_margin"].to_numpy(dtype=float) > 0)),
            }
            for grp, g in item_meta.groupby("object_anaphor_type")
        ]
        breakdown_domain = [
            {
                "object_domain":           grp,
                "n_items":                 int(len(g)),
                "mean_delta_margin":       float(np.mean(g["delta_margin"].to_numpy(dtype=float))),
                "prop_delta_positive":     float(np.mean(g["delta_margin"].to_numpy(dtype=float) > 0)),
            }
            for grp, g in item_meta.groupby("object_domain")
        ]

        opp = item_df[item_df["delta_margin"] <= 0][
            ["item_id", "delta_margin", "bare_margin", "yek_margin"]
        ]

        summary["models"][model_key] = {
            "n_items":                            int(len(item_df)),
            "mean_delta_margin":                  float(np.mean(deltas)),
            "proportion_delta_margin_positive":   prop_pos,
            "bootstrap95_mean_delta_margin":      {"low": mean_ci[0], "high": mean_ci[1]},
            "bootstrap95_prop_delta_positive":    {"low": prop_ci[0], "high": prop_ci[1]},
            "mean_bare_margin":                   float(item_df["bare_margin"].mean()),
            "mean_yek_margin":                    float(item_df["yek_margin"].mean()),
            "mean_delta_prior_margin":            float(item_df["delta_prior_margin"].mean()),
            "breakdown_by_object_anaphor_type":   sorted(breakdown_anaphor, key=lambda x: x["object_anaphor_type"]),
            "breakdown_by_object_domain":         sorted(breakdown_domain, key=lambda x: x["object_domain"]),
            "opposite_direction_items":           opp.to_dict(orient="records"),
        }

    # Cross-model consistency.
    by_item_model = (
        all_scored[["model_key", "item_id", "delta_margin"]]
        .drop_duplicates()
        .pivot(index="item_id", columns="model_key", values="delta_margin")
    )
    avail = sorted(by_item_model.columns.tolist())
    all_pos = np.ones(len(by_item_model), dtype=bool)
    any_neg = np.zeros(len(by_item_model), dtype=bool)
    for mk in avail:
        arr       = by_item_model[mk].to_numpy(dtype=float)
        all_pos  &= arr > 0
        any_neg  |= arr <= 0

    summary["cross_model_consistency"] = {
        "models":                           avail,
        "n_items":                          int(len(by_item_model)),
        "n_items_positive_all_models":      int(np.sum(all_pos)),
        "n_items_with_any_nonpositive":     int(np.sum(any_neg)),
        "items_positive_all_models":        by_item_model.index[all_pos].tolist(),
    }
    return summary


# Report

def build_report(summary: Dict, all_scored: pd.DataFrame) -> str:
    L: List[str] = []
    L += [
        "# Persian ye(k) continuation benchmark v5.5.3 - standard-PLL robustness report",
        "",
        "Standard single-position PLL robustness scoring of the frozen 48-item v5.5.3 stimuli.",
        "",
        "## Setup",
        f"- Input: {summary['input_csv']}",
        f"- Items: {summary['n_item_ids']}",
        f"- Models: {', '.join(summary['models'].keys())}",
        "",
        "## Model Summary",
    ]
    for mk, m in summary["models"].items():
        L += [
            f"### {mk}",
            f"- mean delta_margin: {m['mean_delta_margin']:.4f}",
            f"- 95% CI (mean delta): [{m['bootstrap95_mean_delta_margin']['low']:.4f}, {m['bootstrap95_mean_delta_margin']['high']:.4f}]",
            f"- proportion delta > 0: {m['proportion_delta_margin_positive']:.3f}",
            f"- 95% CI (prop>0): [{m['bootstrap95_prop_delta_positive']['low']:.3f}, {m['bootstrap95_prop_delta_positive']['high']:.3f}]",
            f"- mean bare margin: {m['mean_bare_margin']:.4f}",
            f"- mean yek margin: {m['mean_yek_margin']:.4f}",
            f"- mean delta prior margin: {m['mean_delta_prior_margin']:.4f}",
            "",
        ]

    return "\n".join(L)


# Main

def main() -> None:
    for p in [OUT_CSV.parent, OUT_REPORT.parent]:
        p.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    validate_input(df)

    all_model_rows: List[pd.DataFrame] = []
    for key, name in MODELS.items():
        print(f"\nScoring model: {key} ({name})", flush=True)
        scored = score_model(df, model_key=key, model_name=name)
        all_model_rows.append(scored)
        print(f"  Done: {len(scored)} rows scored.", flush=True)

    all_scored = (
        pd.concat(all_model_rows, ignore_index=True)
        .sort_values(["model_key", "item_id", "condition"])
        .reset_index(drop=True)
    )
    all_scored.to_csv(OUT_CSV, index=False)

    summary = summarize(all_scored)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    report = build_report(summary, all_scored)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report)

    print("\nSaved:")
    for p in [OUT_CSV, OUT_JSON, OUT_REPORT]:
        print(f"  {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

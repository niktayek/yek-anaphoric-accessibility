#!/usr/bin/env python3
"""Score Persian object-marker, full-NP and subject-control spans.

Writes raw span scores, item deltas and summary tables for each model.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


DEFAULT_MODELS = {
    "mbert": "bert-base-multilingual-cased",
    "xlm-r": "xlm-roberta-base",
    "parsbert": "HooshvareLab/bert-fa-base-uncased",
}


COLUMN_CANDIDATES = {
    "item_id": ["item_id", "id", "item"],
    "condition": ["condition", "cond", "context_condition"],
    "context": ["context", "context_sentence", "prompt", "context_text"],
    "object_continuation": [
        "object_continuation",
        "object_continuing_continuation",
        "continuation_object",
        "obj_continuation",
        "object_alt",
        "object_anaphoric_continuation",
    ],
    "subject_continuation": [
        "subject_continuation",
        "subject_continuing_continuation",
        "continuation_subject",
        "subj_continuation",
        "subject_alt",
    ],
    "object_np": ["object_np", "object", "object_text", "object_noun", "object_phrase"],
    "subject_np": ["subject_np", "subject", "subject_text", "subject_name", "subj", "name"],
    "anaphor_type": [
        "object_anaphor_type",
        "anaphor_type",
        "continuation_anaphor_type",
        "object_continuation_type",
    ],
}


PERSIAN_ARABIC_NORMALIZATION = str.maketrans({
    "ي": "ی",
    "ك": "ک",
    "ۀ": "ه",
    "ة": "ه",
    "ؤ": "و",
    "إ": "ا",
    "أ": "ا",
    "ٱ": "ا",
    "آ": "آ",
})


def normalize_text(s: str) -> str:
    """Light normalization for matching spans without changing meaning."""
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    s = str(s)
    s = s.replace("\u200c", " ")  # ZWNJ
    s = s.translate(PERSIAN_ARABIC_NORMALIZATION)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def compact_for_match(s: str) -> str:
    """Normalize text and remove spaces for robust Persian span matching."""
    return re.sub(r"\s+", "", normalize_text(s))


def find_column(df: pd.DataFrame, key: str, required: bool = True) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for cand in COLUMN_CANDIDATES[key]:
        if cand.lower() in lower:
            return lower[cand.lower()]
    # substring fallback
    for c in df.columns:
        cl = c.lower()
        for cand in COLUMN_CANDIDATES[key]:
            if cand.lower() in cl:
                return c
    if required:
        raise ValueError(
            f"Could not find required column for {key}. "
            f"Tried {COLUMN_CANDIDATES[key]}. Available columns: {list(df.columns)}"
        )
    return None


def detect_columns(df: pd.DataFrame, overrides: Dict[str, Optional[str]]) -> Dict[str, str]:
    cols = {}
    for key in COLUMN_CANDIDATES:
        if overrides.get(key):
            if overrides[key] not in df.columns:
                raise ValueError(f"Override --{key}-col={overrides[key]!r} not in CSV columns.")
            cols[key] = overrides[key]
        else:
            required = key not in {"anaphor_type"}
            found = find_column(df, key, required=required)
            if found:
                cols[key] = found
    return cols


def canonical_condition(value: str) -> str:
    v = str(value).strip().lower()
    if v in {"bare", "b", "0", "no_yek", "no-yek", "bare_object"}:
        return "bare"
    if v in {"yek", "ye", "ye(k)", "y", "1", "with_yek", "with-yek"}:
        return "yek"
    if "bare" in v:
        return "bare"
    if "yek" in v or "ye(k)" in v or "ye_" in v:
        return "yek"
    raise ValueError(f"Cannot canonicalize condition value: {value!r}")


def infer_anaphor_type(row: pd.Series, cols: Dict[str, str]) -> str:
    if "anaphor_type" in cols:
        raw = str(row[cols["anaphor_type"]]).strip().lower()
        if "demo" in raw or "این" in raw or raw in {"in", "this"}:
            return "demonstrative_np"
        if "same" in raw or "همان" in raw or "haman" in raw:
            return "same_np"
    obj_cont = normalize_text(row[cols["object_continuation"]])
    if obj_cont.startswith("این") or obj_cont.startswith("in "):
        return "demonstrative_np"
    if obj_cont.startswith("همان") or obj_cont.startswith("haman "):
        return "same_np"
    # Persian marker may be after punctuation/space
    if "این" in obj_cont[:20]:
        return "demonstrative_np"
    if "همان" in obj_cont[:25]:
        return "same_np"
    raise ValueError(
        f"Could not infer anaphor type for item {row.get(cols.get('item_id', ''), '<unknown>')}. "
        f"Object continuation begins: {obj_cont[:80]!r}"
    )


def marker_for_anaphor_type(anaphor_type: str, continuation: str) -> str:
    # Prefer the script actually present in the continuation.
    cont = normalize_text(continuation)
    if anaphor_type == "demonstrative_np":
        if "این" in cont[:30]:
            return "این"
        return "in"
    if anaphor_type == "same_np":
        if "همان" in cont[:40]:
            return "همان"
        return "haman"
    raise ValueError(f"Unknown anaphor_type: {anaphor_type}")


def find_span_char(text: str, target: str, region_start: int = 0) -> Optional[Tuple[int, int]]:
    """
    Find target char span in text, trying exact normalized matching first.
    Returns original-text-ish offsets when possible.

    Because normalization can change ZWNJ/spaces, the fallback uses compact
    matching and maps back approximately to original positions.
    """
    text_s = str(text)
    target_s = str(target)
    if not target_s.strip():
        return None

    # Exact direct search
    idx = text_s.find(target_s, region_start)
    if idx >= 0:
        return idx, idx + len(target_s)

    # Normalized direct search; works if only Arabic/Persian variants differ.
    norm_text = normalize_text(text_s)
    norm_target = normalize_text(target_s)
    idx = norm_text.find(norm_target, max(0, region_start - 5))
    if idx >= 0:
        return idx, idx + len(norm_target)

    # Compact match fallback. This returns approximate offsets in normalized text.
    compact_text = compact_for_match(text_s)
    compact_target = compact_for_match(target_s)
    idx = compact_text.find(compact_target)
    if idx < 0:
        return None

    # Map compact offsets back to normalized-text offsets.
    norm = normalize_text(text_s)
    compact_pos = 0
    start = None
    end = None
    for i, ch in enumerate(norm):
        if ch.isspace():
            continue
        if compact_pos == idx and start is None:
            start = i
        compact_pos += 1
        if compact_pos == idx + len(compact_target):
            end = i + 1
            break
    if start is not None and end is not None:
        return start, end
    return None


@dataclass
class SpanRequest:
    item_id: str
    condition: str
    anaphor_type: str
    target_kind: str
    context: str
    continuation: str
    target_text: str
    full_text: str
    char_start: Optional[int]
    char_end: Optional[int]
    match_ok: bool
    notes: str = ""


def build_span_requests(df: pd.DataFrame, cols: Dict[str, str]) -> List[SpanRequest]:
    rows: List[SpanRequest] = []
    for _, row in df.iterrows():
        item_id = str(row[cols["item_id"]])
        condition = canonical_condition(row[cols["condition"]])
        context = normalize_text(row[cols["context"]])
        obj_cont = normalize_text(row[cols["object_continuation"]])
        subj_cont = normalize_text(row[cols["subject_continuation"]])
        object_np = normalize_text(row[cols["object_np"]])
        subject_np = normalize_text(row[cols["subject_np"]])
        anaphor_type = infer_anaphor_type(row, cols)
        marker = marker_for_anaphor_type(anaphor_type, obj_cont)

        # Object marker span in object continuation
        full_text_obj = context + " " + obj_cont
        cont_start_obj = len(context) + 1
        marker_span = find_span_char(full_text_obj, marker, region_start=cont_start_obj)
        rows.append(
            SpanRequest(
                item_id=item_id,
                condition=condition,
                anaphor_type=anaphor_type,
                target_kind="object_marker",
                context=context,
                continuation=obj_cont,
                target_text=marker,
                full_text=full_text_obj,
                char_start=marker_span[0] if marker_span else None,
                char_end=marker_span[1] if marker_span else None,
                match_ok=marker_span is not None,
                notes="" if marker_span else "marker_not_found",
            )
        )

        # Full object NP span: marker + object NP
        # Prefer exact phrase at the beginning of the continuation.
        full_np_target = normalize_text(f"{marker} {object_np}")
        full_np_span = find_span_char(full_text_obj, full_np_target, region_start=cont_start_obj)
        if full_np_span is None:
            # If the object has minor spelling variants, try to score from marker through the
            # first occurrence of object NP in the continuation.
            marker_only = marker_span
            object_span = find_span_char(full_text_obj, object_np, region_start=cont_start_obj)
            if marker_only and object_span:
                full_np_span = (marker_only[0], object_span[1])
                full_np_target = full_text_obj[full_np_span[0]:full_np_span[1]]
        rows.append(
            SpanRequest(
                item_id=item_id,
                condition=condition,
                anaphor_type=anaphor_type,
                target_kind="object_full_np",
                context=context,
                continuation=obj_cont,
                target_text=full_np_target,
                full_text=full_text_obj,
                char_start=full_np_span[0] if full_np_span else None,
                char_end=full_np_span[1] if full_np_span else None,
                match_ok=full_np_span is not None,
                notes="" if full_np_span else "object_full_np_not_found",
            )
        )

        # Subject span in subject-continuing continuation.
        full_text_subj = context + " " + subj_cont
        cont_start_subj = len(context) + 1
        subj_span = find_span_char(full_text_subj, subject_np, region_start=cont_start_subj)
        rows.append(
            SpanRequest(
                item_id=item_id,
                condition=condition,
                anaphor_type=anaphor_type,
                target_kind="subject_span",
                context=context,
                continuation=subj_cont,
                target_text=subject_np,
                full_text=full_text_subj,
                char_start=subj_span[0] if subj_span else None,
                char_end=subj_span[1] if subj_span else None,
                match_ok=subj_span is not None,
                notes="" if subj_span else "subject_np_not_found_in_subject_continuation",
            )
        )

    return rows


class PLLSpanScorer:
    def __init__(self, model_name: str, device: Optional[str] = None):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if not self.tokenizer.is_fast:
            raise RuntimeError(f"Tokenizer for {model_name} is not fast; offset mapping is required.")
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        if self.tokenizer.mask_token_id is None:
            raise RuntimeError(f"Tokenizer for {model_name} has no mask token.")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score_span(self, text: str, char_start: int, char_end: int) -> Dict[str, object]:
        enc = self.tokenizer(
            text,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=False,
        )
        input_ids = enc["input_ids"][0]
        offsets = enc["offset_mapping"][0].tolist()

        target_token_indices = []
        for i, (s, e) in enumerate(offsets):
            # Special tokens usually have (0, 0)
            if s == e:
                continue
            # Include tokens overlapping the target span.
            if e > char_start and s < char_end:
                target_token_indices.append(i)

        if not target_token_indices:
            return {
                "score_mean": np.nan,
                "score_sum": np.nan,
                "target_token_count": 0,
                "target_token_ids": "",
                "target_tokens": "",
                "score_ok": False,
                "score_notes": "no_target_tokens_from_offsets",
            }

        logps = []
        token_strs = []
        target_ids = []
        for token_idx in target_token_indices:
            original_id = int(input_ids[token_idx].item())
            masked = input_ids.clone()
            masked[token_idx] = self.tokenizer.mask_token_id
            batch = {
                "input_ids": masked.unsqueeze(0).to(self.device),
                "attention_mask": enc["attention_mask"].to(self.device),
            }
            if "token_type_ids" in enc:
                batch["token_type_ids"] = enc["token_type_ids"].to(self.device)

            logits = self.model(**batch).logits[0, token_idx]
            logprob = torch.log_softmax(logits, dim=-1)[original_id].item()
            logps.append(logprob)
            target_ids.append(str(original_id))
            token_strs.append(self.tokenizer.convert_ids_to_tokens([original_id])[0])

        return {
            "score_mean": float(np.mean(logps)),
            "score_sum": float(np.sum(logps)),
            "target_token_count": len(logps),
            "target_token_ids": " ".join(target_ids),
            "target_tokens": " ".join(token_strs),
            "score_ok": True,
            "score_notes": "",
        }


def bootstrap_ci(values: Sequence[float], n_boot: int = 5000, seed: int = 13) -> Tuple[float, float]:
    arr = np.asarray([v for v in values if pd.notna(v)], dtype=float)
    if len(arr) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means[i] = sample.mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize_deltas(deltas: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    rows = []
    for keys, g in deltas.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        vals = g["delta_yek_minus_bare"].dropna().to_numpy(dtype=float)
        lo, hi = bootstrap_ci(vals)
        out = {col: val for col, val in zip(group_cols, keys)}
        out.update({
            "n": int(len(vals)),
            "mean_delta": float(np.mean(vals)) if len(vals) else np.nan,
            "median_delta": float(np.median(vals)) if len(vals) else np.nan,
            "ci_low": lo,
            "ci_high": hi,
            "prop_positive": float(np.mean(vals > 0)) if len(vals) else np.nan,
            "min": float(np.min(vals)) if len(vals) else np.nan,
            "max": float(np.max(vals)) if len(vals) else np.nan,
        })
        rows.append(out)
    return pd.DataFrame(rows)


def paired_condition_deltas(scores: pd.DataFrame) -> pd.DataFrame:
    usable = scores[(scores["match_ok"]) & (scores["score_ok"])].copy()
    pivot = usable.pivot_table(
        index=["model_key", "model_name", "item_id", "anaphor_type", "target_kind"],
        columns="condition",
        values="score_mean",
        aggfunc="first",
    ).reset_index()
    if "bare" not in pivot.columns or "yek" not in pivot.columns:
        raise ValueError(
            f"Need both bare and yek conditions. Found columns after pivot: {list(pivot.columns)}"
        )
    pivot["delta_yek_minus_bare"] = pivot["yek"] - pivot["bare"]
    return pivot


def compute_object_minus_subject(deltas: pd.DataFrame) -> pd.DataFrame:
    wide = deltas.pivot_table(
        index=["model_key", "model_name", "item_id", "anaphor_type"],
        columns="target_kind",
        values="delta_yek_minus_bare",
        aggfunc="first",
    ).reset_index()

    if "object_full_np" in wide.columns and "subject_span" in wide.columns:
        wide["object_full_np_minus_subject_span"] = wide["object_full_np"] - wide["subject_span"]
    if "object_marker" in wide.columns and "subject_span" in wide.columns:
        wide["object_marker_minus_subject_span"] = wide["object_marker"] - wide["subject_span"]
    return wide


def format_table(df: pd.DataFrame, float_cols: Optional[List[str]] = None) -> str:
    if df.empty:
        return "_No rows._"
    out = df.copy()
    if float_cols is None:
        float_cols = [c for c in out.columns if out[c].dtype.kind in "fc"]
    for c in float_cols:
        out[c] = out[c].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    return out.to_markdown(index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stimuli", required=True, help="Path to v5.5.3 stimuli CSV.")
    ap.add_argument("--outdir", default="results/robustness_standard_pll/span_controls", help="Output directory.")
    ap.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS.keys()),
                    help="Model keys or HF identifiers. Defaults: mbert xlm-r parsbert.")
    ap.add_argument("--device", default=None, help="cpu, cuda, mps, or auto if omitted.")
    ap.add_argument("--limit-items", type=int, default=None, help="Debug: limit to first N items.")
    # Optional column overrides
    for key in COLUMN_CANDIDATES:
        ap.add_argument(f"--{key}-col", default=None)
    args = ap.parse_args()

    stimuli_path = Path(args.stimuli)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(stimuli_path)
    overrides = {key: getattr(args, f"{key}_col") for key in COLUMN_CANDIDATES}
    cols = detect_columns(df, overrides)

    df = df.copy()
    df["_condition_canon"] = df[cols["condition"]].map(canonical_condition)
    if args.limit_items:
        keep_items = list(dict.fromkeys(df[cols["item_id"]].astype(str)))[: args.limit_items]
        df = df[df[cols["item_id"]].astype(str).isin(keep_items)].copy()

    with open(outdir / "detected_columns.json", "w", encoding="utf-8") as f:
        json.dump(cols, f, ensure_ascii=False, indent=2)

    requests = build_span_requests(df, cols)
    req_df = pd.DataFrame([r.__dict__ for r in requests])
    req_df.to_csv(outdir / "span_requests_before_scoring.csv", index=False)

    match_summary = (
        req_df.groupby(["target_kind"])["match_ok"]
        .agg(["sum", "count"])
        .reset_index()
        .rename(columns={"sum": "matched", "count": "total"})
    )
    match_summary["match_rate"] = match_summary["matched"] / match_summary["total"]
    match_summary.to_csv(outdir / "target_match_summary.csv", index=False)

    all_scores = []
    for model_arg in args.models:
        if model_arg in DEFAULT_MODELS:
            model_key = model_arg
            model_name = DEFAULT_MODELS[model_arg]
        else:
            model_name = model_arg
            model_key = re.sub(r"[^A-Za-z0-9]+", "_", model_arg).strip("_").lower()

        print(f"\n=== Loading {model_key}: {model_name} ===", flush=True)
        scorer = PLLSpanScorer(model_name, device=args.device)

        for i, req in enumerate(requests, start=1):
            base = req.__dict__.copy()
            base["model_key"] = model_key
            base["model_name"] = model_name
            if not req.match_ok:
                base.update({
                    "score_mean": np.nan,
                    "score_sum": np.nan,
                    "target_token_count": 0,
                    "target_token_ids": "",
                    "target_tokens": "",
                    "score_ok": False,
                    "score_notes": req.notes,
                })
            else:
                try:
                    scored = scorer.score_span(req.full_text, int(req.char_start), int(req.char_end))
                    base.update(scored)
                except Exception as e:
                    base.update({
                        "score_mean": np.nan,
                        "score_sum": np.nan,
                        "target_token_count": 0,
                        "target_token_ids": "",
                        "target_tokens": "",
                        "score_ok": False,
                        "score_notes": f"exception: {type(e).__name__}: {e}",
                    })
            all_scores.append(base)
            if i % 50 == 0:
                print(f"{model_key}: scored {i}/{len(requests)} span requests", flush=True)

        # Free memory before next model
        del scorer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    scores = pd.DataFrame(all_scores)
    scores.to_csv(outdir / "yek_span_control_scores_raw.csv", index=False)

    deltas = paired_condition_deltas(scores)
    deltas.to_csv(outdir / "yek_span_control_deltas.csv", index=False)

    summary_by_model_kind = summarize_deltas(deltas, ["model_key", "target_kind"])
    summary_by_model_kind.to_csv(outdir / "summary_by_model_and_target_kind.csv", index=False)

    summary_by_model_kind_type = summarize_deltas(deltas, ["model_key", "target_kind", "anaphor_type"])
    summary_by_model_kind_type.to_csv(outdir / "summary_by_model_target_kind_and_anaphor_type.csv", index=False)

    obj_subj = compute_object_minus_subject(deltas)
    obj_subj.to_csv(outdir / "object_minus_subject_item_deltas.csv", index=False)

    obj_subj_summaries = []
    for metric in ["object_full_np_minus_subject_span", "object_marker_minus_subject_span"]:
        if metric in obj_subj.columns:
            tmp = obj_subj[["model_key", "model_name", "item_id", "anaphor_type", metric]].copy()
            tmp = tmp.rename(columns={metric: "delta_yek_minus_bare"})
            tmp["target_kind"] = metric
            obj_subj_summaries.append(tmp)
    if obj_subj_summaries:
        obj_subj_long = pd.concat(obj_subj_summaries, ignore_index=True)
        obj_subj_summary = summarize_deltas(obj_subj_long, ["model_key", "target_kind"])
        obj_subj_summary.to_csv(outdir / "summary_object_minus_subject.csv", index=False)
        obj_subj_summary_type = summarize_deltas(obj_subj_long, ["model_key", "target_kind", "anaphor_type"])
        obj_subj_summary_type.to_csv(outdir / "summary_object_minus_subject_by_anaphor_type.csv", index=False)
    else:
        obj_subj_summary = pd.DataFrame()
        obj_subj_summary_type = pd.DataFrame()

    # Cross-model positive counts by target kind.
    cross_rows = []
    for target_kind, g in deltas.groupby("target_kind"):
        wide = g.pivot_table(
            index=["item_id", "anaphor_type"],
            columns="model_key",
            values="delta_yek_minus_bare",
            aggfunc="first",
        ).reset_index()
        model_cols = [m for m in DEFAULT_MODELS.keys() if m in wide.columns]
        if model_cols:
            ok = wide[model_cols].notna().all(axis=1)
            all_pos = (wide[model_cols] > 0).all(axis=1) & ok
            cross_rows.append({
                "target_kind": target_kind,
                "n_items_with_all_models": int(ok.sum()),
                "positive_all_models": int(all_pos.sum()),
                "prop_positive_all_models": float(all_pos.sum() / ok.sum()) if ok.sum() else np.nan,
            })
    cross_model = pd.DataFrame(cross_rows)
    cross_model.to_csv(outdir / "cross_model_positive_counts.csv", index=False)

    md = []
    md.append("# Persian ye(k) span-control scoring report\n")
    md.append(f"Stimuli: `{stimuli_path}`\n")
    md.append("## Detected columns\n")
    md.append("```json\n" + json.dumps(cols, ensure_ascii=False, indent=2) + "\n```\n")

    md.append("## Target matching before scoring\n")
    md.append(format_table(match_summary) + "\n")

    md.append("## Main delta summaries: score(yek) - score(bare)\n")
    md.append(format_table(summary_by_model_kind) + "\n")

    md.append("## Main delta summaries by anaphor type\n")
    md.append(format_table(summary_by_model_kind_type) + "\n")

    md.append("## Object-minus-subject interaction summaries\n")
    if not obj_subj_summary.empty:
        md.append(format_table(obj_subj_summary) + "\n")
    else:
        md.append("_Could not compute object-minus-subject summaries._\n")

    md.append("## Object-minus-subject by anaphor type\n")
    if not obj_subj_summary_type.empty:
        md.append(format_table(obj_subj_summary_type) + "\n")
    else:
        md.append("_Could not compute object-minus-subject by anaphor type._\n")

    md.append("## Cross-model positive counts\n")
    md.append(format_table(cross_model) + "\n")

    (outdir / "YEK_SPAN_CONTROL_REPORT.md").write_text("\n".join(md), encoding="utf-8")

    print(f"\nDone. Wrote outputs to: {outdir}")
    print(f"Read: {outdir / 'YEK_SPAN_CONTROL_REPORT.md'}")


if __name__ == "__main__":
    main()

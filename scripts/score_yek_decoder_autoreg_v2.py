#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def choose_device(force_device=None):
    if force_device:
        return force_device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(model_id, device):
    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    print(f"Loading model: {model_id}")
    dtype = torch.float16 if device in {"mps", "cuda"} else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        trust_remote_code=True,
    )

    model.to(device)
    model.eval()
    return tokenizer, model


def score_continuation(tokenizer, model, device, context, continuation):
    """Score a continuation with an autoregressive language model.

    Context and continuation are tokenized separately and then concatenated.
    The continuation is tokenized with a leading space.
    """
    context = str(context).strip()
    continuation = str(continuation).strip()

    context_ids = tokenizer(
        context,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"]

    continuation_text = " " + continuation

    continuation_ids = tokenizer(
        continuation_text,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"]

    if continuation_ids.shape[1] <= 0:
        raise ValueError(
            f"Continuation has no tokens. Context={context!r}, continuation={continuation!r}"
        )

    full_ids = torch.cat([context_ids, continuation_ids], dim=1)
    context_len = context_ids.shape[1]
    continuation_len = continuation_ids.shape[1]
    full_len = full_ids.shape[1]

    input_ids = full_ids.to(device)

    with torch.no_grad():
        logits = model(input_ids).logits

    # Token at position k is predicted by logits at position k-1.
    # Continuation tokens start at index context_len.
    pred_logits = logits[:, context_len - 1 : full_len - 1, :].float()
    target_ids = input_ids[:, context_len:full_len]

    if pred_logits.shape[1] != target_ids.shape[1]:
        raise RuntimeError(
            f"Shape mismatch: pred_logits={pred_logits.shape}, target_ids={target_ids.shape}"
        )

    log_probs = torch.log_softmax(pred_logits, dim=-1)
    token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

    sum_logprob = float(token_log_probs.sum().detach().cpu())
    mean_logprob = float(token_log_probs.mean().detach().cpu())

    return {
        "mean_logprob": mean_logprob,
        "sum_logprob": sum_logprob,
        "token_count": int(continuation_len),
        "continuation_text_scored": continuation_text,
    }


def bootstrap_ci(values, n_boot=10000, seed=13):
    values = pd.Series(values).dropna().to_numpy()
    if len(values) == 0:
        return (math.nan, math.nan)

    gen = torch.Generator()
    gen.manual_seed(seed)

    vals = torch.tensor(values, dtype=torch.float32)
    n = vals.numel()

    means = []
    for _ in range(n_boot):
        idx = torch.randint(0, n, (n,), generator=gen)
        means.append(vals[idx].mean().item())

    means = sorted(means)
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return lo, hi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="stimuli/continuation_yek_v5_5_3_same_np_smoothing_48.csv",
    )
    parser.add_argument(
        "--outdir",
        default="results/decoder/whole_continuation/qwen25_0_5b",
    )
    parser.add_argument(
        "--model_id",
        default="Qwen/Qwen2.5-0.5B",
    )
    parser.add_argument(
        "--model_key",
        default="qwen25_0_5b",
    )
    parser.add_argument(
        "--device",
        default=None,
    )
    parser.add_argument(
        "--max_items",
        type=int,
        default=None,
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    print(f"Device: {device}")

    df = pd.read_csv(args.input)

    required = [
        "item_id",
        "condition",
        "context",
        "continuation_subject_continuing",
        "continuation_object_anaphoric",
        "subject",
        "object",
        "verb",
        "object_anaphor_type",
        "object_domain",
        "target_span",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    stimuli = (
        df[required]
        .drop_duplicates(["item_id", "condition"])
        .sort_values(["item_id", "condition"])
        .reset_index(drop=True)
    )

    if args.max_items is not None:
        keep_items = stimuli["item_id"].drop_duplicates().head(args.max_items)
        stimuli = stimuli[stimuli["item_id"].isin(keep_items)].reset_index(drop=True)

    print(f"Stimulus condition rows to score: {len(stimuli)}")
    print(f"Unique items: {stimuli['item_id'].nunique()}")

    tokenizer, model = load_model(args.model_id, device)

    raw_rows = []

    for _, row in tqdm(stimuli.iterrows(), total=len(stimuli), desc="Scoring rows"):
        context = row["context"]

        subj = score_continuation(
            tokenizer,
            model,
            device,
            context,
            row["continuation_subject_continuing"],
        )

        obj = score_continuation(
            tokenizer,
            model,
            device,
            context,
            row["continuation_object_anaphoric"],
        )

        object_preference_margin = obj["mean_logprob"] - subj["mean_logprob"]

        raw_rows.append(
            {
                "decoder_model_key": args.model_key,
                "decoder_model_id": args.model_id,
                "item_id": row["item_id"],
                "condition": row["condition"],
                "subject": row["subject"],
                "object": row["object"],
                "verb": row["verb"],
                "object_anaphor_type": row["object_anaphor_type"],
                "object_domain": row["object_domain"],
                "target_span": row["target_span"],
                "context": row["context"],
                "continuation_subject_continuing": row["continuation_subject_continuing"],
                "continuation_object_anaphoric": row["continuation_object_anaphoric"],
                "subject_score": subj["mean_logprob"],
                "subject_sum_logprob": subj["sum_logprob"],
                "subject_token_count": subj["token_count"],
                "object_score": obj["mean_logprob"],
                "object_sum_logprob": obj["sum_logprob"],
                "object_token_count": obj["token_count"],
                "object_preference_margin": object_preference_margin,
                "object_preferred": object_preference_margin > 0,
            }
        )

    raw = pd.DataFrame(raw_rows)
    raw_path = outdir / "decoder_scores_raw.csv"
    raw.to_csv(raw_path, index=False)

    bare = raw[raw["condition"] == "bare"].copy()
    yek = raw[raw["condition"] == "yek"].copy()

    merged = bare.merge(
        yek,
        on=[
            "decoder_model_key",
            "decoder_model_id",
            "item_id",
            "subject",
            "object",
            "verb",
            "object_anaphor_type",
            "object_domain",
            "target_span",
        ],
        suffixes=("_bare", "_yek"),
    )

    item = pd.DataFrame(
        {
            "decoder_model_key": merged["decoder_model_key"],
            "decoder_model_id": merged["decoder_model_id"],
            "item_id": merged["item_id"],
            "subject": merged["subject"],
            "object": merged["object"],
            "verb": merged["verb"],
            "object_anaphor_type": merged["object_anaphor_type"],
            "object_domain": merged["object_domain"],
            "target_span": merged["target_span"],
            "bare_subject_score": merged["subject_score_bare"],
            "bare_object_score": merged["object_score_bare"],
            "bare_margin": merged["object_preference_margin_bare"],
            "yek_subject_score": merged["subject_score_yek"],
            "yek_object_score": merged["object_score_yek"],
            "yek_margin": merged["object_preference_margin_yek"],
        }
    )

    item["delta_margin"] = item["yek_margin"] - item["bare_margin"]
    item["predicted_direction"] = item["delta_margin"] > 0

    item_path = outdir / "decoder_item_deltas.csv"
    item.to_csv(item_path, index=False)

    summaries = []

    def summarize(label, g):
        vals = g["delta_margin"].dropna()
        ci_lo, ci_hi = bootstrap_ci(vals)
        return {
            "group": label,
            "n": len(vals),
            "mean_delta": vals.mean(),
            "median_delta": vals.median(),
            "ci_low": ci_lo,
            "ci_high": ci_hi,
            "prop_positive": (vals > 0).mean(),
            "min_delta": vals.min(),
            "max_delta": vals.max(),
        }

    summaries.append(summarize("all", item))

    for typ, g in item.groupby("object_anaphor_type"):
        summaries.append(summarize(f"anaphor_type={typ}", g))

    summary = pd.DataFrame(summaries)
    summary_path = outdir / "decoder_summary.csv"
    summary.to_csv(summary_path, index=False)

    metadata = {
        "model_id": args.model_id,
        "model_key": args.model_key,
        "device": device,
        "n_condition_rows": int(len(raw)),
        "n_items": int(item["item_id"].nunique()),
    }

    meta_path = outdir / "decoder_run_metadata.json"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nWrote:")
    print(f"  {raw_path}")
    print(f"  {item_path}")
    print(f"  {summary_path}")
    print(f"  {meta_path}")

    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

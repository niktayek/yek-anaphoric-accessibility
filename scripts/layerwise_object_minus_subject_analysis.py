from pathlib import Path
import math
import warnings

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel


ROOT = Path(__file__).resolve().parents[1]

OUT_DIR = ROOT / "results" / "layerwise"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WHOLE_FILE = (
    ROOT
    / "results"
    / "primary_pll_word_l2r"
    / "whole_continuation"
    / "continuation_yek_v5_5_3_pll_word_l2r_scores.csv"
)

BEHAVIOR_FILE = (
    ROOT
    / "results"
    / "primary_pll_word_l2r"
    / "span_controls"
    / "yek_span_control_deltas.csv"
)

MODELS = [
    ("mbert", "bert-base-multilingual-cased"),
    ("xlm-r", "xlm-roberta-base"),
    ("parsbert", "HooshvareLab/bert-fa-base-uncased"),
]

if torch.cuda.is_available():
    DEVICE = "cuda"
elif (
    hasattr(torch.backends, "mps")
    and torch.backends.mps.is_available()
):
    DEVICE = "mps"
else:
    DEVICE = "cpu"
SEED = 13
N_BOOT = 10000

# Inclusion thresholds for the layer-wise analysis.
EARLY_MAX_LAYER = 2


def bootstrap_ci(values, n_boot=N_BOOT, seed=SEED):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boots.append(np.mean(sample))
    return np.percentile(boots, [2.5, 97.5])


def safe_corr(x, y, method="pearson"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        return np.nan
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    if method == "spearman":
        xr = pd.Series(x).rank(method="average").to_numpy()
        yr = pd.Series(y).rank(method="average").to_numpy()
        return float(np.corrcoef(xr, yr)[0, 1])

    return float(np.corrcoef(x, y)[0, 1])


def cosine_distance(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return np.nan
    return float(1.0 - np.dot(a, b) / denom)


def normalize_space(s):
    return " ".join(str(s).split())


def find_span(full_text, context, continuation, target_text):
    """
    Returns char start/end of target_text inside full_text.
    Searches inside the continuation region to avoid accidental context matches.
    """
    context = str(context)
    continuation = str(continuation)
    target_text = str(target_text)

    joiner = " "
    cont_start = len(context) + len(joiner)
    cont_text = full_text[cont_start:]

    idx = cont_text.find(target_text)
    if idx >= 0:
        start = cont_start + idx
        return start, start + len(target_text)

    # Space-normalized fallback.
    norm_target = normalize_space(target_text)
    norm_cont = normalize_space(cont_text)
    idx_norm = norm_cont.find(norm_target)
    if idx_norm < 0:
        return None

    # Recover approximately by searching for first and last words.
    words = norm_target.split()
    if not words:
        return None
    first = words[0]
    last = words[-1]

    first_idx = cont_text.find(first)
    if first_idx < 0:
        return None

    search_from = first_idx
    last_idx = cont_text.find(last, search_from)
    if last_idx < 0:
        return None

    start = cont_start + first_idx
    end = cont_start + last_idx + len(last)
    return start, end


def token_indices_for_char_span(offsets, start, end):
    idxs = []
    for i, (a, b) in enumerate(offsets):
        # Special tokens often have (0, 0).
        if a == b:
            continue
        # Token overlaps the target char span.
        if b > start and a < end:
            idxs.append(i)
    return idxs


def pooled_hidden_for_span(model, tokenizer, full_text, span_start, span_end):
    enc = tokenizer(
        full_text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=512,
    )
    offsets = enc.pop("offset_mapping")[0].tolist()
    token_idxs = token_indices_for_char_span(offsets, span_start, span_end)

    if not token_idxs:
        return None, []

    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)

    hidden_states = out.hidden_states  # embedding + layers
    pooled_by_layer = []

    for hs in hidden_states:
        # hs shape: [batch, seq, hidden]
        span_hs = hs[0, token_idxs, :]
        pooled = span_hs.mean(dim=0).detach().cpu().numpy()
        pooled_by_layer.append(pooled)

    return pooled_by_layer, token_idxs


def load_items():
    if not WHOLE_FILE.exists():
        raise FileNotFoundError(f"Missing file: {WHOLE_FILE}")

    df = pd.read_csv(WHOLE_FILE)

    needed = [
        "item_id",
        "condition",
        "subject",
        "object",
        "context",
        "continuation_subject_continuing",
        "continuation_object_anaphoric",
        "target_span",
        "object_anaphor_type",
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {WHOLE_FILE}: {missing}")

    # One row per item/condition is enough; model rows duplicate the same stimuli.
    stim = df[needed].drop_duplicates()

    rows = []
    for item_id, g in stim.groupby("item_id", sort=True):
        bare = g[g["condition"].eq("bare")]
        yek = g[g["condition"].eq("yek")]

        if len(bare) != 1 or len(yek) != 1:
            print(f"Skipping {item_id}: expected exactly one bare and one yek row after dedup.")
            continue

        bare = bare.iloc[0].to_dict()
        yek = yek.iloc[0].to_dict()

        # Object NP target: target_span + object, e.g. این کوله پشتی / همان لیوان.
        marker = str(bare["target_span"])
        obj = str(bare["object"])
        obj_np = f"{marker} {obj}"

        # Subject target: subject name in subject-continuing continuation.
        subj = str(bare["subject"])

        rows.append({
            "item_id": item_id,
            "subject": subj,
            "object": obj,
            "object_anaphor_type": bare["object_anaphor_type"],
            "object_np_target": obj_np,
            "subject_target": subj,
            "bare_context": bare["context"],
            "yek_context": yek["context"],
            "object_continuation": bare["continuation_object_anaphoric"],
            "subject_continuation": bare["continuation_subject_continuing"],
        })

    out = pd.DataFrame(rows)
    if len(out) != 48:
        print(f"WARNING: expected 48 items, got {len(out)}.")
    return out


def load_behavioral_objnp():
    if not BEHAVIOR_FILE.exists():
        print(f"WARNING: missing behavioral file: {BEHAVIOR_FILE}")
        return pd.DataFrame(columns=["model_key", "item_id", "behavior_delta_objNP"])

    b = pd.read_csv(BEHAVIOR_FILE)
    needed = {"model_key", "item_id", "target_kind", "delta_yek_minus_bare"}
    missing = needed - set(b.columns)
    if missing:
        print(f"WARNING: missing behavioral columns: {missing}")
        return pd.DataFrame(columns=["model_key", "item_id", "behavior_delta_objNP"])

    b = b[b["target_kind"].eq("object_full_np")].copy()
    b = b.rename(columns={"delta_yek_minus_bare": "behavior_delta_objNP"})
    return b[["model_key", "item_id", "behavior_delta_objNP"]]


def run_model(model_key, model_id, items):
    print("\n" + "=" * 90)
    print(f"Loading {model_key}: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModel.from_pretrained(model_id)
    model.to(DEVICE)
    model.eval()

    rows = []
    failures = []

    for _, row in items.iterrows():
        item_id = row["item_id"]

        # Object-anaphoric continuation sequences.
        bare_obj_full = str(row["bare_context"]) + " " + str(row["object_continuation"])
        yek_obj_full = str(row["yek_context"]) + " " + str(row["object_continuation"])

        bare_obj_span = find_span(
            bare_obj_full,
            row["bare_context"],
            row["object_continuation"],
            row["object_np_target"],
        )
        yek_obj_span = find_span(
            yek_obj_full,
            row["yek_context"],
            row["object_continuation"],
            row["object_np_target"],
        )

        # Subject-continuing continuation sequences.
        bare_subj_full = str(row["bare_context"]) + " " + str(row["subject_continuation"])
        yek_subj_full = str(row["yek_context"]) + " " + str(row["subject_continuation"])

        bare_subj_span = find_span(
            bare_subj_full,
            row["bare_context"],
            row["subject_continuation"],
            row["subject_target"],
        )
        yek_subj_span = find_span(
            yek_subj_full,
            row["yek_context"],
            row["subject_continuation"],
            row["subject_target"],
        )

        if not all([bare_obj_span, yek_obj_span, bare_subj_span, yek_subj_span]):
            failures.append({
                "model": model_key,
                "item_id": item_id,
                "reason": "char_span_not_found",
                "object_np_target": row["object_np_target"],
                "subject_target": row["subject_target"],
                "object_continuation": row["object_continuation"],
                "subject_continuation": row["subject_continuation"],
            })
            continue

        bare_obj_h, bare_obj_toks = pooled_hidden_for_span(
            model, tokenizer, bare_obj_full, bare_obj_span[0], bare_obj_span[1]
        )
        yek_obj_h, yek_obj_toks = pooled_hidden_for_span(
            model, tokenizer, yek_obj_full, yek_obj_span[0], yek_obj_span[1]
        )
        bare_subj_h, bare_subj_toks = pooled_hidden_for_span(
            model, tokenizer, bare_subj_full, bare_subj_span[0], bare_subj_span[1]
        )
        yek_subj_h, yek_subj_toks = pooled_hidden_for_span(
            model, tokenizer, yek_subj_full, yek_subj_span[0], yek_subj_span[1]
        )

        if any(x is None for x in [bare_obj_h, yek_obj_h, bare_subj_h, yek_subj_h]):
            failures.append({
                "model": model_key,
                "item_id": item_id,
                "reason": "token_span_not_found",
                "object_np_target": row["object_np_target"],
                "subject_target": row["subject_target"],
            })
            continue

        n_layers = min(len(bare_obj_h), len(yek_obj_h), len(bare_subj_h), len(yek_subj_h))

        for layer in range(n_layers):
            obj_dist = cosine_distance(bare_obj_h[layer], yek_obj_h[layer])
            subj_dist = cosine_distance(bare_subj_h[layer], yek_subj_h[layer])
            contrast = obj_dist - subj_dist

            rows.append({
                "model": model_key,
                "model_id": model_id,
                "item_id": item_id,
                "object_anaphor_type": row["object_anaphor_type"],
                "layer": layer,
                "is_embedding_layer": layer == 0,
                "object_np_target": row["object_np_target"],
                "subject_target": row["subject_target"],
                "bare_obj_token_count": len(bare_obj_toks),
                "yek_obj_token_count": len(yek_obj_toks),
                "bare_subj_token_count": len(bare_subj_toks),
                "yek_subj_token_count": len(yek_subj_toks),
                "obj_distance": obj_dist,
                "subj_distance": subj_dist,
                "obj_minus_subj_distance": contrast,
            })

    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return pd.DataFrame(rows), pd.DataFrame(failures)


def summarize_layers(df):
    rows = []
    for (model, layer), g in df.groupby(["model", "layer"], sort=True):
        vals = g["obj_minus_subj_distance"].to_numpy()
        ci_low, ci_high = bootstrap_ci(vals)
        rows.append({
            "model": model,
            "layer": layer,
            "n": len(g),
            "mean_obj_distance": g["obj_distance"].mean(),
            "mean_subj_distance": g["subj_distance"].mean(),
            "mean_obj_minus_subj_distance": np.mean(vals),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "positive_items": int((g["obj_minus_subj_distance"] > 0).sum()),
        })
    return pd.DataFrame(rows)


def peak_summary(layer_summary):
    rows = []
    for model, g in layer_summary.groupby("model", sort=True):
        g = g.copy()
        max_layer = int(g["layer"].max())
        non_early = g[g["layer"] > EARLY_MAX_LAYER].copy()

        peak_all = g.loc[g["mean_obj_minus_subj_distance"].idxmax()]
        peak_non_early = non_early.loc[non_early["mean_obj_minus_subj_distance"].idxmax()]

        early = g[g["layer"] <= EARLY_MAX_LAYER]
        mid_upper = g[g["layer"] > EARLY_MAX_LAYER]

        rows.append({
            "model": model,
            "max_layer": max_layer,
            "peak_layer_all": int(peak_all["layer"]),
            "peak_mean_all": peak_all["mean_obj_minus_subj_distance"],
            "peak_ci_low_all": peak_all["ci_low"],
            "peak_ci_high_all": peak_all["ci_high"],
            "peak_layer_non_early": int(peak_non_early["layer"]),
            "peak_mean_non_early": peak_non_early["mean_obj_minus_subj_distance"],
            "peak_ci_low_non_early": peak_non_early["ci_low"],
            "peak_ci_high_non_early": peak_non_early["ci_high"],
            "mean_early_layers_0_2": early["mean_obj_minus_subj_distance"].mean(),
            "mean_non_early_layers": mid_upper["mean_obj_minus_subj_distance"].mean(),
            "non_early_peak_ci_above_zero": bool(peak_non_early["ci_low"] > 0),
            "peak_not_early": bool(int(peak_all["layer"]) > EARLY_MAX_LAYER),
            "best_non_early_positive": bool(peak_non_early["mean_obj_minus_subj_distance"] > 0),
        })

    return pd.DataFrame(rows)


def behavior_correlations(item_df, behavioral):
    if behavioral.empty:
        return pd.DataFrame()

    merged = item_df.merge(
        behavioral,
        left_on=["model", "item_id"],
        right_on=["model_key", "item_id"],
        how="left",
    )

    rows = []
    for (model, layer), g in merged.groupby(["model", "layer"], sort=True):
        if g["behavior_delta_objNP"].notna().sum() < 3:
            continue

        rows.append({
            "model": model,
            "layer": layer,
            "n": int(g["behavior_delta_objNP"].notna().sum()),
            "pearson_rep_contrast_vs_behavior_objNP": safe_corr(
                g["obj_minus_subj_distance"],
                g["behavior_delta_objNP"],
                method="pearson",
            ),
            "spearman_rep_contrast_vs_behavior_objNP": safe_corr(
                g["obj_minus_subj_distance"],
                g["behavior_delta_objNP"],
                method="spearman",
            ),
            "pearson_obj_distance_vs_behavior_objNP": safe_corr(
                g["obj_distance"],
                g["behavior_delta_objNP"],
                method="pearson",
            ),
            "spearman_obj_distance_vs_behavior_objNP": safe_corr(
                g["obj_distance"],
                g["behavior_delta_objNP"],
                method="spearman",
            ),
        })

    return pd.DataFrame(rows)


def write_decision(peak_df, corr_df):
    n_models = len(peak_df)

    ci_positive = int(peak_df["non_early_peak_ci_above_zero"].sum())
    peak_not_early = int(peak_df["peak_not_early"].sum())
    best_non_early_positive = int(peak_df["best_non_early_positive"].sum())

    criterion_1_pass = ci_positive >= 2
    criterion_2_pass = peak_not_early >= 2
    criterion_3_pass = best_non_early_positive == n_models

    include = (
        criterion_1_pass
        and criterion_2_pass
        and criterion_3_pass
    )

    lines = [
        f"n_models={n_models}",
        "non_early_peak_ci_positive_required=2",
        f"non_early_peak_ci_positive_observed={ci_positive}",
        f"non_early_peak_ci_positive_pass={str(criterion_1_pass).lower()}",
        "peak_not_early_required=2",
        f"peak_not_early_observed={peak_not_early}",
        f"peak_not_early_pass={str(criterion_2_pass).lower()}",
        f"best_non_early_positive_required={n_models}",
        f"best_non_early_positive_observed={best_non_early_positive}",
        f"best_non_early_positive_pass={str(criterion_3_pass).lower()}",
        f"include={str(include).lower()}",
        "peak_summary_file=layerwise_peak_summary.csv",
        "behavior_correlations_file=layerwise_behavior_correlations.csv",
    ]

    text = "\n".join(lines) + "\n"
    (OUT_DIR / "layerwise_include_drop_decision.txt").write_text(
        text,
        encoding="utf-8",
    )
    print("\n" + text)
    return include


def maybe_plot(layer_summary):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"Plot skipped: matplotlib unavailable ({e})")
        return

    plt.figure(figsize=(8, 5))
    for model, g in layer_summary.groupby("model", sort=True):
        plt.plot(
            g["layer"],
            g["mean_obj_minus_subj_distance"],
            marker="o",
            label=model,
        )
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Layer (0 = embedding layer)")
    plt.ylabel("Object-minus-subject representational distance")
    plt.title("Layer-wise ye(k) representational signature at uptake site")
    plt.legend()
    plt.tight_layout()
    out = OUT_DIR / "layerwise_obj_minus_subject_distance.png"
    plt.savefig(out, dpi=200)
    print(f"Wrote: {out}")


def main():
    print(f"Device: {DEVICE}")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    items = load_items()
    behavioral = load_behavioral_objnp()

    all_rows = []
    all_failures = []

    for model_key, model_id in MODELS:
        try:
            item_df, fail_df = run_model(model_key, model_id, items)
            all_rows.append(item_df)
            all_failures.append(fail_df)
        except Exception as e:
            print(f"ERROR while running {model_key}: {e}")
            raise

    item_df = pd.concat(all_rows, ignore_index=True)
    fail_df = pd.concat(all_failures, ignore_index=True) if all_failures else pd.DataFrame()

    item_path = OUT_DIR / "layerwise_item_distances.csv"
    fail_path = OUT_DIR / "layerwise_failures.csv"

    item_df.to_csv(item_path, index=False)
    fail_df.to_csv(fail_path, index=False)

    print(f"\nWrote: {item_path}")
    print(f"Wrote: {fail_path}")
    print(f"Failures: {len(fail_df)}")

    layer_summary = summarize_layers(item_df)
    peak_df = peak_summary(layer_summary)
    corr_df = behavior_correlations(item_df, behavioral)

    layer_summary_path = OUT_DIR / "layerwise_summary_by_model_layer.csv"
    peak_path = OUT_DIR / "layerwise_peak_summary.csv"
    corr_path = OUT_DIR / "layerwise_behavior_correlations.csv"

    layer_summary.to_csv(layer_summary_path, index=False)
    peak_df.to_csv(peak_path, index=False)
    corr_df.to_csv(corr_path, index=False)

    print(f"Wrote: {layer_summary_path}")
    print(f"Wrote: {peak_path}")
    print(f"Wrote: {corr_path}")

    print("\nPeak summary:")
    print(peak_df.to_string(index=False))

    print("\nLayer summary, first rows:")
    print(layer_summary.head(20).to_string(index=False))

    if len(corr_df):
        print("\nBehavior correlation summary, first rows:")
        print(corr_df.head(20).to_string(index=False))

    write_decision(peak_df, corr_df)
    maybe_plot(layer_summary)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()

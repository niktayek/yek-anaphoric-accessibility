#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np

STIM = Path("stimuli/continuation_yek_v5_5_3_same_np_smoothing_48.csv")
SPAN_DIR = Path("results/primary_pll_word_l2r/span_controls")
DELTAS = SPAN_DIR / "yek_span_control_deltas.csv"
INTERACTIONS = SPAN_DIR / "object_minus_subject_item_deltas.csv"
OUT = Path("results/diagnostics/span_effect_by_verb_domain_v5_5_3.md")
OUT.parent.mkdir(parents=True, exist_ok=True)

stim = pd.read_csv(STIM)
deltas = pd.read_csv(DELTAS)
inter = pd.read_csv(INTERACTIONS)

stim_items = (
    stim[["item_id", "subject", "object", "verb", "object_domain", "object_anaphor_type"]]
    .drop_duplicates("item_id")
)

deltas = deltas.merge(stim_items, on="item_id", how="left")
inter = inter.merge(stim_items, on="item_id", how="left")

def summarize(vals):
    vals = pd.Series(vals).dropna()
    return pd.Series({
        "n_items": int(len(vals)),
        "mean_delta": float(vals.mean()) if len(vals) else np.nan,
        "median_delta": float(vals.median()) if len(vals) else np.nan,
        "positive_items": int((vals > 0).sum()),
        "prop_positive": float((vals > 0).mean()) if len(vals) else np.nan,
        "min_delta": float(vals.min()) if len(vals) else np.nan,
        "max_delta": float(vals.max()) if len(vals) else np.nan,
    })

def group_summary(df, group_cols, value_col):
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        s = summarize(g[value_col])
        row = {col: val for col, val in zip(group_cols, keys)}
        row.update(s.to_dict())
        rows.append(row)
    return pd.DataFrame(rows)

def leave_one_out(df, group_col, value_col, extra_group_cols):
    rows = []
    for keys, g0 in df.groupby(extra_group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        for left_out in sorted(g0[group_col].dropna().unique()):
            g = g0[g0[group_col] != left_out]
            s = summarize(g[value_col])
            row = {col: val for col, val in zip(extra_group_cols, keys)}
            row[f"left_out_{group_col}"] = left_out
            row.update(s.to_dict())
            rows.append(row)
    return pd.DataFrame(rows)

def md(df, max_rows=300):
    if df is None or df.empty:
        return "_No data._"
    return df.head(max_rows).to_markdown(index=False, floatfmt=".4f")

# Main span summaries
by_model_target = group_summary(
    deltas,
    ["model_key", "target_kind"],
    "delta_yek_minus_bare"
)

by_model_target_anaphor = group_summary(
    deltas,
    ["model_key", "target_kind", "anaphor_type"],
    "delta_yek_minus_bare"
)

by_model_target_verb = group_summary(
    deltas,
    ["model_key", "target_kind", "verb"],
    "delta_yek_minus_bare"
)

by_model_target_domain = group_summary(
    deltas,
    ["model_key", "target_kind", "object_domain"],
    "delta_yek_minus_bare"
)

loo_target_verb = leave_one_out(
    deltas,
    "verb",
    "delta_yek_minus_bare",
    ["model_key", "target_kind"]
)

loo_target_domain = leave_one_out(
    deltas,
    "object_domain",
    "delta_yek_minus_bare",
    ["model_key", "target_kind"]
)

# Object-minus-subject interaction summaries
interaction_cols = [
    "object_full_np_minus_subject_span",
    "object_marker_minus_subject_span",
]

long_inter = inter.melt(
    id_vars=["model_key", "model_name", "item_id", "anaphor_type", "verb", "object_domain"],
    value_vars=interaction_cols,
    var_name="interaction_kind",
    value_name="interaction_delta",
)

by_interaction = group_summary(
    long_inter,
    ["model_key", "interaction_kind"],
    "interaction_delta"
)

by_interaction_anaphor = group_summary(
    long_inter,
    ["model_key", "interaction_kind", "anaphor_type"],
    "interaction_delta"
)

by_interaction_verb = group_summary(
    long_inter,
    ["model_key", "interaction_kind", "verb"],
    "interaction_delta"
)

by_interaction_domain = group_summary(
    long_inter,
    ["model_key", "interaction_kind", "object_domain"],
    "interaction_delta"
)

loo_interaction_verb = leave_one_out(
    long_inter,
    "verb",
    "interaction_delta",
    ["model_key", "interaction_kind"]
)

loo_interaction_domain = leave_one_out(
    long_inter,
    "object_domain",
    "interaction_delta",
    ["model_key", "interaction_kind"]
)

# Flags
flags = []

for _, row in loo_interaction_verb.iterrows():
    if row["mean_delta"] <= 0:
        flags.append(
            f"{row['model_key']} / {row['interaction_kind']}: non-positive after leaving out verb `{row['left_out_verb']}` "
            f"(mean={row['mean_delta']:.4f})"
        )

for _, row in loo_interaction_domain.iterrows():
    if row["mean_delta"] <= 0:
        flags.append(
            f"{row['model_key']} / {row['interaction_kind']}: non-positive after leaving out domain `{row['left_out_object_domain']}` "
            f"(mean={row['mean_delta']:.4f})"
        )

md_parts = []
md_parts.append("# Span-effect robustness by predicate and domain — v5.5.3 PLL-word-l2r\n")
md_parts.append(f"Stimulus file: `{STIM}`\n")
md_parts.append(f"Span deltas: `{DELTAS}`\n")
md_parts.append(f"Object-minus-subject interactions: `{INTERACTIONS}`\n")

md_parts.append("## Main target-kind summary\n")
md_parts.append(md(by_model_target))

md_parts.append("\n## Target-kind summary by anaphor type\n")
md_parts.append(md(by_model_target_anaphor))

md_parts.append("\n## Object-minus-subject interaction summary\n")
md_parts.append(md(by_interaction))

md_parts.append("\n## Object-minus-subject interaction by anaphor type\n")
md_parts.append(md(by_interaction_anaphor))

md_parts.append("\n## Target-kind summary by verb\n")
md_parts.append(md(by_model_target_verb, max_rows=500))

md_parts.append("\n## Target-kind summary by object domain\n")
md_parts.append(md(by_model_target_domain, max_rows=800))

md_parts.append("\n## Leave-one-verb-out: target kinds\n")
md_parts.append(md(loo_target_verb, max_rows=500))

md_parts.append("\n## Leave-one-domain-out: target kinds\n")
md_parts.append(md(loo_target_domain, max_rows=800))

md_parts.append("\n## Leave-one-verb-out: object-minus-subject interactions\n")
md_parts.append(md(loo_interaction_verb, max_rows=500))

md_parts.append("\n## Leave-one-domain-out: object-minus-subject interactions\n")
md_parts.append(md(loo_interaction_domain, max_rows=800))

md_parts.append("\n## Automatic flags\n")
if flags:
    for f in flags:
        md_parts.append(f"- {f}")
else:
    md_parts.append("- none")


OUT.write_text("\n".join(md_parts), encoding="utf-8")
print(f"Wrote {OUT}")

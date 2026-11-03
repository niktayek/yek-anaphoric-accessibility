#!/usr/bin/env python3
"""Score the English stimulus set with encoder or decoder models.

Writes item-level scores and a per-model summary to ``english/results/``.
"""
from __future__ import annotations
import argparse, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(__import__('os').cpu_count())

if torch.cuda.is_available():
    DEVICE = "cuda"
elif (
    hasattr(torch.backends, "mps")
    and torch.backends.mps.is_available()
):
    DEVICE = "mps"
else:
    DEVICE = "cpu"


# ----------------------------- span location -----------------------------
def _span_token_indices(offsets, span_char_start, span_char_end):
    """Return indices of tokens whose character span overlaps [start, end)."""
    idx = []
    for i, (a, b) in enumerate(offsets):
        if a == b:            # special tokens have (0,0)
            continue
        if b > span_char_start and a < span_char_end:
            idx.append(i)
    return idx


def _find_span_char_range(full_text, span_text, search_from):
    pos = full_text.find(span_text, search_from)
    if pos < 0:
        raise ValueError(f"span {span_text!r} not found after {search_from} in {full_text!r}")
    return pos, pos + len(span_text)


# ----------------------------- decoder scoring -----------------------------
def decoder_scores(model, tok, context, continuation, span_text=None):
    """Return (whole_mean_logprob, span_mean_logprob or None) for a causal LM."""
    ctx = context.strip() + " "
    cont = continuation.strip()
    full = ctx + cont
    enc = tok(full, return_tensors="pt", return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")[0].tolist()
    ids = enc["input_ids"].to(DEVICE)
    ctx_ids = tok(ctx, return_tensors="pt").input_ids
    n_ctx = ctx_ids.shape[1]
    if ids.shape[1] <= n_ctx:
        raise ValueError(f"empty continuation: {full!r}")

    with torch.inference_mode():
        logits = model(ids).logits
    logp = torch.log_softmax(logits[:, :-1, :], dim=-1)
    tgt = ids[:, 1:]
    tok_logp = logp.gather(2, tgt.unsqueeze(-1)).squeeze(-1)[0]  # len = L-1, position t = logp of token t+1

    pos = torch.arange(1, ids.shape[1], device=DEVICE)
    cont_mask = pos >= n_ctx
    whole = float(tok_logp[cont_mask].mean().cpu())

    span = None
    if span_text is not None:
        cs, ce = _find_span_char_range(full, span_text, len(ctx) - 1)
        span_tok = _span_token_indices(offsets, cs, ce)          # indices into full ids
        sel = [t - 1 for t in span_tok if t - 1 >= 0]            # tok_logp is shifted by 1
        if not sel:
            raise ValueError(f"no span tokens for {span_text!r}")
        span = float(tok_logp[sel].mean().cpu())
    return whole, span


# ----------------------------- encoder scoring (PLL-word-l2r) -----------------------------
def _word_groups(offsets, tok_ids, tok):
    """Group subtoken indices into words using the leading-space / ## heuristics via offsets."""
    # Words = maximal runs separated by whitespace in the ORIGINAL string.
    # Rebuild word membership from offsets. A new word starts after whitespace
    # or at the first real token.
    return offsets


def encoder_scores(model, tok, context, continuation, full_string, text_for_offsets,
                   score_char_start, score_char_end, mask_id):
    """
    PLL-word-l2r over the tokens whose characters fall in [score_char_start, score_char_end).
    For each target subtoken, mask it AND all later subtokens of the SAME word, then read
    the log prob of the true subtoken. Returns mean over scored subtokens.
    """
    enc = tok(full_string, return_tensors="pt", return_offsets_mapping=True)
    offsets = enc["offset_mapping"][0].tolist()
    ids = enc["input_ids"][0].tolist()

    # identify target subtoken indices in the scored region
    targets = [i for i, (a, b) in enumerate(offsets)
               if b > a and b > score_char_start and a < score_char_end]
    if not targets:
        raise ValueError(f"no target subtokens in [{score_char_start},{score_char_end}) for {full_string!r}")

    # assign each target a word id: new word when a gap (whitespace) precedes its char start
    word_of = {}
    for i in targets:
        a, _ = offsets[i]
        prev_ws = (a == 0) or (full_string[a - 1].isspace())
        # a token belongs to the same word as previous target if not preceded by whitespace
        if prev_ws:
            word_of[i] = i               # word starts here
        else:
            # inherit the most recent word-start <= i
            starts = [w for w in word_of.values() if w <= i]
            word_of[i] = max(starts) if starts else i

    # for word-l2r: future subtokens of the same word are those with same word id and larger index
    base = torch.tensor([ids], device=DEVICE)
    logps = []
    for i in targets:
        wid = word_of[i]
        future_same_word = [j for j in targets if word_of[j] == wid and j >= i]
        masked = base.clone()
        for j in future_same_word:
            masked[0, j] = mask_id
        with torch.inference_mode():
            logits = model(masked).logits
        lp = torch.log_softmax(logits[0, i], dim=-1)[ids[i]]
        logps.append(float(lp.cpu()))
    return float(np.mean(logps))


def encoder_whole_and_span(model, tok, context, continuation, span_text, mask_id):
    ctx = context.strip() + " "
    cont = continuation.strip()
    full = ctx + cont
    whole = encoder_scores(model, tok, context, continuation, full, full,
                           len(ctx), len(full), mask_id)     # continuation region
    span = None
    if span_text is not None:
        cs, ce = _find_span_char_range(full, span_text, len(ctx) - 1)
        span = encoder_scores(model, tok, context, continuation, full, full, cs, ce, mask_id)
    return whole, span


# ----------------------------- driver -----------------------------
def bootstrap_ci(vals, n_boot=10000, seed=13):
    rng = random.Random(seed); v = list(vals); m = []
    for _ in range(n_boot):
        s = [rng.choice(v) for _ in v]; m.append(float(np.mean(s)))
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def summarize(df):
    rows = []
    for (mdl, form, meas), sub in df.groupby(["model", "anaphor_form", "measure"]):
        vals = sub["value"].tolist()
        lo, hi = bootstrap_ci(vals)
        rows.append(dict(model=mdl, anaphor_form=form, measure=meas, n=len(vals),
                         mean=float(np.mean(vals)), median=float(np.median(vals)),
                         prop_positive=float(np.mean([x > 0 for x in vals])),
                         ci_low=lo, ci_high=hi,
                         n_pos=int(np.sum([x > 0 for x in vals])),
                         n_neg=int(np.sum([x < 0 for x in vals]))))
    return pd.DataFrame(rows).sort_values(["measure", "anaphor_form", "model"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-class", required=True, choices=["decoder", "encoder"])
    ap.add_argument("--stimuli", default="stimuli/english_discourse_v50.csv")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    items = pd.read_csv(args.stimuli)

    if args.model_class == "decoder":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model).to(DEVICE).eval()
        def score(ctx, cont, span=None): return decoder_scores(model, tok, ctx, cont, span)
        mask_id = None
    else:
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForMaskedLM.from_pretrained(args.model).to(DEVICE).eval()
        mask_id = tok.mask_token_id
        def score(ctx, cont, span=None): return encoder_whole_and_span(model, tok, ctx, cont, span, mask_id)

    rows = []
    for _, it in items.iterrows():
        forms = {"pron": (it.cont_pron, it.anaphor_span_pron),
                 "demNP": (it.cont_demNP, it.anaphor_span_demNP)}
        # Context keys used in the stimulus file.
        ctx_keys = ["indef", "this", "def", "neg"]
        neu = {c: score(getattr(it, f"{c}_context"), it.cont_neutral, None)[0]
               for c in ctx_keys}
        for form, (cont, span) in forms.items():
            w = {}; sp = {}
            for c in ctx_keys:
                wc, spc = score(getattr(it, f"{c}_context"), cont, span)
                w[c], sp[c] = wc, spc
            M = {c: w[c] - neu[c] for c in w}          # within-context object preference
            def add(measure, value):
                rows.append(dict(model=args.model, model_class=args.model_class,
                                 item_id=it.item_id, anaphor_form=form,
                                 measure=measure, value=value))
            base = len(rows)
            # this versus a
            add("spec_span",   sp["this"] - sp["indef"])
            add("spec_whole",  M["this"]  - M["indef"])
            # a versus the
            add("def_span",    sp["indef"] - sp["def"])
            add("def_whole",   M["indef"]  - M["def"])
            # affirmative versus negated a
            add("exist_whole", M["indef"]  - M["neg"])
            rows[base].update(dict(  # Store component scores on the spec_span row
                M_a=M["indef"], M_this=M["this"], M_def=M["def"], M_neg=M["neg"],
                span_a=sp["indef"], span_this=sp["this"], span_def=sp["def"], span_neg=sp["neg"]))
    df = pd.DataFrame(rows)
    ip = out / f"english_discourse_{args.tag}_item_scores.csv"
    sp_ = out / f"english_discourse_{args.tag}_summary.csv"
    df.to_csv(ip, index=False)
    summarize(df).to_csv(sp_, index=False)
    print(f"wrote {ip}\nwrote {sp_}")
    print(summarize(df).to_string(index=False))


if __name__ == "__main__":
    main()

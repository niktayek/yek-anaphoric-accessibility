# English comparison: indefinite *this* versus *a*

Files for §6.6 and Tables 6–7 of the paper.

## Files

- `stimuli/english_discourse_v50.csv`: 48 items.
  - Contexts: *a/an NP*, indefinite *this NP*, *the NP* and negated *a/an NP*.
  - Continuations, held fixed across contexts: a pronoun, a demonstrative NP and an item-matched neutral one.
- `scripts/score_english_discourse.py`: scores one model and writes the item-level results.
- `scripts/combine_english_results.py`: rebuilds `results/english_all_models_summary.csv` from the item scores (`--check` compares with the released file).
- `results/english_discourse_<tag>_item_scores.csv`: item-level scores for each of the 9 models.
- `results/english_all_models_summary.csv`: per-model summaries with bootstrap 95% CIs (10,000 resamples, seed 13).

## Measures

| `measure` | Definition | Used in |
|---|---|---|
| `spec_span` | Anaphor-span score after *this* minus after *a* | Table 6 uses `anaphor_form = pron` |
| `exist_whole` | Neutral-adjusted whole-continuation score, affirmative minus negated *a* context | Table 7 |
| `spec_whole` | Neutral-adjusted whole-continuation version of `spec_span` | not reported |
| `def_span`, `def_whole` | *a* minus *the* | not reported |

See §6.6 of the paper for the selection of reported rows.

## Models

These are `bert-base-cased`, `bert-base-multilingual-cased` and `xlm-roberta-base` (encoders), plus `Qwen/Qwen2.5-{0.5B,1.5B,3B}` and `facebook/xglm-{1.7B,2.9B,4.5B}` (decoders). Exact revisions are in the main README.

mGPT is not part of the English model set.

## Reproduce

Install `../requirements.txt`, then follow "English comparison" under "Full reruns" in the main README. Run commands from this `english/` directory.

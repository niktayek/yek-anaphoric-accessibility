# Data files

All files are UTF-8 CSV unless noted. Persian text is in Persian script. Scores are log probabilities (decoders) or pseudo-log-likelihoods (encoders), so higher means more expected. Every effect is a per-item difference: *ye(k)* minus bare, or *this* minus *a*.

## Persian stimuli

`stimuli/continuation_yek_v5_5_3_same_np_smoothing_48.csv` has 96 rows: 48 items × 2 conditions.

| Column | Meaning |
|---|---|
| `item_id` | Item identifier (`yek5_1_001` … `yek5_1_048`); shared by the two rows of a pair |
| `condition` | `bare` (bare object) or `yek` (*ye(k)*-marked object) |
| `subject`, `object`, `verb` | Lexical frame of the context sentence |
| `context` | Context sentence. The two conditions differ only in *ye(k)*. |
| `continuation_subject_continuing` | Continuation that keeps the subject as topic (control) |
| `continuation_object_anaphoric` | Continuation that refers back to the object. Identical across conditions. |
| `object_anaphor_type` | `demonstrative_np` (*in* 'this' + NP) or `same_np` (*hamān* 'the same' + NP); 24 items each |
| `object_domain` | Semantic domain of the object noun (used in the domain robustness audit) |
| `target_span` | Anaphoric marker that opens the object-anaphoric continuation (`این` or `همان`) |
| `notes` | Revision history of the item across development versions (v5.2 to v5.5.3). Kept because the validator checks it; not used in scoring. |

`stimuli/continuation_yek_v5_5_3_neutral_control_48.csv` is the same file plus `continuation_neutral`, an item-matched continuation with no object anaphor. It is used for the decoder fluency control (Table 5).

## English stimuli

`english/stimuli/english_discourse_v50.csv` has 48 rows, one per item.

| Column | Meaning |
|---|---|
| `item_id`, `subject`, `object_noun` | Item identifier and lexical frame |
| `indef_context`, `this_context`, `def_context`, `neg_context` | Context with *a/an NP*, indefinite *this NP*, *the NP*, and negated *a/an NP* |
| `cont_pron`, `cont_demNP`, `cont_neutral` | Pronoun, demonstrative-NP, and neutral continuations; identical across contexts |
| `anaphor_span_pron`, `anaphor_span_demNP` | The anaphor span scored locally (for example `it`, `that notebook`) |

## Persian encoder results

**Whole continuation:** `results/primary_pll_word_l2r/whole_continuation/` (primary, PLL-word-l2r) and `results/robustness_standard_pll/whole_continuation/` (standard PLL).

- `*_scores.csv` has one row per model × item × condition, with the stimulus columns plus:
  - `subject_score`, `object_score`: length-normalised score of each continuation after the context.
  - `object_preference_margin`: `object_score − subject_score`.
  - `prior_*`: the same scores for the continuations without any context.
  - `context_minus_prior_margin`: the margin minus its prior.
  - `bare_margin`, `yek_margin`, `delta_margin`: the margins per condition and their difference, *ye(k)* − bare (Δwhole, Table 3). These are repeated on both rows of a pair.
  - `predicted_direction`: whether `delta_margin` > 0.
- `*_summary.json`: per-model means, bootstrap CIs, proportions positive, and breakdowns by anaphor type and domain.

**Local spans:** `results/primary_pll_word_l2r/span_controls/` and `results/robustness_standard_pll/span_controls/`.

| File | Contents |
|---|---|
| `span_requests_before_scoring.csv` | Every span to be scored: `target_kind`, `target_text`, character offsets in `full_text`, and `match_ok`. This file is also the input to the decoder span controls. |
| `yek_span_control_scores_raw.csv` | Score of each span per model × item × condition (`score_mean`, `score_sum`, tokenisation) |
| `yek_span_control_deltas.csv` | `bare`, `yek` and `delta_yek_minus_bare` per model × item × `target_kind` |
| `object_minus_subject_item_deltas.csv` | Per item: the three span deltas, plus `object_full_np_minus_subject_span` and `object_marker_minus_subject_span` (the H2 contrasts) |
| `summary_*.csv`, `cross_model_positive_counts.csv` | Bootstrap summaries at different aggregation levels |

`target_kind` values:
- `object_marker`: the anaphoric marker (Δmarker).
- `object_full_np`: the full anaphoric NP (ΔobjNP).
- `subject_span`: the subject mention in the subject-continuing continuation (Δsubj).

## Persian decoder results

Per-model outputs are written by the three decoder scripts into `results/decoder/<analysis>/<model_key>/`:

| Analysis directory | Files |
|---|---|
| `whole_continuation/` | `decoder_scores_raw.csv` (one row per item × condition; mean and summed log probability and token count for each continuation), `decoder_item_deltas.csv` (`bare_margin`, `yek_margin`, `delta_margin` per item), `decoder_summary.csv`, `decoder_run_metadata.json` |
| `span_controls/` | `decoder_span_scores_raw.csv` (span log probability per item × condition × `target_kind`), `decoder_span_deltas.csv` (`delta_yek_minus_bare`) |
| `neutral_control/` | `decoder_neutral_scores_raw.csv`, `decoder_neutral_item_deltas.csv` (`neutral_delta_yek_minus_bare`), `decoder_neutral_summary.csv` |

`decoder_run_metadata.json` records the model ID, model key, device, and row counts for each decoder run.

`scripts/aggregate_decoder_results.py` combines these into `results/all_decoder_*.csv`:

| File | Contents |
|---|---|
| `all_decoder_summary_readable.csv`, `all_decoder_summary_all_items_only.csv` | Whole-continuation Δ per model, overall and by anaphor type (Table 3) |
| `all_decoder_span_scores_raw.csv`, `all_decoder_span_deltas.csv` | All models' span scores and deltas stacked |
| `all_decoder_span_summary_by_target_kind*.csv` | Span deltas summarised by model × target kind (× anaphor type) |
| `all_decoder_object_minus_subject_item_deltas.csv`, `all_decoder_span_summary_object_minus_subject*.csv` | ObjNP − Subject and Marker − Subject contrasts, per item and summarised (Table 5) |
| `all_decoder_neutral_summary.csv` | Neutral-continuation Δ per model |
| `all_decoder_object_neutral_item_contrasts.csv`, `all_decoder_object_neutral_summary.csv` | Span deltas minus the neutral Δ, per item and summarised (Table 5) |

In `all_decoder_neutral_summary.csv`, the `token_count_*` columns are filled for six of the seven models. They were recorded in the original run; the released neutral-control script does not write them.

Summary columns: `n`, `mean`/`mean_delta`, `median`/`median_delta`, `ci_low`, `ci_high` (95% item bootstrap), `prop_positive`, `positive_items` (count out of `n`, where present), `min`, `max`.

## Layer-wise diagnostic

`results/layerwise/` holds the Appendix A outputs.

| File | Contents |
|---|---|
| `layerwise_item_distances.csv` | Per model × item × layer: *ye(k)*-versus-bare representational distance at the object NP (`obj_distance`) and at the subject span (`subj_distance`), and their difference |
| `layerwise_summary_by_model_layer.csv`, `layerwise_peak_summary.csv` | Summaries |
| `layerwise_behavior_correlations.csv` | Correlations with the behavioural span effects |
| `layerwise_failures.csv` | Items that could not be aligned (empty: none failed) |
| `layerwise_include_drop_decision.txt` | Criterion counts and the include/drop flag for the layer-wise analysis |
| `layerwise_obj_minus_subject_distance.png` | Plot |

## English results

`english/results/english_discourse_<tag>_item_scores.csv` has one row per item × `anaphor_form` (`pron`, `demNP`) × `measure`, with the effect in `value`.

On `spec_span` rows only, the component scores are also given:
- `M_a`, `M_this`, `M_def`, `M_neg`: neutral-adjusted whole-continuation scores per context.
- `span_a`, `span_this`, `span_def`, `span_neg`: anaphor-span scores per context.

See `english/README.md` for which measures the paper reports. `english_all_models_summary.csv` has 90 rows: 9 models × 2 forms × 5 measures.

## Validation and audit

| File | Contents |
|---|---|
| `results/continuation_yek_v5_5_3_validation_summary.json` | Structured output of `validate_continuation_yek_v553.py` |
| `scripts/audit_span_effect_by_verb_domain_v553.py` | Predicate/domain robustness analysis |

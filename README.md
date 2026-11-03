# Object-anaphoric accessibility in language models: Persian *ye(k)* and English indefinite *this*

Stimuli, item-level scores and code for

> Nikta Yekrang Safakar and Sebastian Padó. 2026.
> [Testing Language Model Sensitivity to Object-Anaphoric Accessibility: Persian ye(k) and English indefinite this](https://aclanthology.org/2026.mrl-1.14/).
> In *Proceedings of the 6th Workshop on Multilingual Representation Learning (MRL 2026)*, pages 183–195, Budapest, Hungary. Association for Computational Linguistics.

Archived release: [https://doi.org/10.5281/zenodo.23922623](https://doi.org/10.5281/zenodo.23922623)

## Contents

Code, stimuli, and model outputs for the paper.

- `stimuli/`: Persian stimuli
- `english/`: English stimuli, scripts and results
- `results/`: released model outputs
- `scripts/`: Persian scoring and analysis scripts

Column descriptions are in [`DATA.md`](DATA.md).

## Where each paper result lives

| Paper | File(s) |
|---|---|
| Table 1 (example items) | `stimuli/continuation_yek_v5_5_3_same_np_smoothing_48.csv`, `english/stimuli/english_discourse_v50.csv` |
| Table 3, encoders | `results/primary_pll_word_l2r/whole_continuation/continuation_yek_v5_5_3_pll_word_l2r_summary.json` (item level: `…_scores.csv`) |
| Table 3, decoders | `results/all_decoder_summary_all_items_only.csv` (item level: `results/decoder/whole_continuation/<model>/decoder_item_deltas.csv`) |
| Table 4 | `results/primary_pll_word_l2r/span_controls/summary_by_model_and_target_kind.csv` and `summary_object_minus_subject.csv` (item level: `yek_span_control_deltas.csv`, `object_minus_subject_item_deltas.csv`) |
| Table 5, ObjNP − Neutral | `results/all_decoder_object_neutral_summary.csv`, rows `contrast = object_full_np_minus_neutral` (item level: `results/all_decoder_object_neutral_item_contrasts.csv`) |
| Table 5, ObjNP − Subject | `results/all_decoder_span_summary_object_minus_subject.csv`, rows `contrast = object_full_np_minus_subject_span` (item level: `results/all_decoder_object_minus_subject_item_deltas.csv`) |
| Table 6 | `english/results/english_all_models_summary.csv`, rows `measure = spec_span`, `anaphor_form = pron` |
| Table 7 | `english/results/english_all_models_summary.csv`, rows `measure = exist_whole` (both anaphor forms) |
| Appendix A (layer-wise diagnostic) | `results/layerwise/`, including the include/drop decision in `layerwise_include_drop_decision.txt` |
| Predicate and domain robustness | `scripts/audit_span_effect_by_verb_domain_v553.py` |
| Standard-PLL robustness | `results/robustness_standard_pll/` |

Files not listed are intermediate scores or alternative aggregation levels for the same analyses.

## Quick checks (no model downloads)

Run everything from the repository root.

```bash
python3.9 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -m compileall -q scripts english/scripts

# 1. Validate the frozen Persian stimuli. Must print: all_required_checks_pass=True
python scripts/validate_continuation_yek_v553.py

# 2. Regenerate the leave-one-predicate / leave-one-domain audit
python scripts/audit_span_effect_by_verb_domain_v553.py

# 3. Rebuild the combined decoder tables (Tables 3 and 5) from the per-model
#    outputs and compare with the released files. Must print: decoder_aggregation_matches_released=True
python scripts/aggregate_decoder_results.py --check

# 4. Rebuild the English summary (Tables 6 and 7) from the item scores.
#    Must print: english_summary_matches_released=True
(cd english && python scripts/combine_english_results.py --check)
```

The validation script updates the timestamp in `results/continuation_yek_v5_5_3_validation_summary.json`.

## Models

The exact Hugging Face revisions are the commits cached on the machine used for the runs.

| Arm | Model | Hugging Face ID | Revision |
|---|---|---|---|
| Persian + English | mBERT | `bert-base-multilingual-cased` | `3f076fdb1ab68d5b2880cb87a0886f315b8146f8` |
| Persian + English | XLM-R | `xlm-roberta-base` | `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089` |
| Persian | ParsBERT | `HooshvareLab/bert-fa-base-uncased` | `a04aa40c97bcdde570ae11986a534542c2995a62` |
| English | BERT (en) | `bert-base-cased` | `cd5ef92a9fb2f889e972770a36d4ed042daf221e` |
| Persian + English | Qwen2.5-0.5B | `Qwen/Qwen2.5-0.5B` | `060db6499f32faf8b98477b0a26969ef7d8b9987` |
| Persian + English | Qwen2.5-1.5B | `Qwen/Qwen2.5-1.5B` | `8faed761d45a263340a0528343f099c05c9a4323` |
| Persian + English | Qwen2.5-3B | `Qwen/Qwen2.5-3B` | `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b` |
| Persian | mGPT-Persian 1.3B | `ai-forever/mGPT-1.3B-persian` | `53cffe6434f6bdff75d8d26adf74d3cfb74bcf71` |
| Persian + English | XGLM-1.7B | `facebook/xglm-1.7B` | `d23a5e8e2164af31a84a26756b9b17f925143050` |
| Persian + English | XGLM-2.9B | `facebook/xglm-2.9B` | `33c659ae27de09c0a54123d3902dac48cbb8592a` |
| Persian + English | XGLM-4.5B | `facebook/xglm-4.5B` | `1c2905c6ab3ea065e09e61daa12e3af0d246ca7e` |

The scripts load models by ID, not by revision. For ParsBERT, mGPT-Persian, XGLM-1.7B and XGLM-2.9B, whose main branch stores only `pytorch_model.bin`, `transformers` loaded the Hub's automatic safetensors conversion of the same weights.

mGPT-Persian was used only for the Persian experiments.

## Full reruns

Reruns download the checkpoints above and need substantial memory and time. The decoder runs used an Apple-silicon GPU (MPS), as recorded in `decoder_run_metadata.json`; expect small floating-point differences on CUDA or CPU. Rerun outputs overwrite the released files at the same paths, so use `git diff` to compare.

**Persian encoders (Tables 3 and 4)**

```bash
python scripts/validate_continuation_yek_v553.py
python scripts/score_continuation_yek_v553_pll_word_l2r.py
python scripts/score_yek_span_controls_pll_word_l2r.py \
  --stimuli stimuli/continuation_yek_v5_5_3_same_np_smoothing_48.csv \
  --outdir results/primary_pll_word_l2r/span_controls \
  --models mbert xlm-r parsbert --device auto
```

**Standard-PLL robustness**

```bash
python scripts/score_continuation_yek_v553.py
python scripts/score_yek_span_controls.py \
  --stimuli stimuli/continuation_yek_v5_5_3_same_np_smoothing_48.csv \
  --outdir results/robustness_standard_pll/span_controls \
  --models mbert xlm-r parsbert
```

**Persian decoders (Tables 3 and 5).** Span controls read the span positions written by the encoder span run above.

```bash
while read -r MODEL_ID MODEL_KEY; do
  python scripts/score_yek_decoder_autoreg_v2.py \
    --model_id "$MODEL_ID" --model_key "$MODEL_KEY" \
    --outdir "results/decoder/whole_continuation/$MODEL_KEY"
  python scripts/score_yek_decoder_span_controls_v553.py \
    --model_id "$MODEL_ID" --model_key "$MODEL_KEY" \
    --outdir "results/decoder/span_controls/$MODEL_KEY"
  python scripts/score_yek_decoder_neutral_control_v553.py \
    --model_id "$MODEL_ID" --model_key "$MODEL_KEY" \
    --outdir "results/decoder/neutral_control/$MODEL_KEY"
done <<'EOF'
Qwen/Qwen2.5-0.5B qwen25_0_5b
Qwen/Qwen2.5-1.5B qwen25_1_5b
Qwen/Qwen2.5-3B qwen25_3b
ai-forever/mGPT-1.3B-persian mgpt_1_3b_persian
facebook/xglm-1.7B xglm_1_7b
facebook/xglm-2.9B xglm_2_9b
facebook/xglm-4.5B xglm_4_5b
EOF

python scripts/aggregate_decoder_results.py
```

**English comparison (Tables 6 and 7)**

```bash
cd english
while read -r MODEL CLASS TAG; do
  python scripts/score_english_discourse.py --model "$MODEL" --model-class "$CLASS" --tag "$TAG"
done <<'EOF'
bert-base-cased encoder bert_base
bert-base-multilingual-cased encoder mbert
xlm-roberta-base encoder xlm_r
Qwen/Qwen2.5-0.5B decoder qwen25_0_5b
Qwen/Qwen2.5-1.5B decoder qwen25_1_5b
Qwen/Qwen2.5-3B decoder qwen25_3b
facebook/xglm-1.7B decoder xglm_1_7b
facebook/xglm-2.9B decoder xglm_2_9b
facebook/xglm-4.5B decoder xglm_4_5b
EOF
python scripts/combine_english_results.py
cd ..
```

The scorer also writes one `english_discourse_<tag>_summary.csv` per model. These are not released, because `english_all_models_summary.csv` stacks the same rows.

**Layer-wise diagnostic (Appendix A)**

```bash
python scripts/layerwise_object_minus_subject_analysis.py
```

## Bootstrap settings

| Analysis | Resamples | Seed | Defined in |
|---|---:|---:|---|
| Encoder whole continuation (primary and standard PLL) | 5,000 | 7 | `score_continuation_yek_v553.py` |
| Encoder span controls (primary and standard PLL) | 5,000 | 13 | `score_yek_span_controls.py` |
| Decoder whole continuation, span and neutral controls | 10,000 | 13 | `score_yek_decoder_autoreg_v2.py` |
| English | 10,000 | 13 | `english/scripts/score_english_discourse.py` |
| Layer-wise analysis | 10,000 | 13 | `layerwise_object_minus_subject_analysis.py` |

## Environment

Python 3.9 with the pinned versions in `requirements.txt` (`torch` 2.8.0, `transformers` 4.57.6). Some pins do not install on Python 3.13 or later.

## Licence

- **Code** (`scripts/`, `english/scripts/`): MIT; see [`LICENSE`](LICENSE).
- **Stimuli, scores and documentation** (everything else): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); see [`LICENSE-DATA`](LICENSE-DATA).

## Citation

Please cite the paper. [`CITATION.cff`](CITATION.cff) has the same information for GitHub's "Cite this repository" button.

```bibtex
@inproceedings{safakar-pado-2026-testing,
    title     = "Testing Language Model Sensitivity to Object-Anaphoric Accessibility: {P}ersian ye(k) and {E}nglish indefinite this",
    author    = "Safakar, Nikta Yekrang and Pad{\'o}, Sebastian",
    booktitle = "Proceedings of the 6th Workshop on Multilingual Representation Learning (MRL 2026)",
    month     = oct,
    year      = "2026",
    address   = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    url       = "https://aclanthology.org/2026.mrl-1.14/",
    pages     = "183--195",
    isbn      = "979-8-89176-469-9",
}
```

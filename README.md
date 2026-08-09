# Post-Hoc Reasoning in Chain of Thought

Code for **"Post-Hoc Reasoning in Chain of Thought: Decoding and Steering
Pre-Committed Answers"** (ICML 2026 Mechanistic Interpretability Workshop).

Kyle Cox, Darius Kianersi, Adrià Garriga-Alonso

We study *post-hoc reasoning*: cases where a model has already committed to an
answer before its chain of thought begins. We (1) test CoT sensitivity with
truncation/corruption interventions, (2) train linear probes that decode the
final answer from pre-CoT activations, (3) flip answers with activation
steering while the CoT is still being generated, and (4) classify steered CoT
traces to see how the model rationalizes a steered answer.

## Setup

```bash
pip install -r requirements.txt
```

Experiments run on models from Hugging Face via TransformerLens or nnsight
(Gemma-2, Qwen-2.5, Phi-3, Llama-2 families; see `configs/`). The CoT
classification step calls the OpenAI API — set `OPENAI_API_KEY` in `.env`.

## Repository layout

| Path | Contents |
|---|---|
| `src/` | Library: data loading, probe training, steering methods, experiment runners (TransformerLens and nnsight backends), caching |
| `configs/` | YAML experiment configs (model / datasets / split seeds) |
| `data/` | Question datasets (anachronisms, sports understanding, logical deduction, social chemistry, …) — see `DATASETS.md` |
| `scripts/` | Analysis and figure scripts |
| `results/` | CoT sensitivity result tables (CSV) |
| `figs/` | Generated figures, including the published paper figures |
| `consistency_analysis/` | LLM classification-consistency runs (paper appendix) |

## Reproducing the paper

**Probes and answer steering** (Sections: Pre-CoT Probes, Answer Steering):

```bash
python run_transformer_lens_experiments.py --config configs/transformer_lens_by_model/gemma-2-2b-it.yaml
python run_nnsight_experiments.py --config <config>          # nnsight backend
python run_orthogonal_steering_experiments.py --config <config>  # random-orthogonal baseline
```

**CoT sensitivity** (truncation / ellipses / incorrect-CoT interventions):
`cot_sensitivity_experiments.ipynb`; summary tables via
`scripts/generate_cot_summary_table.py`.

**CoT classification** (LLM-classifying steered rollouts):
`classify_rollouts.ipynb`, then `scripts/reclassify_rollouts.py`; the paper's
classification figure comes from `scripts/visualize_rollout_yes_only.py`.

**Figures**: probe AUC / steering figures are produced in
`visualize_results.ipynb`; published copies live in `figs/`.

## Experiment caches

Raw experiment caches (activations, steered generations, classified rollouts)
are multiple GB and are not tracked in this repository. The classification
figures read from `final_cache/cache/rollout_classification_v2/`, which should
contain **197 jsonl files** — the plotting scripts run without error on
partial copies but produce wrong figures, so check the count before
regenerating. Caches are available from the authors on request.

## Citation

```bibtex
@inproceedings{cox2026posthoc,
  title     = {Post-Hoc Reasoning in Chain of Thought: Decoding and Steering Pre-Committed Answers},
  author    = {Cox, Kyle and Kianersi, Darius and Garriga-Alonso, Adri\`{a}},
  booktitle = {ICML Workshop on Mechanistic Interpretability},
  year      = {2026}
}
```

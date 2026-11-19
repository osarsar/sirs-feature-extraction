# SIRS Feature Extraction: From Keywords to Contextual Understanding

Code, data and results for the paper:

> **From Keywords to Contextual Understanding: Transformer-Based Feature Extraction for Societal Risk Scoring of Misinformation**
> Oussama Sarsar, Mounir Soufyane, Yassine Maleh
> LaSTI Laboratory, ENSA Khouribga, Sultan Moulay Slimane University

The paper measures, for the first time, how the SIRS societal-risk framework performs **from raw text**. It keeps the SIRS base formula and its published weights fixed and replaces only the four feature extractors (topic sensitivity, actionability, audience scope, temporal sensitivity).

## Main results

End-to-end Spearman ρ against 26 test claims rated by 16 annotators (gold = mean of 16). The SIRS base formula uses the published weights w = (0.20, 0.10, 0.55, 0.15). CamemBERT values are the mean ± std over 3 seeds; the 95% CIs come from a paired bootstrap (B = 10,000).

| Extraction | ρ | 95% CI |
|---|---|---|
| Rule-based (original Paper 1 extractors) | −0.302 | [−0.64, 0.11] |
| CamemBERT, separate encoders | 0.467 ± 0.025 | [0.09, 0.74] |
| CamemBERT, unified encoder | 0.454 ± 0.056 | [0.04, 0.76] |
| LLM zero-shot (Claude Sonnet 4) | 0.793 | [0.52, 0.94] |
| Human reference (leave-one-out) | 0.465 | — |

With the formula weights **re-fitted on the validation split** (never on the test set), the results are:
- CamemBERT separate: 0.657
- CamemBERT unified: 0.632
- LLM: 0.865

On every seed, audience scope falls to the minimum weight (0.05).

The full output, including the paired differences and the median-gold robustness check, is in `results/paper_analysis_output.txt`.

## Repository structure

```
scripts/
  sirs_paper1/                 Original Paper 1 rule-based extractors (baseline), unchanged
  train_corrected_v4.py        Main training: separate + unified CamemBERT, 3 seeds; saves val/test predictions
  optimize_weights_val.py      Re-fits the formula weights on the validation split (Paper 1 grid, 829 combos)
  paper_analysis.py            Reproduces every human-evaluation number of the paper
  architecture_ablation.py     Head-depth × objective ablation (18 configs × 3 seeds); selection on validation
  train_camembert_extractors.py  Earlier extractor script (Paper 1 weights and baseline fixed)
data/
  training_data_with_dimensions.json   974 Hespress claims with LLM silver labels (x1..x4 + overall risk)
  gold_test_with_dimensions.json       26 test claims: gold risk (mean of 16) + LLM dimension scores
  test_claims.json                     26 test claims with category
  human_ratings_16.csv                 Anonymised ratings of the 16 annotators (0.15 / 0.50 / 0.85)
results/
  predictions_v4.json                  Per-claim val/test predictions, 3 seeds, separate + unified
  corrected_results_v4.json            Per-dimension and end-to-end metrics (v4 run)
  weights_val_optimised.json           Weights selected on validation, per seed
  architecture_ablation_results.json   Ablation (val and test ρ per config and seed)
  human_agreement_16.json              Inter-annotator agreement
  paper_analysis_output.txt            Output of scripts/paper_analysis.py
slurm/                                 Job scripts used on the Toubkal HPC (adapt to your cluster)
legacy/                                Scripts and results of the 16 Nov 2026 draft (superseded, see below)
```

## Reproducing the paper

```bash
pip install torch transformers scipy scikit-learn numpy krippendorff

# 1. Train separate + unified extractors (3 seeds) and save predictions (~5 min on one H100)
python3 scripts/train_corrected_v4.py

# 2. Re-fit the formula weights on the validation split
python3 scripts/optimize_weights_val.py

# 3. Reproduce all human-evaluation numbers (agreement, Tables 5, 6, 8, robustness)
python3 scripts/paper_analysis.py
```

Paths default to `data/` and `results/` inside the repository. Override them with `SIRS_DATA_DIR` and `SIRS_RESULTS_DIR`.

## Correction notice (supersedes the 16 Nov 2026 version)

The first version of this repository, and the draft it accompanied, contained errors that are corrected here:
- **Formula weights.** The code used (0.30, 0.25, 0.25, 0.20), described as the Paper 1 weights. The actual Paper 1 weights are (0.20, 0.10, 0.55, 0.15).
- **Baseline.** The "Paper 1 keywords" baseline was a simplified re-implementation. It is replaced by the original Paper 1 extractors (`scripts/sirs_paper1/`).
- **Head depth.** The depth was selected on the test set. Selection now uses the validation split, which picks one hidden layer for all four dimensions.
- **Human statistics.** These were computed on the first 13 responses, while the gold labels used all 16. Everything is now computed on the same 16 annotators, with the gold label being the mean of 16 and exact ties preserved.

As a result, the headline numbers changed from the earlier draft (for example, CamemBERT separate went from 0.694 to 0.467, and the LLM from 0.850 to 0.793). The superseded scripts and results are kept in `legacy/` for transparency.

## License

See `LICENSE`.

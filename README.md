# SIRS Feature Extraction: From Keywords to Contextual Understanding

Code, data, and results for the paper:

> **From Keywords to Contextual Understanding: Distilling LLM Knowledge into Transformer-Based Risk Feature Extractors**
> Oussama Sarsar, Mounir Soufyane, Yassine Maleh
> LaSTI Laboratory, ENSA Khouribga, Sultan Moulay Slimane University

## Overview

This repository implements a knowledge distillation pipeline that transfers LLM extraction capability to local CamemBERT models for multi-dimensional misinformation risk scoring (SIRS framework).

**Key results:**
- LLM zero-shot extraction: ρ = 0.850
- Distilled CamemBERT (separate): ρ = 0.694 (82% of LLM quality)
- Distilled CamemBERT (unified): ρ = 0.609
- Rule-based baseline: ρ = 0.165

## Repository Structure

```
├── scripts/
│   ├── train_corrected.py            # Main training: separate + unified models (3 seeds)
│   ├── architecture_ablation.py      # 18-config ablation (depth × objective)
│   ├── train_camembert_extractors.py # Individual dimension extractors
│   └── train_camembert_unified_gpu.py# Unified model training
├── data/
│   ├── training_data_with_dimensions.json  # 974 claims with LLM silver labels
│   └── gold_test_with_dimensions.json      # 26 claims with human gold labels (13 annotators)
├── results/
│   ├── corrected_results.json              # Final results (3 seeds × separate + unified)
│   └── architecture_ablation_results.json  # Full ablation results (18 configs × 3 seeds)
├── slurm/
│   ├── run_corrected.slurm           # SLURM script for main training (H100)
│   └── run_ablation_h100.slurm       # SLURM script for ablation (H100)
└── README.md
```

## Architecture

Four SIRS dimensions are extracted from raw claim text:
- **x1** (Topic Sensitivity): 3-layer regression head [256, 128, 64]
- **x2** (Actionability): 1-layer regression head [256]
- **x3** (Audience Scope): 1-layer regression head [128]
- **x4** (Temporal Sensitivity): 1-layer regression head [128]

All heads use pure regression (MSE loss) on top of CamemBERT-base. Architecture selected via ablation over 18 configurations.

## Requirements

```
torch>=2.0
transformers>=4.30
scipy
scikit-learn
```

## Usage

### Training (separate + unified, 3 seeds)

```bash
python scripts/train_corrected.py
```

### Architecture ablation

```bash
python scripts/architecture_ablation.py
```

### Data format

**Training data** (`training_data_with_dimensions.json`):
```json
{
  "claim": "Le Maroc a signé un accord...",
  "risk_score": 0.65,
  "x1": 0.82,
  "x2": 0.35,
  "x3": 0.70,
  "x4": 0.50
}
```

**Gold test** (`gold_test_with_dimensions.json`): Same format, with dimension values derived from the median of 13 human annotators.

## Results

### Main comparison (end-to-end Spearman ρ)

| Method | ρ |
|---|---|
| Rule-based (keywords) | 0.165 |
| Human ceiling (individual) | 0.474 |
| CamemBERT separate (4 encoders) | **0.694** |
| CamemBERT unified (1 encoder) | 0.609 |
| LLM zero-shot (Claude) | 0.850 |

### Architecture ablation (per-dimension ρ, pure regression)

| Dimension | 1 layer | 2 layers | 3 layers |
|---|---|---|---|
| x1: Topic | 0.723 | 0.723 | **0.750** |
| x2: Actionability | **0.766** | 0.206 | 0.093 |
| x3: Audience scope | **0.307** | 0.223 | 0.131 |
| x4: Temporal | **0.781** | 0.671 | 0.650 |

## Citation

```bibtex
@article{sarsar2026keywords,
  title={From Keywords to Contextual Understanding: Distilling LLM Knowledge into Transformer-Based Risk Feature Extractors},
  author={Sarsar, Oussama and Soufyane, Mounir and Maleh, Yassine},
  journal={Social Network Analysis and Mining},
  year={2026}
}
```

## License

MIT License

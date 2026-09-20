# Pareto-Efficient Intrusion Detection Across Heterogeneous IoT Threat Surfaces

[![Paper](https://img.shields.io/badge/paper-PDF-blue)](paper/cose.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)

Code and reproducibility artefacts for the paper:

> **Security Evaluation of DL IDS**
> Christopher Kahola, Stephen Wambura
> Department of Computer Studies, Dar es Salaam Institute of Technology

## Summary

This repository contains the full evaluation pipeline for seven deep learning intrusion detection models evaluated across three structurally different IoT threat surfaces:

| Dataset | Granularity | Features | Classes | Threat Surface |
|---------|-------------|----------|---------|----------------|
| **BoT-IoT** | Network flow | 32 (selected) | 5 | Network-level botnet traffic |
| **N-BaIoT** | Per-device statistical | 115 (selected 32) | 11 | Device-level botnet behavior |
| **IoTID20** | Network flow | 32 (selected) | 5 | Smart-home gateway intrusion |

Seven models are evaluated: CNN-LSTM (reference), CNN-LSTM (non-optimized), CNN-only, LSTM-only, Transformer, CNN-GRU, CNN-BiLSTM. All models consume the identical 8-flow sliding-window representation of shape (N, 8, 32).

## Key Findings

- CNN-LSTM family Pareto-dominates all three literature baselines on every dataset
- 55.5–81.6% parameter reduction at 0.61–5.46 percentage-point macro-F1 cost
- Reference models fit in 31.6–77.2 KB (float32), sub-3 ms per-sample inference
- Three dataset-specific security weaknesses identified (Theft/Reconnaissance, Gafgyt UDP/TCP, Scan-magnet)

## Repository Structure

```
paper/         LaTeX source and compiled PDF
figures/       Figure generation scripts and output
data/          Dataset download instructions and split definitions
src/           Preprocessing, models, training, evaluation
scripts/       Orchestration shell scripts
results/       Metrics, significance tests, optimization trials
notebooks/     Exploratory and analysis notebooks
docs/          Data card, model card, reproducibility notes
```

## Quick Start

```bash
# 1. Clone
git clone https://github.com/<your-username>/pareto-iot-ids-eval.git
cd pareto-iot-ids-eval

# 2. Environment
conda env create -f environment.yml
conda activate pareto-iot-ids

# 3. Download datasets (see data/README.md for manual steps)
bash scripts/download_datasets.sh

# 4. Run full pipeline (~12h on a single T4 GPU)
bash scripts/run_all.sh
```

## Reproducing the Paper

| Paper Section | Artefact |
|---------------|----------|
| Section 3.2 (Datasets) | `data/README.md` |
| Section 3.3 (Preprocessing) | `src/preprocessing.py` |
| Section 3.4 (Models) | `src/models/`, `src/optimize.py` |
| Section 3.5 (Metrics, Stats) | `src/metrics.py`, `src/statistics.py` |
| Section 3.6 (Sensitivity, Pareto) | `src/pareto.py`, `src/evaluate.py` |
| Section 4 (Results) | `results/`, `notebooks/03_results_analysis.ipynb` |
| Figures 1–7 | `figures/source/` |

See `docs/reproducibility.md` for the exact commands and expected runtime.

## Hardware

- All experiments were run on an NVIDIA Tesla T4 (16 GB VRAM) on Kaggle.
- Total compute: ~6 GPU-hours for optimization (150 trials across 3 datasets), plus ~4 GPU-hours for training and evaluation.

## Citation

If you use this code or these findings, please cite:

```bibtex
@article{kahola2026pareto,
  title   = {Security Evaluation of DL IDS...},
  author  = {Kahola, Christopher and Wambura, Stephen},
  journal = {},
  year    = {2026},
  note    = {Under review}
}
```

## License

Code: MIT (see `LICENSE`).
Datasets: governed by their original distribution terms (see `data/README.md`).
Paper: © the authors.

## Contact

Christopher Kahola — christopher.kahola@dit.ac.tz  stephen.wambura@dit.ac.tz

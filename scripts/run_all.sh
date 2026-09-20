#!/usr/bin/env bash
set -euo pipefail

echo "[1/6] Preprocessing"
python -m src.preprocessing --dataset all

echo "[2/6] Optimization (150 trials)"
python -m src.optimize --dataset all --trials 50 --seed 42

echo "[3/6] Training seven models on three datasets"
python -m src.train --dataset all --models all

echo "[4/6] Evaluation"
python -m src.evaluate --dataset all

echo "[5/6] Statistical tests"
python -m src.statistics
python -m src.pareto

echo "[6/6] Figures"
make figures

echo "Done. See results/ and figures/."

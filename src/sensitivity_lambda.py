"""
Sensitivity sweep over the composite-objective penalty weight lambda.
Produces results/sensitivity/lambda_sweep.csv and figures/fig8_lambda_*.pdf.

Sweeps lambda_lat = lambda_pam in {0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30}
over 30 trials per (dataset, lambda) pair. Seed fixed at 42.
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import optuna
import matplotlib.pyplot as plt

DATASETS = ["bot_iot", "n_baiot", "iot_id20"]
LAMBDAS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
TRIALS_PER_POINT = 30
SEED = 42
OUT_DIR = Path("results/sensitivity")
FIG_DIR = Path("figures")


def composite_objective(f1_macro: float, latency_ms: float,
                        n_params: int, lam: float) -> float:
    import math
    return (f1_macro
            - lam * math.tanh(latency_ms / 10.0)
            - lam * math.tanh(n_params / 30_000.0))


def train_and_evaluate(dataset: str, params: dict) -> dict:
    """
    Placeholder: replace with a call to src.train.train_one_config
    that returns {'macro_f1': ..., 'latency_ms': ..., 'n_params': ...}.
    """
    raise NotImplementedError("Hook up the training pipeline here.")


def make_objective(dataset: str, lam: float):
    def objective(trial: optuna.Trial) -> float:
        from src.optimize import suggest
        params = suggest(trial)
        metrics = train_and_evaluate(dataset, params)
        return composite_objective(
            metrics["macro_f1"], metrics["latency_ms"], metrics["n_params"], lam)
    return objective


def run_sweep(dataset: str) -> pd.DataFrame:
    rows = []
    for lam in LAMBDAS:
        sampler = optuna.samplers.TPESampler(seed=SEED)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(make_objective(dataset, lam), n_trials=TRIALS_PER_POINT)
        best = study.best_trial
        rows.append({
            "dataset": dataset,
            "lambda": lam,
            "macro_f1": best.value,
            "params": best.params.get("n_params"),
            "latency_ms": best.params.get("latency_ms"),
            "best_params": json.dumps(best.params),
        })
        print(f"[sweep] {dataset} lambda={lam:.2f} f1={best.value:.4f}")
    return pd.DataFrame(rows)


def plot_panel(df: pd.DataFrame, dataset: str, ax) -> None:
    sub = df[df["dataset"] == dataset].sort_values("lambda")
    ax.plot(sub["params"], sub["macro_f1"], marker="o", label="Pareto-selected")
    anchor = sub[sub["lambda"] == 0.00].iloc[0]
    ax.axhline(anchor["macro_f1"], color="grey", linestyle="--",
               label=r"$\lambda = 0.00$ anchor")
    for _, row in sub.iterrows():
        ax.annotate(f"{row['lambda']:.2f}",
                    (row["params"], row["macro_f1"]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (log)")
    ax.set_ylabel("Macro-F1")
    ax.set_title(dataset.replace("_", "-"))
    ax.grid(alpha=0.3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all",
                        choices=DATASETS + ["all"])
    args = parser.parse_args()
    datasets = DATASETS if args.dataset == "all" else [args.dataset]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    frames = [run_sweep(ds) for ds in datasets]
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(OUT_DIR / "lambda_sweep.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)
    for ax, ds in zip(axes, datasets):
        plot_panel(df, ds, ax)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    for ds in datasets:
        single, single_ax = plt.subplots(figsize=(5, 4))
        plot_panel(df, ds, single_ax)
        single.tight_layout()
        single.savefig(FIG_DIR / f"fig8_lambda_{ds}.pdf", dpi=300)
        plt.close(single)
    print(f"[sweep] done -> {OUT_DIR / 'lambda_sweep.csv'}")


if __name__ == "__main__":
    main()

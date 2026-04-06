"""Lambda tuning for parsimony pressure.

Sweeps lambda values and reports the tradeoff between break strength
(raw_fitness) and pipeline complexity (active stages). The output is a
tradeoff curve whose elbow gives the right lambda for production runs.

Usage:
    python experiments/scripts/tune_lambda.py --config configs/adversary.yaml

Outputs (in results/lambda_sweep/<timestamp>/):
    sweep_results.json   - per-lambda best results
    tradeoff_curve.png   - active stages vs raw_fitness
    per_lambda_summary.csv
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import yaml

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.icl.eval import get_model_from_run
from src.adversary.evaluate import GenomeEvaluator
from src.adversary.search import cma_search
from src.adversary.pipeline_genome import PipelineGenome


DEFAULT_LAMBDA_VALUES = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0]


def run_lambda_sweep(
    model,
    n_dims: int,
    n_points: int,
    task_name: str = "noisy_linear_regression",
    batch_size: int = 64,
    num_batches: int = 10,
    baselines: list[str] | None = None,
    lambda_values: list[float] | None = None,
    c_base: float = 1.0,
    c_stage: float = 1.0,
    c_affine: float = 1.0,
    budget_per_lambda: int = 2000,
    pop_size: int = 32,
    sigma_init: float = 0.3,
    seed: int = 0,
) -> dict:
    """Run short CMA-ES at each lambda, return results dict."""
    if lambda_values is None:
        lambda_values = DEFAULT_LAMBDA_VALUES
    if baselines is None:
        baselines = ["ridge", "least_squares", "averaging"]

    results = {}

    for lam in lambda_values:
        print(f"\n{'='*60}")
        print(f"Lambda = {lam}")
        print(f"{'='*60}")

        evaluator = GenomeEvaluator(
            icl_model=model,
            task_name=task_name,
            n_dims=n_dims,
            n_points=n_points,
            batch_size=batch_size,
            num_batches=num_batches,
            baseline_names=baselines,
            parsimony_lambda=lam,
            c_base=c_base,
            c_stage=c_stage,
            c_affine=c_affine,
        )

        all_results = cma_search(
            evaluator=evaluator,
            n_dims=n_dims,
            budget=budget_per_lambda,
            pop_size=pop_size,
            sigma_init=sigma_init,
            num_restarts=1,
            save_dir=None,
            seed=seed,
            genome_cls=PipelineGenome,
        )

        # Find best valid result
        valid = [r for r in all_results if r.is_valid]
        if not valid:
            print(f"  WARNING: No valid results for lambda={lam}")
            results[lam] = {
                "lambda": lam,
                "effective_fitness": 0.0,
                "raw_fitness": 0.0,
                "num_active_stages": 0,
                "base_distribution": "none",
                "active_transforms": [],
                "complexity": 0.0,
            }
            continue

        best = max(valid, key=lambda r: r.fitness)
        genome = best.genome

        entry = {
            "lambda": lam,
            "effective_fitness": best.fitness,
            "raw_fitness": best.raw_fitness,
            "num_active_stages": genome.num_active_stages(),
            "base_distribution": genome.base_distribution_name(),
            "active_transforms": [name for _, name in genome.active_stages()],
            "complexity": best.complexity,
        }
        results[lam] = entry

        print(f"  Best effective_fitness: {best.fitness:.4f}")
        print(f"  Best raw_fitness: {best.raw_fitness:.4f}")
        print(f"  Active stages: {genome.num_active_stages()}")
        print(f"  Base: {genome.base_distribution_name()}")
        print(f"  Transforms: {[name for _, name in genome.active_stages()]}")
        print(f"  Complexity: {best.complexity:.4f}")

    return results


def plot_tradeoff(results: dict, output_dir: str):
    """Plot active stages vs raw_fitness, colored by lambda."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    lambdas = sorted(results.keys())
    active_stages = [results[l]["num_active_stages"] for l in lambdas]
    raw_fitnesses = [results[l]["raw_fitness"] for l in lambdas]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    scatter = ax.scatter(
        active_stages, raw_fitnesses,
        c=[np.log10(l + 1e-4) for l in lambdas],
        s=100, cmap="viridis", edgecolors="black",
    )
    for l, x, y in zip(lambdas, active_stages, raw_fitnesses):
        ax.annotate(f"  λ={l}", (x, y), fontsize=9)

    ax.set_xlabel("Average Active Stages", fontsize=12)
    ax.set_ylabel("Raw Fitness (without penalty)", fontsize=12)
    ax.set_title("Parsimony Tradeoff: Complexity vs Break Strength", fontsize=13)
    plt.colorbar(scatter, label="log10(lambda)")
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "tradeoff_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved tradeoff curve to {path}")


def check_sanity(results: dict) -> list[str]:
    """Return list of warning messages if sanity checks fail."""
    warnings = []
    lambdas = sorted(results.keys())

    if not lambdas:
        return ["No results to check"]

    # At lambda=0: should use >0 active stages
    if 0.0 in results:
        r = results[0.0]
        if r["num_active_stages"] == 0 and r["raw_fitness"] < 0.01:
            warnings.append(
                "WARN: At lambda=0, adversary uses 0 active stages and has near-zero fitness. "
                "Search may be broken independent of lambda."
            )

    # At max lambda: should use 0 active stages (identity)
    max_lam = max(lambdas)
    if max_lam > 0:
        r = results[max_lam]
        if r["num_active_stages"] > 0:
            warnings.append(
                f"WARN: At lambda={max_lam}, adversary still uses "
                f"{r['num_active_stages']} active stages. Lambda may not be high enough."
            )

    return warnings


def save_results(results: dict, output_dir: str):
    """Save results to JSON and CSV."""
    os.makedirs(output_dir, exist_ok=True)

    # JSON
    json_path = os.path.join(output_dir, "sweep_results.json")
    # Convert keys to strings for JSON
    json_results = {str(k): v for k, v in results.items()}
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"Saved results to {json_path}")

    # CSV
    csv_path = os.path.join(output_dir, "per_lambda_summary.csv")
    with open(csv_path, "w") as f:
        f.write("lambda,effective_fitness,raw_fitness,num_active_stages,base_distribution,complexity\n")
        for lam in sorted(results.keys()):
            r = results[lam]
            f.write(
                f"{r['lambda']},{r['effective_fitness']:.6f},{r['raw_fitness']:.6f},"
                f"{r['num_active_stages']},{r['base_distribution']},{r['complexity']:.6f}\n"
            )
    print(f"Saved summary to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Lambda tuning for parsimony pressure")
    parser.add_argument("--config", type=str, required=True, help="Path to adversary config YAML")
    parser.add_argument("--budget", type=int, default=2000, help="Budget per lambda value")
    parser.add_argument(
        "--lambdas", type=float, nargs="+",
        default=DEFAULT_LAMBDA_VALUES,
        help="Lambda values to sweep",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load ICL model
    model_config = config["icl_model"]
    run_path = model_config["run_path"]
    step = model_config.get("step", -1)
    print(f"Loading ICL model from {run_path}...")
    model, conf = get_model_from_run(run_path, step)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    # Config
    task_config = config.get("task", {})
    n_dims = task_config.get("n_dims", conf.model.n_dims)
    n_points = task_config.get("n_points", 41)
    task_name = task_config.get("name", "noisy_linear_regression")

    eval_config = config.get("eval", {})
    batch_size = eval_config.get("batch_size", 64)
    num_batches = eval_config.get("num_batches", 10)
    baselines = eval_config.get("baselines", ["ridge", "least_squares", "averaging"])

    parsimony_config = config.get("parsimony", {})
    c_base = parsimony_config.get("c_base", 1.0)
    c_stage = parsimony_config.get("c_stage", 1.0)
    c_affine = parsimony_config.get("c_affine", 1.0)

    search_config = config.get("search", {})
    pop_size = search_config.get("pop_size", 32)
    sigma_init = search_config.get("sigma_init", 0.3)
    seed = search_config.get("seed", 0)

    # Output dir
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("results", "lambda_sweep", timestamp)

    print(f"\nSweeping lambda values: {args.lambdas}")
    print(f"Budget per lambda: {args.budget}")
    print(f"Output: {output_dir}")

    # Run sweep
    results = run_lambda_sweep(
        model=model,
        n_dims=n_dims,
        n_points=n_points,
        task_name=task_name,
        batch_size=batch_size,
        num_batches=num_batches,
        baselines=baselines,
        lambda_values=args.lambdas,
        c_base=c_base,
        c_stage=c_stage,
        c_affine=c_affine,
        budget_per_lambda=args.budget,
        pop_size=pop_size,
        sigma_init=sigma_init,
        seed=seed,
    )

    # Save and plot
    save_results(results, output_dir)
    plot_tradeoff(results, output_dir)

    # Sanity checks
    warnings = check_sanity(results)
    if warnings:
        print("\n--- Sanity Check Warnings ---")
        for w in warnings:
            print(f"  {w}")
    else:
        print("\nAll sanity checks passed.")


if __name__ == "__main__":
    main()

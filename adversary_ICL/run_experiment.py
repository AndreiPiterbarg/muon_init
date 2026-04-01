"""
End-to-end experiment: train ICL model, run adversarial search, analyze results.

Usage: python run_experiment.py
"""

import os
import sys
import time
import pickle
import json

import numpy as np
import torch
import yaml

# Project root
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.icl.models import build_model, TransformerModel, LeastSquaresModel, AveragingModel
from src.icl.schema import AttrDict, load_config
from src.icl.train import train
from src.adversary.genome import Genome
from src.adversary.evaluate import GenomeEvaluator, EvalResult
from src.adversary.search import cma_search


# ===========================================================================
# STEP 1: Train an ICL model
# ===========================================================================

def train_icl_model(out_dir, train_steps=5000, n_dims=10, n_layer=4, n_head=2, n_embd=64):
    """Train a small ICL model for linear regression."""
    os.makedirs(out_dir, exist_ok=True)

    config = {
        "out_dir": out_dir,
        "test_run": True,  # disables wandb
        "model": {
            "family": "gpt2",
            "n_dims": n_dims,
            "n_positions": 51,
            "n_embd": n_embd,
            "n_layer": n_layer,
            "n_head": n_head,
        },
        "training": {
            "task": "linear_regression",
            "task_kwargs": {},
            "data": "gaussian",
            "batch_size": 64,
            "learning_rate": 3e-4,
            "train_steps": train_steps,
            "save_every_steps": train_steps,  # save at end
            "keep_every_steps": -1,
            "resume_id": None,
            "num_tasks": None,
            "num_training_examples": None,
            "curriculum": {
                "dims": {"start": n_dims, "end": n_dims, "inc": 1, "interval": 2000},
                "points": {"start": 11, "end": 11, "inc": 2, "interval": 2000},
            },
        },
        "wandb": {
            "project": "test", "entity": "test", "notes": "", "name": "test",
            "log_every_steps": 100,
        },
    }

    args = load_config(config)
    # Override test_run behavior: we DO want to save
    args.test_run = False
    args.out_dir = out_dir

    # Save config for later loading
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.dump(args.toDict(), f, default_flow_style=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args.model)
    model.to(device)
    model.train()

    print(f"Training ICL model: {n_layer} layers, {n_head} heads, {n_embd} embd, {n_dims} dims")
    print(f"  Device: {device}")
    print(f"  Steps: {train_steps}")

    train(model, args)

    # Verify checkpoint saved
    state_path = os.path.join(out_dir, "state.pt")
    assert os.path.exists(state_path), f"Checkpoint not saved at {state_path}"
    print(f"  Model saved to {out_dir}")

    return model, args


# ===========================================================================
# STEP 2: Quick baseline evaluation (verify model learned something)
# ===========================================================================

def quick_eval(model, n_dims, n_points=21, batch_size=64, num_batches=5):
    """Quick check that the trained model does ICL on standard Gaussian inputs."""
    from src.icl.samplers import GaussianSampler
    from src.icl.tasks import get_task_sampler

    device = next(model.parameters()).device
    model.eval()

    sampler = GaussianSampler(n_dims)
    task_sampler = get_task_sampler("linear_regression", n_dims, batch_size)

    all_icl_err = []
    all_ols_err = []

    ols = LeastSquaresModel()

    for _ in range(num_batches):
        xs = sampler.sample_xs(n_points, batch_size)
        task = task_sampler()
        ys = task.evaluate(xs)

        with torch.no_grad():
            pred_icl = model(xs.to(device), ys.to(device)).cpu()
        pred_ols = ols(xs, ys)

        icl_err = ((pred_icl - ys) ** 2).mean(dim=0)
        ols_err = ((pred_ols - ys) ** 2).mean(dim=0)

        all_icl_err.append(icl_err)
        all_ols_err.append(ols_err)

    icl_err = torch.stack(all_icl_err).mean(dim=0).numpy()
    ols_err = torch.stack(all_ols_err).mean(dim=0).numpy()

    print(f"\n--- Baseline Evaluation (standard Gaussian inputs) ---")
    print(f"  {'k':>3s}  {'ICL_err':>10s}  {'OLS_err':>10s}  {'Ratio':>8s}")
    for k in [0, 2, 5, 10, 15, 20]:
        if k < len(icl_err):
            ratio = icl_err[k] / (ols_err[k] + 1e-8)
            print(f"  {k:3d}  {icl_err[k]:10.4f}  {ols_err[k]:10.4f}  {ratio:8.2f}x")

    # Check: at k=n_dims, ICL should be within ~2x of OLS for a trained model
    ratio_at_d = icl_err[min(n_dims, len(icl_err)-1)] / (ols_err[min(n_dims, len(icl_err)-1)] + 1e-8)
    print(f"\n  ICL/OLS ratio at k={n_dims}: {ratio_at_d:.2f}x")
    return icl_err, ols_err


# ===========================================================================
# STEP 3: Run adversarial search
# ===========================================================================

def run_adversary(model, n_dims, n_points=21, budget=1600, pop_size=16, save_dir=None):
    """Run the adversarial CMA-ES search."""
    model.eval()

    evaluator = GenomeEvaluator(
        icl_model=model,
        task_name="noisy_linear_regression",
        n_dims=n_dims,
        n_points=n_points,
        batch_size=32,
        num_batches=5,
        baseline_names=["least_squares", "averaging"],
    )

    results = cma_search(
        evaluator=evaluator,
        n_dims=n_dims,
        budget=budget,
        pop_size=pop_size,
        sigma_init=0.5,
        save_dir=save_dir,
        save_interval=20,
        seed=42,
    )

    return results


# ===========================================================================
# STEP 4: Analyze results
# ===========================================================================

def analyze_results(results, n_dims, output_dir):
    """Analyze and print detailed results."""
    os.makedirs(output_dir, exist_ok=True)

    valid = [r for r in results if r.is_valid]
    invalid = [r for r in results if not r.is_valid]

    print(f"\n{'='*70}")
    print(f"ADVERSARIAL SEARCH RESULTS")
    print(f"{'='*70}")
    print(f"Total evaluations: {len(results)}")
    print(f"Valid: {len(valid)}, Invalid: {len(invalid)}")

    if not valid:
        print("No valid results found!")
        return

    fitnesses = np.array([r.fitness for r in valid])
    print(f"\nFitness distribution:")
    print(f"  Min:    {fitnesses.min():.4f}")
    print(f"  Median: {np.median(fitnesses):.4f}")
    print(f"  Mean:   {fitnesses.mean():.4f}")
    print(f"  Max:    {fitnesses.max():.4f}")
    print(f"  Std:    {fitnesses.std():.4f}")

    # Top failures
    valid.sort(key=lambda r: r.fitness, reverse=True)
    top_k = min(10, len(valid))

    print(f"\n--- Top {top_k} Failure Modes ---")
    for i, r in enumerate(valid[:top_k]):
        g = r.genome
        spectrum = r.covariance_spectrum
        print(f"\n  #{i+1}: fitness={r.fitness:.3f}")
        print(f"    Condition number (train): {g.condition_number('L_train'):.1f}")
        print(f"    Effective rank (train):   {g.effective_rank('L_train'):.2f}")
        print(f"    Condition number (test):  {g.condition_number('L_test'):.1f}")
        print(f"    Noise std:                {g.decode_noise_std():.4f}")
        print(f"    Weight norm:              {np.linalg.norm(g.decode_weights().numpy()):.3f}")
        print(f"    Top 5 eigenvalues:        {spectrum[:5].round(3)}")
        if r.descriptors:
            print(f"    Train-test divergence:    {r.descriptors.get('train_test_divergence', 0):.3f}")
            print(f"    Weight-cov alignment:     {r.descriptors.get('weight_alignment', 0):.3f}")
            print(f"    Peak failure position:    {r.descriptors.get('peak_failure_position', 0):.2f}")

    # Detailed learning curves for top 3
    print(f"\n--- Detailed Learning Curves (Top 3) ---")
    for i, r in enumerate(valid[:3]):
        print(f"\n  Failure #{i+1} (fitness={r.fitness:.3f}):")
        print(f"    {'k':>4s}  {'ICL_err':>10s}  {'OLS_err':>10s}  {'Avg_err':>10s}  {'Ratio':>8s}")
        for k in range(0, len(r.icl_curve), max(1, len(r.icl_curve) // 10)):
            icl_e = r.icl_curve[k]
            ols_e = r.baseline_curves.get("least_squares", np.zeros_like(r.icl_curve))[k]
            avg_e = r.baseline_curves.get("averaging", np.zeros_like(r.icl_curve))[k]
            best_bl = min(ols_e, avg_e)
            ratio = icl_e / (best_bl + 1e-8)
            print(f"    {k:4d}  {icl_e:10.4f}  {ols_e:10.4f}  {avg_e:10.4f}  {ratio:8.2f}x")

    # Descriptor correlation analysis
    print(f"\n--- What Predicts Failure? (Spearman Correlations) ---")
    from scipy.stats import spearmanr

    desc_keys = list(valid[0].descriptors.keys()) if valid[0].descriptors else []
    extra_features = {
        "condition_number": [r.genome.condition_number("L_train") for r in valid],
        "noise_std": [r.genome.decode_noise_std() for r in valid],
        "weight_norm": [np.linalg.norm(r.genome.decode_weights().numpy()) for r in valid],
    }

    all_features = {}
    for key in desc_keys:
        all_features[key] = [r.descriptors.get(key, 0) for r in valid]
    all_features.update(extra_features)

    correlations = []
    for name, values in all_features.items():
        rho, pval = spearmanr(values, fitnesses)
        correlations.append((name, rho, pval))

    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    print(f"  {'Feature':>30s}  {'Spearman rho':>12s}  {'p-value':>10s}")
    for name, rho, pval in correlations:
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        print(f"  {name:>30s}  {rho:12.4f}  {pval:10.6f} {sig}")

    # Plot top failures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Fitness over time
        fig, ax = plt.subplots(figsize=(10, 4))
        all_fit = [r.fitness for r in results if r.is_valid]
        running_best = np.maximum.accumulate(all_fit) if all_fit else []
        ax.scatter(range(len(all_fit)), all_fit, s=1, alpha=0.3, label="Individual")
        ax.plot(running_best, color="red", linewidth=2, label="Running best")
        ax.set_xlabel("Evaluation #")
        ax.set_ylabel("Fitness (ICL/baseline ratio)")
        ax.set_title("Adversarial Search Progress")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "fitness_over_time.png"), dpi=150)
        plt.close()
        print(f"\n  Saved: {output_dir}/fitness_over_time.png")

        # Top 5 learning curves
        fig, axes = plt.subplots(1, min(5, len(valid)), figsize=(4 * min(5, len(valid)), 4))
        if min(5, len(valid)) == 1:
            axes = [axes]
        for i, (ax, r) in enumerate(zip(axes, valid[:5])):
            x = np.arange(1, len(r.icl_curve) + 1)
            ax.plot(x, r.icl_curve, label="ICL", linewidth=2)
            for name, curve in r.baseline_curves.items():
                ax.plot(x, curve, label=name, linestyle="--")
            ax.set_xlabel("k")
            ax.set_ylabel("Squared error")
            ax.set_title(f"#{i+1} fit={r.fitness:.1f}\ncond={r.genome.condition_number('L_train'):.0f}")
            ax.legend(fontsize=7)
            ax.set_yscale("log")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "top_failures.png"), dpi=150)
        plt.close()
        print(f"  Saved: {output_dir}/top_failures.png")

        # Eigenvalue spectra of top 5
        fig, axes = plt.subplots(1, min(5, len(valid)), figsize=(4 * min(5, len(valid)), 3))
        if min(5, len(valid)) == 1:
            axes = [axes]
        for i, (ax, r) in enumerate(zip(axes, valid[:5])):
            ax.bar(range(len(r.covariance_spectrum)), r.covariance_spectrum)
            ax.set_xlabel("Index")
            ax.set_ylabel("Eigenvalue")
            ax.set_title(f"#{i+1} cond={r.genome.condition_number('L_train'):.0f}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "top_spectra.png"), dpi=150)
        plt.close()
        print(f"  Saved: {output_dir}/top_spectra.png")

        # Scatter: condition number vs fitness
        fig, ax = plt.subplots(figsize=(8, 5))
        conds = [r.genome.condition_number("L_train") for r in valid]
        ax.scatter(conds, fitnesses, s=5, alpha=0.5)
        ax.set_xlabel("Condition number (train covariance)")
        ax.set_ylabel("Fitness")
        ax.set_title("Condition Number vs ICL Failure")
        ax.set_xscale("log")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "cond_vs_fitness.png"), dpi=150)
        plt.close()
        print(f"  Saved: {output_dir}/cond_vs_fitness.png")

    except Exception as e:
        print(f"  Plotting failed: {e}")

    # Save summary
    summary = {
        "total_evals": len(results),
        "valid_evals": len(valid),
        "best_fitness": float(fitnesses.max()),
        "mean_fitness": float(fitnesses.mean()),
        "top_5": [
            {
                "fitness": float(r.fitness),
                "cond_train": float(r.genome.condition_number("L_train")),
                "cond_test": float(r.genome.condition_number("L_test")),
                "eff_rank": float(r.genome.effective_rank("L_train")),
                "noise_std": float(r.genome.decode_noise_std()),
                "weight_norm": float(np.linalg.norm(r.genome.decode_weights().numpy())),
                "descriptors": r.descriptors,
            }
            for r in valid[:5]
        ],
        "correlations": {name: float(rho) for name, rho, _ in correlations},
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {output_dir}/summary.json")

    return valid


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    N_DIMS = 5
    TRAIN_STEPS = 50000
    ADVERSARY_BUDGET = 3200  # 200 generations * 16 pop
    POP_SIZE = 16

    checkpoint_dir = os.path.join(ROOT, "results", "checkpoints", "experiment_run")
    adversary_dir = os.path.join(ROOT, "results", "adversary_runs", "experiment_run")
    analysis_dir = os.path.join(ROOT, "results", "analysis")

    print("=" * 70)
    print("STEP 1: Training ICL Transformer")
    print("=" * 70)
    t0 = time.time()
    model, args = train_icl_model(
        out_dir=checkpoint_dir,
        train_steps=TRAIN_STEPS,
        n_dims=N_DIMS,
        n_layer=4,
        n_head=2,
        n_embd=64,
    )
    t_train = time.time() - t0
    print(f"\nTraining time: {t_train:.1f}s")

    print("\n" + "=" * 70)
    print("STEP 2: Baseline Evaluation")
    print("=" * 70)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    icl_err, ols_err = quick_eval(model, N_DIMS, n_points=11)

    print("\n" + "=" * 70)
    print("STEP 3: Adversarial Search")
    print("=" * 70)
    t0 = time.time()
    results = run_adversary(
        model, N_DIMS, n_points=11,
        budget=ADVERSARY_BUDGET, pop_size=POP_SIZE,
        save_dir=adversary_dir,
    )
    t_search = time.time() - t0
    print(f"\nSearch time: {t_search:.1f}s")

    print("\n" + "=" * 70)
    print("STEP 4: Analysis")
    print("=" * 70)
    analyze_results(results, N_DIMS, analysis_dir)

    print(f"\n{'='*70}")
    print(f"EXPERIMENT COMPLETE")
    print(f"  Training: {t_train:.1f}s")
    print(f"  Search:   {t_search:.1f}s")
    print(f"  Results:  {analysis_dir}")
    print(f"{'='*70}")

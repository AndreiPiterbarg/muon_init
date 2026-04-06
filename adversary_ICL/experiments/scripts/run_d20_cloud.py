"""Launch d=20 adversarial search on cloud GPU.

Runs pre-flight checks (GPU available, model loads, model converged on
isotropic data, quick adversary smoke test) then starts the full search.

Usage:
    python experiments/scripts/run_d20_cloud.py
    python experiments/scripts/run_d20_cloud.py --smoke-test  # quick 200-eval test
"""

import argparse
import os
import sys
import time

import torch
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.icl.eval import get_model_from_run
from src.icl.samplers import GaussianSampler
from src.icl.tasks import get_task_sampler
from src.icl.models import RidgeRegressionModel
from src.adversary.evaluate import GenomeEvaluator
from src.adversary.search import cma_search
from src.adversary.pipeline_genome import PipelineGenome


CONFIG = {
    "run_path": "results/checkpoints/d20_12layer_500k",
    "n_dims": 20,
    "n_points": 41,
    "task_name": "noisy_linear_regression",
    "budget": 200_000,
    "pop_size": 64,
    "num_restarts": 10,
    "sigma_init": 0.3,
    "batch_size": 64,
    "num_batches": 10,
    "save_dir": "results/adversary_runs/d20_pipeline",
}


def preflight_check_gpu():
    """Verify CUDA is available and report GPU info."""
    if not torch.cuda.is_available():
        print("FATAL: No CUDA GPU detected. Aborting.")
        sys.exit(1)
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    return "cuda"


def preflight_check_model(device):
    """Load model and verify it converged on isotropic Gaussian data."""
    run_path = CONFIG["run_path"]
    n_dims = CONFIG["n_dims"]
    n_points = CONFIG["n_points"]

    print(f"\nLoading model from {run_path}...")
    model, conf = get_model_from_run(run_path, step=-1)
    model = model.to(device).eval()

    # Check convergence: ICL error should decrease with k on isotropic data
    print("Checking model convergence on isotropic Gaussian...")
    sampler = GaussianSampler(n_dims)
    ridge = RidgeRegressionModel(alpha=1.0)

    icl_errors = []
    ridge_errors = []
    n_eval_batches = 5

    with torch.no_grad():
        for _ in range(n_eval_batches):
            xs = sampler.sample_xs(n_points, CONFIG["batch_size"])
            task = get_task_sampler(
                CONFIG["task_name"], n_dims, CONFIG["batch_size"]
            )()
            ys = task.evaluate(xs)

            pred_icl = model(xs.to(device), ys.to(device)).detach().cpu()
            pred_ridge = ridge(xs, ys)

            icl_err = ((pred_icl - ys) ** 2).mean(dim=0).numpy()
            ridge_err = ((pred_ridge - ys) ** 2).mean(dim=0).numpy()

            icl_errors.append(icl_err)
            ridge_errors.append(ridge_err)

    icl_curve = np.mean(icl_errors, axis=0)
    ridge_curve = np.mean(ridge_errors, axis=0)

    # Check: ICL error at k=d should be much less than at k=1
    k_early = 2
    k_late = min(n_dims, n_points - 1)
    icl_ratio = icl_curve[k_late] / (icl_curve[k_early] + 1e-10)
    icl_vs_ridge_at_d = icl_curve[k_late] / (ridge_curve[k_late] + 1e-10)

    print(f"  ICL error at k={k_early}: {icl_curve[k_early]:.4f}")
    print(f"  ICL error at k={k_late}: {icl_curve[k_late]:.4f}")
    print(f"  Ridge error at k={k_late}: {ridge_curve[k_late]:.4f}")
    print(f"  ICL late/early ratio: {icl_ratio:.4f} (should be << 1 if converged)")
    print(f"  ICL/Ridge at k=d: {icl_vs_ridge_at_d:.2f}x")

    if icl_ratio > 0.9:
        print("WARNING: ICL error is not decreasing with context — model may not have converged!")
        print("         The adversary will find trivial failures. Consider retraining.")
        response = input("Continue anyway? [y/N]: ")
        if response.lower() != "y":
            sys.exit(1)
    else:
        print("  Model convergence: OK")

    return model, conf


def preflight_smoke_test(model, device):
    """Quick 200-eval adversary search to verify pipeline works at d=20."""
    print("\nRunning smoke test (200 evals, 1 restart)...")
    n_dims = CONFIG["n_dims"]

    evaluator = GenomeEvaluator(
        icl_model=model,
        task_name=CONFIG["task_name"],
        n_dims=n_dims,
        n_points=CONFIG["n_points"],
        batch_size=CONFIG["batch_size"],
        num_batches=CONFIG["num_batches"],
        baseline_names=["ridge", "least_squares", "averaging"],
    )

    # Time a single genome evaluation
    genome = PipelineGenome.random_structured(n_dims)
    t0 = time.time()
    result = evaluator.evaluate(genome)
    t1 = time.time()
    print(f"  Single genome eval: {t1 - t0:.2f}s (fitness={result.fitness:.4f})")

    # Estimate total time
    evals_per_sec = 1.0 / (t1 - t0)
    total_hours = CONFIG["budget"] / evals_per_sec / 3600
    print(f"  Estimated throughput: {evals_per_sec:.1f} evals/sec")
    print(f"  Estimated total time for {CONFIG['budget']} evals: {total_hours:.1f} hours")

    # Short search
    t0 = time.time()
    results = cma_search(
        evaluator=evaluator,
        n_dims=n_dims,
        budget=200,
        pop_size=32,
        sigma_init=CONFIG["sigma_init"],
        num_restarts=1,
        save_dir=None,
        seed=42,
        genome_cls=PipelineGenome,
    )
    t1 = time.time()
    valid = [r for r in results if r.is_valid]
    if valid:
        best = max(r.fitness for r in valid)
        print(f"  Smoke test: {len(results)} evals in {t1-t0:.1f}s, best fitness={best:.4f}")
    else:
        print("  WARNING: No valid results in smoke test!")
    print("  Smoke test: OK")


def run_full_search(model):
    """Run the full d=20 adversarial search."""
    device = next(model.parameters()).device
    n_dims = CONFIG["n_dims"]

    if hasattr(torch, "compile"):
        print("\nCompiling model with torch.compile...")
        model = torch.compile(model)

    evaluator = GenomeEvaluator(
        icl_model=model,
        task_name=CONFIG["task_name"],
        n_dims=n_dims,
        n_points=CONFIG["n_points"],
        batch_size=CONFIG["batch_size"],
        num_batches=CONFIG["num_batches"],
        baseline_names=["ridge", "least_squares", "averaging"],
    )

    os.makedirs(CONFIG["save_dir"], exist_ok=True)

    print(f"\nStarting full search: budget={CONFIG['budget']}, restarts={CONFIG['num_restarts']}")
    print(f"  Genome size: {PipelineGenome.flat_size(n_dims)}")
    print(f"  Save dir: {CONFIG['save_dir']}")

    t0 = time.time()
    results = cma_search(
        evaluator=evaluator,
        n_dims=n_dims,
        budget=CONFIG["budget"],
        pop_size=CONFIG["pop_size"],
        sigma_init=CONFIG["sigma_init"],
        num_restarts=CONFIG["num_restarts"],
        save_dir=CONFIG["save_dir"],
        save_interval=50,
        seed=0,
        genome_cls=PipelineGenome,
    )
    elapsed = time.time() - t0

    valid = [r for r in results if r.is_valid]
    if valid:
        best = max(valid, key=lambda r: r.fitness)
        print(f"\n=== SEARCH COMPLETE ===")
        print(f"Total time: {elapsed / 3600:.1f} hours")
        print(f"Total evals: {len(results)} ({len(valid)} valid)")
        print(f"Best fitness: {best.fitness:.4f}")
        print(f"Best genome: {best.genome}")
    else:
        print("No valid results found.")


def main():
    parser = argparse.ArgumentParser(description="d=20 adversarial search (cloud)")
    parser.add_argument("--smoke-test", action="store_true", help="Run only pre-flight + 200-eval smoke test")
    parser.add_argument("--skip-checks", action="store_true", help="Skip pre-flight convergence check")
    args = parser.parse_args()

    print("=" * 60)
    print("d=20 Adversarial Search — Cloud GPU Launch")
    print("=" * 60)

    device = preflight_check_gpu()

    if args.skip_checks:
        print("\nSkipping convergence check (--skip-checks)...")
        model, conf = get_model_from_run(CONFIG["run_path"], step=-1)
        model = model.to(device).eval()
    else:
        model, conf = preflight_check_model(device)

    if args.smoke_test:
        preflight_smoke_test(model, device)
        print("\nSmoke test complete. Run without --smoke-test for full search.")
        return

    preflight_smoke_test(model, device)
    run_full_search(model)


if __name__ == "__main__":
    main()

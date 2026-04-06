"""Quick lambda sweep for d=5 with multiple restarts per lambda."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
from src.icl.eval import get_model_from_run
from src.adversary.evaluate import GenomeEvaluator
from src.adversary.search import cma_search
from src.adversary.pipeline_genome import PipelineGenome

model, conf = get_model_from_run("results/checkpoints/d5_6layer_150k")
model = model.eval()

lambda_values = [0.0, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3]
n_dims = 5
n_points = 11

print("  Lambda |  Eff Fit |  Raw Fit | Stg | Cmplx | Pipeline")
print("-" * 80)

for lam in lambda_values:
    evaluator = GenomeEvaluator(
        icl_model=model,
        task_name="noisy_linear_regression",
        n_dims=n_dims,
        n_points=n_points,
        batch_size=64,
        num_batches=10,
        baseline_names=["ridge", "least_squares", "averaging"],
        parsimony_lambda=lam,
        c_base=1.0,
        c_stage=1.0,
        c_affine=1.0,
    )

    all_results = cma_search(
        evaluator=evaluator,
        n_dims=n_dims,
        budget=3000,
        pop_size=32,
        sigma_init=0.3,
        num_restarts=3,
        save_dir=None,
        seed=42,
        genome_cls=PipelineGenome,
    )

    valid = [r for r in all_results if r.is_valid]
    if not valid:
        print(f"  {lam:>5.3f} | no valid results")
        continue

    best = max(valid, key=lambda r: r.fitness)
    g = best.genome
    transforms = [name for _, name in g.active_stages()]
    desc = g.base_distribution_name() + " -> " + str(transforms)

    print(
        f"  {lam:>5.3f} | {best.fitness:>8.4f} | {best.raw_fitness:>8.4f} "
        f"| {g.num_active_stages():>3} | {best.complexity:>5.2f} | {desc}"
    )

    # Show stage count diversity across the 3 restarts
    chunk = len(valid) // 3
    stages_per_restart = []
    fits_per_restart = []
    for i in range(3):
        start = i * chunk
        end = (i + 1) * chunk if i < 2 else len(valid)
        rv = valid[start:end]
        if rv:
            rb = max(rv, key=lambda r: r.fitness)
            stages_per_restart.append(rb.genome.num_active_stages())
            fits_per_restart.append(rb.fitness)
    uniq = sorted(set(stages_per_restart))
    if len(uniq) > 1:
        detail = ", ".join(
            f"restart{i}: {s}stg(fit={f:.3f})"
            for i, (s, f) in enumerate(zip(stages_per_restart, fits_per_restart))
        )
        print(f"          ** stage diversity: {detail}")

print("\nDone.")

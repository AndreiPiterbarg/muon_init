"""Re-evaluate saved round-1 best genome with the oracle baseline added.

Loads the saved checkpoint from round 1 of d5_dual_v1, extracts the best
genome, and re-scores it against (a) the old baseline set {ridge(1), OLS,
averaging} and (b) the new set with the Bayes-optimal oracle included.

Prints both fitness numbers so we can see the delta introduced by the
oracle baseline on a known break.
"""
import pickle
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.adversary.evaluate import GenomeEvaluator  # noqa: E402
from src.icl.eval import get_model_from_run  # noqa: E402


def compute_fitness(baseline_curves, icl_curve, d, eps=1e-8):
    """Fitness formula from GenomeEvaluator."""
    best = np.minimum.reduce(list(baseline_curves.values()))
    k_max = min(d, len(icl_curve) - 1)
    log_ratios = []
    for k in range(1, k_max + 1):
        denom = max(best[k], eps)
        ratio = icl_curve[k] / denom
        log_ratios.append(np.log(max(ratio, eps)))
    return max(float(np.mean(log_ratios)), 0.0)


def main():
    ckpt = PROJECT_ROOT / "results/retrain_loop/d5_dual_v1/round_1_attack/checkpoint.pkl"
    with open(ckpt, "rb") as f:
        data = pickle.load(f)
    best_result = data["best_result"]
    best_genome = best_result.genome

    print("Saved best genome from round 1:")
    print(f"  {best_genome}")
    print(f"  saved fitness (old baseline set): {best_result.fitness:.4f}")

    model_dir = PROJECT_ROOT / "results/checkpoints/d5_6layer_150k"
    model, conf = get_model_from_run(str(model_dir))
    model.eval()
    d = int(conf.model.n_dims)
    n_points = int(conf.model.n_positions)

    evaluator = GenomeEvaluator(
        icl_model=model,
        task_name="noisy_linear_regression",
        n_dims=d,
        n_points=n_points,
        batch_size=64,
        num_batches=10,
    )

    # Re-evaluate with the new (oracle-including) baseline set
    result = evaluator.evaluate(best_genome)
    assert result.is_valid, "re-eval invalid"

    curves_with_oracle = dict(result.baseline_curves)
    curves_without_oracle = {k: v for k, v in curves_with_oracle.items() if k != "oracle"}

    fitness_new = compute_fitness(curves_with_oracle, result.icl_curve, d)
    fitness_old_on_base = compute_fitness(curves_without_oracle, result.icl_curve, d)

    sigma = best_genome.decode_noise_std()
    alpha_oracle = d * sigma ** 2

    print()
    print(f"Re-evaluated on base model (d5_6layer_150k), sigma={sigma:.4f}, alpha*={alpha_oracle:.4f}:")
    print(f"  fitness WITHOUT oracle in baselines  = {fitness_old_on_base:.4f}")
    print(f"  fitness WITH oracle in baselines     = {fitness_new:.4f}")
    print(f"  delta (oracle tightening)            = {fitness_old_on_base - fitness_new:+.4f}")
    print()
    print("Per-k curves (underdetermined regime):")
    print(f"  {'k':>3} | {'ICL':>10} | {'ridge(1)':>10} | {'OLS':>10} | {'avg':>10} | {'oracle':>10}")
    for k in range(1, d + 1):
        row = [
            result.icl_curve[k],
            curves_with_oracle.get("ridge", np.zeros(d + 1))[k],
            curves_with_oracle.get("least_squares", np.zeros(d + 1))[k],
            curves_with_oracle.get("averaging", np.zeros(d + 1))[k],
            curves_with_oracle.get("oracle", np.zeros(d + 1))[k],
        ]
        print(f"  {k:>3} | " + " | ".join(f"{v:>10.4f}" for v in row))


if __name__ == "__main__":
    main()

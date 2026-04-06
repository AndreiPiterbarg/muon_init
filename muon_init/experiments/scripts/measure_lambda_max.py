"""Measure lambda_max(alpha) at initialization for any model/config.

Phase 1 of Direction D (Sharpness-Aware Initialization): characterize how
the top Hessian eigenvalue depends on initialization scale alpha.

Usage:
    python -m experiments.scripts.measure_lambda_max \
        --config experiments/configs/mlp_cifar10.yaml \
        --alphas 0.1 0.2 0.3 0.4 0.5 0.75 1.0 1.414 1.5 2.0 2.5 3.0 \
        --seeds 42 43 44 45 46 \
        --num_batches 10
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
import yaml

from models import build_model
from data import build_dataloaders
from initializations.implementations.scaled_orthogonal import scaled_orthogonal
from evaluation.metrics.hessian.hessian_top_eigenvalue import compute_lambda_max


class _LogitsOnlyWrapper(nn.Module):
    """Wraps a GPT model so forward() returns only logits (no loss tuple).

    The HVP code does: logits = model(inputs); loss = loss_fn(logits, targets).
    GPT.forward() returns (logits, loss), which breaks this pattern.
    This wrapper makes GPT compatible with the HVP code.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        logits, _ = self.model(x)
        return logits


def wrap_model_if_lm(model, model_name):
    """Wrap LM models so forward() returns logits only."""
    if model_name in ("nanogpt", "deep_narrow_gpt"):
        return _LogitsOnlyWrapper(model)
    return model


class _FlatCELoss(nn.Module):
    """CrossEntropyLoss that flattens 3D logits (B,T,V) to (B*T,V)."""
    def forward(self, logits, targets):
        return nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1)
        )


def make_subset_loader(data_loader, num_batches):
    """Collect a fixed subset of batches into a simple list-based loader."""
    batches = []
    for i, batch in enumerate(data_loader):
        if i >= num_batches:
            break
        batches.append(batch)
    return batches


def measure_one(config, alpha, seed, subset_loader, device, num_iterations=100):
    """Initialize model at given alpha, measure lambda_max. Returns float."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = build_model(config["model"]).to(device)
    scaled_orthogonal(model, alpha=alpha)

    model_name = config["model"]["name"]
    model_for_hvp = wrap_model_if_lm(model, model_name)

    is_lm = model_name in ("nanogpt", "deep_narrow_gpt")
    loss_fn = _FlatCELoss() if is_lm else nn.CrossEntropyLoss()
    lam = compute_lambda_max(model_for_hvp, loss_fn, subset_loader,
                             num_iterations=num_iterations)
    return lam


def main():
    parser = argparse.ArgumentParser(description="Measure lambda_max(alpha) at init")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0,
                                 math.sqrt(2), 1.5, 2.0, 2.5, 3.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--num_batches", type=int, default=10,
                        help="Number of data batches to average Hessian over")
    parser.add_argument("--num_iterations", type=int, default=100,
                        help="Power iteration steps for lambda_max")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: auto-generated)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = config["model"]["name"]
    lr_muon = config["training"].get("lr_muon", 0.02)
    eos_threshold = 2.0 / lr_muon

    print(f"Model: {model_name} | Device: {device}")
    print(f"lr_muon: {lr_muon} | EoS threshold (2/eta): {eos_threshold:.1f}")
    print(f"Alphas: {args.alphas}")
    print(f"Seeds: {args.seeds}")
    print(f"Batches for Hessian: {args.num_batches}")
    print()

    # Load data once.
    train_loader, _ = build_dataloaders(config["data"])
    subset_loader = make_subset_loader(train_loader, args.num_batches)
    print(f"Loaded {len(subset_loader)} batches for Hessian computation")

    # Measure.
    results = {
        "config_path": args.config,
        "model_name": model_name,
        "lr_muon": lr_muon,
        "eos_threshold": eos_threshold,
        "num_batches": args.num_batches,
        "num_iterations": args.num_iterations,
        "measurements": [],
    }

    for alpha in args.alphas:
        lam_values = []
        for seed in args.seeds:
            t0 = time.time()
            lam = measure_one(config, alpha, seed, subset_loader, device,
                              num_iterations=args.num_iterations)
            elapsed = time.time() - t0
            lam_values.append(lam)
            print(f"  alpha={alpha:.4f}  seed={seed}  lambda_max={lam:.2f}  "
                  f"eta*lam={lr_muon * lam:.4f}  ({elapsed:.1f}s)")

        mean_lam = sum(lam_values) / len(lam_values)
        std_lam = (sum((v - mean_lam) ** 2 for v in lam_values) / len(lam_values)) ** 0.5
        stable = mean_lam < eos_threshold

        entry = {
            "alpha": alpha,
            "lambda_max_mean": mean_lam,
            "lambda_max_std": std_lam,
            "lambda_max_per_seed": lam_values,
            "eta_lambda_max": lr_muon * mean_lam,
            "stable": stable,
        }
        results["measurements"].append(entry)
        marker = "OK" if stable else "UNSTABLE"
        print(f"  => mean={mean_lam:.2f} +/- {std_lam:.2f}  "
              f"eta*lam={lr_muon * mean_lam:.4f}  [{marker}]")
        print()

    # Summary table.
    print("=" * 70)
    print(f"{'alpha':>8}  {'lambda_max':>12}  {'std':>8}  {'eta*lam':>10}  {'status':>8}")
    print("-" * 70)
    for m in results["measurements"]:
        status = "OK" if m["stable"] else "UNSTABLE"
        print(f"{m['alpha']:>8.4f}  {m['lambda_max_mean']:>12.2f}  "
              f"{m['lambda_max_std']:>8.2f}  {m['eta_lambda_max']:>10.4f}  "
              f"{status:>8}")
    print("=" * 70)
    print(f"EoS threshold: lambda_max < {eos_threshold:.1f}  (eta={lr_muon})")

    # Find alpha* (largest alpha that is stable).
    stable_alphas = [m["alpha"] for m in results["measurements"] if m["stable"]]
    if stable_alphas:
        alpha_star = max(stable_alphas)
        print(f"alpha*_EoS (largest stable): {alpha_star:.4f}")
        results["alpha_star_eos"] = alpha_star
    else:
        print("WARNING: No alpha is stable at this learning rate!")
        results["alpha_star_eos"] = None

    # Save.
    if args.output is None:
        save_dir = "experiments/results/phase1_lambda_max"
        os.makedirs(save_dir, exist_ok=True)
        args.output = os.path.join(save_dir, f"lambda_max_{model_name}.json")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

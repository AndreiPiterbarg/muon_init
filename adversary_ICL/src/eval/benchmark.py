"""Fixed evaluation benchmark for measuring ICL generalization.

A test suite of covariance structures spanning the space the adversary
explores. Evaluated before and after each retraining round — never used
during training. This is the ruler we measure all retraining against.

Distributions from RESEARCH_PLAN.md Step 1.
"""

import numpy as np
import torch

from ..icl.samplers import GaussianSampler
from ..icl import models


def _make_cholesky(eigenvalues, n_dims):
    """Build a trace-normalized Cholesky factor from eigenvalues.

    Returns L such that L^T @ L has the given eigenvalue structure,
    trace-normalized to n_dims, with a random orthogonal rotation.
    """
    eigenvalues = np.array(eigenvalues, dtype=np.float64)
    # Trace-normalize: sum of eigenvalues = n_dims
    eigenvalues = eigenvalues * (n_dims / eigenvalues.sum())
    # Random rotation
    Q, _ = np.linalg.qr(np.random.default_rng(42).standard_normal((n_dims, n_dims)))
    Sigma = Q @ np.diag(eigenvalues) @ Q.T
    # Ensure PSD (numerical)
    Sigma = (Sigma + Sigma.T) / 2
    eigvals = np.linalg.eigvalsh(Sigma)
    if eigvals.min() < 0:
        Sigma += (-eigvals.min() + 1e-10) * np.eye(n_dims)
        Sigma = Sigma * (n_dims / np.trace(Sigma))
    L = np.linalg.cholesky(Sigma)
    return torch.tensor(L, dtype=torch.float32)


def build_benchmark_distributions(n_dims):
    """Return list of (name, Cholesky_factor) for the fixed test suite.

    All covariances are trace-normalized (tr(Sigma) = n_dims).
    """
    eps = 1e-4  # small floor for near-zero eigenvalues
    dists = []

    # 1. Isotropic: Sigma = I
    dists.append(("isotropic", torch.eye(n_dims)))

    # 2. Rank-1: lambda_1 = d, rest ~ eps
    eigs = np.full(n_dims, eps)
    eigs[0] = n_dims
    dists.append(("rank_1", _make_cholesky(eigs, n_dims)))

    # 3. Rank-2: lambda_{1,2} = d/2, rest ~ eps
    eigs = np.full(n_dims, eps)
    eigs[0] = n_dims / 2
    eigs[1] = n_dims / 2
    dists.append(("rank_2", _make_cholesky(eigs, n_dims)))

    # 4. Rank d/2: top d/2 eigenvalues = 2, rest ~ eps
    eigs = np.full(n_dims, eps)
    k = max(n_dims // 2, 1)
    eigs[:k] = 2.0
    dists.append(("rank_half", _make_cholesky(eigs, n_dims)))

    # 5. Exponential decay: lambda_i = c * exp(-i/tau), tau=d/4
    tau = max(n_dims / 4, 1)
    eigs = np.array([np.exp(-i / tau) for i in range(n_dims)])
    dists.append(("exp_decay", _make_cholesky(eigs, n_dims)))

    # 6. Power law: lambda_i = c * (i+1)^{-2}
    eigs = np.array([1.0 / (i + 1) ** 2 for i in range(n_dims)])
    dists.append(("power_law", _make_cholesky(eigs, n_dims)))

    # 7. Step function: top k=d/3 eigenvalues = d/k, rest ~ eps
    k = max(n_dims // 3, 1)
    eigs = np.full(n_dims, eps)
    eigs[:k] = n_dims / k
    dists.append(("step_function", _make_cholesky(eigs, n_dims)))

    # 8. Random Wishart: Sigma ~ W_d(I, d) / d
    rng = np.random.default_rng(123)
    A = rng.standard_normal((n_dims, n_dims))
    Sigma_w = A @ A.T / n_dims
    Sigma_w = Sigma_w * (n_dims / np.trace(Sigma_w))
    L_w = np.linalg.cholesky(Sigma_w)
    dists.append(("wishart", torch.tensor(L_w, dtype=torch.float32)))

    # 9-12. Condition number sweep
    for kappa in [2, 10, 100, 1000]:
        eigs = np.ones(n_dims)
        eigs[0] = kappa
        dists.append((f"condition_{kappa}", _make_cholesky(eigs, n_dims)))

    return dists


def evaluate_benchmark(model, n_dims, n_points, batch_size=64, num_batches=10, seed=12345):
    """Evaluate model on all benchmark distributions.

    Uses a fixed seed so that all rounds are evaluated on identical test
    data (same task weights, same xs draws). This makes round-to-round
    comparisons clean and eliminates evaluation variance.

    Returns dict: dist_name -> {
        icl_curve, ridge_curve, icl_ridge_ratio_at_d, icl_ridge_ratio_at_d_half
    }
    """
    device = next(model.parameters()).device
    dists = build_benchmark_distributions(n_dims)
    ridge = models.RidgeRegressionModel(alpha=1.0)

    # Pre-generate all test data with fixed seed so every call gets identical inputs
    torch.manual_seed(seed)
    all_test_data = {}
    for dist_idx, (name, L) in enumerate(dists):
        batches = []
        for batch_idx in range(num_batches):
            xs = torch.randn(batch_size, n_points, n_dims) @ L
            w = torch.randn(batch_size, n_dims, 1)
            w = w / w.norm(dim=1, keepdim=True)
            noise = torch.randn(batch_size, n_points) * 0.1
            batches.append((xs, w, noise))
        all_test_data[name] = batches

    results = {}

    for name, L in dists:
        all_icl_err = []
        all_ridge_err = []

        for batch_idx in range(num_batches):
            xs, w, noise = all_test_data[name][batch_idx]

            # Compute ys directly from pre-generated data for full reproducibility
            # y = w^T x + noise (noisy linear regression)
            ys = (xs @ w).squeeze(-1) + noise

            # ICL
            with torch.no_grad():
                pred_icl = model(xs.to(device), ys.to(device)).cpu()
            icl_err = ((pred_icl - ys) ** 2).mean(dim=0)
            all_icl_err.append(icl_err)

            # Ridge
            pred_ridge = ridge(xs, ys)
            ridge_err = ((pred_ridge - ys) ** 2).mean(dim=0)
            all_ridge_err.append(ridge_err)

        icl_curve = torch.stack(all_icl_err).mean(dim=0).numpy()
        ridge_curve = torch.stack(all_ridge_err).mean(dim=0).numpy()

        # ICL/ridge ratio at key points
        eps = 1e-8
        d_idx = min(n_dims, len(icl_curve) - 1)
        d_half_idx = min(n_dims // 2, len(icl_curve) - 1)

        ratio_at_d = float(icl_curve[d_idx] / max(ridge_curve[d_idx], eps))
        ratio_at_d_half = float(icl_curve[d_half_idx] / max(ridge_curve[d_half_idx], eps))

        results[name] = {
            "icl_curve": icl_curve.tolist(),
            "ridge_curve": ridge_curve.tolist(),
            "icl_ridge_ratio_at_d": ratio_at_d,
            "icl_ridge_ratio_at_d_half": ratio_at_d_half,
        }

    return results


def print_benchmark(results, label=""):
    """Print benchmark results in a readable table."""
    if label:
        print(f"\n{'='*60}")
        print(f"  Benchmark: {label}")
        print(f"{'='*60}")

    print(f"  {'Distribution':<20} {'ICL/Ridge @d':>14} {'ICL/Ridge @d/2':>16}")
    print(f"  {'-'*20} {'-'*14} {'-'*16}")
    for name, r in results.items():
        rd = r["icl_ridge_ratio_at_d"]
        rh = r["icl_ridge_ratio_at_d_half"]
        flag = " ***" if rd > 5.0 else ""
        print(f"  {name:<20} {rd:>14.2f} {rh:>16.2f}{flag}")

    # Mean ratio (excluding isotropic)
    non_iso = [r["icl_ridge_ratio_at_d"] for n, r in results.items() if n != "isotropic"]
    if non_iso:
        print(f"  {'MEAN (non-iso)':<20} {np.mean(non_iso):>14.2f}")


def print_benchmark_comparison(before, after, label_before="Before", label_after="After"):
    """Print side-by-side comparison."""
    print(f"\n{'='*70}")
    print(f"  Benchmark Comparison: {label_before} vs {label_after}")
    print(f"{'='*70}")
    print(f"  {'Distribution':<18} {label_before+' @d':>12} {label_after+' @d':>12} {'Change':>10}")
    print(f"  {'-'*18} {'-'*12} {'-'*12} {'-'*10}")

    for name in before:
        if name not in after:
            continue
        rb = before[name]["icl_ridge_ratio_at_d"]
        ra = after[name]["icl_ridge_ratio_at_d"]
        change = ra - rb
        arrow = "improved" if change < -0.5 else ("degraded" if change > 0.5 else "~same")
        print(f"  {name:<18} {rb:>12.2f} {ra:>12.2f} {change:>+10.2f}  {arrow}")

    # Summary
    before_mean = np.mean([r["icl_ridge_ratio_at_d"] for n, r in before.items() if n != "isotropic"])
    after_mean = np.mean([r["icl_ridge_ratio_at_d"] for n, r in after.items() if n != "isotropic"])
    print(f"  {'MEAN (non-iso)':<18} {before_mean:>12.2f} {after_mean:>12.2f} {after_mean-before_mean:>+10.2f}")

    # Isotropic check (catastrophic forgetting)
    if "isotropic" in before and "isotropic" in after:
        iso_b = before["isotropic"]["icl_ridge_ratio_at_d"]
        iso_a = after["isotropic"]["icl_ridge_ratio_at_d"]
        if iso_a > iso_b * 1.5:
            print(f"\n  WARNING: Isotropic performance degraded {iso_b:.2f} -> {iso_a:.2f} (possible catastrophic forgetting)")
        else:
            print(f"\n  Isotropic retention OK: {iso_b:.2f} -> {iso_a:.2f}")

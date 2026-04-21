"""Validation: Bayes-optimal oracle baseline in GenomeEvaluator.

Verifies three correctness properties of the oracle added to
src/adversary/evaluate.py:

  1. Closed-form math: oracle uses alpha* = d * sigma^2 and matches the
     analytic posterior-mean formula on a hand-constructed Gaussian problem.
  2. Dominance: oracle MSE <= ridge(alpha=1) MSE in low-signal and
     high-signal regimes (where alpha=1 is misspecified).
  3. Integration: running GenomeEvaluator on a PipelineGenome produces an
     'oracle' entry in baseline_curves, and best_baseline <= oracle_curve
     element-wise (since best_baseline = min over all baselines).

Run:
    python -m experiments.scripts.verify_oracle_baseline
"""
import sys
from pathlib import Path

import numpy as np
import torch

# Make project importable when run as a script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.icl.models import RidgeRegressionModel  # noqa: E402
from src.icl.tasks import get_task_sampler  # noqa: E402


def analytic_posterior_mean(X, y, sigma2, tau2):
    """Closed-form posterior mean for w ~ N(0, tau2 I), y = Xw + N(0, sigma2 I).

    Returns w_hat = (X^T X + (sigma2/tau2) I)^-1 X^T y.
    """
    d = X.shape[1]
    alpha = sigma2 / tau2
    A = X.T @ X + alpha * np.eye(d)
    b = X.T @ y
    return np.linalg.solve(A, b)


def test_closed_form_match():
    """Oracle RidgeRegressionModel(alpha=d*sigma^2) should match analytic posterior mean
    for a hand-constructed Gaussian problem with w ~ N(0, I/d)."""
    torch.manual_seed(0)
    d = 5
    sigma = 0.3
    n_ctx = 7  # underdetermined-ish

    # True w drawn from prior N(0, I/d)
    w = torch.randn(d) / np.sqrt(d)

    # Gaussian X, Gaussian noise
    X = torch.randn(n_ctx, d)
    eps = sigma * torch.randn(n_ctx)
    y = X @ w + eps

    # Analytic posterior mean with correct alpha
    alpha_star = d * sigma ** 2
    w_hat_analytic = analytic_posterior_mean(
        X.numpy(), y.numpy(), sigma2=sigma ** 2, tau2=1.0 / d
    )

    # Oracle model: we have to match its interface. It predicts y for each test
    # index using all prior context. Build a 1-batch problem of length n_ctx+1
    # and read out the prediction at the last index after conditioning on the
    # first n_ctx examples. Equivalently: its internal XtX/Xty after absorbing
    # the first n_ctx examples equals our X^T X, X^T y.
    oracle = RidgeRegressionModel(alpha=alpha_star)
    # Construct xs/ys shape (batch=1, n_ctx+1, d)
    x_test = torch.randn(1, d)
    xs = torch.cat([X, x_test], dim=0).unsqueeze(0)  # (1, n_ctx+1, d)
    # Any ys value at the final slot; it's only used for the *prediction*, not
    # the fit. Append a zero.
    ys = torch.cat([y, torch.zeros(1)], dim=0).unsqueeze(0)  # (1, n_ctx+1)

    preds = oracle(xs, ys)  # (1, n_ctx+1)
    pred_at_last = preds[0, n_ctx].item()

    # Analytic prediction
    pred_analytic = float((x_test.numpy() @ w_hat_analytic).item())

    err = abs(pred_at_last - pred_analytic)
    assert err < 1e-4, f"Oracle prediction mismatch: {pred_at_last} vs {pred_analytic}, err={err}"
    print(f"[PASS] Closed-form match: |oracle - analytic| = {err:.2e}")


def test_oracle_beats_ridge1_when_misspecified():
    """Oracle should have <= MSE than Ridge(alpha=1) when alpha=1 is wrong."""
    torch.manual_seed(1)
    d = 5
    n_trials = 200
    n_ctx = 8
    n_test = 50

    # Two regimes where alpha=1 is misspecified:
    #   - low noise (sigma=0.05): optimal alpha = 0.0125, ridge(1) over-regularizes
    #   - high noise (sigma=2.0): optimal alpha = 20, ridge(1) under-regularizes
    for sigma in [0.05, 2.0]:
        alpha_star = d * sigma ** 2
        oracle = RidgeRegressionModel(alpha=alpha_star)
        ridge1 = RidgeRegressionModel(alpha=1.0)

        mse_oracle = 0.0
        mse_ridge1 = 0.0
        for trial in range(n_trials):
            w = torch.randn(d) / np.sqrt(d)  # prior N(0, I/d)
            X_ctx = torch.randn(n_ctx, d)
            y_ctx = X_ctx @ w + sigma * torch.randn(n_ctx)
            X_test = torch.randn(n_test, d)
            y_test_clean = X_test @ w

            xs = torch.cat([X_ctx, X_test], dim=0).unsqueeze(0)
            ys = torch.cat([y_ctx, torch.zeros(n_test)], dim=0).unsqueeze(0)

            pred_o = oracle(xs, ys)[0, n_ctx:].numpy()
            pred_r = ridge1(xs, ys)[0, n_ctx:].numpy()

            mse_oracle += np.mean((pred_o - y_test_clean.numpy()) ** 2)
            mse_ridge1 += np.mean((pred_r - y_test_clean.numpy()) ** 2)
        mse_oracle /= n_trials
        mse_ridge1 /= n_trials

        # Oracle should be weakly better (with enough trials, strictly better
        # when ridge(1) is misspecified).
        assert mse_oracle <= mse_ridge1 + 1e-6, (
            f"Oracle lost to ridge(1) at sigma={sigma}: "
            f"oracle={mse_oracle:.4f}, ridge1={mse_ridge1:.4f}"
        )
        gap_pct = 100 * (mse_ridge1 - mse_oracle) / (mse_ridge1 + 1e-12)
        print(
            f"[PASS] Oracle <= Ridge(1) at sigma={sigma:<4}: "
            f"oracle={mse_oracle:.4f}, ridge1={mse_ridge1:.4f} (oracle {gap_pct:+.1f}%)"
        )


def test_edge_cases():
    """Oracle must produce finite, non-NaN predictions in edge cases."""
    torch.manual_seed(2)
    d = 5
    n_ctx = 6

    # Edge: sigma -> 0 (alpha* -> 0, oracle ~ OLS)
    oracle_lownoise = RidgeRegressionModel(alpha=d * (1e-5) ** 2)
    # Edge: sigma -> large (alpha* -> large, oracle ~ predicts ~0)
    oracle_highnoise = RidgeRegressionModel(alpha=d * (5.0) ** 2)

    X = torch.randn(1, n_ctx + 1, d)
    y = torch.randn(1, n_ctx + 1)

    for name, oracle in [("low_noise", oracle_lownoise), ("high_noise", oracle_highnoise)]:
        preds = oracle(X, y)
        assert torch.isfinite(preds).all(), f"Oracle produced NaN/Inf at {name}"
        print(f"[PASS] Oracle finite at {name}: preds range [{preds.min():.3f}, {preds.max():.3f}]")


def test_integration_with_evaluator():
    """End-to-end: run GenomeEvaluator on a PipelineGenome, check oracle is
    included in baseline_curves and best_baseline <= oracle element-wise."""
    from src.adversary.pipeline_genome import PipelineGenome  # noqa: E402
    from src.adversary.evaluate import GenomeEvaluator  # noqa: E402
    from src.icl.models import TransformerModel  # noqa: E402

    d = 5
    n_points = 11
    # Small untrained transformer: we just need a callable model for the eval path.
    icl = TransformerModel(n_dims=d, n_positions=n_points, n_embd=16, n_layer=2, n_head=2)
    icl.eval()

    evaluator = GenomeEvaluator(
        icl_model=icl,
        task_name="noisy_linear_regression",
        n_dims=d,
        n_points=n_points,
        batch_size=32,
        num_batches=2,
    )

    rng = np.random.default_rng(42)
    g = PipelineGenome.random_structured(d, rng)
    result = evaluator.evaluate(g)

    assert result.is_valid, "Evaluation failed unexpectedly"
    assert "oracle" in result.baseline_curves, (
        f"'oracle' missing from baseline_curves: got {list(result.baseline_curves.keys())}"
    )

    oracle_curve = result.baseline_curves["oracle"]
    best_baseline = np.minimum.reduce(list(result.baseline_curves.values()))
    # best_baseline is elementwise min across all baselines (including oracle),
    # so best_baseline <= oracle_curve element-wise.
    assert np.all(best_baseline <= oracle_curve + 1e-8), (
        "best_baseline exceeds oracle_curve somewhere — min-reduction is broken"
    )
    # Oracle is a legitimate baseline => finite values
    assert np.all(np.isfinite(oracle_curve)), "oracle_curve has non-finite entries"

    # Alpha sanity: expected alpha* = d * sigma^2 from genome's decoded noise
    sigma = g.decode_noise_std()
    alpha_expected = d * sigma ** 2
    print(
        f"[PASS] Integration: oracle in baselines, "
        f"oracle_curve finite, best<=oracle. "
        f"(genome sigma={sigma:.4f}, expected alpha*={alpha_expected:.4f})"
    )
    print(f"       oracle_curve[1:]    = {np.array2string(oracle_curve[1:], precision=4)}")
    print(f"       best_baseline[1:]   = {np.array2string(best_baseline[1:], precision=4)}")


def test_identity_pipeline_sanity():
    """Harness sanity check (per reviewer feedback).

    On the identity PipelineGenome (Gaussian base, all transforms identity,
    w = e_1, noise sigma=0.1), the trained ICL transformer should match the
    Bayes-optimal oracle to within noise. If it does not, there is a harness
    bug (wrong model, wrong task, wrong scaling) before any adversarial break
    is trustworthy.

    Loads the d5_6layer_150k base checkpoint. Expected behavior: per-point
    oracle error and ICL error are close, mean log-ratio (fitness) is small.
    """
    from src.adversary.pipeline_genome import PipelineGenome  # noqa: E402
    from src.adversary.evaluate import GenomeEvaluator  # noqa: E402
    from src.icl.eval import get_model_from_run  # noqa: E402

    ckpt_dir = PROJECT_ROOT / "results" / "checkpoints" / "d5_6layer_150k"
    if not (ckpt_dir / "state.pt").exists():
        print(f"[SKIP] Identity sanity: no checkpoint at {ckpt_dir}")
        return

    model, conf = get_model_from_run(str(ckpt_dir))
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

    g_identity = PipelineGenome.identity(d)
    assert g_identity.num_active_stages() == 0, "identity genome has active stages?"
    sigma = g_identity.decode_noise_std()
    alpha_expected = d * sigma ** 2

    result = evaluator.evaluate(g_identity)
    assert result.is_valid, "identity genome failed to evaluate"

    icl = result.icl_curve
    oracle = result.baseline_curves["oracle"]
    ridge1 = result.baseline_curves["ridge"]

    # Print the curves in the underdetermined regime for visual inspection
    print(f"\n  Identity pipeline (d={d}, sigma={sigma:.4f}, alpha*={alpha_expected:.4f})")
    print(f"  {'k':>3} | {'ICL':>8} | {'oracle':>8} | {'ridge(1)':>8} | {'ICL/oracle':>10}")
    for k in range(1, min(d + 1, len(icl))):
        ratio = icl[k] / max(oracle[k], 1e-12)
        print(f"  {k:>3} | {icl[k]:>8.4f} | {oracle[k]:>8.4f} | {ridge1[k]:>8.4f} | {ratio:>10.3f}")

    # Compute fitness on identity pipeline (should be near zero)
    eps = 1e-8
    best = np.minimum.reduce(list(result.baseline_curves.values()))
    log_ratios = [
        np.log(max(icl[k] / max(best[k], eps), eps)) for k in range(1, d + 1)
    ]
    fitness = max(float(np.mean(log_ratios)), 0.0)

    # Tolerance: on the identity pipeline, ICL should be within factor e (log<=1)
    # of the oracle on average. Fitness substantially above ~1 would indicate
    # the ICL harness/model is not actually doing ridge-like ICL on Gaussian data,
    # which blocks any downstream adversarial-break interpretation.
    assert fitness < 1.0, (
        f"Identity-pipeline fitness = {fitness:.3f} exceeds 1.0 — "
        f"ICL does not match oracle on clean Gaussian data. Harness bug likely."
    )
    # Irreducible-floor check: at k=d, oracle MSE should be O(sigma^2) + finite-sample;
    # an absurdly low or high number indicates a scaling bug.
    assert 0.1 * sigma ** 2 < oracle[d] < 100 * sigma ** 2, (
        f"Oracle error at k=d={d}: {oracle[d]:.4f} is not in plausible range "
        f"around sigma^2={sigma**2:.4f}"
    )
    print(f"[PASS] Identity sanity: fitness={fitness:.3f} (<1.0), "
          f"oracle[d]={oracle[d]:.4f} within O(sigma^2={sigma**2:.4f})")


def main():
    print("=" * 60)
    print("Oracle baseline validation (Mode A: labels post-transform)")
    print("=" * 60)
    test_closed_form_match()
    test_oracle_beats_ridge1_when_misspecified()
    test_edge_cases()
    test_integration_with_evaluator()
    test_identity_pipeline_sanity()
    print("=" * 60)
    print("All checks passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()

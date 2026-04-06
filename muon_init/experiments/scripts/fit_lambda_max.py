"""Phase 2: Fit lambda_max(alpha) and derive alpha*_EoS.

Loads Phase 1 measurements, fits parametric models, solves for the
initialization scale alpha where eta * lambda_max(alpha) = 2.

Usage:
    python -m experiments.scripts.fit_lambda_max \
        --results experiments/results/phase1_lambda_max/lambda_max_deep_mlp.json \
                   experiments/results/phase1_lambda_max/lambda_max_deep_mlp_fine.json \
        --model_name deep_mlp \
        --num_layers 8
"""

import argparse
import json
import math
import os

import numpy as np
from scipy.optimize import curve_fit, brentq


# ── Parametric models ──────────────────────────────────────────────

def power_law(alpha, C, k):
    """lambda_max = C * alpha^k"""
    return C * np.power(alpha, k)


def power_law_with_floor(alpha, C, k, floor):
    """lambda_max = C * alpha^k + floor

    The floor captures the data-dependent baseline Hessian curvature that
    exists even at alpha -> 0 (from the loss function + data, not weights).
    """
    return C * np.power(alpha, k) + floor


def exponential(alpha, a, b):
    """lambda_max = a * exp(b * alpha)"""
    return a * np.exp(b * alpha)


# ── Loading and merging ────────────────────────────────────────────

def load_and_merge(result_paths):
    """Load multiple result JSONs, merge measurements, deduplicate by alpha."""
    all_measurements = {}
    meta = {}

    for path in result_paths:
        with open(path) as f:
            data = json.load(f)
        if not meta:
            meta = {k: v for k, v in data.items() if k != "measurements"}
        for m in data["measurements"]:
            alpha = round(m["alpha"], 4)
            # Keep the measurement with more seeds or lower std
            if alpha not in all_measurements:
                all_measurements[alpha] = m
            else:
                existing = all_measurements[alpha]
                if len(m["lambda_max_per_seed"]) > len(existing["lambda_max_per_seed"]):
                    all_measurements[alpha] = m

    # Sort by alpha
    sorted_measurements = sorted(all_measurements.values(), key=lambda m: m["alpha"])
    return sorted_measurements, meta


# ── Fitting ────────────────────────────────────────────────────────

def fit_models(alphas, lambdas, weights=None):
    """Fit all parametric models, return dict of {name: (params, r_squared, func)}.

    R^2 is computed in log space for power-law fits (since the data spans
    many orders of magnitude, linear R^2 is dominated by the largest values).
    """
    results = {}

    def r_squared_log(y_true, y_pred):
        """R^2 in log space — appropriate for data spanning orders of magnitude."""
        mask = (y_true > 0) & (y_pred > 0)
        if np.sum(mask) < 2:
            return 0.0
        log_true = np.log(y_true[mask])
        log_pred = np.log(y_pred[mask])
        ss_res = np.sum((log_true - log_pred) ** 2)
        ss_tot = np.sum((log_true - np.mean(log_true)) ** 2)
        if ss_tot < 1e-15:
            return 0.0
        return 1.0 - ss_res / ss_tot

    # Identify the non-floor regime: points where lambda rises above the
    # data-dependent floor. The floor is the minimum lambda_max, typically
    # from the smallest alpha values where weights barely affect curvature.
    floor_val = np.min(lambdas)
    # "Above floor" = at least 2x the minimum (clearly in the power-law regime)
    above_floor = lambdas > 2 * floor_val
    n_above = np.sum(above_floor)

    # 1. Power law (log-log linear regression on above-floor data only)
    if n_above >= 3:
        log_a = np.log(alphas[above_floor])
        log_l = np.log(lambdas[above_floor])
        coeffs = np.polyfit(log_a, log_l, 1)
        k_fit = coeffs[0]
        C_fit = np.exp(coeffs[1])
        pred = power_law(alphas[above_floor], C_fit, k_fit)
        r2 = r_squared_log(lambdas[above_floor], pred)
        results["power_law"] = {
            "params": {"C": C_fit, "k": k_fit},
            "r_squared": r2,
            "fitted_range": f"alpha >= {alphas[above_floor][0]:.2f} ({n_above} points)",
            "func": lambda a, C=C_fit, k=k_fit: power_law(a, C, k),
        }

    # 2. Power law with floor (fits entire range)
    try:
        popt, _ = curve_fit(
            power_law_with_floor, alphas, lambdas,
            p0=[1.0, 16.0, floor_val],
            bounds=([0, 0.1, 0], [1e10, 50, 10 * floor_val]),
            sigma=weights,
            maxfev=10000,
        )
        C_fit, k_fit, floor_fit = popt
        pred = power_law_with_floor(alphas, *popt)
        r2 = r_squared_log(lambdas, pred)
        results["power_law_floor"] = {
            "params": {"C": C_fit, "k": k_fit, "floor": floor_fit},
            "r_squared": r2,
            "func": lambda a, C=C_fit, k=k_fit, f=floor_fit: power_law_with_floor(a, C, k, f),
        }
    except RuntimeError:
        pass

    # 3. Exponential (fits entire range)
    try:
        popt, _ = curve_fit(
            exponential, alphas, lambdas,
            p0=[0.01, 10.0],
            bounds=([0, 0], [1e6, 100]),
            sigma=weights,
            maxfev=10000,
        )
        a_fit, b_fit = popt
        pred = exponential(alphas, *popt)
        r2 = r_squared_log(lambdas, pred)
        results["exponential"] = {
            "params": {"a": a_fit, "b": b_fit},
            "r_squared": r2,
            "func": lambda a, a_f=a_fit, b_f=b_fit: exponential(a, a_f, b_f),
        }
    except RuntimeError:
        pass

    return results


def solve_alpha_star(fit_func, threshold, bracket=(0.01, 10.0)):
    """Solve fit_func(alpha) = threshold using Brent's method."""
    f = lambda a: fit_func(a) - threshold
    # Check bracket validity
    try:
        fa, fb = f(bracket[0]), f(bracket[1])
    except (ValueError, OverflowError):
        return None
    if fa * fb > 0:
        # No sign change — threshold may not be crossed in bracket
        # Try to find a valid bracket
        for upper in [5.0, 3.0, 2.0, 1.5]:
            try:
                if f(bracket[0]) * f(upper) < 0:
                    bracket = (bracket[0], upper)
                    break
            except (ValueError, OverflowError):
                continue
        else:
            return None
    try:
        return brentq(f, bracket[0], bracket[1], xtol=1e-6)
    except ValueError:
        return None


# ── Analytical derivation ──────────────────────────────────────────

def analytical_alpha_star(num_layers, c_phi, threshold, C_data):
    """Analytical alpha* from the Gauss-Newton approximation.

    Theory (see analysis_phase2.md for derivation):

    For an L-layer MLP with activation gain c_phi and scaled orthogonal
    init W_l = alpha * Q_l, the Gauss-Newton approximation gives:

        lambda_max ≈ C_data * (alpha^2 * c_phi)^L

    where C_data depends on data statistics and loss function (measured
    empirically from a single lambda_max value at a known alpha).

    Solving lambda_max = threshold:
        alpha* = (threshold / C_data) ^ (1/(2L)) / sqrt(c_phi)

    Parameters
    ----------
    num_layers : int
        Number of layers L.
    c_phi : float
        Activation gain E[phi(z)^2] for z ~ N(0,1). ReLU: 0.5.
    threshold : float
        Target lambda_max (= 2/eta).
    C_data : float
        Data-dependent constant, estimated from a reference measurement.
    """
    exponent = 1.0 / (2 * num_layers)
    return (threshold / C_data) ** exponent / math.sqrt(c_phi)


def estimate_C_data(alpha_ref, lambda_ref, num_layers, c_phi):
    """Estimate C_data from a single (alpha, lambda_max) measurement.

    C_data = lambda_ref / (alpha_ref^2 * c_phi)^L
    """
    return lambda_ref / (alpha_ref ** 2 * c_phi) ** num_layers


# ── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fit lambda_max(alpha) and derive alpha*_EoS")
    parser.add_argument("--results", type=str, nargs="+", required=True,
                        help="Paths to Phase 1 JSON result files")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_layers", type=int, default=None,
                        help="Number of layers (for analytical derivation)")
    parser.add_argument("--c_phi", type=float, default=0.5,
                        help="Activation gain E[phi(z)^2] (default: 0.5 for ReLU)")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    measurements, meta = load_and_merge(args.results)
    lr = meta.get("lr_muon", 0.02)
    threshold = 2.0 / lr

    print(f"Model: {args.model_name}")
    print(f"lr_muon: {lr} | EoS threshold: {threshold}")
    print(f"Loaded {len(measurements)} unique alpha values")
    print()

    # Extract arrays
    alphas = np.array([m["alpha"] for m in measurements])
    lambdas = np.array([m["lambda_max_mean"] for m in measurements])
    stds = np.array([m["lambda_max_std"] for m in measurements])

    # Filter out negative lambda_max values (e.g. ResNet with BatchNorm)
    valid = lambdas > 0
    if not np.all(valid):
        print(f"WARNING: {np.sum(~valid)} measurements have negative lambda_max (removed)")
        alphas = alphas[valid]
        lambdas = lambdas[valid]
        stds = stds[valid]

    if len(alphas) < 3:
        print("ERROR: Not enough valid measurements to fit")
        return

    # Use 1/std as weights (more precise measurements count more),
    # but cap weights to avoid division by near-zero
    weights = np.clip(stds, 0.01, None)

    # ── Fit parametric models ──────────────────────────────────────
    print("=" * 70)
    print("PARAMETRIC FITS")
    print("=" * 70)

    fits = fit_models(alphas, lambdas, weights)

    best_name, best_r2 = None, -1
    for name, fit in sorted(fits.items(), key=lambda x: -x[1]["r_squared"]):
        r2 = fit["r_squared"]
        params_str = ", ".join(f"{k}={v:.6f}" for k, v in fit["params"].items())
        print(f"  {name:25s}  R²={r2:.6f}  ({params_str})")
        if r2 > best_r2:
            best_name, best_r2 = name, r2

    print(f"\n  Best fit: {best_name} (R²={best_r2:.6f})")
    best_fit = fits[best_name]

    # ── Solve for alpha* ───────────────────────────────────────────
    print()
    print("=" * 70)
    print("ALPHA*_EoS FROM FITS")
    print("=" * 70)

    for name, fit in sorted(fits.items(), key=lambda x: -x[1]["r_squared"]):
        alpha_star = solve_alpha_star(fit["func"], threshold)
        if alpha_star is not None:
            lam_at_star = fit["func"](alpha_star)
            print(f"  {name:25s}  alpha*={alpha_star:.6f}  "
                  f"(lambda_max={lam_at_star:.2f}, eta*lam={lr * lam_at_star:.4f})")
        else:
            print(f"  {name:25s}  alpha*=UNSOLVABLE")

    alpha_star_best = solve_alpha_star(best_fit["func"], threshold)

    # ── Analytical derivation (if num_layers provided) ─────────────
    alpha_star_analytical = None
    if args.num_layers is not None:
        print()
        print("=" * 70)
        print("ANALYTICAL DERIVATION")
        print("=" * 70)

        L = args.num_layers
        c_phi = args.c_phi
        print(f"  Layers L={L}, activation gain c_phi={c_phi}")
        print(f"  Theory: lambda_max ~ C_data * (alpha^2 * c_phi)^L")
        print(f"  Predicted exponent k = 2L = {2*L}")
        print()

        # Estimate C_data from the measurement closest to alpha=1.0
        ref_idx = np.argmin(np.abs(alphas - 1.0))
        alpha_ref = alphas[ref_idx]
        lambda_ref = lambdas[ref_idx]
        C_data = estimate_C_data(alpha_ref, lambda_ref, L, c_phi)

        print(f"  Reference: alpha={alpha_ref:.4f}, lambda_max={lambda_ref:.4f}")
        print(f"  Estimated C_data = {C_data:.6f}")
        print()

        # Analytical alpha*
        alpha_star_analytical = analytical_alpha_star(L, c_phi, threshold, C_data)
        lam_predicted = C_data * (alpha_star_analytical ** 2 * c_phi) ** L
        print(f"  alpha*_analytical = (threshold/C_data)^(1/(2L)) / sqrt(c_phi)")
        print(f"                    = ({threshold}/{C_data:.4f})^(1/{2*L}) / {math.sqrt(c_phi):.4f}")
        print(f"                    = {alpha_star_analytical:.6f}")
        print(f"  Predicted lambda_max at alpha*: {lam_predicted:.2f}")
        print(f"  eta * lambda_max: {lr * lam_predicted:.4f}")
        print()

        # Compare analytical predictions to all measurements
        print("  Analytical model vs. data:")
        print(f"  {'alpha':>8}  {'measured':>12}  {'predicted':>12}  {'ratio':>8}")
        print(f"  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*8}")
        for a, l in zip(alphas, lambdas):
            pred = C_data * (a ** 2 * c_phi) ** L
            ratio = l / pred if pred > 1e-10 else float('inf')
            print(f"  {a:>8.4f}  {l:>12.4f}  {pred:>12.4f}  {ratio:>8.2f}")

        # Also check: does the fitted exponent k match 2L?
        if "power_law" in fits:
            k_fitted = fits["power_law"]["params"]["k"]
            print(f"\n  Fitted exponent k = {k_fitted:.2f} vs. predicted 2L = {2*L}")
            print(f"  Ratio k/(2L) = {k_fitted/(2*L):.3f}")
            if abs(k_fitted - 2*L) / (2*L) < 0.2:
                print(f"  MATCH: fitted exponent is within 20% of 2L")
            else:
                print(f"  MISMATCH: fitted exponent differs from 2L by "
                      f"{abs(k_fitted - 2*L) / (2*L) * 100:.0f}%")

    # ── Summary ────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Model:                {args.model_name}")
    print(f"  EoS threshold:        lambda_max < {threshold:.1f}  (eta={lr})")
    if alpha_star_best is not None:
        print(f"  alpha*_EoS (empirical, {best_name}): {alpha_star_best:.6f}")
    if alpha_star_analytical is not None:
        print(f"  alpha*_EoS (analytical):            {alpha_star_analytical:.6f}")
    if alpha_star_best is not None and alpha_star_analytical is not None:
        diff_pct = abs(alpha_star_best - alpha_star_analytical) / alpha_star_best * 100
        print(f"  Difference:                         {diff_pct:.1f}%")

    # Grid search alpha* (most reliable when fits are poor)
    stable_alphas = [m["alpha"] for m in measurements
                     if m["lambda_max_mean"] > 0 and m["lambda_max_mean"] < threshold]
    alpha_star_grid = max(stable_alphas) if stable_alphas else None
    if alpha_star_grid is not None:
        print(f"  alpha*_EoS (grid search):            {alpha_star_grid:.4f}")

    # ── Save results ───────────────────────────────────────────────

    output = {
        "model_name": args.model_name,
        "lr_muon": lr,
        "eos_threshold": threshold,
        "num_measurements": len(measurements),
        "fits": {},
        "alpha_star_eos_grid": alpha_star_grid,
        "alpha_star_eos_empirical": alpha_star_best,
        "alpha_star_eos_analytical": alpha_star_analytical,
    }
    for name, fit in fits.items():
        alpha_star = solve_alpha_star(fit["func"], threshold)
        output["fits"][name] = {
            "params": {k: float(v) for k, v in fit["params"].items()},
            "r_squared": fit["r_squared"],
            "alpha_star": alpha_star,
        }
    if args.num_layers is not None:
        output["analytical"] = {
            "num_layers": args.num_layers,
            "c_phi": args.c_phi,
            "C_data": float(C_data),
            "alpha_ref": float(alpha_ref),
            "lambda_ref": float(lambda_ref),
            "predicted_exponent": 2 * args.num_layers,
        }
        if "power_law" in fits:
            output["analytical"]["fitted_exponent"] = fits["power_law"]["params"]["k"]

    if args.output is None:
        save_dir = "experiments/results/phase2_fits"
        os.makedirs(save_dir, exist_ok=True)
        args.output = os.path.join(save_dir, f"fit_{args.model_name}.json")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

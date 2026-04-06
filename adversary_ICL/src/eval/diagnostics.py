"""Diagnostics for understanding ICL model behavior and failure modes.

Eight diagnostics, each implemented as a class with a shared interface.
These help distinguish interesting failures (ICL learned a narrow algorithm
that breaks on non-standard data) from trivial ones (model hasn't learned
anything, or the distribution is hard for everyone).

References are documented in expressiveness_solutions.md.
"""

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import Ridge as SklearnRidge

from ..icl import models
from ..icl.samplers import GaussianSampler


@dataclass
class DiagnosticResult:
    """Output of a single diagnostic run."""
    name: str
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    plots: list[str] = field(default_factory=list)


# --- Shared utilities ---


def _generate_standard_data(n_dims, n_points, batch_size):
    """Generate standard Gaussian data for baseline comparison."""
    sampler = GaussianSampler(n_dims)
    xs = sampler.sample_xs(n_points, batch_size)
    w = torch.randn(n_dims)
    w = w / w.norm()
    ys = (xs @ w).squeeze(-1) if xs.dim() == 3 else xs @ w
    if ys.dim() == 3:
        ys = ys.squeeze(-1)
    # Add small noise
    ys = ys + 0.1 * torch.randn_like(ys)
    return xs, ys, w


def _compute_ridge_estimate(xs, ys, alpha=1.0):
    """Compute ridge regression estimate: w = (X^T X + alpha I)^-1 X^T y."""
    # xs: (batch, k, d), ys: (batch, k)
    XtX = xs.transpose(-2, -1) @ xs  # (batch, d, d)
    d = xs.shape[-1]
    reg = alpha * torch.eye(d, device=xs.device).unsqueeze(0)
    Xty = xs.transpose(-2, -1) @ ys.unsqueeze(-1)  # (batch, d, 1)
    w_ridge = torch.linalg.solve(XtX + reg, Xty).squeeze(-1)  # (batch, d)
    return w_ridge


# =============================================================================
# Diagnostic 1: Learning Curve Comparison
# =============================================================================


class LearningCurveComparison:
    """Compare ICL vs baselines across the learning curve.

    Classifies the failure pattern:
    - 'icl_tracks_then_diverges': ICL matches ridge on standard, fails on adversarial (interesting)
    - 'icl_matches_averaging': ICL hasn't learned much (not interesting)
    - 'all_methods_fail': Distribution is hard for everyone (not ICL-specific)
    """

    name = "learning_curve"

    def run(self, model, xs, ys, genome, output_dir=None, **kwargs) -> DiagnosticResult:
        evaluator = kwargs.get("evaluator")
        if evaluator is None:
            return DiagnosticResult(
                name=self.name,
                summary="Evaluator required but not provided",
            )

        # Evaluate on adversarial data
        from ..adversary.evaluate import EvalResult
        adv_result = evaluator.evaluate(genome)

        # Evaluate on standard Gaussian
        from ..adversary.genome import Genome
        std_genome = Genome.identity(genome.n_dims)
        std_result = evaluator.evaluate(std_genome)

        # Classification logic
        icl_adv = adv_result.icl_curve
        icl_std = std_result.icl_curve
        ridge_adv = adv_result.baseline_curves.get("ridge", icl_adv)
        ridge_std = std_result.baseline_curves.get("ridge", icl_std)
        avg_adv = adv_result.baseline_curves.get("averaging", icl_adv)

        # Correlation between ICL and ridge on standard data
        n_dims = genome.n_dims
        k_range = range(1, min(n_dims, len(icl_std) - 1) + 1)
        if len(list(k_range)) > 2:
            icl_std_slice = np.array([icl_std[k] for k in k_range])
            ridge_std_slice = np.array([ridge_std[k] for k in k_range])
            icl_adv_slice = np.array([icl_adv[k] for k in k_range])
            ridge_adv_slice = np.array([ridge_adv[k] for k in k_range])

            corr_std = float(np.corrcoef(icl_std_slice, ridge_std_slice)[0, 1])
            corr_adv = float(np.corrcoef(icl_adv_slice, ridge_adv_slice)[0, 1])

            # Ratio of ICL to averaging on standard data
            avg_std = std_result.baseline_curves.get("averaging", icl_std)
            avg_std_slice = np.array([avg_std[k] for k in k_range])
            icl_vs_avg_std = float(np.mean(icl_std_slice / (avg_std_slice + 1e-8)))
        else:
            corr_std = 0.0
            corr_adv = 0.0
            icl_vs_avg_std = 1.0

        # Classify
        if corr_std > 0.9 and corr_adv < 0.5:
            classification = "icl_tracks_then_diverges"
            summary = "ICL tracks ridge on standard data but diverges on adversarial — interesting failure"
        elif icl_vs_avg_std > 0.8:
            classification = "icl_matches_averaging"
            summary = "ICL approximately matches averaging baseline — model hasn't learned much"
        elif adv_result.fitness < 0.1:
            classification = "all_methods_fail"
            summary = "All methods perform similarly on adversarial data — not ICL-specific"
        else:
            classification = "icl_tracks_then_diverges"
            summary = f"ICL-ridge corr: std={corr_std:.2f}, adv={corr_adv:.2f}"

        return DiagnosticResult(
            name=self.name,
            summary=summary,
            data={
                "classification": classification,
                "corr_standard": corr_std,
                "corr_adversarial": corr_adv,
                "icl_vs_averaging_standard": icl_vs_avg_std,
                "icl_curve_adversarial": icl_adv.tolist(),
                "icl_curve_standard": icl_std.tolist(),
                "ridge_curve_adversarial": ridge_adv.tolist(),
                "ridge_curve_standard": ridge_std.tolist(),
                "fitness": adv_result.fitness,
            },
        )


# =============================================================================
# Diagnostic 2: Algorithm Identification
# =============================================================================


class AlgorithmIdentification:
    """Test which algorithm the ICL model has learned (ridge vs OLS vs GD).

    Three sub-tests:
    1. Ridge vs OLS: rank-deficient data, check null-space predictions
    2. Regularization adaptation: vary noise, check if model adapts shrinkage
    3. Preconditioned GD: isotropic vs anisotropic convergence
    """

    name = "algorithm_id"

    def run(self, model, xs, ys, genome, output_dir=None, **kwargs) -> DiagnosticResult:
        n_dims = genome.n_dims
        device = next(model.parameters()).device

        ridge_vs_ols = self._ridge_vs_ols_test(model, n_dims, device)
        reg_adapt = self._regularization_test(model, n_dims, device)
        precon_gd = self._preconditioned_gd_test(model, n_dims, device)

        # Synthesize
        if ridge_vs_ols["null_space_norm_ratio"] < 0.3:
            algo = "ridge-like"
        elif ridge_vs_ols["null_space_norm_ratio"] > 0.7:
            algo = "OLS-like"
        else:
            algo = "intermediate"

        adapts = reg_adapt["adapts_regularization"]
        summary = f"Model is {algo}, {'adapts' if adapts else 'does not adapt'} regularization to noise"

        return DiagnosticResult(
            name=self.name,
            summary=summary,
            data={
                "identified_algorithm": algo,
                "ridge_vs_ols": ridge_vs_ols,
                "regularization_adaptation": reg_adapt,
                "preconditioned_gd": precon_gd,
            },
        )

    def _ridge_vs_ols_test(self, model, n_dims, device, batch_size=32):
        """Feed rank-deficient data, check null-space predictions."""
        k = max(n_dims // 2, 2)  # k < d examples
        n_points = n_dims + 1  # need enough positions for the model

        w_true = torch.randn(n_dims)
        w_true = w_true / w_true.norm()

        # Create rank-deficient xs: only first k dimensions have signal
        xs = torch.zeros(batch_size, n_points, n_dims)
        xs[:, :k, :k] = torch.randn(batch_size, k, k)
        ys = (xs @ w_true).squeeze(-1) if xs.dim() == 3 else xs @ w_true
        if ys.dim() == 3:
            ys = ys.squeeze(-1)

        # Predict on null-space direction (a vector orthogonal to data subspace)
        null_vec = torch.zeros(n_dims)
        null_vec[k] = 1.0  # dimension k is in null space

        # Run model with k context points, predict on null_vec
        with torch.no_grad():
            # Build test sequence: k context points + null_vec as test
            xs_test = xs.clone()
            xs_test[:, k, :] = null_vec
            pred = model(xs_test.to(device), ys.to(device)).cpu()
            null_pred = pred[:, k]  # prediction at the null-space test point

        null_norm = float(null_pred.abs().mean())
        # Ridge predicts ~0 in null space, OLS predicts large values
        # Normalize by typical prediction magnitude
        pred_all = pred[:, :k]
        typical_norm = float(pred_all.abs().mean()) + 1e-8
        ratio = null_norm / typical_norm

        return {
            "null_space_pred_mean_abs": null_norm,
            "typical_pred_mean_abs": float(typical_norm),
            "null_space_norm_ratio": ratio,
            "interpretation": "ridge" if ratio < 0.3 else ("OLS" if ratio > 0.7 else "intermediate"),
        }

    def _regularization_test(self, model, n_dims, device, batch_size=32):
        """Vary noise_std, check if model adapts its shrinkage."""
        n_points = 2 * n_dims + 1
        noise_levels = [0.01, 0.1, 0.5, 1.0]
        shrinkage_estimates = []

        w_true = torch.randn(n_dims)
        w_true = w_true / w_true.norm()

        for noise_std in noise_levels:
            xs = torch.randn(batch_size, n_points, n_dims)
            ys = (xs @ w_true) + noise_std * torch.randn(batch_size, n_points)

            with torch.no_grad():
                pred = model(xs.to(device), ys.to(device)).cpu()

            # Measure shrinkage: compare model prediction norms to OLS prediction norms
            # at the last point (most context)
            pred_last = pred[:, -1]  # (batch,)
            # OLS prediction at last point
            ols_w = torch.linalg.lstsq(
                xs[:, :-1], ys[:, :-1].unsqueeze(-1)
            ).solution.squeeze(-1)
            ols_pred = (xs[:, -1] * ols_w).sum(dim=-1)

            shrinkage = float((pred_last.abs() / (ols_pred.abs() + 1e-8)).mean())
            shrinkage_estimates.append(shrinkage)

        # If model adapts: shrinkage should increase (less regularization) at low noise
        # and decrease (more regularization) at high noise
        adapts = shrinkage_estimates[0] > shrinkage_estimates[-1] * 1.1

        return {
            "noise_levels": noise_levels,
            "shrinkage_estimates": shrinkage_estimates,
            "adapts_regularization": adapts,
        }

    def _preconditioned_gd_test(self, model, n_dims, device, batch_size=32):
        """Compare convergence on isotropic vs anisotropic inputs."""
        n_points = 2 * n_dims + 1
        w_true = torch.randn(n_dims)
        w_true = w_true / w_true.norm()

        errors = {}
        for name, cov_diag in [("isotropic", None), ("anisotropic", None)]:
            xs = torch.randn(batch_size, n_points, n_dims)
            if name == "anisotropic":
                # Condition number ~100
                scales = torch.logspace(-1, 1, n_dims)
                xs = xs * scales.unsqueeze(0).unsqueeze(0)

            ys = (xs @ w_true) + 0.1 * torch.randn(batch_size, n_points)

            with torch.no_grad():
                pred = model(xs.to(device), ys.to(device)).cpu()

            # Per-point error
            err = ((pred - ys) ** 2).mean(dim=0).numpy()
            errors[name] = err.tolist()

        # Preconditioned GD adapts to covariance and has similar convergence
        # Vanilla GD is slower on anisotropic data
        iso_final = errors["isotropic"][-1]
        aniso_final = errors["anisotropic"][-1]
        ratio = aniso_final / (iso_final + 1e-8)

        return {
            "isotropic_errors": errors["isotropic"],
            "anisotropic_errors": errors["anisotropic"],
            "final_error_ratio": float(ratio),
            "interpretation": "preconditioned" if ratio < 2.0 else "vanilla_gd",
        }


# =============================================================================
# Diagnostic 3: Weight Recovery Probe
# =============================================================================


class WeightRecoveryProbe:
    """Recover the model's implicit weight estimate by predicting on basis vectors.

    For each basis vector e_i, predict y to get w_hat[i]. Compare w_hat to
    the true w and the ridge estimate w_ridge.
    """

    name = "weight_recovery"

    def run(self, model, xs, ys, genome, output_dir=None, **kwargs) -> DiagnosticResult:
        n_dims = genome.n_dims
        device = next(model.parameters()).device
        batch_size = min(xs.shape[0], 16)

        w_true = genome.decode_weights()

        # Test at different context lengths
        k_values = [1, 5, min(10, xs.shape[1] - 1), min(n_dims, xs.shape[1] - 1)]
        k_values = sorted(set(k for k in k_values if k > 0))

        results_per_k = {}

        for k in k_values:
            # Use first k context points from the provided data
            xs_ctx = xs[:batch_size, :k]
            ys_ctx = ys[:batch_size, :k]

            # Probe: predict y for each basis vector
            w_hat_list = []
            for i in range(n_dims):
                e_i = torch.zeros(batch_size, 1, n_dims)
                e_i[:, 0, i] = 1.0

                # Build full sequence: k context + 1 test
                xs_probe = torch.cat([xs_ctx, e_i], dim=1)
                ys_probe = torch.cat([ys_ctx, torch.zeros(batch_size, 1)], dim=1)

                with torch.no_grad():
                    pred = model(xs_probe.to(device), ys_probe.to(device)).cpu()
                y_hat = pred[:, k]  # prediction at the test point
                w_hat_list.append(y_hat.mean().item())

            w_hat = torch.tensor(w_hat_list)

            # Ridge estimate
            if k >= 2:
                w_ridge = _compute_ridge_estimate(xs_ctx, ys_ctx, alpha=1.0).mean(dim=0)
            else:
                w_ridge = torch.zeros(n_dims)

            # Metrics
            cos_sim_true = float(torch.nn.functional.cosine_similarity(
                w_hat.unsqueeze(0), w_true.unsqueeze(0)
            ))
            cos_sim_ridge = float(torch.nn.functional.cosine_similarity(
                w_hat.unsqueeze(0), w_ridge.unsqueeze(0)
            ))
            l2_to_ridge = float((w_hat - w_ridge).norm())

            results_per_k[k] = {
                "w_hat": w_hat.tolist(),
                "cos_sim_to_true_w": cos_sim_true,
                "cos_sim_to_ridge": cos_sim_ridge,
                "l2_distance_to_ridge": l2_to_ridge,
            }

        # Summary based on final k
        final_k = k_values[-1]
        final = results_per_k[final_k]
        summary = (
            f"At k={final_k}: cos_sim(w_hat, w_true)={final['cos_sim_to_true_w']:.3f}, "
            f"cos_sim(w_hat, w_ridge)={final['cos_sim_to_ridge']:.3f}"
        )

        return DiagnosticResult(
            name=self.name,
            summary=summary,
            data={"per_k": results_per_k},
        )


# =============================================================================
# Diagnostic 4: Attention Pattern Analysis
# =============================================================================


class AttentionPatternAnalysis:
    """Extract and compare attention patterns on standard vs adversarial data.

    Requires model to support forward_with_attention (added to TransformerModel).
    Falls back to hook-based extraction if not available.
    """

    name = "attention"

    def run(self, model, xs, ys, genome, output_dir=None, **kwargs) -> DiagnosticResult:
        device = next(model.parameters()).device
        batch_size = min(xs.shape[0], 8)

        # Check for forward_with_attention method
        if not hasattr(model, "forward_with_attention"):
            return DiagnosticResult(
                name=self.name,
                summary="Model does not support forward_with_attention. Add method to TransformerModel.",
            )

        # Adversarial data
        xs_adv = xs[:batch_size]
        ys_adv = ys[:batch_size]

        # Standard data
        xs_std, ys_std, _ = _generate_standard_data(genome.n_dims, xs.shape[1], batch_size)

        with torch.no_grad():
            _, attn_adv = model.forward_with_attention(xs_adv.to(device), ys_adv.to(device))
            _, attn_std = model.forward_with_attention(xs_std.to(device), ys_std.to(device))

        # Analyze per layer × head
        n_layers = len(attn_adv)
        n_heads = attn_adv[0].shape[1]
        analysis = {}

        for l in range(n_layers):
            for h in range(n_heads):
                key = f"layer{l}_head{h}"
                a_adv = attn_adv[l][:, h].cpu().float()  # (batch, seq, seq)
                a_std = attn_std[l][:, h].cpu().float()

                # Per-head entropy
                eps = 1e-10
                entropy_adv = float(-(a_adv * (a_adv + eps).log()).sum(-1).mean())
                entropy_std = float(-(a_std * (a_std + eps).log()).sum(-1).mean())

                # Max attention weight
                max_attn_adv = float(a_adv.max(dim=-1).values.mean())
                max_attn_std = float(a_std.max(dim=-1).values.mean())

                analysis[key] = {
                    "entropy_adversarial": entropy_adv,
                    "entropy_standard": entropy_std,
                    "entropy_change": entropy_adv - entropy_std,
                    "max_attn_adversarial": max_attn_adv,
                    "max_attn_standard": max_attn_std,
                }

        # Find heads with biggest entropy change
        entropy_changes = {
            k: abs(v["entropy_change"]) for k, v in analysis.items()
        }
        most_changed = sorted(entropy_changes, key=entropy_changes.get, reverse=True)[:3]

        summary = f"Most affected heads: {', '.join(most_changed)}"

        # Save attention maps if output_dir provided
        plots = []
        if output_dir:
            try:
                import matplotlib.pyplot as plt
                os.makedirs(output_dir, exist_ok=True)
                for key in most_changed:
                    l, h = int(key.split("_")[0][5:]), int(key.split("_")[1][4:])
                    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                    axes[0].imshow(attn_std[l][0, h].cpu().numpy(), aspect="auto")
                    axes[0].set_title(f"Standard - {key}")
                    axes[1].imshow(attn_adv[l][0, h].cpu().numpy(), aspect="auto")
                    axes[1].set_title(f"Adversarial - {key}")
                    path = os.path.join(output_dir, f"attention_{key}.png")
                    fig.savefig(path, dpi=100, bbox_inches="tight")
                    plt.close(fig)
                    plots.append(path)
            except ImportError:
                pass

        return DiagnosticResult(
            name=self.name,
            summary=summary,
            data={"per_head": analysis, "most_changed_heads": most_changed},
            plots=plots,
        )


# =============================================================================
# Diagnostic 5: Linear Probing of Internal Representations
# =============================================================================


class LinearProbing:
    """Train linear probes at each layer to recover w from hidden states.

    Reports R^2 per layer for standard vs adversarial data.
    """

    name = "linear_probing"

    def run(self, model, xs, ys, genome, output_dir=None, **kwargs) -> DiagnosticResult:
        device = next(model.parameters()).device
        n_dims = genome.n_dims
        batch_size = min(xs.shape[0], 32)

        w_true = genome.decode_weights().numpy()

        # Get hidden states for adversarial data
        r2_adv = self._probe_all_layers(model, xs[:batch_size], ys[:batch_size], w_true, device)

        # Get hidden states for standard data
        xs_std, ys_std, w_std = _generate_standard_data(n_dims, xs.shape[1], batch_size)
        r2_std = self._probe_all_layers(model, xs_std, ys_std, w_std.numpy(), device)

        # Find layer where adversarial R^2 drops
        if r2_adv and r2_std:
            drop_layer = None
            for l in range(len(r2_adv)):
                if r2_std[l] > 0.5 and r2_adv[l] < r2_std[l] * 0.5:
                    drop_layer = l
                    break

            if drop_layer is not None:
                summary = f"Adversarial R^2 drops at layer {drop_layer} (std={r2_std[drop_layer]:.3f}, adv={r2_adv[drop_layer]:.3f})"
            else:
                summary = f"No significant R^2 drop. Final layer: std={r2_std[-1]:.3f}, adv={r2_adv[-1]:.3f}"
        else:
            summary = "Could not extract hidden states"

        # Plot
        plots = []
        if output_dir and r2_adv:
            try:
                import matplotlib.pyplot as plt
                os.makedirs(output_dir, exist_ok=True)
                fig, ax = plt.subplots(figsize=(8, 5))
                layers = list(range(len(r2_std)))
                ax.plot(layers, r2_std, "b-o", label="Standard")
                ax.plot(layers, r2_adv, "r-o", label="Adversarial")
                ax.set_xlabel("Layer")
                ax.set_ylabel("R^2 (linear probe)")
                ax.set_title("Weight Recovery by Layer")
                ax.legend()
                ax.grid(True, alpha=0.3)
                path = os.path.join(output_dir, "linear_probing.png")
                fig.savefig(path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                plots.append(path)
            except ImportError:
                pass

        return DiagnosticResult(
            name=self.name,
            summary=summary,
            data={
                "r2_per_layer_standard": r2_std,
                "r2_per_layer_adversarial": r2_adv,
            },
            plots=plots,
        )

    def _probe_all_layers(self, model, xs, ys, w_true, device):
        """Extract hidden states at each layer and probe for w."""
        # Run forward with hidden states
        zs = model._combine(xs.to(device), ys.to(device))
        embeds = model._read_in(zs)

        with torch.no_grad():
            output = model._backbone(inputs_embeds=embeds, output_hidden_states=True)

        hidden_states = output.hidden_states  # tuple of (batch, seq_len, embd)
        batch_size = xs.shape[0]

        r2_scores = []
        for hidden in hidden_states:
            # Take hidden state at last token position (most context seen)
            h = hidden[:, -1, :].cpu().numpy()  # (batch, embd)

            # Train linear probe to predict w from h
            # We need multiple samples, so we treat each batch element as a sample
            # with the same true w
            if batch_size < 4:
                r2_scores.append(0.0)
                continue

            # Expand w_true to match batch
            W_true = np.tile(w_true, (batch_size, 1))

            try:
                probe = SklearnRidge(alpha=1.0)
                probe.fit(h, W_true)
                r2 = float(probe.score(h, W_true))
                r2_scores.append(max(r2, 0.0))
            except Exception:
                r2_scores.append(0.0)

        return r2_scores


# =============================================================================
# Diagnostic 6: Gradient Attribution (Meta-Gradients)
# =============================================================================


class GradientAttribution:
    """Compute per-example sensitivity: which in-context examples matter most.

    On standard data, sensitivity should be roughly uniform. On adversarial
    data, concentration on few examples indicates outlier domination or
    attention collapse.
    """

    name = "gradient_attribution"

    def run(self, model, xs, ys, genome, output_dir=None, **kwargs) -> DiagnosticResult:
        device = next(model.parameters()).device
        batch_size = min(xs.shape[0], 16)

        # Adversarial sensitivity
        sens_adv = self._compute_sensitivity(model, xs[:batch_size], ys[:batch_size], device)

        # Standard sensitivity
        xs_std, ys_std, _ = _generate_standard_data(genome.n_dims, xs.shape[1], batch_size)
        sens_std = self._compute_sensitivity(model, xs_std, ys_std, device)

        # Metrics
        metrics = {}
        for name, sens in [("adversarial", sens_adv), ("standard", sens_std)]:
            if sens is not None:
                # Normalize to distribution
                sens_norm = sens / (sens.sum(dim=-1, keepdim=True) + 1e-8)
                # Gini coefficient
                sorted_s, _ = sens_norm.sort(dim=-1)
                n = sorted_s.shape[-1]
                idx = torch.arange(1, n + 1, device=sorted_s.device, dtype=torch.float32)
                gini = float((2 * (idx * sorted_s).sum(-1) / (n * sorted_s.sum(-1) + 1e-8) - (n + 1) / n).mean())
                # Max/mean ratio
                max_mean = float((sens.max(dim=-1).values / (sens.mean(dim=-1) + 1e-8)).mean())
                # Top-3 mass
                top3, _ = sens.topk(min(3, sens.shape[-1]), dim=-1)
                top3_frac = float((top3.sum(-1) / (sens.sum(-1) + 1e-8)).mean())

                metrics[name] = {
                    "gini_coefficient": gini,
                    "max_mean_ratio": max_mean,
                    "top3_mass_fraction": top3_frac,
                    "sensitivity_per_point": sens.mean(dim=0).tolist(),
                }
            else:
                metrics[name] = {"error": "Could not compute gradients"}

        # Compare
        if "gini_coefficient" in metrics.get("adversarial", {}) and "gini_coefficient" in metrics.get("standard", {}):
            gini_diff = metrics["adversarial"]["gini_coefficient"] - metrics["standard"]["gini_coefficient"]
            summary = (
                f"Sensitivity Gini: std={metrics['standard']['gini_coefficient']:.3f}, "
                f"adv={metrics['adversarial']['gini_coefficient']:.3f} (diff={gini_diff:+.3f})"
            )
        else:
            summary = "Could not compare sensitivities"

        return DiagnosticResult(
            name=self.name,
            summary=summary,
            data=metrics,
        )

    def _compute_sensitivity(self, model, xs, ys, device):
        """Compute d(prediction)/d(xs) norm per example."""
        xs_var = xs.clone().requires_grad_(True).to(device)
        ys_dev = ys.to(device)

        try:
            pred = model(xs_var, ys_dev)[:, -1]  # prediction at last point
            pred.sum().backward()
            sensitivity = xs_var.grad.norm(dim=-1).cpu()  # (batch, n_points)
            return sensitivity
        except Exception:
            return None


# =============================================================================
# Diagnostic 7: Pipeline Ablation
# =============================================================================


class PipelineAblation:
    """Test each pipeline component in isolation to find the load-bearing one.

    Given a pipeline [base + stage_i + stage_j + ...], creates variants:
    - base only (all stages identity)
    - stage_i only (Gaussian base, other stages off)
    - etc.
    """

    name = "pipeline_ablation"

    def run(self, model, xs, ys, genome, output_dir=None, **kwargs) -> DiagnosticResult:
        evaluator = kwargs.get("evaluator")
        if evaluator is None:
            return DiagnosticResult(
                name=self.name,
                summary="Evaluator required but not provided",
            )

        from ..adversary.pipeline_genome import PipelineGenome
        if not isinstance(genome, PipelineGenome):
            return DiagnosticResult(
                name=self.name,
                summary="Ablation only applicable to PipelineGenome",
            )

        n_dims = genome.n_dims
        full_result = evaluator.evaluate(genome)
        full_fitness = full_result.raw_fitness

        ablation_results = {}

        # Test 1: base only (all stages identity)
        base_only = self._make_base_only(genome)
        base_result = evaluator.evaluate(base_only)
        ablation_results["base_only"] = {
            "base": genome.base_distribution_name(),
            "fitness": base_result.raw_fitness,
            "fraction_of_full": base_result.raw_fitness / (full_fitness + 1e-8),
        }

        # Test 2: each active stage in isolation
        for stage_idx, transform_name in genome.active_stages():
            variant = self._make_single_stage(genome, stage_idx)
            variant_result = evaluator.evaluate(variant)
            ablation_results[f"stage_{stage_idx}_{transform_name}"] = {
                "transform": transform_name,
                "fitness": variant_result.raw_fitness,
                "fraction_of_full": variant_result.raw_fitness / (full_fitness + 1e-8),
            }

        # Find load-bearing component
        if ablation_results:
            best_component = max(ablation_results, key=lambda k: ablation_results[k]["fitness"])
            best_fitness = ablation_results[best_component]["fitness"]
            summary = (
                f"Load-bearing component: {best_component} "
                f"(fitness={best_fitness:.4f}, {ablation_results[best_component]['fraction_of_full']:.0%} of full)"
            )
        else:
            summary = "No active components to ablate"

        return DiagnosticResult(
            name=self.name,
            summary=summary,
            data={
                "full_fitness": full_fitness,
                "ablations": ablation_results,
            },
        )

    def _make_base_only(self, genome):
        """Copy genome but set all stages to identity."""
        from ..adversary.pipeline_genome import PipelineGenome
        g = genome.copy()
        for s in range(PipelineGenome.N_STAGES):
            logits = np.zeros(PipelineGenome.N_TRANSFORMS)
            logits[0] = 10.0  # identity
            g._set_block(f"stage_{s}_logits", logits)
        return g

    def _make_single_stage(self, genome, keep_stage_idx):
        """Copy genome with Gaussian base and only one active stage."""
        from ..adversary.pipeline_genome import PipelineGenome
        g = genome.copy()

        # Set base to Gaussian
        base_logits = np.zeros(PipelineGenome.N_BASE_DISTS)
        base_logits[0] = 10.0
        g._set_block("base_logits", base_logits)

        # Set all stages to identity except the kept one
        for s in range(PipelineGenome.N_STAGES):
            if s != keep_stage_idx:
                logits = np.zeros(PipelineGenome.N_TRANSFORMS)
                logits[0] = 10.0
                g._set_block(f"stage_{s}_logits", logits)

        return g


# =============================================================================
# Diagnostic 8: Retrained Model Test
# =============================================================================


class RetrainedModelTest:
    """Test model retrained on adversarial distributions.

    Checks:
    1. Does retrained model still work on standard Gaussian?
    2. Does it resist the original adversarial distribution?
    3. Does the adversary find new breaks?
    """

    name = "retrained_model"

    def run(self, model, xs, ys, genome, output_dir=None, **kwargs) -> DiagnosticResult:
        evaluator = kwargs.get("evaluator")
        train_fn = kwargs.get("train_fn")  # optional: function to retrain model

        if train_fn is None:
            return DiagnosticResult(
                name=self.name,
                summary="Retraining requires train_fn kwarg (expensive diagnostic, skipped by default)",
                data={"skipped": True, "reason": "no train_fn provided"},
            )

        from ..adversary.pipeline_genome import PipelineGenome
        from ..adversary.genome import Genome

        n_dims = genome.n_dims

        # 1. Evaluate original model
        pre_adv = evaluator.evaluate(genome)
        std_genome = Genome.identity(n_dims)
        pre_std = evaluator.evaluate(std_genome)

        # 2. Retrain (caller provides this)
        retrained_model = train_fn(model, genome)

        # 3. Evaluate retrained model
        post_evaluator = GenomeEvaluator(
            icl_model=retrained_model,
            task_name=evaluator.task_name,
            n_dims=n_dims,
            n_points=evaluator.n_points,
            batch_size=evaluator.batch_size,
            num_batches=evaluator.num_batches,
            baseline_names=[name for name, _ in evaluator.baselines],
        )

        post_adv = post_evaluator.evaluate(genome)
        post_std = post_evaluator.evaluate(std_genome)

        summary = (
            f"Pre-retrain: std={pre_std.fitness:.4f}, adv={pre_adv.fitness:.4f}. "
            f"Post-retrain: std={post_std.fitness:.4f}, adv={post_adv.fitness:.4f}"
        )

        return DiagnosticResult(
            name=self.name,
            summary=summary,
            data={
                "pre_standard_fitness": pre_std.fitness,
                "pre_adversarial_fitness": pre_adv.fitness,
                "post_standard_fitness": post_std.fitness,
                "post_adversarial_fitness": post_adv.fitness,
                "resistance_gained": pre_adv.fitness - post_adv.fitness,
                "standard_performance_change": post_std.fitness - pre_std.fitness,
            },
        )


# =============================================================================
# Registry
# =============================================================================


DIAGNOSTICS = {
    "learning_curve": LearningCurveComparison,
    "algorithm_id": AlgorithmIdentification,
    "weight_recovery": WeightRecoveryProbe,
    "attention": AttentionPatternAnalysis,
    "linear_probing": LinearProbing,
    "gradient_attribution": GradientAttribution,
    "pipeline_ablation": PipelineAblation,
    "retrained_model": RetrainedModelTest,
}

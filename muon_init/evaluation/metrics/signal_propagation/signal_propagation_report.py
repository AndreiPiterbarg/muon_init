"""Combined signal propagation diagnostic report.

Convenience wrapper that runs all signal-propagation metrics and produces a
summary dict plus optional matplotlib visualisation.
"""

from __future__ import annotations

from typing import Any, Callable, Iterator

import torch
from torch import Tensor
import torch.nn as nn

from .activation_statistics import ActivationStats, compute_activation_stats
from .angular_gradient_snr import AngularGSNRResult, compute_angular_gsnr
from .gradient_flow import GradientFlowStats, compute_gradient_flow
from .jacobian_spectrum import (
    JacobianStats,
    compute_jacobian_singular_values,
    dynamical_isometry_score,
    estimate_jacobian_singular_values,
    jacobian_stats,
)
from .layer_jacobians import LayerJacobianInfo, compute_layer_jacobians


def run_signal_propagation_diagnostics(
    model: nn.Module,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    data_loader: Iterator[tuple[Tensor, Tensor]],
    *,
    exact_jacobian: bool = True,
    num_jacobian_projections: int = 64,
    num_angular_microbatches: int = 8,
) -> dict[str, Any]:
    """Run all signal propagation diagnostics and return a summary dict.

    Args:
        model: Network to analyse.
        loss_fn: Loss function ``(preds, targets) -> scalar``.
        data_loader: Iterable yielding ``(inputs, targets)`` tuples.
            Must yield at least ``num_angular_microbatches + 1`` batches.
        exact_jacobian: If True, compute the full Jacobian (small models).
            Otherwise use stochastic estimation.
        num_jacobian_projections: Number of random projections for stochastic
            Jacobian estimation.
        num_angular_microbatches: Number of micro-batches for angular GSNR.

    Returns:
        Dict with keys:
            ``"activation_stats"``, ``"gradient_flow"``, ``"jacobian_stats"``,
            ``"dynamical_isometry_score"``, ``"layer_jacobians"``,
            ``"angular_gsnr"``.
    """
    # Draw one batch for static diagnostics
    batches: list[tuple[Tensor, Tensor]] = []
    loader_iter = iter(data_loader)
    for _ in range(num_angular_microbatches + 1):
        try:
            batches.append(next(loader_iter))
        except StopIteration:
            break

    if not batches:
        raise ValueError("data_loader yielded no batches")

    input_batch, target_batch = batches[0]

    # 1. Activation statistics
    act_stats = compute_activation_stats(model, input_batch)

    # 2. Gradient flow
    grad_flow = compute_gradient_flow(model, loss_fn, input_batch, target_batch)

    # 3. Jacobian singular values
    if exact_jacobian:
        jac_svs = compute_jacobian_singular_values(model, input_batch)
    else:
        jac_svs = estimate_jacobian_singular_values(
            model, input_batch, num_projections=num_jacobian_projections
        )
    jac_summary = jacobian_stats(jac_svs)
    di_score = dynamical_isometry_score(jac_svs)

    # 4. Layer Jacobians
    layer_jacs = compute_layer_jacobians(model, input_batch)

    # 5. Angular GSNR (needs multiple micro-batches)
    angular_result: AngularGSNRResult | None = None
    if len(batches) >= 2:

        def _micro_loader():
            for b in batches[1:]:
                yield b

        angular_result = compute_angular_gsnr(
            model, loss_fn, _micro_loader(), num_microbatches=len(batches) - 1
        )

    return {
        "activation_stats": act_stats,
        "gradient_flow": grad_flow,
        "jacobian_stats": jac_summary,
        "dynamical_isometry_score": di_score,
        "layer_jacobians": layer_jacs,
        "angular_gsnr": angular_result,
    }


def plot_signal_propagation(
    diagnostics: dict[str, Any],
    save_path: str | None = None,
) -> Any:
    """Visualise signal propagation diagnostics as a multi-panel figure.

    Args:
        diagnostics: Output of :func:`run_signal_propagation_diagnostics`.
        save_path: If provided, save the figure to this path.

    Returns:
        The matplotlib Figure object.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Signal Propagation Diagnostics", fontsize=14)

    # Panel 1: Activation variance across layers
    ax = axes[0, 0]
    act_stats: dict[str, ActivationStats] = diagnostics["activation_stats"]
    names = list(act_stats.keys())
    variances = [act_stats[n].variance for n in names]
    ax.bar(range(len(names)), variances, color="steelblue", alpha=0.8)
    ax.set_ylabel("Activation Variance")
    ax.set_title("Per-Layer Activation Variance")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(range(len(names)))
    ax.set_xlabel("Layer index")
    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="target=1")
    ax.legend()

    # Panel 2: Gradient norms across layers
    ax = axes[0, 1]
    gf: GradientFlowStats = diagnostics["gradient_flow"]
    ax.bar(range(len(gf.gradient_norms)), gf.gradient_norms, color="coral", alpha=0.8)
    ax.set_ylabel("Gradient Norm")
    ax.set_title(f"Per-Layer Gradient Norms (CV={gf.coefficient_of_variation:.3f})")
    ax.set_xlabel("Parameter index")

    # Panel 3: Jacobian singular values (histogram)
    ax = axes[1, 0]
    jac: JacobianStats = diagnostics["jacobian_stats"]
    svs = jac.singular_values.flatten().cpu().numpy()
    ax.hist(svs, bins=50, color="seagreen", alpha=0.8, edgecolor="black")
    ax.axvline(x=1.0, color="red", linestyle="--", label="isometry target")
    ax.set_xlabel("Singular Value")
    ax.set_ylabel("Count")
    di = diagnostics["dynamical_isometry_score"]
    ax.set_title(f"Jacobian SV Distribution (DI score={di:.3f})")
    ax.legend()

    # Panel 4: Layer Jacobian spectral norms
    ax = axes[1, 1]
    lj: list[LayerJacobianInfo] = diagnostics["layer_jacobians"]
    if lj:
        lj_names = [info.layer_name for info in lj]
        lj_norms = [info.spectral_norm for info in lj]
        ax.bar(range(len(lj_norms)), lj_norms, color="mediumpurple", alpha=0.8)
        ax.set_ylabel("Spectral Norm")
        ax.set_title("Per-Layer Jacobian Spectral Norms")
        ax.set_xticks(range(len(lj_names)))
        ax.set_xticklabels(range(len(lj_names)))
        ax.set_xlabel("Layer index")
        ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5)
    else:
        ax.text(0.5, 0.5, "No layer Jacobians", ha="center", va="center")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig

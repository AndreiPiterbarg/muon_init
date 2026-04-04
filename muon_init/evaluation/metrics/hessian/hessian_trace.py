"""Hessian trace via Hutchinson's stochastic estimator.

Computes Tr(H) = E[v^T H v] where v is drawn from the Rademacher
distribution ({-1, +1} with equal probability).  Also provides the
"spikiness" ratio lambda_max / Tr(H).

Reference:
    Yao, Z., Gholami, A., Keutzer, K., & Mahoney, M. (NeurIPS 2020).
    "PyHessian: Neural Networks Through the Lens of the Hessian."

    Hutchinson, M. F. (1990). "A Stochastic Estimator of the Trace of the
    Influence Matrix for Laplacian Smoothing Splines."
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ._hessian_vector_product import hessian_vector_product, gather_params


def compute_hessian_trace(
    model: nn.Module,
    loss_fn: nn.Module,
    data_loader: DataLoader,
    num_samples: int = 100,
) -> float:
    """Estimate the Hessian trace via Hutchinson's estimator.

    Parameters
    ----------
    model : nn.Module
        The model whose Hessian trace we estimate.
    loss_fn : nn.Module
        Loss function.
    data_loader : DataLoader
        Data batches used to compute the loss.
    num_samples : int
        Number of random Rademacher vectors to draw.

    Returns
    -------
    float
        Estimated Hessian trace (sum of eigenvalues).
    """
    device = next(model.parameters()).device
    params = gather_params(model)

    trace_sum = 0.0
    for _ in range(num_samples):
        # Rademacher random vector: each element is +1 or -1 uniformly
        v = [torch.randint_like(p, high=2) * 2.0 - 1.0 for p in params]

        # Average Hv over data
        hv = _hessian_vector_product_over_data(model, loss_fn, data_loader, params, v)

        # v^T H v  (unbiased estimate of Tr(H) because E[v^T H v] = Tr(H))
        trace_sum += sum(float(torch.sum(vi * hvi)) for vi, hvi in zip(v, hv))

    return trace_sum / num_samples


def compute_spikiness(lambda_max: float, trace: float) -> float:
    """Compute the curvature spikiness ratio lambda_max / Tr(H).

    A high ratio means curvature is concentrated in a few directions
    (ill-conditioned); a ratio near 1/d (d = number of parameters) means
    the curvature is isotropic.

    Parameters
    ----------
    lambda_max : float
        Largest Hessian eigenvalue.
    trace : float
        Hessian trace.

    Returns
    -------
    float
        Spikiness ratio.
    """
    if abs(trace) < 1e-12:
        return float("inf")
    return lambda_max / trace


# ------------------------------------------------------------------
# Internal helper
# ------------------------------------------------------------------

def _hessian_vector_product_over_data(
    model: nn.Module,
    loss_fn: nn.Module,
    data_loader: DataLoader,
    params: list[torch.Tensor],
    v: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Average Hv over the full data loader."""
    device = params[0].device
    num_batches = 0
    hv_acc: list[torch.Tensor] | None = None

    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        hv = hessian_vector_product(model, loss_fn, inputs, targets, params, v)
        if hv_acc is None:
            hv_acc = hv
        else:
            hv_acc = [a + b for a, b in zip(hv_acc, hv)]
        num_batches += 1

    assert hv_acc is not None, "data_loader must yield at least one batch"
    return [h / num_batches for h in hv_acc]

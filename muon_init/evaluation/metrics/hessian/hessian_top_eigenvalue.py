"""Hessian top eigenvalue (lambda_max) via power iteration on Hessian-vector products.

Uses the standard double-backward approach: compute the gradient g = dL/dw, then
compute d(g^T v)/dw to get Hv without ever forming the full Hessian.

Reference:
    Yao, Z., Gholami, A., Keutzer, K., & Mahoney, M. (NeurIPS 2020).
    "PyHessian: Neural Networks Through the Lens of the Hessian."
    https://github.com/amirgholami/PyHessian
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ._hessian_vector_product import hessian_vector_product, gather_params


def compute_lambda_max(
    model: nn.Module,
    loss_fn: nn.Module,
    data_loader: DataLoader,
    num_iterations: int = 100,
    tol: float = 1e-6,
) -> float:
    """Estimate the largest eigenvalue of the Hessian via power iteration.

    Parameters
    ----------
    model : nn.Module
        The model whose Hessian we analyze.
    loss_fn : nn.Module
        Loss function (e.g. ``nn.CrossEntropyLoss()``).
    data_loader : DataLoader
        Data used to compute the loss.  The full Hessian is approximated
        over one pass of the data loader (or a single batch for speed).
    num_iterations : int
        Maximum number of power-iteration steps.
    tol : float
        Convergence tolerance on the relative change of the eigenvalue.

    Returns
    -------
    float
        Estimated largest Hessian eigenvalue (lambda_max).

    Notes
    -----
    Following PyHessian, the Hessian-vector product is computed via the
    double-backward trick (``torch.autograd.grad`` twice) and the loss is
    averaged over all batches in ``data_loader``.
    """
    device = next(model.parameters()).device
    params = gather_params(model)

    # Random unit vector in parameter space
    v = [torch.randn_like(p) for p in params]
    v = _normalize(v)

    eigenvalue = 0.0
    for _ in range(num_iterations):
        hv = _hessian_vector_product_over_data(model, loss_fn, data_loader, params, v)

        eigenvalue_new = _inner_product(hv, v)

        v = _normalize(hv)

        if abs(eigenvalue_new - eigenvalue) / (abs(eigenvalue) + 1e-10) < tol:
            eigenvalue = eigenvalue_new
            break
        eigenvalue = eigenvalue_new

    return float(eigenvalue)


# ------------------------------------------------------------------
# Internal helpers
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


def _inner_product(a: list[torch.Tensor], b: list[torch.Tensor]) -> float:
    return float(sum(torch.sum(x * y) for x, y in zip(a, b)))


def _normalize(v: list[torch.Tensor]) -> list[torch.Tensor]:
    norm = max(_inner_product(v, v) ** 0.5, 1e-12)
    return [x / norm for x in v]

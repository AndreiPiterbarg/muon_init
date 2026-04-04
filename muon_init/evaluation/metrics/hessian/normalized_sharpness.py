"""Scale-invariant (normalized) sharpness metrics.

Raw sharpness metrics like lambda_max and Tr(H) are *not* reparameterization-
invariant: Dinh et al. (ICML 2017) showed that for ReLU networks, any flat
minimum can be reparameterized to appear arbitrarily sharp without changing
the function.  Normalizing by the squared Frobenius norm of the weights
resolves this.

Metrics implemented:
    - lambda_max / ||w||_F^2   (Neyshabur et al., NeurIPS 2017)
    - Tr(H)     / ||w||_F^2   (Tsuzuku et al., 2020)

References:
    Neyshabur, B., Bhojanapalli, S., McAllester, D., & Srebro, N. (2017).
    "Exploring Generalization in Deep Networks." NeurIPS.

    Tsuzuku, Y., Sato, I., & Sugiyama, M. (2020).
    "Normalized Flat Minima: Exploring Scale Invariant Definition of Flat Minima."

    Dinh, L., Pascanu, R., Bengio, S., & Bengio, Y. (ICML 2017).
    "Sharp Minima Can Generalize for Deep Nets."
"""

from __future__ import annotations

import torch
import torch.nn as nn


def compute_weight_norm_squared(model: nn.Module) -> float:
    """Compute the squared Frobenius norm of all trainable parameters.

    Parameters
    ----------
    model : nn.Module
        The model.

    Returns
    -------
    float
        ``sum_i ||w_i||_F^2`` over all parameters with ``requires_grad``.
    """
    return float(sum(
        torch.sum(p ** 2) for p in model.parameters() if p.requires_grad
    ))


def compute_normalized_lambda_max(lambda_max: float, model: nn.Module) -> float:
    """Normalized top Hessian eigenvalue: ``lambda_max / ||w||_F^2``.

    Parameters
    ----------
    lambda_max : float
        Precomputed largest Hessian eigenvalue.
    model : nn.Module
        The model (used to compute the weight norm).

    Returns
    -------
    float
        Normalized sharpness value.
    """
    w_norm_sq = compute_weight_norm_squared(model)
    if w_norm_sq < 1e-12:
        return float("inf")
    return lambda_max / w_norm_sq


def compute_normalized_trace(trace: float, model: nn.Module) -> float:
    """Normalized Hessian trace: ``Tr(H) / ||w||_F^2``.

    Parameters
    ----------
    trace : float
        Precomputed Hessian trace.
    model : nn.Module
        The model.

    Returns
    -------
    float
        Normalized flatness value.
    """
    w_norm_sq = compute_weight_norm_squared(model)
    if w_norm_sq < 1e-12:
        return float("inf")
    return trace / w_norm_sq

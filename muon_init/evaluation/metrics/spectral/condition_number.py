"""Condition Number of weight matrices.

The condition number measures the ratio of the largest to smallest singular value:

    kappa(W) = sigma_max / sigma_min

A condition number of 1 indicates an isometry (all singular values equal).
Large condition numbers indicate ill-conditioning, where the matrix amplifies
some directions far more than others.

For Muon, tracking condition numbers reveals how the optimizer reshapes spectral
structure — Muon's updates have condition number exactly 1 (orthogonalized).

References:
    Jordan et al. (2024). "Muon: An Optimizer for Hidden Layers."
        Notes that SGD/Adam updates have very high condition number.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor

from ._utils import apply_metric_all_layers, singular_values

# Threshold below which a singular value is considered effectively zero
_SIGMA_MIN_FLOOR = 1e-7


def condition_number(W: Tensor, floor: float = _SIGMA_MIN_FLOOR) -> float:
    """Compute the condition number of a weight matrix.

    For rank-deficient matrices where sigma_min < floor, sigma_min is clamped
    to `floor` to avoid division by zero. The returned value is then
    sigma_max / floor, which signals extreme ill-conditioning.

    Args:
        W: Weight tensor (2D or higher; reshaped to 2D if needed).
        floor: Minimum value for sigma_min to prevent inf. Default 1e-7.

    Returns:
        Condition number as a float (>= 1.0 for non-degenerate matrices).
    """
    sv = singular_values(W)
    if len(sv) == 0:
        return float("inf")
    sigma_max = sv[0].item()
    sigma_min = sv[-1].item()
    if sigma_max < 1e-12:
        return float("inf")
    sigma_min = max(sigma_min, floor)
    return sigma_max / sigma_min


def condition_number_all_layers(model: nn.Module) -> Dict[str, float]:
    """Compute condition number for all 2D+ weight matrices in a model.

    Args:
        model: PyTorch model.

    Returns:
        Dictionary mapping parameter name to condition number.
    """
    return apply_metric_all_layers(model, condition_number)

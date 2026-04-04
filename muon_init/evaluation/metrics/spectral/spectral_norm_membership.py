"""Spectral Norm Ball Membership check.

Muon with decoupled weight decay implicitly solves a constrained optimization
where ||W||_op <= 1/lambda_wd. Convergence has two phases:
  1. Constraint satisfaction: parameters rapidly enter the spectral norm ball.
  2. Optimization within the constrained region.

An initialization that already satisfies ||W||_op <= 1/lambda_wd skips phase 1,
potentially eliminating the need for warmup.

This module checks whether each layer's spectral norm (operator norm = largest
singular value) is within the ball defined by the weight decay coefficient.

References:
    "Muon Optimizes Under Spectral Norm Constraints." arXiv:2506.15054.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from ._utils import iter_weight_matrices, singular_values


def spectral_norm_ratio(W: Tensor, weight_decay: float) -> float:
    """Compute ||W||_op * lambda_wd.

    Values < 1 mean W is inside the spectral norm ball.
    Values >= 1 mean W violates the constraint.

    Args:
        W: Weight tensor (2D or higher; reshaped to 2D if needed).
        weight_decay: Weight decay coefficient (lambda_wd).

    Returns:
        The ratio ||W||_op * lambda_wd.
    """
    sv = singular_values(W)
    if len(sv) == 0:
        return 0.0
    return (sv[0].item() * weight_decay)


def check_spectral_norm_ball(
    model: nn.Module, weight_decay: float
) -> Dict[str, Tuple[bool, float]]:
    """Check spectral norm ball membership for all weight matrices.

    For each 2D+ parameter, checks whether ||W||_op <= 1/lambda_wd,
    i.e., whether ||W||_op * lambda_wd <= 1.

    Args:
        model: PyTorch model.
        weight_decay: Weight decay coefficient (lambda_wd). Must be > 0.

    Returns:
        Dictionary mapping parameter name to (inside_ball, ratio) where
        inside_ball is True if ||W||_op * lambda_wd <= 1.0, and ratio
        is the value ||W||_op * lambda_wd.
    """
    if weight_decay <= 0:
        raise ValueError(f"weight_decay must be > 0, got {weight_decay}")
    results = {}
    for name, W in iter_weight_matrices(model):
        ratio = spectral_norm_ratio(W, weight_decay)
        results[name] = (ratio <= 1.0, ratio)
    return results

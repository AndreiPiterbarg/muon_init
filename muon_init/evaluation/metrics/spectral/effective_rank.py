"""Effective Rank (eRank) of weight matrices.

The effective rank is a continuous measure of the "dimensionality" of a matrix's
singular value spectrum, defined as the exponential of the Shannon entropy of the
normalized singular values:

    eRank(W) = exp(H(p))
    where p_i = sigma_i / sum(sigma_j),  H(p) = -sum(p_i * log(p_i))

Ranges from 1 (one dominant singular value) to min(m, n) (uniform spectrum).

References:
    Roy & Bhattacharya (2007). "Effective Rank: A Measure of Effective
        Dimensionality." European Signal Processing Conference (EUSIPCO).
    Huh et al. "The Low-Rank Simplicity Bias in Deep Networks."
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor

from ._utils import apply_metric_all_layers, singular_values


def effective_rank(W: Tensor) -> float:
    """Compute the effective rank of a weight matrix.

    Args:
        W: Weight tensor (2D or higher; reshaped to 2D if needed).

    Returns:
        Effective rank as a float. Minimum 1.0, maximum min(m, n).
    """
    sv = singular_values(W)
    # Filter out near-zero singular values for numerical stability
    sv = sv[sv > 1e-12]
    if len(sv) == 0:
        return 0.0
    # Normalize to form a probability distribution
    p = sv / sv.sum()
    # Shannon entropy: H(p) = -sum(p_i * log(p_i))
    entropy = -(p * torch.log(p)).sum()
    return torch.exp(entropy).item()


def effective_rank_all_layers(model: nn.Module) -> Dict[str, float]:
    """Compute effective rank for all 2D+ weight matrices in a model.

    Args:
        model: PyTorch model.

    Returns:
        Dictionary mapping parameter name to effective rank.
    """
    return apply_metric_all_layers(model, effective_rank)

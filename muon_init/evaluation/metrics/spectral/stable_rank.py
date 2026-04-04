"""Stable Rank of weight matrices.

The stable rank is a robust, continuous relaxation of algebraic rank:

    srank(W) = ||W||_F^2 / ||W||_2^2 = sum(sigma_i^2) / sigma_max^2

Always <= algebraic rank. Appears directly in generalization bounds and is
cheaper/more stable than effective rank since it doesn't require full SVD
(only the top singular value and the Frobenius norm).

References:
    Sanyal, Torr & Dokania (ICLR 2020). "Stable Rank Normalization for
        Improved Generalization in Neural Networks and GANs." arXiv:1906.04659.
    Neyshabur et al. (NeurIPS 2017). "Exploring Generalization in Deep Learning."
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor

from ._utils import apply_metric_all_layers, singular_values


def stable_rank(W: Tensor) -> float:
    """Compute the stable rank of a weight matrix.

    Args:
        W: Weight tensor (2D or higher; reshaped to 2D if needed).

    Returns:
        Stable rank as a float. Minimum 1.0 (rank-1 matrix), maximum min(m, n).
    """
    if W.ndim > 2:
        W = W.reshape(W.shape[0], -1)
    W = W.float()
    fro_sq = torch.linalg.norm(W, ord="fro") ** 2
    spectral_sq = torch.linalg.svdvals(W)[0] ** 2
    if spectral_sq < 1e-12:
        return 0.0
    return (fro_sq / spectral_sq).item()


def stable_rank_all_layers(model: nn.Module) -> Dict[str, float]:
    """Compute stable rank for all 2D+ weight matrices in a model.

    Args:
        model: PyTorch model.

    Returns:
        Dictionary mapping parameter name to stable rank.
    """
    return apply_metric_all_layers(model, stable_rank)

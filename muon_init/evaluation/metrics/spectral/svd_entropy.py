"""SVD Entropy of weight matrices.

SVD entropy measures the diversity of a weight matrix's spectral energy
distribution. Following Moonlight's convention, it uses squared singular
values to form the probability distribution:

    H(W) = -sum(p_i * log(p_i))
    where p_i = sigma_i^2 / sum(sigma_j^2)

Higher entropy = more uniform spectral energy = more diverse optimization
directions. Moonlight found that "Muon achieves higher SVD entropy than
AdamW, verifying more diverse optimization directions."

Note: This differs from effective rank which uses sigma_i (not sigma_i^2)
for the probability distribution.

References:
    Liu, Su et al. (2025). "Muon is Scalable for LLM Training."
        arXiv:2502.16982. [Moonlight]
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor

from ._utils import apply_metric_all_layers, singular_values


def svd_entropy(W: Tensor) -> float:
    """Compute the SVD entropy of a weight matrix.

    Uses squared singular values following the Moonlight convention.

    Args:
        W: Weight tensor (2D or higher; reshaped to 2D if needed).

    Returns:
        SVD entropy as a float (>= 0). Maximum is log(min(m, n)) for a
        matrix with uniform singular values.
    """
    sv = singular_values(W)
    sv_sq = sv ** 2
    # Filter out near-zero values
    sv_sq = sv_sq[sv_sq > 1e-24]
    if len(sv_sq) == 0:
        return 0.0
    # Normalize to probability distribution
    p = sv_sq / sv_sq.sum()
    entropy = -(p * torch.log(p)).sum()
    return entropy.item()


def svd_entropy_all_layers(model: nn.Module) -> Dict[str, float]:
    """Compute SVD entropy for all 2D+ weight matrices in a model.

    Args:
        model: PyTorch model.

    Returns:
        Dictionary mapping parameter name to SVD entropy.
    """
    return apply_metric_all_layers(model, svd_entropy)

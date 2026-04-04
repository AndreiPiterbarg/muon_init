"""Shared utilities for spectral metrics."""

from __future__ import annotations

from typing import Dict, Iterator, Tuple

import torch
import torch.nn as nn
from torch import Tensor


def iter_weight_matrices(model: nn.Module) -> Iterator[Tuple[str, Tensor]]:
    """Iterate over all 2D+ parameter tensors in a model, yielding (name, weight_2d).

    For tensors with more than 2 dimensions (e.g., conv filters), reshapes to
    (fan_out, fan_in) where fan_out = shape[0] and fan_in = product of remaining dims.
    Skips 1D parameters (biases, LayerNorm scales, etc.).
    """
    for name, param in model.named_parameters():
        if param.ndim < 2:
            continue
        W = param.data
        if W.ndim > 2:
            W = W.reshape(W.shape[0], -1)
        yield name, W


def singular_values(W: Tensor) -> Tensor:
    """Compute singular values of a 2D tensor, reshaping if needed.

    Uses torch.linalg.svdvals which is faster than full SVD when only
    singular values are needed.
    """
    if W.ndim > 2:
        W = W.reshape(W.shape[0], -1)
    return torch.linalg.svdvals(W.float())


def apply_metric_all_layers(
    model: nn.Module, metric_fn: callable
) -> Dict[str, float]:
    """Apply a single-matrix metric function to all weight matrices in a model."""
    results = {}
    for name, W in iter_weight_matrices(model):
        results[name] = metric_fn(W)
    return results

"""Gradient flow health metrics.

Runs a single forward-backward pass and records per-layer gradient norms
to diagnose vanishing / exploding gradient pathologies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import Tensor
import torch.nn as nn


@dataclass
class GradientFlowStats:
    """Per-layer gradient flow diagnostics."""

    layer_names: list[str]
    gradient_norms: list[float]
    gradient_ratios: list[float]  # norm[l] / norm[l-1] for adjacent layers
    coefficient_of_variation: float  # std / mean of gradient norms
    dead_neuron_pct: dict[str, float]  # per-layer dead neuron %


def compute_gradient_flow(
    model: nn.Module,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    input_batch: Tensor,
    target_batch: Tensor,
) -> GradientFlowStats:
    """Compute gradient flow statistics from one forward-backward pass.

    Args:
        model: Network to analyse.  Gradients are zeroed before and after.
        loss_fn: Loss function ``(predictions, targets) -> scalar loss``.
        input_batch: Input tensor, shape ``(B, *input_dims)``.
        target_batch: Target tensor matching the loss function's expectation.

    Returns:
        :class:`GradientFlowStats` with per-layer norms, ratios, CV, and
        dead-neuron percentages.
    """
    model.train()
    model.zero_grad()

    output = model(input_batch)
    loss = loss_fn(output, target_batch)
    loss.backward()

    layer_names: list[str] = []
    gradient_norms: list[float] = []
    dead_neuron_pct: dict[str, float] = {}

    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        norm = grad.norm().item()
        layer_names.append(name)
        gradient_norms.append(norm)

        # Dead neuron detection: for weight matrices (2-D), a "dead" output
        # neuron has an all-zero gradient row.
        if grad.ndim >= 2:
            row_norms = grad.reshape(grad.shape[0], -1).norm(dim=1)
            dead_frac = (row_norms == 0).float().mean().item()
            dead_neuron_pct[name] = dead_frac

    # Ratios between adjacent layers
    gradient_ratios: list[float] = []
    for i in range(1, len(gradient_norms)):
        prev = gradient_norms[i - 1]
        ratio = gradient_norms[i] / max(prev, 1e-12)
        gradient_ratios.append(ratio)

    # Coefficient of variation
    norms_t = torch.tensor(gradient_norms)
    mean_norm = norms_t.mean().item()
    std_norm = norms_t.std().item()
    cv = std_norm / max(mean_norm, 1e-12)

    model.zero_grad()

    return GradientFlowStats(
        layer_names=layer_names,
        gradient_norms=gradient_norms,
        gradient_ratios=gradient_ratios,
        coefficient_of_variation=cv,
        dead_neuron_pct=dead_neuron_pct,
    )

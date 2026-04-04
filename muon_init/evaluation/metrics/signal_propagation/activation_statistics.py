"""Per-layer activation statistics for signal propagation diagnostics.

Uses forward hooks to non-invasively record intermediate activations and
compute summary statistics that characterise signal propagation health.

References:
    - Glorot & Bengio (2010), "Understanding the Difficulty of Training
      Deep Feedforward Neural Networks"
    - He et al. (2015), "Delving Deep into Rectifiers" (Kaiming init)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn as nn


@dataclass
class ActivationStats:
    """Summary statistics for a single layer's activations."""

    layer_name: str
    mean: float
    variance: float
    kurtosis: float
    nonzero_fraction: float  # fraction of activations > 0 (meaningful for ReLU)


def _kurtosis(x: Tensor) -> float:
    """Compute excess kurtosis of a flattened tensor."""
    x_flat = x.flatten().float()
    if x_flat.numel() < 4:
        return 0.0
    mean = x_flat.mean()
    var = x_flat.var(correction=1)
    if var < 1e-12:
        return 0.0
    m4 = ((x_flat - mean) ** 4).mean()
    return (m4 / var**2 - 3.0).item()


def compute_activation_stats(
    model: nn.Module,
    input_batch: Tensor,
) -> dict[str, ActivationStats]:
    """Record and summarise per-layer activation statistics.

    Attaches temporary forward hooks to every non-container module that
    produces a tensor output (Linear, Conv2d, activations, norms, etc.),
    runs one forward pass, then removes the hooks.

    Args:
        model: Network to analyse (not modified).
        input_batch: Calibration batch, shape ``(B, *input_dims)``.

    Returns:
        Ordered dict mapping ``layer_name -> ActivationStats``.
    """
    model.eval()

    activations: dict[str, Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHook] = []

    def _make_hook(name: str):
        def hook(_module: nn.Module, _inp: tuple, out: Tensor | tuple) -> None:
            tensor = out if isinstance(out, Tensor) else out[0]
            activations[name] = tensor.detach()

        return hook

    # Attach hooks to leaf modules only (avoids double-counting containers)
    for name, mod in model.named_modules():
        if len(list(mod.children())) == 0:
            handles.append(mod.register_forward_hook(_make_hook(name)))

    with torch.no_grad():
        model(input_batch)

    for h in handles:
        h.remove()

    results: dict[str, ActivationStats] = {}
    for name, act in activations.items():
        act_flat = act.flatten().float()
        results[name] = ActivationStats(
            layer_name=name,
            mean=act_flat.mean().item(),
            variance=act_flat.var().item(),
            kurtosis=_kurtosis(act),
            nonzero_fraction=(act_flat != 0).float().mean().item(),
        )

    return results

"""Angular Gradient Signal-to-Noise Ratio (Muon-specific).

Muon discards gradient magnitudes during orthogonalisation (it keeps only
the polar factor UV^T).  This means gradient *direction* is the true signal,
and directional noise across micro-batches directly degrades optimiser quality.

This module defines the Angular GSNR: the consistency of gradient direction
across micro-batches, measured via cosine similarity in Frobenius inner-product
space for matrix-valued gradients.

This is a NOVEL metric — see Section 6 of the survey:
    "Since Muon discards gradient magnitudes (only keeps singular vectors),
     the angular signal-to-noise ratio matters more than magnitude-based GSNR."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import torch
from torch import Tensor
import torch.nn as nn


@dataclass
class AngularGSNRResult:
    """Angular gradient SNR per layer."""

    per_layer: dict[str, float]  # layer_name -> angular SNR
    per_layer_mean_cosine: dict[str, float]  # layer_name -> mean cosine sim


def _cosine_similarity_flat(a: Tensor, b: Tensor) -> float:
    """Cosine similarity between two tensors treated as flat vectors."""
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    denom = a_flat.norm() * b_flat.norm()
    if denom < 1e-12:
        return 0.0
    return (a_flat @ b_flat / denom).item()


def _collect_gradients(
    model: nn.Module,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    inputs: Tensor,
    targets: Tensor,
) -> dict[str, Tensor]:
    """Run forward-backward on a single micro-batch, return per-param grads."""
    model.zero_grad()
    output = model(inputs)
    loss = loss_fn(output, targets)
    loss.backward()

    grads: dict[str, Tensor] = {}
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.ndim >= 2:
            grads[name] = param.grad.detach().clone()
    return grads


def compute_angular_gsnr(
    model: nn.Module,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    data_loader: Iterator[tuple[Tensor, Tensor]],
    num_microbatches: int = 8,
) -> AngularGSNRResult:
    """Compute per-layer angular gradient signal-to-noise ratio.

    Splits data into *K* micro-batches, computes per-micro-batch gradients,
    then measures pairwise cosine similarity of gradient matrices (Frobenius
    inner product).

    **Angular SNR definition**::

        mean_cos = mean of pairwise cosine similarities
        angular_snr = mean_cos / (1 - mean_cos + eps)

    High angular SNR means gradient direction is consistent across
    micro-batches (good signal); low means directions are noisy.

    Args:
        model: Network to analyse.
        loss_fn: Loss callable ``(predictions, targets) -> scalar``.
        data_loader: Iterable yielding ``(input_batch, target_batch)`` tuples.
            At least ``num_microbatches`` batches must be available.
        num_microbatches: Number of micro-batches to draw.

    Returns:
        :class:`AngularGSNRResult` with per-layer angular SNR values.
    """
    model.train()

    # Collect per-micro-batch gradients
    all_grads: list[dict[str, Tensor]] = []
    loader_iter = iter(data_loader)
    for _ in range(num_microbatches):
        inputs, targets = next(loader_iter)
        grads = _collect_gradients(model, loss_fn, inputs, targets)
        all_grads.append(grads)

    if not all_grads or not all_grads[0]:
        return AngularGSNRResult(per_layer={}, per_layer_mean_cosine={})

    # Compute pairwise cosine similarities per layer
    layer_names = list(all_grads[0].keys())
    per_layer_snr: dict[str, float] = {}
    per_layer_cos: dict[str, float] = {}

    for name in layer_names:
        cosines: list[float] = []
        for i in range(num_microbatches):
            for j in range(i + 1, num_microbatches):
                if name in all_grads[i] and name in all_grads[j]:
                    cos = _cosine_similarity_flat(
                        all_grads[i][name], all_grads[j][name]
                    )
                    cosines.append(cos)

        if cosines:
            mean_cos = sum(cosines) / len(cosines)
            snr = mean_cos / max(1.0 - mean_cos, 1e-8)
        else:
            mean_cos = 0.0
            snr = 0.0

        per_layer_cos[name] = mean_cos
        per_layer_snr[name] = snr

    model.zero_grad()

    return AngularGSNRResult(
        per_layer=per_layer_snr,
        per_layer_mean_cosine=per_layer_cos,
    )

"""Per-layer (partial) Jacobian analysis.

Computes the Jacobian of each layer's output with respect to the previous
layer's output, enabling identification of which layers break dynamical
isometry.

Reference:
    "Critical Initialization of Wide and Deep Neural Networks through
    Partial Jacobians" (ICLR).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn as nn


@dataclass
class LayerJacobianInfo:
    """Singular values and metadata for a single layer's local Jacobian."""

    layer_name: str
    singular_values: Tensor  # (min(out_dim, in_dim),) averaged over batch
    condition_number: float
    spectral_norm: float  # largest singular value


def _get_target_layers(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Return layers whose local Jacobians are meaningful (Linear, Conv2d)."""
    targets: list[tuple[str, nn.Module]] = []
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Linear, nn.Conv2d)):
            targets.append((name, mod))
    return targets


def compute_layer_jacobians(
    model: nn.Module,
    input_batch: Tensor,
) -> list[LayerJacobianInfo]:
    """Compute per-layer local Jacobian singular values.

    For each target layer *l* (Linear or Conv2d), this records the layer's
    input activation *a_{l-1}* and output activation *a_l* via forward hooks,
    then computes ``d a_l / d a_{l-1}`` and its singular values.

    Args:
        model: Network to analyse.
        input_batch: Calibration batch, shape ``(B, *input_dims)``.

    Returns:
        List of :class:`LayerJacobianInfo`, one per target layer, in forward
        order.
    """
    model.eval()
    target_layers = _get_target_layers(model)
    if not target_layers:
        return []

    # Storage for activations captured by hooks
    layer_inputs: dict[str, Tensor] = {}
    layer_outputs: dict[str, Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHook] = []

    def _make_hook(name: str):
        def hook(module: nn.Module, inp: tuple, out: Tensor) -> None:
            # inp is a tuple; take the first element
            layer_inputs[name] = inp[0].detach()
            layer_outputs[name] = out.detach()

        return hook

    for name, mod in target_layers:
        handles.append(mod.register_forward_hook(_make_hook(name)))

    # Forward pass to capture activations
    with torch.no_grad():
        model(input_batch)

    for h in handles:
        h.remove()

    # Now compute local Jacobians for each layer
    results: list[LayerJacobianInfo] = []

    for name, mod in target_layers:
        inp_act = layer_inputs[name].requires_grad_(True)
        out_act = mod(inp_act)
        out_flat = out_act.reshape(out_act.shape[0], -1)
        inp_flat_dim = inp_act.reshape(inp_act.shape[0], -1).shape[1]
        out_flat_dim = out_flat.shape[1]

        # Accumulate per-sample Jacobian SVDs (average over batch for summary)
        batch_svs: list[Tensor] = []
        for b in range(min(inp_act.shape[0], 8)):  # cap at 8 samples
            jac_rows: list[Tensor] = []
            for j in range(out_flat_dim):
                grad = torch.autograd.grad(
                    out_flat[b, j],
                    inp_act,
                    retain_graph=True,
                )[0]
                jac_rows.append(grad[b].reshape(-1))
            jac = torch.stack(jac_rows)  # (out_dim, in_dim)
            svs = torch.linalg.svdvals(jac)
            batch_svs.append(svs)

        mean_svs = torch.stack(batch_svs).mean(dim=0)
        cond = (mean_svs[0] / mean_svs[-1].clamp(min=1e-12)).item()

        results.append(
            LayerJacobianInfo(
                layer_name=name,
                singular_values=mean_svs,
                condition_number=cond,
                spectral_norm=mean_svs[0].item(),
            )
        )

    return results

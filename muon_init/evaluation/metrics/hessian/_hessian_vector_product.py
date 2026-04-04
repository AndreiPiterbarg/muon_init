"""Shared Hessian-vector product utility.

Implements the standard double-backward approach following PyHessian
(Yao et al., NeurIPS 2020).  The key idea: given loss L(w) with gradient
g = dL/dw, the Hessian-vector product Hv = d(g^T v)/dw, which can be
computed with a single backward pass through the computation graph of g^T v.

Reference:
    https://github.com/amirgholami/PyHessian/blob/master/pyhessian/hessian.py
"""

from __future__ import annotations

import torch
import torch.nn as nn


def gather_params(model: nn.Module) -> list[torch.Tensor]:
    """Return a list of all parameters that require gradients."""
    return [p for p in model.parameters() if p.requires_grad]


def hessian_vector_product(
    model: nn.Module,
    loss_fn: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    params: list[torch.Tensor],
    v: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Compute the Hessian-vector product Hv via the double-backward trick.

    Parameters
    ----------
    model : nn.Module
        Network.
    loss_fn : nn.Module
        Loss function mapping (logits, targets) -> scalar loss.
    inputs, targets : torch.Tensor
        A single batch of data.
    params : list[torch.Tensor]
        Parameters with respect to which we differentiate (from ``gather_params``).
    v : list[torch.Tensor]
        Direction vector (same shapes as *params*).

    Returns
    -------
    list[torch.Tensor]
        Hessian-vector product, one tensor per parameter.
    """
    model.zero_grad()
    logits = model(inputs)
    loss = loss_fn(logits, targets)

    # First backward: compute gradients g = dL/dw
    grads = torch.autograd.grad(loss, params, create_graph=True)

    # Dot product g^T v (scalar)
    dot = sum(torch.sum(g * vi) for g, vi in zip(grads, v))

    # Second backward: d(g^T v)/dw = Hv
    hv = torch.autograd.grad(dot, params)

    return [h.detach() for h in hv]

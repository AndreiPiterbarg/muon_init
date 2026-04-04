"""Analysis of Muon's First Optimization Step from a Given Initialization.

Mathematical Background
-----------------------
Muon's update rule for weight matrices is fundamentally different from
standard optimizers.  Given gradient G for a weight matrix W:

  1. Accumulate momentum: M = beta * M_prev + G  (at step 0, M = G)
  2. Orthogonalize via polar decomposition: U = polar(M) = U_svd @ V_svd^T
     (from SVD of M = U_svd @ S @ V_svd^T, discard singular values)
  3. Update: W_new = W - lr * U

The key insight is that the update direction U is always a partial isometry
(all singular values = 1), regardless of the gradient's magnitude spectrum.
This means:

  - Muon treats all singular-value directions of the gradient equally.
  - The "shape" of the gradient (its singular vectors) matters, not its
    "magnitude" (its singular values).
  - Jeremy Bernstein's interpretation: Muon = spectral steepest descent,
    replacing G = U S V^T with U V^T.

Why First-Step Analysis Matters
-------------------------------
The first Muon step reveals how the optimizer will initially reshape the
weight matrices.  Analyzing it shows:

  1. **Update-weight alignment**: Is the first update reinforcing the
     initial structure or moving away from it?
  2. **Effective rank of the update**: Should be full-rank for Muon (since
     all singular values = 1), but the gradient at init may be degenerate.
  3. **Spectral change**: How singular values of W change after one step —
     does the init lead to a balanced or unbalanced spectral shift?

A well-designed initialization should produce a first Muon step that is
well-conditioned, full-rank, and moves the weights toward a productive
region of parameter space.

References
----------
- Muon optimizer: https://github.com/KellerJordan/Muon
- Jeremy Bernstein's derivation: Muon = spectral descent + momentum
- "Muon Optimizes Under Spectral Norm Constraints", arXiv 2506.15054
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from evaluation.metrics.spectral._utils import iter_weight_matrices
from evaluation.metrics.muon_specific.polar_error import polar_factor


# ---------------------------------------------------------------------------
# Dataclasses for results
# ---------------------------------------------------------------------------

@dataclass
class LayerFirstStepDiagnostics:
    """Diagnostics for a single layer's first Muon update."""
    layer_name: str

    # Gradient properties
    gradient_frobenius_norm: float
    gradient_spectral_norm: float
    gradient_effective_rank: float

    # Polar factor of gradient (= Muon's update direction)
    update_effective_rank: float       # should be full rank for Muon

    # Alignment between update and current weights
    update_weight_cosine: float        # cosine similarity (Frobenius inner product)
    update_weight_angle_deg: float     # angle in degrees

    # Spectral change after one Muon step
    sv_before: List[float]             # singular values of W (top-k)
    sv_after: List[float]              # singular values of W - lr * polar(G) (top-k)
    sv_mean_change: float              # mean absolute change in singular values
    sv_max_change: float               # max absolute change
    condition_number_before: float
    condition_number_after: float


@dataclass
class FirstStepAnalysis:
    """Full first-step analysis across all layers."""
    layers: Dict[str, LayerFirstStepDiagnostics]
    mean_update_weight_cosine: float
    mean_gradient_effective_rank: float
    mean_sv_change: float


# ---------------------------------------------------------------------------
# Helper: effective rank
# ---------------------------------------------------------------------------

def _effective_rank(sv: Tensor) -> float:
    """Compute effective rank from singular values.

    eRank(M) = exp(H(p))  where  p_i = sigma_i / sum(sigma_j)
    and H(p) = -sum(p_i * log(p_i)) is the Shannon entropy.

    Parameters
    ----------
    sv : Tensor
        1-D tensor of singular values (non-negative).

    Returns
    -------
    float
        Effective rank (continuous, >= 1 if any sv > 0).
    """
    sv = sv[sv > 0].float()
    if len(sv) == 0:
        return 0.0
    p = sv / sv.sum()
    entropy = -(p * torch.log(p)).sum()
    return math.exp(entropy.item())


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def _analyze_layer(
    name: str,
    W: Tensor,
    G: Tensor,
    lr: float,
    num_sv: int = 20,
) -> LayerFirstStepDiagnostics:
    """Analyze the first Muon step for a single layer.

    Parameters
    ----------
    name : str
        Layer name.
    W : Tensor
        Weight matrix, shape (m, n).
    G : Tensor
        Gradient matrix (same shape as W, already reshaped to 2D).
    lr : float
        Learning rate.
    num_sv : int
        Number of top singular values to record.
    """
    W_f = W.float()
    G_f = G.float()

    # --- Gradient properties ---
    g_fro = torch.linalg.norm(G_f, ord="fro").item()
    g_spec = torch.linalg.norm(G_f, ord=2).item()
    g_sv = torch.linalg.svdvals(G_f)
    g_erank = _effective_rank(g_sv)

    # --- Polar factor of gradient = Muon's update direction ---
    U_update = polar_factor(G_f)
    u_sv = torch.linalg.svdvals(U_update)
    u_erank = _effective_rank(u_sv)

    # --- Alignment: cosine similarity between update and weight ---
    # cos(W, U) = <W, U>_F / (||W||_F * ||U||_F)
    w_fro = torch.linalg.norm(W_f, ord="fro")
    u_fro = torch.linalg.norm(U_update, ord="fro")
    if w_fro > 0 and u_fro > 0:
        cos_sim = (torch.sum(W_f * U_update) / (w_fro * u_fro)).item()
    else:
        cos_sim = 0.0
    cos_sim = max(-1.0, min(1.0, cos_sim))  # clamp for numerical safety
    angle_deg = math.degrees(math.acos(cos_sim))

    # --- Spectral change after one step ---
    W_new = W_f - lr * U_update
    sv_before = torch.linalg.svdvals(W_f)
    sv_after = torch.linalg.svdvals(W_new)

    k = min(num_sv, len(sv_before))
    sv_before_top = sv_before[:k]
    sv_after_top = sv_after[:k]

    sv_diff = torch.abs(sv_before_top - sv_after_top)
    sv_mean_change = sv_diff.mean().item()
    sv_max_change = sv_diff.max().item()

    eps = 1e-10
    cond_before = (sv_before[0] / max(sv_before[-1].item(), eps)).item()
    cond_after = (sv_after[0] / max(sv_after[-1].item(), eps)).item()

    return LayerFirstStepDiagnostics(
        layer_name=name,
        gradient_frobenius_norm=g_fro,
        gradient_spectral_norm=g_spec,
        gradient_effective_rank=g_erank,
        update_effective_rank=u_erank,
        update_weight_cosine=cos_sim,
        update_weight_angle_deg=angle_deg,
        sv_before=sv_before_top.tolist(),
        sv_after=sv_after_top.tolist(),
        sv_mean_change=sv_mean_change,
        sv_max_change=sv_max_change,
        condition_number_before=cond_before,
        condition_number_after=cond_after,
    )


def analyze_first_muon_step(
    model: nn.Module,
    loss_fn: Callable,
    data_loader: DataLoader,
    lr: float = 0.02,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None,
    num_sv: int = 20,
) -> FirstStepAnalysis:
    """Analyze what Muon's first optimization step looks like from init.

    Performs one forward-backward pass to compute gradients, then simulates
    Muon's first update (momentum = gradient at step 0) for each weight
    matrix.  Computes per-layer diagnostics including update-weight alignment,
    effective rank, and spectral changes.

    Parameters
    ----------
    model : nn.Module
        The model at initialization.
    loss_fn : callable
        Loss function: ``loss_fn(model_output, targets) -> scalar``.
    data_loader : DataLoader
        Provides (input, target) batches.  Only the first batch is used.
    lr : float
        Muon learning rate for the simulated step.
    weight_decay : float
        If > 0, applies decoupled weight decay: G_eff = G + wd * W.
    device : torch.device, optional
        Device.  Inferred from model if not given.
    num_sv : int
        Number of top singular values to record per layer.

    Returns
    -------
    FirstStepAnalysis
        Per-layer and aggregate diagnostics of the first Muon step.
    """
    if device is None:
        device = next(model.parameters()).device

    model.train()
    model.zero_grad()

    # Forward-backward pass
    inputs, targets = next(iter(data_loader))
    inputs, targets = inputs.to(device), targets.to(device)
    output = model(inputs)
    loss = loss_fn(output, targets)
    loss.backward()

    # Analyze each weight matrix
    layer_diagnostics: Dict[str, LayerFirstStepDiagnostics] = {}

    for name, param in model.named_parameters():
        if param.ndim < 2 or param.grad is None:
            continue

        W = param.data
        G = param.grad.data

        # Reshape to 2D if needed
        if W.ndim > 2:
            orig_shape = W.shape
            W = W.reshape(W.shape[0], -1)
            G = G.reshape(G.shape[0], -1)

        # Apply decoupled weight decay to gradient
        if weight_decay > 0:
            G = G + weight_decay * W

        diag = _analyze_layer(name, W, G, lr, num_sv=num_sv)
        layer_diagnostics[name] = diag

    # Aggregate
    if layer_diagnostics:
        cosines = [d.update_weight_cosine for d in layer_diagnostics.values()]
        eranks = [d.gradient_effective_rank for d in layer_diagnostics.values()]
        sv_changes = [d.sv_mean_change for d in layer_diagnostics.values()]
        mean_cos = sum(cosines) / len(cosines)
        mean_erank = sum(eranks) / len(eranks)
        mean_sv = sum(sv_changes) / len(sv_changes)
    else:
        mean_cos = mean_erank = mean_sv = 0.0

    model.zero_grad()

    return FirstStepAnalysis(
        layers=layer_diagnostics,
        mean_update_weight_cosine=mean_cos,
        mean_gradient_effective_rank=mean_erank,
        mean_sv_change=mean_sv,
    )

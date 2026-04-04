"""Input-output Jacobian singular value analysis for dynamical isometry evaluation.

Computes the full end-to-end Jacobian of a network and extracts its singular
value spectrum. Provides both exact computation (small models) and stochastic
estimation via random projections (large models).

References:
    - Pennington, Schoenholz & Ganguli (NeurIPS 2017), "Resurrecting the Sigmoid"
    - Xiao et al. (ICML 2018), "Dynamical Isometry and Mean Field Theory of CNNs"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor
import torch.nn as nn


@dataclass
class JacobianStats:
    """Summary statistics of the Jacobian singular value spectrum."""

    singular_values: Tensor  # (num_samples, min(out_dim, in_dim))
    mean: float
    std: float
    min: float
    max: float
    condition_number: float  # sigma_max / sigma_min


def compute_jacobian_singular_values(
    model: nn.Module,
    input_batch: Tensor,
    output_indices: Sequence[int] | None = None,
) -> Tensor:
    """Compute singular values of the input-output Jacobian (exact).

    Uses ``torch.autograd.functional.jacobian`` — suitable for small models
    where the full Jacobian matrix fits in memory.

    Args:
        model: Network to analyse (set to eval mode externally if desired).
        input_batch: Input tensor of shape ``(B, *input_dims)``.
        output_indices: If provided, only compute the Jacobian for these
            output dimensions (useful when the output is high-dimensional).

    Returns:
        Tensor of shape ``(B, min(out_dim, in_dim))`` containing the singular
        values for each sample in the batch.
    """
    model.eval()
    device = input_batch.device
    all_svs: list[Tensor] = []

    for i in range(input_batch.shape[0]):
        x = input_batch[i : i + 1].detach().requires_grad_(True)

        def func(inp: Tensor) -> Tensor:
            out = model(inp)
            out_flat = out.reshape(inp.shape[0], -1)
            if output_indices is not None:
                out_flat = out_flat[:, output_indices]
            return out_flat.squeeze(0)

        jac = torch.autograd.functional.jacobian(func, x, vectorize=True)
        # jac shape: (out_dim, 1, *input_dims) — squeeze the batch dim
        jac_2d = jac.reshape(jac.shape[0], -1)
        svs = torch.linalg.svdvals(jac_2d)
        all_svs.append(svs)

    return torch.stack(all_svs).to(device)


def estimate_jacobian_singular_values(
    model: nn.Module,
    input_batch: Tensor,
    num_projections: int = 64,
    output_indices: Sequence[int] | None = None,
) -> Tensor:
    """Stochastic estimation of Jacobian singular values via randomized SVD.

    Uses the two-pass randomized SVD algorithm (Halko, Martinsson & Tropp 2011):

    1. **Range finder (VJPs):** draw random Gaussian probes in output space,
       compute ``Omega^T J`` via vector-Jacobian products to get a sketch
       ``Y = J^T Omega`` whose columns approximate the row space of *J*.
    2. **Orthogonalise:** ``Q, _ = QR(Y)`` to get an orthonormal basis for
       the row space of *J*.
    3. **Project (VJPs):** compute ``B = J Q`` by solving from the sketch:
       ``B = Omega_pinv @ (Omega^T J) @ Q`` — or equivalently, re-probe with
       a fresh set of output-space vectors and project.  Here we use the
       simpler approach of computing ``B_i = q_i^T J^T`` via VJPs applied to
       basis vectors in the *input* projected space, then transposing.

    In practice we use a streamlined variant: compute the sketch ``S = V J``
    where ``V`` is ``(k, m)`` standard Gaussian, form ``Q, R = QR(S^T)``,
    then use VJPs with the columns of ``Q`` (which live in input space) to
    build ``C = J Q`` (shape ``(m, r)``), and return ``svdvals(C)``.

    Suitable for models where the full Jacobian is too large to store.

    Args:
        model: Network to analyse.
        input_batch: Input tensor of shape ``(B, *input_dims)``.
        num_projections: Number of random probe vectors.  More probes give
            more accurate estimates of the top singular values.
        output_indices: Optional subset of output indices.

    Returns:
        Tensor of shape ``(B, K)`` with estimated singular values, where
        ``K = min(num_projections, out_dim, in_dim)``.
    """
    model.eval()
    device = input_batch.device
    all_svs: list[Tensor] = []

    for i in range(input_batch.shape[0]):
        x = input_batch[i : i + 1].detach().requires_grad_(True)
        out = model(x)
        out_flat = out.reshape(1, -1)
        if output_indices is not None:
            out_flat = out_flat[:, output_indices]
        out_flat = out_flat.squeeze(0)
        out_dim = out_flat.shape[0]
        in_dim = x.reshape(-1).shape[0]

        # --- Pass 1: sketch the row space of J via VJPs ---
        # Omega is (num_projections, out_dim) standard Gaussian
        omega = torch.randn(num_projections, out_dim, device=device)

        # Compute S = Omega @ J, shape (num_projections, in_dim)
        sketch_rows: list[Tensor] = []
        for v in omega:
            (vjp,) = torch.autograd.grad(
                out_flat, x, grad_outputs=v, retain_graph=True
            )
            sketch_rows.append(vjp.reshape(-1))
        sketch = torch.stack(sketch_rows)  # (k, in_dim)

        # QR on sketch^T to get orthonormal basis for row space of J
        # sketch^T is (in_dim, k); Q is (in_dim, r), R is (r, k)
        r = min(num_projections, in_dim, out_dim)
        Q, _R = torch.linalg.qr(sketch.T)
        Q = Q[:, :r]  # (in_dim, r)

        # --- Pass 2: compute B = J @ Q via JVPs ---
        # Use torch.autograd.functional.jvp for forward-mode differentiation.
        # For each column q_j of Q, compute J @ q_j.
        def func(inp: Tensor) -> Tensor:
            o = model(inp)
            o_flat = o.reshape(inp.shape[0], -1)
            if output_indices is not None:
                o_flat = o_flat[:, output_indices]
            return o_flat.squeeze(0)

        x_base = input_batch[i : i + 1].detach()
        B_cols: list[Tensor] = []
        for j in range(r):
            tangent = Q[:, j].reshape(x_base.shape)
            _, jvp_out = torch.autograd.functional.jvp(
                func, (x_base,), (tangent,)
            )
            B_cols.append(jvp_out.reshape(-1))

        B = torch.stack(B_cols, dim=1)  # (out_dim, r)
        svs = torch.linalg.svdvals(B)
        all_svs.append(svs)

    return torch.stack(all_svs).to(device)


def jacobian_stats(singular_values: Tensor) -> JacobianStats:
    """Compute summary statistics from a singular value tensor.

    Args:
        singular_values: Tensor of shape ``(B, K)`` (output of
            ``compute_jacobian_singular_values`` or the stochastic variant).

    Returns:
        :class:`JacobianStats` with aggregated statistics across the batch.
    """
    flat = singular_values.flatten().float()
    return JacobianStats(
        singular_values=singular_values,
        mean=flat.mean().item(),
        std=flat.std().item(),
        min=flat.min().item(),
        max=flat.max().item(),
        condition_number=flat.max().item() / max(flat.min().item(), 1e-12),
    )


def dynamical_isometry_score(singular_values: Tensor) -> float:
    """Quantify how close the Jacobian spectrum is to perfect dynamical isometry.

    Perfect isometry means all singular values equal 1.  Two complementary
    sub-scores are combined:

    1. **Log-spread**: ``std(log(sigma_i))`` — should be 0 for perfect isometry.
       Mapped to [0, 1] via ``exp(-log_spread)``.
    2. **Fraction in band**: fraction of singular values in ``[0.5, 2.0]``.

    The final score is the geometric mean of both sub-scores, in [0, 1]
    (1 = perfect isometry).

    Args:
        singular_values: Tensor of any shape containing positive singular values.
    """
    flat = singular_values.flatten().float()
    flat = flat.clamp(min=1e-12)

    # Sub-score 1: concentration of log-spectrum around 0
    log_svs = torch.log(flat)
    log_spread = log_svs.std().item()
    score_spread = float(torch.exp(torch.tensor(-log_spread)))

    # Sub-score 2: fraction within [0.5, 2.0]
    in_band = ((flat >= 0.5) & (flat <= 2.0)).float().mean().item()

    return (score_spread * in_band) ** 0.5

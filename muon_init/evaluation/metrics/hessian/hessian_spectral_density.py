"""Full Hessian spectral density via Stochastic Lanczos Quadrature (SLQ).

Estimates the eigenvalue density of the Hessian without explicitly forming it.
The algorithm:
  1. Run the Lanczos algorithm using Hessian-vector products to build a
     tridiagonal matrix T.
  2. Diagonalize T to get Ritz values (eigenvalue estimates) and weights
     (from the first components of the eigenvectors).
  3. Represent the spectral density as a mixture of delta functions at
     the Ritz values, smoothed with Gaussian kernels.
  4. Average over multiple random starting vectors.

References:
    Yao, Z., Gholami, A., Keutzer, K., & Mahoney, M. (NeurIPS 2020).
    "PyHessian: Neural Networks Through the Lens of the Hessian."

    Ghorbani, B., Krishnan, S., & Xiao, Y. (ICML 2019).
    "An Investigation into Neural Net Optimization via Hessian Eigenvalue Density."

    Ubaru, S., Chen, J., & Saad, Y. (2017). "Fast Estimation of
    tr(f(A)) via Stochastic Lanczos Quadrature."
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ._hessian_vector_product import hessian_vector_product, gather_params


def compute_spectral_density(
    model: nn.Module,
    loss_fn: nn.Module,
    data_loader: DataLoader,
    num_eigenvalues: int = 100,
    num_lanczos_steps: int = 200,
    num_random_vectors: int = 1,
    sigma_squared: float = 1e-5,
    num_density_points: int = 1024,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate the Hessian spectral density via Stochastic Lanczos Quadrature.

    Parameters
    ----------
    model : nn.Module
        The model.
    loss_fn : nn.Module
        Loss function.
    data_loader : DataLoader
        Data batches for the loss.
    num_eigenvalues : int
        Not used directly — kept for API compatibility.  The number of Ritz
        values is determined by ``num_lanczos_steps``.
    num_lanczos_steps : int
        Number of Lanczos iterations (controls resolution of the density).
    num_random_vectors : int
        Number of random starting vectors to average over.
    sigma_squared : float
        Bandwidth of the Gaussian kernel for density smoothing.
    num_density_points : int
        Number of grid points for the output density curve.

    Returns
    -------
    (eigenvalues, density) : Tuple[np.ndarray, np.ndarray]
        ``eigenvalues`` is a 1-D grid of shape ``(num_density_points,)``
        and ``density`` is the estimated spectral density evaluated at
        those points.
    """
    params = gather_params(model)
    device = params[0].device

    all_ritz_values: list[np.ndarray] = []
    all_weights: list[np.ndarray] = []

    for _ in range(num_random_vectors):
        ritz_vals, weights = _lanczos(
            model, loss_fn, data_loader, params, device, num_lanczos_steps
        )
        all_ritz_values.append(ritz_vals)
        all_weights.append(weights)

    ritz_values = np.concatenate(all_ritz_values)
    weights = np.concatenate(all_weights)
    weights /= weights.sum()  # re-normalize across all random vectors

    # Build density on a grid
    grid_min = ritz_values.min() - abs(ritz_values.min()) * 0.1 - 1e-6
    grid_max = ritz_values.max() + abs(ritz_values.max()) * 0.1 + 1e-6
    grid = np.linspace(grid_min, grid_max, num_density_points)

    # Gaussian kernel density from the Ritz values / weights
    density = np.zeros_like(grid)
    for val, w in zip(ritz_values, weights):
        density += w * np.exp(-((grid - val) ** 2) / (2 * sigma_squared))
    density /= np.sqrt(2 * np.pi * sigma_squared)

    return grid, density


# ------------------------------------------------------------------
# Lanczos tridiagonalization
# ------------------------------------------------------------------

def _lanczos(
    model: nn.Module,
    loss_fn: nn.Module,
    data_loader: DataLoader,
    params: list[torch.Tensor],
    device: torch.device,
    num_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run Lanczos iteration and return Ritz values + weights.

    Returns
    -------
    (ritz_values, weights) : Tuple[np.ndarray, np.ndarray]
        Ritz values (eigenvalue estimates) and their associated quadrature
        weights (first component of eigenvectors squared).
    """
    # Random starting vector, normalized
    v = [torch.randn_like(p) for p in params]
    norm = _vec_norm(v)
    v = [x / norm for x in v]

    # Store all basis vectors for full reorthogonalization (following PyHessian).
    # This costs O(num_steps) memory but prevents loss of orthogonality that
    # plagues the simple three-term recurrence in finite precision.
    basis: list[list[torch.Tensor]] = [v]

    # Tridiagonal matrix entries
    alphas: list[float] = []  # diagonal
    betas: list[float] = []   # off-diagonal

    for j in range(num_steps):
        # w = H @ v
        w = _hv_over_data(model, loss_fn, data_loader, params, v)

        alpha = _inner(w, v)
        alphas.append(alpha)

        # Three-term recurrence: w = w - alpha * v - beta * v_prev
        if j == 0:
            w = [wi - alpha * vi for wi, vi in zip(w, v)]
        else:
            w = [wi - alpha * vi - betas[-1] * bj
                 for wi, vi, bj in zip(w, v, basis[j - 1])]

        # Full reorthogonalization against all previous basis vectors
        for b in basis:
            coeff = _inner(w, b)
            w = [wi - coeff * bi for wi, bi in zip(w, b)]

        beta = _vec_norm(w)
        if beta < 1e-10:
            # Invariant subspace found — stop early
            break
        betas.append(beta)

        v = [wi / beta for wi in w]
        basis.append(v)

    # Build tridiagonal matrix T and diagonalize
    k = len(alphas)
    T = np.zeros((k, k), dtype=np.float64)
    for i, a in enumerate(alphas):
        T[i, i] = a
    for i, b in enumerate(betas):
        if i + 1 < k:
            T[i, i + 1] = b
            T[i + 1, i] = b

    eigenvalues, eigenvectors = np.linalg.eigh(T)

    # Quadrature weights = first component of each eigenvector, squared
    weights = eigenvectors[0, :] ** 2

    return eigenvalues, weights


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _hv_over_data(
    model: nn.Module,
    loss_fn: nn.Module,
    data_loader: DataLoader,
    params: list[torch.Tensor],
    v: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Average Hv over the full data loader."""
    device = params[0].device
    num_batches = 0
    hv_acc: list[torch.Tensor] | None = None

    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        hv = hessian_vector_product(model, loss_fn, inputs, targets, params, v)
        if hv_acc is None:
            hv_acc = hv
        else:
            hv_acc = [a + b for a, b in zip(hv_acc, hv)]
        num_batches += 1

    assert hv_acc is not None
    return [h / num_batches for h in hv_acc]


def _inner(a: list[torch.Tensor], b: list[torch.Tensor]) -> float:
    return float(sum(torch.sum(x * y) for x, y in zip(a, b)))


def _vec_norm(v: list[torch.Tensor]) -> float:
    return max(_inner(v, v) ** 0.5, 1e-12)

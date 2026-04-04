"""Polar Error at Initialization — a novel metric for Muon optimizer evaluation.

Mathematical Background
-----------------------
The polar decomposition of a matrix M is M = U @ H where U is orthogonal
(or a partial isometry for rectangular matrices) and H is symmetric positive
semi-definite. Muon's core operation is computing the polar factor of the
momentum matrix via Newton-Schulz (NS) iterations, which approximate:

    polar(M) = U @ V^T   (from SVD:  M = U S V^T)

This is the closest orthogonal matrix to M in Frobenius norm.

Why This Metric Matters
-----------------------
At initialization, the first Muon step must orthogonalize the gradient/momentum
via NS iterations. The "polar error" measures how far the initial weight matrix
(or the gradient computed at init) is from its polar factor. A high polar error
means NS must do more work on the first step, which:

  1. Increases approximation error if NS iterations are truncated (Muon uses
     a fixed number, typically 5-10 iterations).
  2. Can cause the first update to be poorly conditioned.
  3. Motivates the need for learning rate warmup.

The Turbo-Muon paper (arXiv 2512.04632) shows that the AOL preconditioner
reduces polar error, directly improving NS convergence — validating that
polar error at initialization is a meaningful quality metric.

Metrics Provided
----------------
- ``compute_polar_error``: Relative Frobenius distance from a matrix to its polar factor.
- ``compute_polar_error_all_layers``: Per-layer polar error for an entire model.
- ``newton_schulz_convergence_steps``: Number of NS iterations needed to converge,
  using the same iteration formula as Muon (KellerJordan/Muon).

References
----------
- Muon optimizer: https://github.com/KellerJordan/Muon
- Turbo-Muon / AOL preconditioner: arXiv 2512.04632
- scipy.linalg.polar for ground-truth polar decomposition
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from evaluation.metrics.spectral._utils import iter_weight_matrices


# ---------------------------------------------------------------------------
# Ground-truth polar factor via SVD
# ---------------------------------------------------------------------------

def polar_factor(M: Tensor) -> Tensor:
    """Compute the polar factor (closest orthogonal/partial-isometry matrix).

    Given M = U S V^T (full SVD), the polar factor is U V^T.
    This is the unique matrix that minimises ||M - Q||_F over all (partial)
    isometries Q.

    Parameters
    ----------
    M : Tensor
        2-D matrix of shape (m, n).

    Returns
    -------
    Tensor
        The polar factor, same shape as M.
    """
    U, _S, Vh = torch.linalg.svd(M.float(), full_matrices=False)
    return U @ Vh


# ---------------------------------------------------------------------------
# Polar error
# ---------------------------------------------------------------------------

def compute_polar_error(M: Tensor) -> float:
    """Relative Frobenius distance from M to its polar factor.

    Defined as::

        polar_error(M) = ||M - polar(M)||_F / ||M||_F

    A value of 0 means M is already a (partial) isometry (all singular values
    equal).  Larger values mean Newton-Schulz must do more work.

    Parameters
    ----------
    M : Tensor
        2-D weight or momentum matrix.

    Returns
    -------
    float
        Polar error in [0, +inf).  Exactly 0 for orthogonal / partial-isometry
        matrices.
    """
    M_f = M.float()
    if M_f.ndim > 2:
        M_f = M_f.reshape(M_f.shape[0], -1)

    norm_M = torch.linalg.norm(M_f, ord="fro")
    if norm_M == 0:
        return 0.0

    P = polar_factor(M_f)
    return (torch.linalg.norm(M_f - P, ord="fro") / norm_M).item()


def compute_polar_error_all_layers(model: nn.Module) -> Dict[str, float]:
    """Compute polar error for every weight matrix in a model.

    Skips 1-D parameters (biases, LayerNorm scales).  Higher-dimensional
    tensors (e.g., conv filters) are reshaped to (fan_out, fan_in).

    Parameters
    ----------
    model : nn.Module
        Any PyTorch model.

    Returns
    -------
    Dict[str, float]
        Mapping from parameter name to polar error.
    """
    results: Dict[str, float] = {}
    for name, W in iter_weight_matrices(model):
        results[name] = compute_polar_error(W)
    return results


# ---------------------------------------------------------------------------
# Newton-Schulz iteration (Muon's exact formula)
# ---------------------------------------------------------------------------

def _muon_newton_schulz_step(X: Tensor) -> Tensor:
    """One step of Muon's *approximate* Newton-Schulz iteration.

    Exact reproduction of ``zeropower_via_newtonschulz5`` from
    KellerJordan/Muon.  Applies the quintic polynomial::

        A = X @ X^T
        B = b * A + c * A @ A
        X_new = a * X + B @ X

    with ``a, b, c = 3.4445, -4.7750, 2.0315``.

    **Important:** These coefficients are chosen to maximise the slope at
    zero (fast convergence of small singular values) at the cost of NOT
    converging exactly to the polar factor.  The result after Muon's default
    10 steps is ``U S' V^T`` where ``S' ~ Uniform(0.5, 1.5)`` — an
    *approximate* isometry that works well in practice.  See the docstring
    in KellerJordan/Muon for details.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    A = X @ X.T
    B = b * A + c * A @ A
    return a * X + B @ X


def _convergent_newton_schulz_step(X: Tensor) -> Tensor:
    """One step of the standard *convergent* cubic Newton-Schulz iteration.

    Applies the classical iteration::

        X_{k+1} = (3/2) X_k - (1/2) X_k @ (X_k^T @ X_k)

    which maps each singular value via ``f(σ) = (3σ - σ³) / 2``.
    This has a *stable* fixed point at σ = 1, guaranteeing convergence
    to the true polar factor for all starting singular values in (0, √3).
    """
    return 1.5 * X - 0.5 * X @ (X.T @ X)


def muon_ns_approximation_quality(
    M: Tensor,
    steps: int = 10,
) -> float:
    """Measure how close Muon's (approximate) NS output is to a true isometry.

    Runs Muon's exact NS iteration for the given number of steps and returns
    the mean absolute deviation of the output's singular values from 1.0.

    A value of 0 means the output is a perfect isometry.  Muon's coefficients
    intentionally trade exact convergence for speed, so typical values are
    in the range [0.1, 0.5].

    Parameters
    ----------
    M : Tensor
        2-D matrix (the gradient / momentum that Muon would orthogonalise).
    steps : int
        Number of NS iterations (Muon default: 10).

    Returns
    -------
    float
        Mean |σ_i - 1| of the NS output's singular values.
    """
    M_f = M.float()
    if M_f.ndim > 2:
        M_f = M_f.reshape(M_f.shape[0], -1)

    # Muon normalises by Frobenius norm and transposes if tall
    transposed = M_f.shape[0] > M_f.shape[1]
    X = M_f.T if transposed else M_f.clone()
    X = X / (torch.linalg.norm(X, ord="fro") + 1e-7)

    for _ in range(steps):
        X = _muon_newton_schulz_step(X)

    sv = torch.linalg.svdvals(X)
    return torch.abs(sv - 1.0).mean().item()


def newton_schulz_convergence_steps(
    M: Tensor,
    tol: float = 1e-6,
    max_iter: int = 20,
) -> int:
    """Count convergent NS iterations needed to reach the polar factor.

    Uses the *standard cubic* Newton-Schulz iteration (which provably
    converges), not Muon's approximate quintic.  The input is normalised
    by spectral norm so all singular values lie in (0, 1], well within the
    convergence basin.

    Convergence is measured by comparing against the ground-truth polar
    factor (computed via SVD)::

        ||X_k - polar(M / ||M||_2)||_F  <  tol

    This metric quantifies how "far from orthogonal" the input matrix is
    in a way that directly relates to NS convergence difficulty.

    Parameters
    ----------
    M : Tensor
        2-D matrix.
    tol : float
        Convergence tolerance on the distance to the true polar factor.
    max_iter : int
        Maximum number of NS iterations.

    Returns
    -------
    int
        Number of iterations to reach tolerance.  Returns ``max_iter`` if
        convergence was not achieved.
    """
    M_f = M.float()
    if M_f.ndim > 2:
        M_f = M_f.reshape(M_f.shape[0], -1)

    spectral_norm = torch.linalg.norm(M_f, ord=2)
    if spectral_norm == 0:
        return 0
    X = M_f / spectral_norm

    # Ground-truth polar factor of the normalised matrix
    target = polar_factor(X)

    for step in range(1, max_iter + 1):
        X = _convergent_newton_schulz_step(X)
        dist = torch.linalg.norm(X - target, ord="fro").item()
        if dist < tol:
            return step

    return max_iter


# ---------------------------------------------------------------------------
# Combined analysis dataclass
# ---------------------------------------------------------------------------

@dataclass
class PolarErrorReport:
    """Per-layer polar error analysis."""
    polar_errors: Dict[str, float]
    ns_convergence_steps: Dict[str, int]
    muon_ns_quality: Dict[str, float]
    mean_polar_error: float
    max_polar_error: float
    mean_ns_steps: float
    mean_muon_ns_quality: float


def analyze_polar_error(
    model: nn.Module,
    ns_tol: float = 1e-6,
    ns_max_iter: int = 20,
    muon_ns_steps: int = 10,
) -> PolarErrorReport:
    """Full polar error analysis for all weight matrices in a model.

    Computes three complementary metrics per layer:

    1. **Polar error**: relative Frobenius distance to the true polar factor.
    2. **NS convergence steps**: how many *convergent* (cubic) NS iterations
       are needed to reach the polar factor (measures structural distance
       from orthogonality).
    3. **Muon NS quality**: how close Muon's *approximate* (quintic) NS
       output is to a true isometry after a fixed number of steps (measures
       the practical approximation quality for the actual optimizer).

    Parameters
    ----------
    model : nn.Module
        Any PyTorch model.
    ns_tol : float
        Tolerance for convergent NS step count.
    ns_max_iter : int
        Maximum convergent NS iterations.
    muon_ns_steps : int
        Number of Muon NS steps for approximation quality.

    Returns
    -------
    PolarErrorReport
        Aggregated polar error diagnostics.
    """
    polar_errors: Dict[str, float] = {}
    ns_steps: Dict[str, int] = {}
    ns_quality: Dict[str, float] = {}

    for name, W in iter_weight_matrices(model):
        polar_errors[name] = compute_polar_error(W)
        ns_steps[name] = newton_schulz_convergence_steps(W, tol=ns_tol, max_iter=ns_max_iter)
        ns_quality[name] = muon_ns_approximation_quality(W, steps=muon_ns_steps)

    pe_vals = list(polar_errors.values()) if polar_errors else [0.0]
    ns_vals = list(ns_steps.values()) if ns_steps else [0]
    nq_vals = list(ns_quality.values()) if ns_quality else [0.0]

    return PolarErrorReport(
        polar_errors=polar_errors,
        ns_convergence_steps=ns_steps,
        muon_ns_quality=ns_quality,
        mean_polar_error=sum(pe_vals) / len(pe_vals),
        max_polar_error=max(pe_vals),
        mean_ns_steps=sum(ns_vals) / len(ns_vals),
        mean_muon_ns_quality=sum(nq_vals) / len(nq_vals),
    )

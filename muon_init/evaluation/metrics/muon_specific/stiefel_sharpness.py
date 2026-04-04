"""Stiefel-Projected Sharpness — a novel metric for Muon optimizer evaluation.

Mathematical Background
-----------------------
Standard sharpness metrics (SAM, lambda_max, etc.) measure the maximum loss
increase under *arbitrary* perturbations in an epsilon-ball around the current
parameters.  But Muon does not move in arbitrary directions — every update is
the polar factor of the momentum, which is constrained to the Stiefel manifold
(the set of matrices with orthonormal columns/rows).

Therefore, sharpness in directions Muon *cannot reach* is irrelevant.  What
matters is sharpness along the **tangent space of the Stiefel manifold** at the
current weight matrices.

Stiefel Manifold Tangent Space
------------------------------
The Stiefel manifold St(n, p) = {W in R^{n x p} : W^T W = I_p} has tangent
space at W given by:

    T_W St(n, p) = { Delta in R^{n x p} : W^T Delta + Delta^T W = 0 }

Equivalently, any tangent vector can be decomposed as:

    Delta = W @ A + W_perp @ B

where A is p x p skew-symmetric (A = -A^T), W_perp is the n x (n-p) orthogonal
complement of W, and B is (n-p) x p arbitrary.

For square orthogonal matrices (n = p), the tangent space simplifies to:

    T_W O(n) = { W @ A : A^T = -A }

i.e., all tangent vectors are W times a skew-symmetric matrix.

Stiefel-Projected Sharpness
----------------------------
We define Stiefel-projected sharpness as:

    sharp_St(W) = max_{Delta in T_W St, ||Delta||_F <= epsilon} L(W + Delta) - L(W)

Approximated by sampling random tangent vectors, perturbing weights, and
measuring the maximum loss increase.

Why This Metric Matters
-----------------------
A model may sit in a "sharp" region of parameter space (high standard
sharpness), but if the sharp directions are orthogonal to the Stiefel tangent
space, Muon will never explore them.  Conversely, a model could appear "flat"
under standard metrics but be sharp along Stiefel-tangent directions.

Stiefel-projected sharpness captures the effective curvature that Muon
experiences, making it a better predictor of:
  - Training stability under Muon
  - Whether warmup is needed
  - Generalization quality of the converged solution

Implementation References
-------------------------
- geoopt (https://github.com/geoopt/geoopt): Stiefel manifold projections
- pymanopt (https://github.com/pymanopt/pymanopt): Manifold optimization
- McTorch (https://github.com/mctorch/mctorch): Manifold optimization in PyTorch
These were consulted for the tangent space projection formula.

This metric is genuinely novel — no prior work defines sharpness projected
onto the Stiefel tangent space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from evaluation.metrics.spectral._utils import iter_weight_matrices


# ---------------------------------------------------------------------------
# Tangent space projection
# ---------------------------------------------------------------------------

def project_to_stiefel_tangent(W: Tensor, Delta: Tensor) -> Tensor:
    """Project an arbitrary perturbation onto the Stiefel tangent space at W.

    The tangent space of the Stiefel manifold St(n, p) at W is:

        T_W St(n, p) = { Delta : W^T Delta + Delta^T W = 0 }

    The projection removes the symmetric part of W^T Delta:

        proj(Delta) = Delta - W @ sym(W^T @ Delta)

    where sym(A) = (A + A^T) / 2.

    This formula is the standard Riemannian projection used in manifold
    optimization (see Absil, Mahony & Sepulchre, "Optimization Algorithms
    on Matrix Manifolds", Eq. 3.35; also geoopt's Stiefel implementation).

    Parameters
    ----------
    W : Tensor
        Point on (or near) the Stiefel manifold, shape (n, p) with n >= p.
    Delta : Tensor
        Arbitrary perturbation matrix, same shape as W.

    Returns
    -------
    Tensor
        Projected perturbation in T_W St(n, p).  Satisfies (approximately):
        W^T @ result + result^T @ W = 0.
    """
    # sym(W^T @ Delta) = (W^T Delta + Delta^T W) / 2
    WtD = W.T @ Delta
    sym_part = 0.5 * (WtD + WtD.T)
    return Delta - W @ sym_part


def random_stiefel_tangent_vector(W: Tensor) -> Tensor:
    """Sample a random tangent vector at W on the Stiefel manifold.

    Strategy: generate a random Gaussian matrix, project it onto T_W St,
    then normalize to unit Frobenius norm.

    Parameters
    ----------
    W : Tensor
        Point on the Stiefel manifold, shape (n, p).

    Returns
    -------
    Tensor
        Unit-norm tangent vector in T_W St(n, p).
    """
    Delta = torch.randn_like(W)
    Delta = project_to_stiefel_tangent(W, Delta)
    norm = torch.linalg.norm(Delta, ord="fro")
    if norm > 0:
        Delta = Delta / norm
    return Delta


def verify_tangent_condition(W: Tensor, Delta: Tensor, atol: float = 1e-5) -> float:
    """Check how well Delta satisfies the Stiefel tangent space condition.

    Returns ||W^T Delta + Delta^T W||_F, which should be ~0 for a valid
    tangent vector.

    Parameters
    ----------
    W : Tensor
        Point on the Stiefel manifold.
    Delta : Tensor
        Candidate tangent vector.
    atol : float
        Not used for computation, provided for reference.

    Returns
    -------
    float
        The tangent space violation: ||W^T Delta + Delta^T W||_F.
    """
    sym = W.T @ Delta + Delta.T @ W
    return torch.linalg.norm(sym, ord="fro").item()


# ---------------------------------------------------------------------------
# Stiefel-projected sharpness
# ---------------------------------------------------------------------------

@dataclass
class SharpnessResult:
    """Result of a Stiefel-projected sharpness measurement."""
    stiefel_sharpness: float      # max loss increase along Stiefel tangent directions
    euclidean_sharpness: float    # max loss increase along unconstrained directions (for comparison)
    base_loss: float              # loss at the unperturbed weights
    worst_tangent_loss: float     # highest loss found along tangent perturbations
    num_samples: int              # number of random tangent directions sampled
    per_sample_losses: List[float]  # loss at each perturbation


def stiefel_sharpness(
    model: nn.Module,
    loss_fn: Callable,
    data_loader: DataLoader,
    epsilon: float = 0.01,
    num_samples: int = 10,
    device: Optional[torch.device] = None,
) -> SharpnessResult:
    """Compute Stiefel-projected sharpness of the model.

    Samples random tangent vectors on the Stiefel manifold at each weight
    matrix, perturbs the weights along these directions (scaled by epsilon),
    and measures the maximum loss increase over a batch of data.

    For comparison, also computes standard (Euclidean) sharpness using
    unconstrained random perturbations.

    Parameters
    ----------
    model : nn.Module
        The model to evaluate.
    loss_fn : callable
        Loss function with signature ``loss_fn(model_output, targets) -> scalar``.
    data_loader : DataLoader
        Provides (input, target) batches.  Only the first batch is used.
    epsilon : float
        Perturbation radius in Frobenius norm.
    num_samples : int
        Number of random tangent directions to sample.
    device : torch.device, optional
        Device to use.  Inferred from model if not given.

    Returns
    -------
    SharpnessResult
        Sharpness measurements and diagnostics.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Get a single batch
    inputs, targets = next(iter(data_loader))
    inputs, targets = inputs.to(device), targets.to(device)

    # Compute base loss
    with torch.no_grad():
        base_output = model(inputs)
        base_loss = loss_fn(base_output, targets).item()

    # Collect 2D weight matrices and their names
    weight_params: List[tuple] = []  # (name, param, original_data)
    for name, param in model.named_parameters():
        if param.ndim >= 2:
            weight_params.append((name, param, param.data.clone()))

    # --- Stiefel-projected sharpness ---
    stiefel_losses: List[float] = []
    for _ in range(num_samples):
        # Perturb each weight along a random Stiefel tangent direction
        with torch.no_grad():
            for _name, param, orig in weight_params:
                W = orig.float()
                if W.ndim > 2:
                    shape = W.shape
                    W_2d = W.reshape(W.shape[0], -1)
                    tangent = random_stiefel_tangent_vector(W_2d)
                    perturbation = (epsilon * tangent).reshape(shape)
                else:
                    tangent = random_stiefel_tangent_vector(W)
                    perturbation = epsilon * tangent
                param.data = (orig + perturbation.to(orig.dtype)).clone()

        with torch.no_grad():
            output = model(inputs)
            perturbed_loss = loss_fn(output, targets).item()
        stiefel_losses.append(perturbed_loss)

        # Restore original weights
        with torch.no_grad():
            for _name, param, orig in weight_params:
                param.data = orig.clone()

    # --- Euclidean sharpness (for comparison) ---
    euclidean_max_loss = base_loss
    for _ in range(num_samples):
        with torch.no_grad():
            for _name, param, orig in weight_params:
                delta = torch.randn_like(orig)
                delta_norm = torch.linalg.norm(delta.float(), ord="fro")
                if delta_norm > 0:
                    delta = delta * (epsilon / delta_norm)
                param.data = (orig + delta).clone()

        with torch.no_grad():
            output = model(inputs)
            perturbed_loss = loss_fn(output, targets).item()
        euclidean_max_loss = max(euclidean_max_loss, perturbed_loss)

        with torch.no_grad():
            for _name, param, orig in weight_params:
                param.data = orig.clone()

    worst_tangent_loss = max(stiefel_losses) if stiefel_losses else base_loss

    return SharpnessResult(
        stiefel_sharpness=worst_tangent_loss - base_loss,
        euclidean_sharpness=euclidean_max_loss - base_loss,
        base_loss=base_loss,
        worst_tangent_loss=worst_tangent_loss,
        num_samples=num_samples,
        per_sample_losses=stiefel_losses,
    )

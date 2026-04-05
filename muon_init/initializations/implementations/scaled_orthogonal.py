"""Scaled orthogonal initialization for Muon.

Initializes weight matrices as W = alpha * Q where Q is a random partial
isometry (Haar-distributed). All singular values equal alpha, giving:
  - Zero polar error (Muon's NS iteration has nothing to fix)
  - Perfect signal propagation when alpha = 1/sqrt(c_phi)
  - Unit condition number

See initializations/theory/optimal_alpha.md for the full derivation.
"""

import math

import torch
import torch.nn as nn


# Activation gain c_phi = E[phi(z)^2] for z ~ N(0,1).
# alpha* = 1 / sqrt(c_phi) for variance-preserving signal propagation.
ACTIVATION_GAINS = {
    "linear": 1.0,
    "relu": 0.5,
    "gelu": 0.4252,
    "silu": 0.3024,
    "tanh": 0.3926,
}


def compute_activation_gain(phi, num_samples=1_000_000):
    """Compute c_phi = E[phi(z)^2] / E[z^2] for z ~ N(0,1)."""
    z = torch.randn(num_samples)
    return (phi(z) ** 2).mean().item()


def optimal_alpha(activation="relu"):
    """Return the optimal scale factor for the given activation.

    Parameters
    ----------
    activation : str
        One of "linear", "relu", "gelu", "silu", "tanh".

    Returns
    -------
    float
        The scale factor alpha* = 1 / sqrt(c_phi).
    """
    if activation not in ACTIVATION_GAINS:
        raise ValueError(
            f"Unknown activation '{activation}'. "
            f"Known: {list(ACTIVATION_GAINS.keys())}. "
            f"Use compute_activation_gain() for custom activations."
        )
    return 1.0 / math.sqrt(ACTIVATION_GAINS[activation])


def scaled_orthogonal(model, alpha=None, activation="relu"):
    """Apply scaled orthogonal initialization to all weight matrices.

    W = alpha * Q where Q is drawn from nn.init.orthogonal_ (Haar measure
    via QR decomposition of a Gaussian matrix).

    Parameters
    ----------
    model : nn.Module
        Model to initialize in-place.
    alpha : float, optional
        Scale factor. If None, uses optimal_alpha(activation).
    activation : str
        Activation function name (used to compute default alpha).
        Ignored if alpha is explicitly provided.
    """
    if alpha is None:
        alpha = optimal_alpha(activation)

    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            nn.init.orthogonal_(module.weight)
            module.weight.data.mul_(alpha)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

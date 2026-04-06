"""Sharpness-aware initialization for Muon.

Derives the initialization scale alpha from the Edge of Stability condition:

    eta * lambda_max(alpha) < 2

Two modes:
1. Analytical: alpha* = (2 / (eta * C_data))^(1/(2L)) / sqrt(c_phi)
   Fast but only accurate for unnormalized MLPs.
2. Empirical: measure lambda_max at a few alpha values, fit, and solve.
   Slower but works for any architecture.

See initializations/theory/sharpness_aware_derivation.md for the full derivation.
"""

import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from initializations.implementations.scaled_orthogonal import (
    scaled_orthogonal, ACTIVATION_GAINS,
)
from evaluation.metrics.hessian.hessian_top_eigenvalue import compute_lambda_max


def sharpness_aware_analytical(
    model: nn.Module,
    lr: float = 0.02,
    num_layers: int = 8,
    activation: str = "relu",
    reference_alpha: float = 1.0,
    loss_fn: nn.Module | None = None,
    data_loader: DataLoader | None = None,
    C_data: float | None = None,
):
    """Apply sharpness-aware initialization using the analytical formula.

    Computes alpha* = (2 / (eta * C_data))^(1/(2L)) / sqrt(c_phi), then
    initializes weights as W = alpha* * Q.

    Parameters
    ----------
    model : nn.Module
        Model to initialize in-place.
    lr : float
        Muon learning rate eta.
    num_layers : int
        Number of weight layers L.
    activation : str
        Activation function name (for c_phi lookup).
    reference_alpha : float
        Alpha value used to estimate C_data (if C_data not provided).
    loss_fn : nn.Module, optional
        Loss function for lambda_max measurement. Required if C_data is None.
    data_loader : DataLoader, optional
        Data for lambda_max measurement. Required if C_data is None.
    C_data : float, optional
        Pre-computed data-dependent constant. If None, estimated from a
        lambda_max measurement at reference_alpha.

    Returns
    -------
    float
        The computed alpha*.
    """
    c_phi = ACTIVATION_GAINS.get(activation, 0.5)
    threshold = 2.0 / lr

    if C_data is None:
        if loss_fn is None or data_loader is None:
            raise ValueError(
                "Must provide loss_fn and data_loader to estimate C_data, "
                "or provide C_data directly."
            )
        # Measure lambda_max at reference_alpha
        scaled_orthogonal(model, alpha=reference_alpha)
        lambda_ref = compute_lambda_max(model, loss_fn, data_loader)
        C_data = lambda_ref / (reference_alpha ** 2 * c_phi) ** num_layers

    alpha_star = (threshold / C_data) ** (1.0 / (2 * num_layers)) / math.sqrt(c_phi)

    # Apply the initialization
    scaled_orthogonal(model, alpha=alpha_star)
    return alpha_star


def sharpness_aware_empirical(
    model: nn.Module,
    loss_fn: nn.Module,
    data_loader: DataLoader,
    lr: float = 0.02,
    probe_alphas: list[float] | None = None,
    num_iterations: int = 30,
    model_builder=None,
    model_config: dict | None = None,
):
    """Apply sharpness-aware initialization using empirical lambda_max measurement.

    Measures lambda_max at several alpha values, fits a power law, and solves
    for alpha where lambda_max = 2/eta. Works for any architecture.

    Parameters
    ----------
    model : nn.Module
        Model to initialize in-place. Must be on the correct device.
    loss_fn : nn.Module
        Loss function.
    data_loader : DataLoader
        Data for Hessian computation (a few batches suffice).
    lr : float
        Muon learning rate eta.
    probe_alphas : list[float], optional
        Alpha values to probe. Default: [0.1, 0.5, 1.0, 1.5, 2.0].
    num_iterations : int
        Power iteration steps for lambda_max.
    model_builder : callable, optional
        Function that returns a fresh model (for re-initialization at each
        probe alpha). If None, the passed model is reused (less accurate
        due to in-place modification).
    model_config : dict, optional
        Config dict passed to model_builder.

    Returns
    -------
    float
        The computed alpha*.
    """
    import numpy as np
    from scipy.optimize import brentq

    if probe_alphas is None:
        probe_alphas = [0.1, 0.5, 1.0, 1.5, 2.0]

    threshold = 2.0 / lr
    device = next(model.parameters()).device

    # Measure lambda_max at each probe alpha
    alphas_arr = []
    lambdas_arr = []
    for alpha in probe_alphas:
        if model_builder is not None:
            probe_model = model_builder(model_config).to(device)
        else:
            probe_model = model
        scaled_orthogonal(probe_model, alpha=alpha)
        lam = compute_lambda_max(
            probe_model, loss_fn, data_loader, num_iterations=num_iterations
        )
        if lam > 0:
            alphas_arr.append(alpha)
            lambdas_arr.append(lam)

    if len(alphas_arr) < 2:
        raise RuntimeError("Not enough valid lambda_max measurements to fit")

    alphas_np = np.array(alphas_arr)
    lambdas_np = np.array(lambdas_arr)

    # Fit power law in log-log space
    # Filter to above-floor regime
    floor = np.min(lambdas_np)
    above_floor = lambdas_np > 2 * floor
    if np.sum(above_floor) >= 2:
        log_a = np.log(alphas_np[above_floor])
        log_l = np.log(lambdas_np[above_floor])
    else:
        log_a = np.log(alphas_np)
        log_l = np.log(lambdas_np)

    coeffs = np.polyfit(log_a, log_l, 1)
    k_fit = coeffs[0]
    C_fit = np.exp(coeffs[1])

    # Solve C * alpha^k = threshold
    alpha_star = (threshold / C_fit) ** (1.0 / k_fit)

    # Clamp to reasonable range
    alpha_star = max(0.01, min(alpha_star, max(probe_alphas)))

    # Apply the initialization
    scaled_orthogonal(model, alpha=alpha_star)
    return alpha_star

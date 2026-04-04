"""Empirical Spectral Density (ESD) of weight matrices.

Computes the distribution of singular values and compares against the
Marchenko-Pastur distribution, the random matrix theory baseline expected
at initialization for i.i.d. Gaussian weights.

The Marchenko-Pastur law for an (m x n) matrix with i.i.d. entries of
variance sigma^2 has density supported on [lambda_-, lambda_+] where:
    lambda_+/- = sigma^2 * (1 +/- sqrt(gamma))^2
    gamma = m / n  (aspect ratio, assuming m <= n)

References:
    Martin & Mahoney (ICML 2019). "Implicit Self-Regularization in Deep
        Neural Networks: Evidence from Random Matrix Theory and Implications
        for Training." arXiv:1810.01075.
    Martin & Mahoney (JMLR 2021). "Heavy-Tailed Self-Regularization in DNN."
    WeightWatcher: https://github.com/CalculatedContent/WeightWatcher
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch import Tensor

from ._utils import singular_values


def _histogram_sv(
    sv: np.ndarray, num_bins: int = 100
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Histogram singular values, handling degenerate (constant) distributions.

    Returns (bin_centers, density, bin_width) where density integrates to ~1.
    """
    sv_range = float(sv.max() - sv.min())
    if sv_range < 1e-10:
        # Degenerate case: all SVs are (nearly) identical.
        # Create bins around the single value; density is a spike.
        center = float(sv.mean())
        half_width = max(abs(center) * 0.1, 0.5)
        bin_edges = np.linspace(center - half_width, center + half_width, num_bins + 1)
    else:
        bin_edges = np.linspace(sv.min(), sv.max(), num_bins + 1)
    counts, bin_edges = np.histogram(sv, bins=bin_edges)
    bin_width = bin_edges[1] - bin_edges[0]
    # Manually normalize to density to avoid numpy divide-by-zero
    total = counts.sum() * bin_width
    density = counts / total if total > 0 else counts.astype(float)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return bin_centers, density, bin_width


def compute_esd(
    W: Tensor, num_bins: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the empirical spectral density of a weight matrix.

    Returns a histogram of singular values (not eigenvalues of W^T W).

    Args:
        W: Weight tensor (2D or higher; reshaped to 2D if needed).
        num_bins: Number of histogram bins.

    Returns:
        Tuple of (bin_centers, density) as numpy arrays, where density
        is normalized so that it integrates to 1 (a proper PDF).
    """
    sv = singular_values(W).cpu().numpy()
    bin_centers, density, _ = _histogram_sv(sv, num_bins=num_bins)
    return bin_centers, density


def _marchenko_pastur_pdf(
    x: np.ndarray, gamma: float, sigma_sq: float
) -> np.ndarray:
    """Evaluate the Marchenko-Pastur density for singular values.

    The MP law is defined for eigenvalues of (1/n) W^T W. To get the
    density over singular values, we apply the change of variables
    lambda = sigma_val^2 and multiply by the Jacobian |d(lambda)/d(sigma_val)| = 2*sigma_val.

    Args:
        x: Singular value points at which to evaluate.
        gamma: Aspect ratio min(m,n)/max(m,n), in (0, 1].
        sigma_sq: Variance of the matrix entries.

    Returns:
        Density values at each point x.
    """
    # Eigenvalue bounds for (1/n) * W^T W
    lambda_plus = sigma_sq * (1 + np.sqrt(gamma)) ** 2
    lambda_minus = sigma_sq * (1 - np.sqrt(gamma)) ** 2
    # Singular value bounds
    sv_plus = np.sqrt(lambda_plus)
    sv_minus = np.sqrt(lambda_minus)

    pdf = np.zeros_like(x)
    mask = (x >= sv_minus) & (x <= sv_plus) & (x > 1e-12)
    lam = x[mask] ** 2  # Convert singular values to eigenvalues
    # MP density in eigenvalue space
    mp_eig = np.sqrt((lambda_plus - lam) * (lam - lambda_minus)) / (
        2 * np.pi * sigma_sq * gamma * lam
    )
    # Jacobian for change of variables: d(lambda)/d(sv) = 2*sv
    pdf[mask] = mp_eig * 2 * x[mask]
    return pdf


def marchenko_pastur_fit(
    W: Tensor,
) -> Tuple[float, float]:
    """Compare the singular value distribution to the Marchenko-Pastur law.

    Estimates sigma^2 from the matrix and computes both the KL divergence
    and the Kolmogorov-Smirnov statistic between the empirical singular
    value distribution and the theoretical MP distribution.

    Args:
        W: Weight tensor (2D or higher; reshaped to 2D if needed).

    Returns:
        Tuple of (kl_divergence, ks_statistic).
        - kl_divergence: KL(empirical || MP). Lower = closer to random init.
        - ks_statistic: KS statistic. Lower = closer to random init.
    """
    if W.ndim > 2:
        W = W.reshape(W.shape[0], -1)
    W = W.float()
    m, n = W.shape
    gamma = min(m, n) / max(m, n)

    sv = torch.linalg.svdvals(W).cpu().numpy()

    # Estimate entry variance: for (m x n) matrix, Var(entry) ≈ ||W||_F^2 / (m*n)
    sigma_sq = np.sum(sv ** 2) / max(m, n)

    # --- KL divergence via histogram ---
    num_bins = min(100, len(sv))
    bin_centers, density_emp, bin_width = _histogram_sv(sv, num_bins=num_bins)
    density_mp = _marchenko_pastur_pdf(bin_centers, gamma, sigma_sq)

    eps = 1e-12
    p = density_emp + eps
    q = density_mp + eps
    p_sum = p.sum() * bin_width
    q_sum = q.sum() * bin_width
    if p_sum > 0 and q_sum > 0:
        p = p / p_sum
        q = q / q_sum
        kl = float(np.sum(p * np.log(p / q)) * bin_width)
    else:
        kl = float("inf")

    # --- KS statistic via empirical CDF on sorted SVs ---
    # This avoids histogram binning artifacts for degenerate distributions.
    sv_sorted = np.sort(sv)
    n_sv = len(sv_sorted)
    # Empirical CDF: F_emp(x) = (rank of x) / n
    cdf_emp = np.arange(1, n_sv + 1) / n_sv
    # Theoretical MP CDF at each observed SV via numerical integration
    # Use a fine grid for the MP PDF and interpolate the CDF
    lambda_plus = sigma_sq * (1 + np.sqrt(gamma)) ** 2
    lambda_minus = sigma_sq * (1 - np.sqrt(gamma)) ** 2
    sv_plus = np.sqrt(lambda_plus)
    sv_minus = np.sqrt(max(lambda_minus, 0.0))
    grid = np.linspace(0, max(sv_sorted.max(), sv_plus) * 1.1, 2000)
    mp_pdf = _marchenko_pastur_pdf(grid, gamma, sigma_sq)
    mp_cdf = np.cumsum(mp_pdf) * (grid[1] - grid[0])
    # Clamp to [0, 1]
    mp_cdf = np.clip(mp_cdf, 0.0, 1.0)
    # Interpolate theoretical CDF at observed SVs
    cdf_mp_at_sv = np.interp(sv_sorted, grid, mp_cdf)
    ks = float(np.max(np.abs(cdf_emp - cdf_mp_at_sv)))

    return kl, ks

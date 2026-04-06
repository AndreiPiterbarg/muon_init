"""Base distribution samplers for the compositional transform pipeline.

All distributions are standardized to empirical mean=0, variance=1 per
dimension so the base choice controls shape (tail weight, skewness, support
bounds) without changing scale.
"""

import torch
import torch.nn.functional as F

BASE_DISTRIBUTIONS = [
    "gaussian",
    "student_t",
    "uniform",
    "laplace",
    "log_normal",
    "beta",
    "exponential",
    "mixture_of_gaussians",
]

# Maximum extra params needed by any base distribution (mixture of Gaussians
# with K_max=4, d=20: 3 K-logits + 4*20 means + 4*20 log-stds = 163).
# For general d: 3 + K_max * d + K_max * d = 3 + 2 * K_max * d.
K_MAX = 4


def base_param_size(n_dims: int) -> int:
    """Total param budget for the base distribution block."""
    return 3 + 2 * K_MAX * n_dims  # 163 for d=20


def sample_base(
    dist_index: int,
    n_points: int,
    batch_size: int,
    n_dims: int,
    params: torch.Tensor,
) -> torch.Tensor:
    """Sample from the selected base distribution.

    Args:
        dist_index: Index into BASE_DISTRIBUTIONS (0-7).
        n_points: Number of points per sequence.
        batch_size: Number of sequences.
        n_dims: Dimensionality of each point.
        params: Flat parameter tensor (size = base_param_size(n_dims)).

    Returns:
        Tensor of shape (batch_size, n_points, n_dims), standardized to
        empirical mean=0, var=1 per dimension.
    """
    shape = (batch_size, n_points, n_dims)

    if dist_index == 0:
        z = _sample_gaussian(shape)
    elif dist_index == 1:
        z = _sample_student_t(shape, params)
    elif dist_index == 2:
        z = _sample_uniform(shape)
    elif dist_index == 3:
        z = _sample_laplace(shape)
    elif dist_index == 4:
        z = _sample_log_normal(shape, params)
    elif dist_index == 5:
        z = _sample_beta(shape, params)
    elif dist_index == 6:
        z = _sample_exponential(shape)
    elif dist_index == 7:
        z = _sample_mixture_of_gaussians(shape, params, n_dims)
    else:
        raise ValueError(f"Unknown base distribution index: {dist_index}")

    return _standardize(z)


def _standardize(z: torch.Tensor) -> torch.Tensor:
    """Per-dimension empirical standardization to mean=0, var=1.

    Operates across the n_points axis (dim=1).
    """
    mean = z.mean(dim=1, keepdim=True)
    std = z.std(dim=1, keepdim=True)
    return (z - mean) / (std + 1e-8)


# --- Individual distribution samplers ---


def _sample_gaussian(shape: tuple) -> torch.Tensor:
    return torch.randn(shape)


def _sample_student_t(shape: tuple, params: torch.Tensor) -> torch.Tensor:
    log_df = params[0].item()
    df = max(min(torch.exp(torch.tensor(log_df)).item(), 100.0), 1.01)
    dist = torch.distributions.StudentT(df)
    return dist.sample(shape)


def _sample_uniform(shape: tuple) -> torch.Tensor:
    return torch.rand(shape) * 2.0 - 1.0


def _sample_laplace(shape: tuple) -> torch.Tensor:
    dist = torch.distributions.Laplace(0.0, 1.0)
    return dist.sample(shape)


def _sample_log_normal(shape: tuple, params: torch.Tensor) -> torch.Tensor:
    mu_ln = params[0].item()
    sigma_ln = F.softplus(params[1]).item()
    sigma_ln = max(sigma_ln, 0.01)  # avoid degenerate case
    dist = torch.distributions.LogNormal(mu_ln, sigma_ln)
    return dist.sample(shape)


def _sample_beta(shape: tuple, params: torch.Tensor) -> torch.Tensor:
    log_a, log_b = params[0].item(), params[1].item()
    a = max(min(torch.exp(torch.tensor(log_a)).item(), 10.0), 0.1)
    b = max(min(torch.exp(torch.tensor(log_b)).item(), 10.0), 0.1)
    dist = torch.distributions.Beta(a, b)
    return dist.sample(shape)


def _sample_exponential(shape: tuple) -> torch.Tensor:
    dist = torch.distributions.Exponential(1.0)
    return dist.sample(shape)


def _sample_mixture_of_gaussians(
    shape: tuple, params: torch.Tensor, n_dims: int
) -> torch.Tensor:
    """Mixture of K Gaussians (K=2,3,4 selected by logits).

    Param layout:
        [0:3]  - K logits (for K=2, 3, 4)
        [3 : 3 + K_MAX*d] - per-component means (K_MAX * n_dims)
        [3 + K_MAX*d : 3 + 2*K_MAX*d] - per-component log-stds (K_MAX * n_dims)
    """
    batch_size, n_points, _ = shape

    # Decode K from 3 logits -> argmax + 2 gives K in {2, 3, 4}
    k_logits = params[0:3]
    K = int(torch.argmax(k_logits).item()) + 2

    # Decode per-component means and stds
    means_flat = params[3: 3 + K_MAX * n_dims]
    log_stds_flat = params[3 + K_MAX * n_dims: 3 + 2 * K_MAX * n_dims]

    # Reshape to (K_MAX, n_dims), take first K components
    means = means_flat.reshape(K_MAX, n_dims)[:K]        # (K, d)
    stds = torch.exp(log_stds_flat.reshape(K_MAX, n_dims)[:K])  # (K, d)
    stds = stds.clamp(min=0.01, max=10.0)

    # Sample component assignments uniformly (equal weight per component)
    # Shape: (batch_size, n_points)
    assignments = torch.randint(0, K, (batch_size, n_points))

    # Sample from each component
    z = torch.randn(shape)
    for k in range(K):
        mask = (assignments == k).unsqueeze(-1).float()  # (batch, n_points, 1)
        component = z * stds[k].unsqueeze(0).unsqueeze(0) + means[k].unsqueeze(0).unsqueeze(0)
        z = z * (1 - mask) + component * mask

    return z

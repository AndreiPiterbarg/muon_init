"""Transform library for the compositional pipeline.

Each transform operates on tensors of shape (batch_size, n_points, n_dims)
and returns the same shape. Per-stage type selection is via hardmax (argmax)
of logits in the genome.
"""

import torch

TRANSFORM_TYPES = [
    "identity",
    "affine",
    "elementwise_power",
    "mixture_injection",
    "soft_clipping",
    "dimension_interaction",
    "feature_sparsity",
    "sorting_rank",
    "sign_flip",
]

# Interaction transform rank (low-rank cross-feature dependencies)
INTERACTION_RANK = 4


def stage_param_size(n_dims: int) -> int:
    """Shared param block per stage, sized for the largest transform (Affine).

    Affine needs d*(d+1)/2 + d params.
    For d=20: 210 + 20 = 230.
    """
    return n_dims * (n_dims + 1) // 2 + n_dims


def apply_transform(
    transform_index: int,
    x: torch.Tensor,
    params: torch.Tensor,
    n_dims: int,
) -> torch.Tensor:
    """Apply the selected transform to x.

    Args:
        transform_index: Index into TRANSFORM_TYPES (0-8).
        x: Input tensor, shape (batch_size, n_points, n_dims).
        params: Flat parameter tensor (size = stage_param_size(n_dims)).
        n_dims: Dimensionality.

    Returns:
        Transformed tensor, same shape as x.
    """
    if transform_index == 0:
        return x  # identity
    elif transform_index == 1:
        return _apply_affine(x, params, n_dims)
    elif transform_index == 2:
        return _apply_elementwise_power(x, params, n_dims)
    elif transform_index == 3:
        return _apply_mixture_injection(x, params, n_dims)
    elif transform_index == 4:
        return _apply_soft_clipping(x, params, n_dims)
    elif transform_index == 5:
        return _apply_dimension_interaction(x, params, n_dims)
    elif transform_index == 6:
        return _apply_feature_sparsity(x, params, n_dims)
    elif transform_index == 7:
        return _apply_sorting_rank(x, n_dims)
    elif transform_index == 8:
        return _apply_sign_flip(x, params, n_dims)
    else:
        raise ValueError(f"Unknown transform index: {transform_index}")


# --- Helper: Cholesky decoding (shared with genome.py logic) ---


def _flat_to_lower_triangular(flat: torch.Tensor, d: int) -> torch.Tensor:
    """Convert flat vector to d x d lower-triangular matrix.
    Diagonal elements are exponentiated (stored in log-space)."""
    L = torch.zeros(d, d, device=flat.device, dtype=flat.dtype)
    idx = 0
    for i in range(d):
        for j in range(i + 1):
            if i == j:
                L[i, j] = torch.exp(flat[idx])
            else:
                L[i, j] = flat[idx]
            idx += 1
    return L


# --- Transform implementations ---


def _apply_affine(x: torch.Tensor, params: torch.Tensor, d: int) -> torch.Tensor:
    """x_out = x @ L + b. Cholesky rotation + bias."""
    tri = d * (d + 1) // 2
    L = _flat_to_lower_triangular(params[:tri], d)
    b = params[tri: tri + d]
    # x: (batch, n_points, d), L: (d, d), b: (d,)
    return x @ L + b.unsqueeze(0).unsqueeze(0)


def _apply_elementwise_power(
    x: torch.Tensor, params: torch.Tensor, d: int
) -> torch.Tensor:
    """sign(x) * |x|^alpha per dimension. Reshapes tails."""
    alpha = params[:d].clamp(0.1, 5.0)
    abs_x = x.abs() + 1e-8  # avoid 0^alpha gradient issues
    return x.sign() * abs_x.pow(alpha.unsqueeze(0).unsqueeze(0))


def _apply_mixture_injection(
    x: torch.Tensor, params: torch.Tensor, d: int
) -> torch.Tensor:
    """With prob p, replace sample with draw from N(mu2, diag(sigma2))."""
    p = torch.sigmoid(params[0])
    mu2 = params[1: d + 1]
    log_sigma2 = params[d + 1: 2 * d + 1]
    sigma2 = torch.exp(log_sigma2).clamp(min=0.01, max=10.0)

    batch_size, n_points, _ = x.shape
    # Bernoulli mask: per-sample (not per-point, to maintain structure within sequence)
    # Actually per-point is more expressive for contamination
    mask = torch.bernoulli(p.expand(batch_size, n_points)).unsqueeze(-1)  # (batch, n_points, 1)

    replacement = torch.randn_like(x) * sigma2.unsqueeze(0).unsqueeze(0) + mu2.unsqueeze(0).unsqueeze(0)
    return x * (1 - mask) + replacement * mask


def _apply_soft_clipping(
    x: torch.Tensor, params: torch.Tensor, d: int
) -> torch.Tensor:
    """tanh(x * beta) / tanh(beta). Compresses tails, bounded support."""
    beta = params[:d].clamp(0.1, 10.0)
    beta_exp = beta.unsqueeze(0).unsqueeze(0)  # (1, 1, d)
    tanh_scale = torch.tanh(beta_exp)
    # Avoid division by near-zero (beta very small -> tanh(beta) ~ beta)
    tanh_scale = tanh_scale.clamp(min=1e-6)
    return torch.tanh(x * beta_exp) / tanh_scale


def _apply_dimension_interaction(
    x: torch.Tensor, params: torch.Tensor, d: int
) -> torch.Tensor:
    """x + (x @ W @ W^T) * x. Nonlinear cross-feature dependencies."""
    rank = min(INTERACTION_RANK, d)
    W = params[: d * rank].reshape(d, rank)
    # x @ W @ W^T gives a linear projection, element-wise multiply with x
    # adds nonlinear cross-feature coupling
    projected = x @ W  # (batch, n_points, rank)
    back_proj = projected @ W.T  # (batch, n_points, d)
    return x + back_proj * x


def _apply_feature_sparsity(
    x: torch.Tensor, params: torch.Tensor, d: int
) -> torch.Tensor:
    """Soft feature masking via sigmoid."""
    temperature = 10.0
    mask = torch.sigmoid(params[:d] * temperature)  # (d,)
    return x * mask.unsqueeze(0).unsqueeze(0)


def _apply_sorting_rank(x: torch.Tensor, d: int) -> torch.Tensor:
    """Replace values with normalized ranks per dimension across n_points.

    Destroys distributional shape while preserving rank order.
    """
    batch_size, n_points, _ = x.shape
    # Sort along n_points axis, get ranks
    _, indices = x.sort(dim=1)
    ranks = torch.zeros_like(x)
    # Scatter ranks
    batch_idx = torch.arange(batch_size, device=x.device).unsqueeze(1).unsqueeze(2).expand_as(indices)
    dim_idx = torch.arange(d, device=x.device).unsqueeze(0).unsqueeze(0).expand_as(indices)
    rank_values = torch.arange(n_points, device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(2).expand_as(indices)
    ranks[batch_idx, indices, dim_idx] = rank_values
    # Normalize ranks to roughly N(0,1)-ish range
    ranks = (ranks - n_points / 2.0) / (n_points / 4.0)
    return ranks


def _apply_sign_flip(
    x: torch.Tensor, params: torch.Tensor, d: int
) -> torch.Tensor:
    """Random sign corruption per dimension per batch.

    Flip probability per dimension is sigmoid(params). The flip is constant
    across n_points within a batch to maintain consistent sign structure.
    """
    flip_prob = torch.sigmoid(params[:d])  # (d,)
    batch_size = x.shape[0]
    # Draw one flip decision per batch per dimension (constant across n_points)
    flips = torch.bernoulli(flip_prob.unsqueeze(0).expand(batch_size, -1))  # (batch, d)
    signs = 2.0 * flips - 1.0  # -1 or +1
    return x * signs.unsqueeze(1)  # broadcast over n_points

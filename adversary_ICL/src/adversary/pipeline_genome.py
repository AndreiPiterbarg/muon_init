"""Pipeline genome: flat vector encoding a compositional transform pipeline.

Replaces the original Genome class with a much more expressive representation:
    z ~ base_dist -> [stage 1] -> ... -> [stage N] -> normalize -> x

The adversary controls the base distribution, N transform stages, task weight,
and noise level. Unused stages default to identity (pass-through), and
CMA-ES adapts step sizes to near-zero on inert dimensions.

Genome layout (d=20, N=8):
    base_logits:       8     (one per base distribution)
    base_params:     163     (sized for mixture of Gaussians)
    stage_0 logits:    9     (one per transform type)
    stage_0 params:  230     (sized for Affine)
    ...
    stage_7 logits:    9
    stage_7 params:  230
    w:                20     (unit-normalized task weight)
    noise_log_sigma:   1
    Total:         2,104
"""

import numpy as np
import torch

from .distributions import (
    BASE_DISTRIBUTIONS,
    base_param_size,
    sample_base,
)
from .transforms import (
    TRANSFORM_TYPES,
    stage_param_size,
    apply_transform,
    _flat_to_lower_triangular,
)


class PipelineGenome:
    """Flat real-valued vector encoding a compositional adversarial pipeline."""

    N_STAGES = 8
    N_BASE_DISTS = len(BASE_DISTRIBUTIONS)   # 8
    N_TRANSFORMS = len(TRANSFORM_TYPES)      # 9

    NOISE_LOG_MIN = -5.0
    NOISE_LOG_MAX = 2.0
    L_DIAG_LOG_MIN = -5.0
    L_DIAG_LOG_MAX = 5.0

    def __init__(self, n_dims: int, raw: np.ndarray | None = None):
        self.n_dims = n_dims
        self._base_param_sz = base_param_size(n_dims)
        self._stage_param_sz = stage_param_size(n_dims)
        self._layout = self._compute_layout(n_dims)
        self._size = self._layout["total"]

        if raw is not None:
            assert raw.shape == (self._size,), f"Expected {self._size}, got {raw.shape}"
            self.raw = raw.copy()
        else:
            self.raw = np.zeros(self._size)

    # --- Layout ---

    def _compute_layout(self, d: int) -> dict:
        blocks = {}
        offset = 0

        # Base distribution
        blocks["base_logits"] = (offset, offset + self.N_BASE_DISTS)
        offset += self.N_BASE_DISTS

        blocks["base_params"] = (offset, offset + self._base_param_sz)
        offset += self._base_param_sz

        # Stages
        for s in range(self.N_STAGES):
            blocks[f"stage_{s}_logits"] = (offset, offset + self.N_TRANSFORMS)
            offset += self.N_TRANSFORMS
            blocks[f"stage_{s}_params"] = (offset, offset + self._stage_param_sz)
            offset += self._stage_param_sz

        # Task weight and noise
        blocks["w"] = (offset, offset + d)
        offset += d
        blocks["noise_log_sigma"] = (offset, offset + 1)
        offset += 1

        blocks["total"] = offset
        return blocks

    @staticmethod
    def flat_size(n_dims: int) -> int:
        n_base_dists = len(BASE_DISTRIBUTIONS)
        n_transforms = len(TRANSFORM_TYPES)
        bp = base_param_size(n_dims)
        sp = stage_param_size(n_dims)
        n_stages = 8
        return (
            n_base_dists + bp
            + n_stages * (n_transforms + sp)
            + n_dims + 1
        )

    def _get_block(self, name: str) -> np.ndarray:
        start, end = self._layout[name]
        return self.raw[start:end]

    def _set_block(self, name: str, values: np.ndarray):
        start, end = self._layout[name]
        self.raw[start:end] = values

    # --- Decoding ---

    def decode_base_distribution(self) -> int:
        """Index of the selected base distribution (argmax of logits)."""
        logits = self._get_block("base_logits")
        return int(np.argmax(logits))

    def decode_base_params(self) -> torch.Tensor:
        """Raw base distribution parameter block."""
        return torch.tensor(self._get_block("base_params"), dtype=torch.float32)

    def decode_stage(self, stage_idx: int) -> tuple[int, torch.Tensor]:
        """Decode a pipeline stage: (transform_index, param_tensor)."""
        logits = self._get_block(f"stage_{stage_idx}_logits")
        transform_idx = int(np.argmax(logits))
        params = torch.tensor(
            self._get_block(f"stage_{stage_idx}_params"), dtype=torch.float32
        )
        return transform_idx, params

    def decode_weights(self) -> torch.Tensor:
        """Unit-normalized task weight vector."""
        w_raw = torch.tensor(self._get_block("w"), dtype=torch.float32)
        norm = torch.linalg.norm(w_raw)
        if norm > 1e-10:
            return w_raw / norm
        w_default = torch.zeros(self.n_dims)
        w_default[0] = 1.0
        return w_default

    def decode_noise_std(self) -> float:
        """Noise standard deviation from log-space."""
        return float(np.exp(np.clip(
            self._get_block("noise_log_sigma")[0],
            self.NOISE_LOG_MIN, self.NOISE_LOG_MAX,
        )))

    # --- Sampling (full pipeline) ---

    def sample_xs(self, n_points: int, batch_size: int) -> torch.Tensor:
        """Run the full pipeline: base -> transforms -> normalize -> x.

        Returns:
            Tensor of shape (batch_size, n_points, n_dims).
        """
        base_idx = self.decode_base_distribution()
        base_params = self.decode_base_params()

        # Sample from base distribution
        z = sample_base(base_idx, n_points, batch_size, self.n_dims, base_params)

        # Apply pipeline stages
        for s in range(self.N_STAGES):
            transform_idx, stage_params = self.decode_stage(s)
            if transform_idx == 0:
                continue  # identity, skip for efficiency
            z = apply_transform(transform_idx, z, stage_params, self.n_dims)
            # NaN guard: if any stage produces NaN, return NaN tensor
            # (the evaluator's NaN check will catch this and return fitness=0)
            if torch.isnan(z).any() or torch.isinf(z).any():
                return z

        # Trace normalization: x *= sqrt(d / tr(cov_empirical))
        z = self._trace_normalize(z)
        return z

    @staticmethod
    def _trace_normalize(x: torch.Tensor) -> torch.Tensor:
        """Normalize so empirical tr(cov) = n_dims per batch element."""
        d = x.shape[-1]
        # Centered data
        x_centered = x - x.mean(dim=1, keepdim=True)
        # Empirical covariance trace = sum of per-dim variances
        # More efficient than computing full covariance matrix
        var_per_dim = (x_centered ** 2).mean(dim=1)  # (batch, d)
        trace = var_per_dim.sum(dim=-1, keepdim=True).unsqueeze(1)  # (batch, 1, 1)
        scale = torch.sqrt(d / (trace + 1e-8))
        return x * scale

    # --- Constraint enforcement ---

    def clamp_(self):
        """In-place: clamp only what's needed to avoid NaN."""
        # Clamp noise log-sigma
        noise = self._get_block("noise_log_sigma")
        noise[0] = np.clip(noise[0], self.NOISE_LOG_MIN, self.NOISE_LOG_MAX)
        self._set_block("noise_log_sigma", noise)

        # Clamp affine diagonal entries in log-space (for any active affine stage)
        d = self.n_dims
        for s in range(self.N_STAGES):
            logits = self._get_block(f"stage_{s}_logits")
            if int(np.argmax(logits)) == 1:  # affine
                params = self._get_block(f"stage_{s}_params")
                idx = 0
                for i in range(d):
                    for j in range(i + 1):
                        if i == j:
                            params[idx] = np.clip(
                                params[idx], self.L_DIAG_LOG_MIN, self.L_DIAG_LOG_MAX
                            )
                        idx += 1
                self._set_block(f"stage_{s}_params", params)

    # --- Factories ---

    @classmethod
    def identity(cls, n_dims: int) -> "PipelineGenome":
        """Baseline: standard Gaussian, all stages identity, w=e_1, low noise."""
        g = cls(n_dims)

        # Base logits: strongly prefer Gaussian (index 0)
        base_logits = np.zeros(cls.N_BASE_DISTS)
        base_logits[0] = 10.0
        g._set_block("base_logits", base_logits)

        # All stage logits: strongly prefer Identity (index 0)
        for s in range(cls.N_STAGES):
            stage_logits = np.zeros(cls.N_TRANSFORMS)
            stage_logits[0] = 10.0
            g._set_block(f"stage_{s}_logits", stage_logits)

        # w = first basis vector
        w = np.zeros(n_dims)
        w[0] = 1.0
        g._set_block("w", w)

        # noise_log_sigma = log(0.1)
        g._set_block("noise_log_sigma", np.array([np.log(0.1)]))

        return g

    @classmethod
    def random(cls, n_dims: int, rng: np.random.Generator | None = None) -> "PipelineGenome":
        """Random initialization with small noise."""
        if rng is None:
            rng = np.random.default_rng()
        g = cls(n_dims)
        g.raw = rng.standard_normal(g._size) * 0.1
        g.clamp_()
        return g

    @classmethod
    def random_structured(
        cls, n_dims: int, rng: np.random.Generator | None = None
    ) -> "PipelineGenome":
        """Structured random init: random base (biased Gaussian), 1-3 active stages."""
        if rng is None:
            rng = np.random.default_rng()
        g = cls(n_dims)

        # Small background noise on everything
        g.raw = rng.standard_normal(g._size) * 0.1

        # Base logits: bias toward Gaussian but allow others
        base_logits = rng.standard_normal(cls.N_BASE_DISTS) * 0.5
        base_logits[0] += 2.0  # Gaussian bias
        g._set_block("base_logits", base_logits)

        # Randomly activate 1-3 stages
        n_active = rng.integers(1, 4)  # 1, 2, or 3
        active_stages = rng.choice(cls.N_STAGES, size=n_active, replace=False)

        for s in range(cls.N_STAGES):
            stage_logits = rng.standard_normal(cls.N_TRANSFORMS) * 0.3
            if s in active_stages:
                # Pick a random non-identity transform for this stage
                transform_idx = rng.integers(1, cls.N_TRANSFORMS)
                stage_logits[transform_idx] += 3.0
            else:
                # Bias toward identity
                stage_logits[0] += 5.0
            g._set_block(f"stage_{s}_logits", stage_logits)

        # Random weight direction
        g._set_block("w", rng.standard_normal(n_dims))

        # Random noise level
        g._set_block("noise_log_sigma", np.array([rng.uniform(-2, 0)]))

        g.clamp_()
        return g

    # --- Utilities ---

    def copy(self) -> "PipelineGenome":
        return PipelineGenome(self.n_dims, self.raw.copy())

    def active_stages(self) -> list[tuple[int, str]]:
        """List of (stage_index, transform_name) for non-identity stages."""
        result = []
        for s in range(self.N_STAGES):
            idx, _ = self.decode_stage(s)
            if idx != 0:
                result.append((s, TRANSFORM_TYPES[idx]))
        return result

    def num_active_stages(self) -> int:
        return len(self.active_stages())

    def base_distribution_name(self) -> str:
        return BASE_DISTRIBUTIONS[self.decode_base_distribution()]

    def complexity(
        self, c_base: float = 1.0, c_stage: float = 1.0, c_affine: float = 1.0
    ) -> float:
        """Compute complexity score for parsimony pressure."""
        comp = 0.0

        # Non-Gaussian base
        if self.decode_base_distribution() != 0:
            comp += c_base

        # Active stages
        for s in range(self.N_STAGES):
            idx, params = self.decode_stage(s)
            if idx != 0:
                comp += c_stage
            # Affine deviation from identity
            if idx == 1:
                d = self.n_dims
                L = _flat_to_lower_triangular(params[: d * (d + 1) // 2], d)
                I = torch.eye(d)
                comp += c_affine * torch.norm(L - I, p="fro").item() / d

        return comp

    def describe(self) -> str:
        """Human-readable summary of the pipeline."""
        parts = [self.base_distribution_name()]
        for s, name in self.active_stages():
            parts.append(name)
        pipeline = " -> ".join(parts)
        return (
            f"PipelineGenome(d={self.n_dims}, "
            f"pipeline=[{pipeline}], "
            f"active={self.num_active_stages()}, "
            f"noise={self.decode_noise_std():.3f})"
        )

    def eigenvalues(self) -> np.ndarray:
        """Eigenvalues of empirical covariance from a sample batch (sorted desc).

        Since the pipeline is nonlinear, covariance must be estimated empirically.
        """
        with torch.no_grad():
            xs = self.sample_xs(n_points=200, batch_size=1)
        if torch.isnan(xs).any():
            return np.ones(self.n_dims)
        xs_np = xs[0].numpy()  # (200, d)
        cov = np.cov(xs_np, rowvar=False)
        eigvals = np.linalg.eigvalsh(cov)[::-1]
        return eigvals

    def condition_number(self) -> float:
        eigvals = self.eigenvalues()
        return float(eigvals[0] / max(eigvals[-1], 1e-10))

    def effective_rank(self) -> float:
        eigvals = self.eigenvalues()
        return float(np.sum(eigvals) ** 2 / np.sum(eigvals ** 2))

    def __repr__(self):
        return self.describe()

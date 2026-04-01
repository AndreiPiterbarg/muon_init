import numpy as np
import torch


class Genome:
    """Flat real-valued vector encoding a complete adversarial configuration.

    Encodes train/test covariance (via Cholesky), means, task weights, and noise.
    The Cholesky parameterization guarantees PSD covariances by construction:
    diagonal elements are stored in log-space, so exp(diag) is always positive.

    Parameter layout for n_dims=d:
        L_train:  d*(d+1)/2  (lower-triangular Cholesky factor, diag in log-space)
        mu_train: d
        L_test:   d*(d+1)/2
        mu_test:  d
        w:        d          (task weight vector)
        noise_log_sigma: 1   (log noise std)
    """

    # Clamp bounds (only to avoid NaN, not to impose structure)
    L_DIAG_LOG_MIN = -5.0
    L_DIAG_LOG_MAX = 5.0
    NOISE_LOG_MIN = -5.0
    NOISE_LOG_MAX = 2.0

    def __init__(self, n_dims: int, raw: np.ndarray | None = None):
        self.n_dims = n_dims
        self._layout = self._compute_layout(n_dims)
        self._size = self._layout["total"]

        if raw is not None:
            assert raw.shape == (self._size,), f"Expected {self._size}, got {raw.shape}"
            self.raw = raw.copy()
        else:
            self.raw = np.zeros(self._size)

    @staticmethod
    def _compute_layout(d: int) -> dict:
        tri = d * (d + 1) // 2
        blocks = {}
        offset = 0
        for name, size in [
            ("L_train", tri),
            ("mu_train", d),
            ("L_test", tri),
            ("mu_test", d),
            ("w", d),
            ("noise_log_sigma", 1),
        ]:
            blocks[name] = (offset, offset + size)
            offset += size
        blocks["total"] = offset
        return blocks

    @staticmethod
    def flat_size(n_dims: int) -> int:
        tri = n_dims * (n_dims + 1) // 2
        return tri * 2 + n_dims * 4 + 1

    def _get_block(self, name: str) -> np.ndarray:
        start, end = self._layout[name]
        return self.raw[start:end]

    def _set_block(self, name: str, values: np.ndarray):
        start, end = self._layout[name]
        self.raw[start:end] = values

    # --- Decoding ---

    def _flat_to_lower_triangular(self, flat: np.ndarray) -> torch.Tensor:
        """Convert flat vector to d x d lower-triangular matrix.
        Diagonal elements are exponentiated (stored in log-space)."""
        d = self.n_dims
        L = torch.zeros(d, d)
        idx = 0
        for i in range(d):
            for j in range(i + 1):
                if i == j:
                    L[i, j] = torch.exp(torch.tensor(flat[idx]))
                else:
                    L[i, j] = flat[idx]
                idx += 1
        return L

    def decode_L(self, block: str) -> torch.Tensor:
        """Decode a Cholesky factor block. Returns d x d lower-triangular."""
        return self._flat_to_lower_triangular(self._get_block(block))

    def decode_covariance(self, block: str) -> torch.Tensor:
        """Decode covariance matrix Sigma = L @ L^T. Always PSD."""
        L = self.decode_L(block)
        return L @ L.T

    def decode_mu(self, block: str) -> torch.Tensor:
        return torch.tensor(self._get_block(block), dtype=torch.float32)

    def decode_weights(self) -> torch.Tensor:
        return torch.tensor(self._get_block("w"), dtype=torch.float32)

    def decode_noise_std(self) -> float:
        return float(np.exp(np.clip(
            self._get_block("noise_log_sigma")[0],
            self.NOISE_LOG_MIN, self.NOISE_LOG_MAX
        )))

    # --- Constraint enforcement (minimal) ---

    def clamp_(self):
        """In-place: clamp only what's needed to avoid NaN."""
        d = self.n_dims
        for block_name in ["L_train", "L_test"]:
            flat = self._get_block(block_name)
            idx = 0
            for i in range(d):
                for j in range(i + 1):
                    if i == j:
                        flat[idx] = np.clip(flat[idx], self.L_DIAG_LOG_MIN, self.L_DIAG_LOG_MAX)
                    idx += 1
            self._set_block(block_name, flat)

        noise = self._get_block("noise_log_sigma")
        noise[0] = np.clip(noise[0], self.NOISE_LOG_MIN, self.NOISE_LOG_MAX)
        self._set_block("noise_log_sigma", noise)

    # --- Factories ---

    @classmethod
    def identity(cls, n_dims: int) -> "Genome":
        """Isotropic Gaussian baseline: Sigma=I, mu=0, w~N(0,1), low noise."""
        g = cls(n_dims)
        d = n_dims
        # L = I means log-diagonal = 0, off-diagonal = 0 (already zero-initialized)
        # mu = 0 (already zero)
        # w = random unit vector
        rng = np.random.default_rng(42)
        w = rng.standard_normal(d)
        w = w / np.linalg.norm(w)
        g._set_block("w", w)
        # noise_log_sigma = log(0.1) for small noise
        g._set_block("noise_log_sigma", np.array([np.log(0.1)]))
        return g

    @classmethod
    def random(cls, n_dims: int, rng: np.random.Generator | None = None) -> "Genome":
        """Random initialization with reasonable scale."""
        if rng is None:
            rng = np.random.default_rng()
        g = cls(n_dims)
        g.raw = rng.standard_normal(g._size) * 0.5
        g.clamp_()
        return g

    @classmethod
    def random_structured(cls, n_dims: int, rng: np.random.Generator | None = None) -> "Genome":
        """Random init biased toward more interesting regions.
        Log-uniform eigenvalues, random orthogonal rotation."""
        if rng is None:
            rng = np.random.default_rng()
        g = cls(n_dims)
        d = n_dims

        for block_name in ["L_train", "L_test"]:
            # Random eigenvalues (log-uniform in [0.1, 10])
            log_eigvals = rng.uniform(-1, 1, size=d)
            # Random orthogonal rotation
            Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
            L_mat = Q @ np.diag(np.exp(log_eigvals * 0.5))  # sqrt of eigenvalues
            # Extract lower-triangular via Cholesky of Q diag Q^T
            Sigma = L_mat @ L_mat.T
            L_chol = np.linalg.cholesky(Sigma)
            # Encode: diagonal in log-space, off-diagonal raw
            flat = np.zeros(d * (d + 1) // 2)
            idx = 0
            for i in range(d):
                for j in range(i + 1):
                    if i == j:
                        flat[idx] = np.log(L_chol[i, j])
                    else:
                        flat[idx] = L_chol[i, j]
                    idx += 1
            g._set_block(block_name, flat)

        # Random means (small)
        g._set_block("mu_train", rng.standard_normal(d) * 0.5)
        g._set_block("mu_test", rng.standard_normal(d) * 0.5)

        # Random weights
        w = rng.standard_normal(d)
        g._set_block("w", w)

        # Random noise
        g._set_block("noise_log_sigma", np.array([rng.uniform(-2, 0)]))

        g.clamp_()
        return g

    # --- Utilities ---

    def copy(self) -> "Genome":
        return Genome(self.n_dims, self.raw.copy())

    def eigenvalues(self, block: str = "L_train") -> np.ndarray:
        """Eigenvalues of the decoded covariance (sorted descending)."""
        Sigma = self.decode_covariance(block).numpy()
        eigvals = np.linalg.eigvalsh(Sigma)[::-1]
        return eigvals

    def condition_number(self, block: str = "L_train") -> float:
        eigvals = self.eigenvalues(block)
        return float(eigvals[0] / max(eigvals[-1], 1e-10))

    def effective_rank(self, block: str = "L_train") -> float:
        eigvals = self.eigenvalues(block)
        return float(np.sum(eigvals) ** 2 / np.sum(eigvals ** 2))

    def __repr__(self):
        return (
            f"Genome(d={self.n_dims}, "
            f"cond_train={self.condition_number('L_train'):.1f}, "
            f"cond_test={self.condition_number('L_test'):.1f}, "
            f"noise={self.decode_noise_std():.3f})"
        )

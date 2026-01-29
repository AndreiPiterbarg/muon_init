"""
SPD Matrix Sampling for Linear System Experiments

Generates symmetric positive definite matrices with controlled condition numbers
for training and testing ICL models on linear system solving.
"""

import torch
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class SPDSampleConfig:
    """Configuration for SPD matrix sampling."""
    d: int = 4                    # Matrix dimension
    kappa_min: float = 1.0        # Minimum condition number
    kappa_max: float = 100.0      # Maximum condition number


class SPDSampler:
    """
    Samples symmetric positive definite matrices with controlled condition numbers.

    Condition number kappa = lambda_max / lambda_min controls problem difficulty.
    Higher kappa means more ill-conditioned (harder) problems.
    """

    def __init__(self, d: int, device: torch.device):
        self.d = d
        self.device = device

    def sample(
        self,
        batch_size: int,
        kappa_min: float = 1.0,
        kappa_max: float = 100.0
    ) -> torch.Tensor:
        """
        Sample SPD matrices with condition numbers in [kappa_min, kappa_max].

        Args:
            batch_size: Number of matrices to sample
            kappa_min: Minimum condition number (>= 1.0)
            kappa_max: Maximum condition number

        Returns:
            Tensor of shape (batch_size, d, d) containing SPD matrices
        """
        return sample_spd(
            batch_size, self.d, self.device, kappa_min, kappa_max
        )

    def sample_linear_system(
        self,
        batch_size: int,
        num_vectors: int,
        kappa_min: float = 1.0,
        kappa_max: float = 100.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample SPD matrix A and solve Ax = b for multiple b vectors.

        Args:
            batch_size: Number of systems
            num_vectors: Number of (b, x) pairs per system
            kappa_min: Minimum condition number
            kappa_max: Maximum condition number

        Returns:
            Tuple of (A, b, x) where:
            - A: (batch_size, d, d) SPD matrices
            - b: (batch_size, num_vectors, d) right-hand side vectors
            - x: (batch_size, num_vectors, d) solution vectors
        """
        A = self.sample(batch_size, kappa_min, kappa_max)
        b = torch.randn(batch_size, num_vectors, self.d, device=self.device)
        x = torch.linalg.solve(A, b.transpose(-2, -1)).transpose(-2, -1)
        return A, b, x


def sample_spd(
    batch_size: int,
    d: int,
    device: torch.device,
    kappa_min: float = 1.0,
    kappa_max: float = 100.0,
) -> torch.Tensor:
    """
    Sample SPD matrices with controlled condition numbers.

    Algorithm:
    1. Sample condition number kappa uniformly on log scale
    2. Sample eigenvalues uniformly on log scale in [1, kappa]
    3. Generate random orthogonal matrix Q via QR decomposition
    4. Construct A = Q @ diag(eigenvalues) @ Q^T

    Args:
        batch_size: Number of matrices to sample
        d: Matrix dimension
        device: PyTorch device
        kappa_min: Minimum condition number
        kappa_max: Maximum condition number

    Returns:
        Tensor of shape (batch_size, d, d) containing SPD matrices
    """
    log_min, log_max = np.log(kappa_min), np.log(kappa_max)

    # Sample condition numbers uniformly on log scale
    u = torch.rand(batch_size, device=device)
    kappas = torch.exp(torch.tensor(log_min, device=device) + u * (log_max - log_min))

    # Sample eigenvalues uniformly on log scale in [1, kappa]
    u_eigs = torch.rand(batch_size, d, device=device)
    eigs = torch.exp(u_eigs * kappas.unsqueeze(-1).log())

    # Generate random orthogonal matrices via QR
    G = torch.randn(batch_size, d, d, device=device)
    Q, _ = torch.linalg.qr(G)

    # Construct A = Q @ diag(eigs) @ Q^T
    A = Q @ torch.diag_embed(eigs) @ Q.transpose(-2, -1)

    # Ensure symmetry (numerical precision)
    return 0.5 * (A + A.transpose(-2, -1))


def compute_condition_number(A: torch.Tensor) -> torch.Tensor:
    """
    Compute condition number of matrices.

    Args:
        A: Tensor of shape (..., d, d)

    Returns:
        Tensor of condition numbers, shape (...)
    """
    eigenvalues = torch.linalg.eigvalsh(A)
    return eigenvalues[..., -1] / eigenvalues[..., 0]

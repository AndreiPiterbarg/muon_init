"""
Component embedders for the curriculum transformer.

This module provides three separate embedding layers that map different types
of mathematical components (vectors, matrices, scalars) to the transformer's
hidden dimension. These embedders are shared across all tasks.
"""

import torch
import torch.nn as nn


class VectorEmbedder(nn.Module):
    """
    Embeds vectors of dimension d to the transformer hidden dimension n_embd.

    A simple linear projection without bias (bias can be absorbed into role embeddings).
    """

    def __init__(self, d: int, n_embd: int):
        """
        Args:
            d: Input vector dimension
            n_embd: Transformer hidden dimension
        """
        super().__init__()
        self.d = d
        self.n_embd = n_embd
        self.linear = nn.Linear(d, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (..., d)

        Returns:
            Embedded tensor of shape (..., n_embd)
        """
        return self.linear(x)


class MatrixEmbedder(nn.Module):
    """
    Embeds matrices of dimension d x d to the transformer hidden dimension n_embd.

    Matrices are flattened row-major before embedding. No bias is used
    (bias can be absorbed into role embeddings).
    """

    def __init__(self, d: int, n_embd: int):
        """
        Args:
            d: Matrix dimension (assumes square d x d matrices)
            n_embd: Transformer hidden dimension
        """
        super().__init__()
        self.d = d
        self.n_embd = n_embd
        self.linear = nn.Linear(d * d, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (..., d, d)

        Returns:
            Embedded tensor of shape (..., n_embd)
        """
        # Flatten the last two dimensions (row-major)
        batch_shape = x.shape[:-2]
        x_flat = x.reshape(*batch_shape, self.d * self.d)
        return self.linear(x_flat)


class ScalarEmbedder(nn.Module):
    """
    Embeds scalars to the transformer hidden dimension n_embd.

    Scalars are treated as 1D tensors. No bias is used
    (bias can be absorbed into role embeddings).
    """

    def __init__(self, n_embd: int):
        """
        Args:
            n_embd: Transformer hidden dimension
        """
        super().__init__()
        self.n_embd = n_embd
        self.linear = nn.Linear(1, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (...,) or (..., 1)

        Returns:
            Embedded tensor of shape (..., n_embd)
        """
        # Ensure x has at least one dimension for the linear layer
        if x.dim() == 0:
            x = x.unsqueeze(0)
        if x.shape[-1] != 1:
            x = x.unsqueeze(-1)
        return self.linear(x)


class ComponentEmbedders(nn.Module):
    """
    Container module holding all three component embedders.

    This provides a convenient way to create and manage all embedders together,
    ensuring they share the same n_embd and can be easily passed around.
    """

    def __init__(self, d: int, n_embd: int):
        """
        Args:
            d: Dimension for vectors and matrices (matrices are d x d)
            n_embd: Transformer hidden dimension
        """
        super().__init__()
        self.d = d
        self.n_embd = n_embd

        self.vector = VectorEmbedder(d, n_embd)
        self.matrix = MatrixEmbedder(d, n_embd)
        self.scalar = ScalarEmbedder(n_embd)

    def embed_vector(self, x: torch.Tensor) -> torch.Tensor:
        """Embed a vector component."""
        return self.vector(x)

    def embed_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Embed a matrix component."""
        return self.matrix(x)

    def embed_scalar(self, x: torch.Tensor) -> torch.Tensor:
        """Embed a scalar component."""
        return self.scalar(x)

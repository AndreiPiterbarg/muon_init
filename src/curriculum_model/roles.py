"""
Role embeddings for the curriculum transformer.

This module defines semantic role embeddings that indicate the function of each
component within a task. Token construction uses additive composition:
    token = embed_component(component) + embed_role(role_index)

The same component (e.g., a vector) can have different roles in different contexts,
and the role embedding captures this semantic information.
"""

import torch
import torch.nn as nn
from enum import IntEnum
from typing import Union


class Role(IntEnum):
    """
    Enumeration of semantic roles for components.

    These roles indicate the function of a component within a task:
    - MATRIX: The operator matrix (e.g., A in Ax = b)
    - VEC_PRIMARY: Primary vector operand (e.g., x in x + z)
    - VEC_SECONDARY: Secondary vector operand (e.g., z in x + z)
    - VEC_BIAS: Bias vector (e.g., b in Ax - b)
    - SCALAR: Scalar value (e.g., α in αx)
    - OUTPUT: The output/target position
    """
    MATRIX = 0
    VEC_PRIMARY = 1
    VEC_SECONDARY = 2
    VEC_BIAS = 3
    SCALAR = 4
    OUTPUT = 5


# Total number of roles
NUM_ROLES = len(Role)


class RoleEmbedding(nn.Module):
    """
    Embedding layer for semantic roles.

    Maps role indices to learned embeddings of dimension n_embd.
    These embeddings are added to component embeddings to create
    the final token representations.
    """

    def __init__(self, n_embd: int):
        """
        Args:
            n_embd: Embedding dimension (must match component embedders)
        """
        super().__init__()
        self.n_embd = n_embd
        self.num_roles = NUM_ROLES
        self.embedding = nn.Embedding(NUM_ROLES, n_embd)

    def forward(self, role_indices: torch.Tensor) -> torch.Tensor:
        """
        Get role embeddings for given indices.

        Args:
            role_indices: Tensor of role indices (values from Role enum)
                         Shape can be any: (), (N,), (B, N), etc.

        Returns:
            Role embeddings with shape (*role_indices.shape, n_embd)
        """
        return self.embedding(role_indices)

    def get_role(self, role: Union[Role, int]) -> torch.Tensor:
        """
        Get embedding for a single role.

        Args:
            role: Role enum value or integer index

        Returns:
            Role embedding of shape (n_embd,)
        """
        idx = torch.tensor(int(role), device=self.embedding.weight.device)
        return self.embedding(idx)

    def get_matrix_role(self) -> torch.Tensor:
        """Get embedding for MATRIX role."""
        return self.get_role(Role.MATRIX)

    def get_primary_role(self) -> torch.Tensor:
        """Get embedding for VEC_PRIMARY role."""
        return self.get_role(Role.VEC_PRIMARY)

    def get_secondary_role(self) -> torch.Tensor:
        """Get embedding for VEC_SECONDARY role."""
        return self.get_role(Role.VEC_SECONDARY)

    def get_bias_role(self) -> torch.Tensor:
        """Get embedding for VEC_BIAS role."""
        return self.get_role(Role.VEC_BIAS)

    def get_scalar_role(self) -> torch.Tensor:
        """Get embedding for SCALAR role."""
        return self.get_role(Role.SCALAR)

    def get_output_role(self) -> torch.Tensor:
        """Get embedding for OUTPUT role."""
        return self.get_role(Role.OUTPUT)


def compose_token(
    component_embedding: torch.Tensor,
    role_embedding: torch.Tensor
) -> torch.Tensor:
    """
    Compose a token from component and role embeddings using addition.

    This is the standard additive composition used throughout the model:
        token = embed_component(component) + embed_role(role)

    Args:
        component_embedding: Embedded component of shape (..., n_embd)
        role_embedding: Role embedding of shape (n_embd,) or (..., n_embd)

    Returns:
        Composed token of shape (..., n_embd)
    """
    return component_embedding + role_embedding

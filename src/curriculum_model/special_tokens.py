"""
Special tokens for the curriculum transformer.

This module defines learned special tokens that serve structural purposes
in the sequence:
- SEP: Marks boundaries between examples in the context
- MASK: Marks the query output position (where prediction happens)

These tokens are complete on their own and do not need role embeddings added.
"""

import torch
import torch.nn as nn
import math


class SpecialTokens(nn.Module):
    """
    Container for special learned tokens.

    Provides SEP and MASK tokens as learned parameters that can be
    used directly in sequence construction.
    """

    def __init__(self, n_embd: int, init_std: float = 0.02):
        """
        Args:
            n_embd: Embedding dimension (must match other embeddings)
            init_std: Standard deviation for initialization (default 0.02,
                     matching typical transformer initialization)
        """
        super().__init__()
        self.n_embd = n_embd

        # SEP token marks example boundaries
        self.sep = nn.Parameter(torch.empty(n_embd))

        # MASK token marks the query output position
        self.mask = nn.Parameter(torch.empty(n_embd))

        # Initialize parameters
        self._init_parameters(init_std)

    def _init_parameters(self, std: float):
        """Initialize special token parameters with normal distribution."""
        nn.init.normal_(self.sep, mean=0.0, std=std)
        nn.init.normal_(self.mask, mean=0.0, std=std)

    def get_sep(self) -> torch.Tensor:
        """
        Get the SEP token embedding.

        Returns:
            SEP token of shape (n_embd,)
        """
        return self.sep

    def get_mask(self) -> torch.Tensor:
        """
        Get the MASK token embedding.

        Returns:
            MASK token of shape (n_embd,)
        """
        return self.mask

    def get_sep_batch(self, batch_size: int) -> torch.Tensor:
        """
        Get SEP token expanded for a batch.

        Args:
            batch_size: Number of sequences in the batch

        Returns:
            SEP tokens of shape (batch_size, n_embd)
        """
        return self.sep.unsqueeze(0).expand(batch_size, -1)

    def get_mask_batch(self, batch_size: int) -> torch.Tensor:
        """
        Get MASK token expanded for a batch.

        Args:
            batch_size: Number of sequences in the batch

        Returns:
            MASK tokens of shape (batch_size, n_embd)
        """
        return self.mask.unsqueeze(0).expand(batch_size, -1)

    def get_sep_sequence(self, num_tokens: int) -> torch.Tensor:
        """
        Get multiple SEP tokens for a sequence.

        Useful when constructing sequences with multiple examples.

        Args:
            num_tokens: Number of SEP tokens needed

        Returns:
            SEP tokens of shape (num_tokens, n_embd)
        """
        return self.sep.unsqueeze(0).expand(num_tokens, -1)

    def get_mask_with_role(self, role_embedding: torch.Tensor) -> torch.Tensor:
        """
        Get MASK token with an added role embedding.

        While special tokens typically don't need roles, the MASK token
        at the query position may optionally have ROLE_OUTPUT added
        to indicate it represents an output position.

        Args:
            role_embedding: Role embedding to add, shape (n_embd,)

        Returns:
            MASK token with role, shape (n_embd,)
        """
        return self.mask + role_embedding

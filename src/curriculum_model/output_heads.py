"""
Output heads for the curriculum transformer.

This module provides dual output heads for the model:
- VectorHead: Projects from hidden dimension to vector output
- ScalarHead: Projects from hidden dimension to scalar output

Both heads are always computed during forward pass. The loss function
receives task metadata indicating which head's output to supervise.
This prevents task information leakage through head selection.
"""

from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn

from .tasks import OutputType


@dataclass
class OutputHeadResult:
    """Result from dual output heads."""
    vector_output: torch.Tensor  # (batch_size, d)
    scalar_output: torch.Tensor  # (batch_size, 1)


class VectorHead(nn.Module):
    """
    Output head for vector predictions.

    Projects from transformer hidden dimension to output vector dimension.
    """

    def __init__(self, n_embd: int, d: int, bias: bool = True):
        """
        Args:
            n_embd: Transformer hidden dimension (input)
            d: Output vector dimension
            bias: Whether to include bias in projection
        """
        super().__init__()
        self.n_embd = n_embd
        self.d = d
        self.linear = nn.Linear(n_embd, d, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project hidden state to vector output.

        Args:
            x: Hidden state of shape (..., n_embd)

        Returns:
            Vector output of shape (..., d)
        """
        return self.linear(x)


class ScalarHead(nn.Module):
    """
    Output head for scalar predictions.

    Projects from transformer hidden dimension to scalar output.
    """

    def __init__(self, n_embd: int, bias: bool = True):
        """
        Args:
            n_embd: Transformer hidden dimension (input)
            bias: Whether to include bias in projection
        """
        super().__init__()
        self.n_embd = n_embd
        self.linear = nn.Linear(n_embd, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project hidden state to scalar output.

        Args:
            x: Hidden state of shape (..., n_embd)

        Returns:
            Scalar output of shape (..., 1)
        """
        return self.linear(x)


class DualOutputHead(nn.Module):
    """
    Combined dual output head module.

    Contains both vector and scalar heads. Both are always computed
    during forward pass to prevent task information leakage.
    """

    def __init__(self, n_embd: int, d: int, bias: bool = True):
        """
        Args:
            n_embd: Transformer hidden dimension (input)
            d: Output vector dimension
            bias: Whether to include bias in projections
        """
        super().__init__()
        self.n_embd = n_embd
        self.d = d

        self.vector_head = VectorHead(n_embd, d, bias=bias)
        self.scalar_head = ScalarHead(n_embd, bias=bias)

    def forward(self, x: torch.Tensor) -> OutputHeadResult:
        """
        Compute both vector and scalar outputs.

        Args:
            x: Hidden state of shape (batch_size, n_embd)

        Returns:
            OutputHeadResult with vector_output (batch_size, d) and
            scalar_output (batch_size, 1)
        """
        vector_out = self.vector_head(x)
        scalar_out = self.scalar_head(x)

        return OutputHeadResult(
            vector_output=vector_out,
            scalar_output=scalar_out,
        )

    def get_output(
        self,
        x: torch.Tensor,
        output_type: OutputType,
    ) -> torch.Tensor:
        """
        Compute both outputs but return only the relevant one.

        Both heads are still computed (for consistent compute graph),
        but only the relevant output is returned.

        Args:
            x: Hidden state of shape (batch_size, n_embd)
            output_type: Which output to return

        Returns:
            Output tensor of shape (batch_size, d) or (batch_size, 1)
        """
        result = self.forward(x)

        if output_type == OutputType.VECTOR:
            return result.vector_output
        else:
            return result.scalar_output


def compute_task_loss(
    predictions: OutputHeadResult,
    target: torch.Tensor,
    output_type: OutputType,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute MSE loss for the appropriate output head.

    Only the relevant head's output is supervised; gradients don't
    flow through the unused head's output.

    Args:
        predictions: Output from DualOutputHead
        target: Target tensor
        output_type: Which output type to supervise
        reduction: Loss reduction method ("mean", "sum", "none")

    Returns:
        MSE loss tensor
    """
    if output_type == OutputType.VECTOR:
        pred = predictions.vector_output
    else:
        pred = predictions.scalar_output

    # Ensure target has same shape as prediction
    if pred.shape != target.shape:
        # Try to match shapes
        if output_type == OutputType.SCALAR and target.dim() == 1:
            target = target.unsqueeze(-1)

    loss = nn.functional.mse_loss(pred, target, reduction=reduction)
    return loss


def compute_loss_from_hidden(
    hidden: torch.Tensor,
    target: torch.Tensor,
    output_type: OutputType,
    output_head: DualOutputHead,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Convenience function to compute loss from hidden state.

    Args:
        hidden: Hidden state from transformer at MASK position
        target: Target tensor
        output_type: Which output type to supervise
        output_head: The dual output head module
        reduction: Loss reduction method

    Returns:
        MSE loss tensor
    """
    predictions = output_head(hidden)
    return compute_task_loss(predictions, target, output_type, reduction)


class OutputHeadWithLoss(nn.Module):
    """
    Output head module with integrated loss computation.

    This module wraps DualOutputHead and provides loss computation,
    making it convenient for training loops.
    """

    def __init__(self, n_embd: int, d: int, bias: bool = True):
        """
        Args:
            n_embd: Transformer hidden dimension
            d: Output vector dimension
            bias: Whether to include bias in projections
        """
        super().__init__()
        self.output_head = DualOutputHead(n_embd, d, bias=bias)

    def forward(
        self,
        hidden: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        output_type: Optional[OutputType] = None,
    ) -> OutputHeadResult:
        """
        Compute outputs from hidden state.

        Args:
            hidden: Hidden state from transformer
            target: Optional target for loss computation (unused in forward)
            output_type: Optional output type (unused in forward)

        Returns:
            OutputHeadResult with both outputs
        """
        return self.output_head(hidden)

    def compute_loss(
        self,
        hidden: torch.Tensor,
        target: torch.Tensor,
        output_type: OutputType,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Compute loss for the given output type.

        Args:
            hidden: Hidden state from transformer at MASK position
            target: Target tensor
            output_type: Which output to supervise
            reduction: Loss reduction method

        Returns:
            MSE loss tensor
        """
        predictions = self.output_head(hidden)
        return compute_task_loss(predictions, target, output_type, reduction)

    def predict(
        self,
        hidden: torch.Tensor,
        output_type: OutputType,
    ) -> torch.Tensor:
        """
        Get prediction for specific output type.

        Args:
            hidden: Hidden state
            output_type: Which output to return

        Returns:
            Prediction tensor
        """
        return self.output_head.get_output(hidden, output_type)

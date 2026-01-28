"""
ComponentTransformerModel for multi-task learning.

Integrates:
- Component embedders (vector, matrix, scalar)
- Role embedding layer
- SEP and MASK special tokens
- Transformer backbone
- Dual output heads (vector and scalar)
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
import torch
import torch.nn as nn

from .embedders import ComponentEmbedders
from .roles import RoleEmbedding
from .special_tokens import SpecialTokens
from .sequence_builder import SequenceBuilder, SequenceOutput, PositionalEncoder
from .output_heads import DualOutputHead, OutputHeadResult, compute_task_loss
from .tasks import OutputType, TaskSpec

# Import backbone from custom_transformer (which redirects to model/)
from custom_transformer import CustomGPTBackbone, TransformerConfig, GPTOutput


@dataclass
class ComponentModelConfig:
    """Configuration for ComponentTransformerModel."""

    # Dimensions
    d: int = 4
    n_embd: int = 128
    n_layer: int = 6
    n_head: int = 4
    n_positions: int = 128

    # Architecture options
    max_examples: int = 64
    dropout: float = 0.0
    output_bias: bool = True

    # Backbone options
    norm_type: str = "layernorm"
    ffn_type: str = "mlp"
    activation: str = "gelu"
    pre_norm: bool = True

    def to_transformer_config(self) -> TransformerConfig:
        """Convert to TransformerConfig for backbone."""
        return TransformerConfig(
            n_embd=self.n_embd,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_positions=self.n_positions,
            pos_encoding_type="none",  # We use our own positional encoding
            norm_type=self.norm_type,
            ffn_type=self.ffn_type,
            activation=self.activation,
            pre_norm=self.pre_norm,
            dropout=self.dropout,
        )


@dataclass
class ComponentModelOutput:
    """Output from ComponentTransformerModel."""

    vector_output: torch.Tensor  # (batch_size, d)
    scalar_output: torch.Tensor  # (batch_size, 1)
    hidden_at_mask: Optional[torch.Tensor] = None
    last_hidden_state: Optional[torch.Tensor] = None
    attention_maps: Optional[List[torch.Tensor]] = None


class ComponentTransformerModel(nn.Module):
    """Component-based transformer for multi-task learning."""

    def __init__(
        self,
        config: Optional[ComponentModelConfig] = None,
        embedders: Optional[ComponentEmbedders] = None,
        role_embedding: Optional[RoleEmbedding] = None,
        special_tokens: Optional[SpecialTokens] = None,
        backbone: Optional[nn.Module] = None,
        output_head: Optional[DualOutputHead] = None,
    ):
        super().__init__()

        if config is None:
            config = ComponentModelConfig()

        self.config = config
        self.d = config.d
        self.n_embd = config.n_embd

        # Component embedders
        self.embedders = embedders or ComponentEmbedders(config.d, config.n_embd)

        # Role embeddings
        self.role_embedding = role_embedding or RoleEmbedding(config.n_embd)

        # Special tokens
        self.special_tokens = special_tokens or SpecialTokens(config.n_embd)

        # Example-level positional encoder
        self.positional_encoder = PositionalEncoder(
            config.n_embd,
            max_examples=config.max_examples,
        )

        # Transformer backbone
        if backbone is not None:
            self.backbone = backbone
        else:
            transformer_config = config.to_transformer_config()
            self.backbone = CustomGPTBackbone(transformer_config)

        # Output heads
        self.output_head = output_head or DualOutputHead(
            config.n_embd,
            config.d,
            bias=config.output_bias,
        )

        # Sequence builder
        self.sequence_builder = SequenceBuilder(
            d=config.d,
            n_embd=config.n_embd,
            embedders=self.embedders,
            role_embedding=self.role_embedding,
            special_tokens=self.special_tokens,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        example_positions: torch.Tensor,
        mask_positions: torch.Tensor,
        return_hidden: bool = False,
        return_attention: bool = False,
    ) -> ComponentModelOutput:
        """
        Forward pass.

        Args:
            tokens: Pre-embedded tokens, shape (batch_size, seq_len, n_embd)
            example_positions: Example index per token, shape (batch_size, seq_len)
            mask_positions: MASK token position, shape (batch_size,)
            return_hidden: Whether to return hidden states
            return_attention: Whether to return attention maps

        Returns:
            ComponentModelOutput with predictions
        """
        batch_size = tokens.shape[0]

        # Add example-level positional encoding
        tokens_with_pos = self.positional_encoder(tokens, example_positions)

        # Pass through transformer backbone
        backbone_output = self.backbone(
            inputs_embeds=tokens_with_pos,
            return_attention=return_attention,
        )

        hidden_states = backbone_output.last_hidden_state

        # Extract hidden state at MASK positions
        batch_indices = torch.arange(batch_size, device=tokens.device)
        hidden_at_mask = hidden_states[batch_indices, mask_positions]

        # Compute outputs from both heads
        output_result: OutputHeadResult = self.output_head(hidden_at_mask)

        return ComponentModelOutput(
            vector_output=output_result.vector_output,
            scalar_output=output_result.scalar_output,
            hidden_at_mask=hidden_at_mask if return_hidden else None,
            last_hidden_state=hidden_states if return_hidden else None,
            attention_maps=backbone_output.attention_maps if return_attention else None,
        )

    def forward_from_sequence_output(
        self,
        seq_output: SequenceOutput,
        return_hidden: bool = False,
        return_attention: bool = False,
    ) -> ComponentModelOutput:
        """Forward from SequenceOutput."""
        return self.forward(
            tokens=seq_output.tokens,
            example_positions=seq_output.example_positions,
            mask_positions=seq_output.mask_positions,
            return_hidden=return_hidden,
            return_attention=return_attention,
        )

    def compute_loss(
        self,
        model_output: ComponentModelOutput,
        target: torch.Tensor,
        output_type: OutputType,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Compute loss for the appropriate output type."""
        result = OutputHeadResult(
            vector_output=model_output.vector_output,
            scalar_output=model_output.scalar_output,
        )
        return compute_task_loss(result, target, output_type, reduction)

    def predict(
        self,
        tokens: torch.Tensor,
        example_positions: torch.Tensor,
        mask_positions: torch.Tensor,
        output_type: OutputType,
    ) -> torch.Tensor:
        """Get prediction for specific output type."""
        output = self.forward(tokens, example_positions, mask_positions)
        if output_type == OutputType.VECTOR:
            return output.vector_output
        return output.scalar_output

    def build_and_forward(
        self,
        context_examples: List[Dict[str, torch.Tensor]],
        query_inputs: Dict[str, torch.Tensor],
        task_spec: TaskSpec,
        return_hidden: bool = False,
        return_attention: bool = False,
    ) -> Tuple[ComponentModelOutput, SequenceOutput]:
        """Build sequence and forward in one call."""
        seq_output = self.sequence_builder.build_sequence(
            context_examples, query_inputs, task_spec
        )
        model_output = self.forward_from_sequence_output(
            seq_output,
            return_hidden=return_hidden,
            return_attention=return_attention,
        )
        return model_output, seq_output


def create_model(
    d: int = 4,
    n_embd: int = 128,
    n_layer: int = 6,
    n_head: int = 4,
    **kwargs,
) -> ComponentTransformerModel:
    """Create a ComponentTransformerModel."""
    config = ComponentModelConfig(
        d=d,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        **kwargs,
    )
    return ComponentTransformerModel(config)

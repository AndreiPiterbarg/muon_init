# custom GPT class that matches GPT2Model interface
import torch.nn as nn
import torch
from dataclasses import dataclass
from typing import List, Optional

from custom_transformer.block import TransformerBlock, get_norm
from custom_transformer.positional import PositionalEncoding
from custom_transformer.config import TransformerConfig


@dataclass
class GPTOutput:
    """Output class to match GPT2Model interface."""
    last_hidden_state: torch.Tensor
    attention_maps: Optional[List[torch.Tensor]] = None  # each: (B, n_heads, T, T)




class CustomGPTBackbone(nn.Module):
    """
    Custom GPT backbone that matches GPT2Model interface.
    Takes inputs_embeds and returns object with .last_hidden_state
    """

    def __init__(self, config: TransformerConfig = None):
        super().__init__()
        if config is None:
            config = TransformerConfig()

        self.config = config
        self.n_embd = config.n_embd
        self.n_layer = config.n_layer
        self.n_head = config.n_head
        self.n_positions = config.n_positions

        # Positional encoding (only for learned, RoPE is handled in attention)
        self.pos_encoding = None
        if config.pos_encoding_type == "learned":
            self.pos_encoding = PositionalEncoding(config.n_embd, max_len=config.n_positions)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config.n_embd, config.n_head, config=config)
            for _ in range(config.n_layer)
        ])

        # Final layer norm
        self.final_norm = get_norm(config.norm_type, config.n_embd)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,  # ignored; causal mask is internal
        return_attention: bool = False
    ) -> GPTOutput:
        """
        Args:
            inputs_embeds: (B, seq_len, n_embd)
            attention_mask: (B, seq_len), currently ignored
            return_attention: if True, also return per-block pre-dropout attn maps

        Returns:
            GPTOutput with:
              - last_hidden_state: (B, seq_len, n_embd)
              - attention_maps: list of (B, n_heads, T, T) or None
        """
        x = inputs_embeds

        # Add positional encoding (only for learned encoding, RoPE is in attention)
        if self.pos_encoding is not None:
            x = self.pos_encoding(x)
        x = self.dropout(x)

        attn_maps: Optional[List[torch.Tensor]] = [] if return_attention else None

        # Pass through transformer blocks
        for block in self.blocks:
            if return_attention:
                x, w = block(x, return_attention=True)
                attn_maps.append(w)
            else:
                x = block(x, return_attention=False)

        # Final layer norm
        x = self.final_norm(x)

        return GPTOutput(last_hidden_state=x, attention_maps=attn_maps)



'''impl:
Pre-norm architecture:                                                                                                                                                                  │
x -> LayerNorm -> Attention -> + residual                                                                                                                                               │
  -> LayerNorm -> FFN -> + residual
     '''

import torch.nn as nn
import torch
from custom_transformer.normalization import LayerNorm, RMSNorm
from custom_transformer.ffn import FeedForward
from custom_transformer.attention import MultiHeadedAttention
from custom_transformer.config import TransformerConfig


def get_norm(norm_type: str, dim: int) -> nn.Module:
    """Factory function to get normalization layer."""
    if norm_type == "rmsnorm":
        return RMSNorm(dim)
    else:
        return LayerNorm(dim)


class TransformerBLock(nn.Module):

    def __init__(self, n_embds, n_heads, config: TransformerConfig = None):
        super().__init__()
        if config is None:
            config = TransformerConfig()

        self.layer_norm1 = get_norm(config.norm_type, n_embds)
        self.layer_norm2 = get_norm(config.norm_type, n_embds)
        self.attention = MultiHeadedAttention(n_embds, n_heads=n_heads, config=config)
        self.ffw = FeedForward(n_embds)

    def forward(self, x: torch.Tensor, return_attention: bool = False):

        if return_attention:
            attn_out, attn_weights = self.attention(self.layer_norm1(x), return_attention=True)
            x = x + attn_out
        else:
            attn_out = self.attention(self.layer_norm1(x), return_attention=False)
            x = x + attn_out

        # FFN (pre-norm) + residual
        x = x + self.ffw(self.layer_norm2(x))

        if return_attention:
            return x, attn_weights  # (B, T, C), (B, n_heads, T, T)
        return x

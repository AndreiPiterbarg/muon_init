
import torch.nn as nn
import torch
from custom_transformer.config import TransformerConfig
from custom_transformer.utils import create_causal_mask
from custom_transformer.positional import RotaryPositionalEmbeddings
import torch.nn.functional as F
import math


class MultiHeadedAttention(nn.Module):

    def __init__(self, emb_dim, n_heads, config: TransformerConfig = None):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_heads = n_heads

        if config is None:
            config = TransformerConfig()
        self.config = config
        self.dropout = nn.Dropout(config.dropout)

        # ensure embedding dimension divisible by num heads
        if emb_dim % n_heads != 0:
            raise Exception("Embedding dimension must be divisible by number of heads")
        self.head_dim = emb_dim // n_heads
        self.linear_Q = nn.Linear(emb_dim, emb_dim)
        self.linear_K = nn.Linear(emb_dim, emb_dim)
        self.linear_V = nn.Linear(emb_dim, emb_dim)

        self.output_projection = nn.Linear(emb_dim, emb_dim)
        self.register_buffer('causal_mask', create_causal_mask(512, torch.device('cpu')))

        # RoPE if configured
        self.rope = None
        if config.pos_encoding_type == "rope":
            self.rope = RotaryPositionalEmbeddings(self.head_dim, max_len=config.n_positions * 2)

    def forward(self, x, return_attention = False):
        batch, seq_len, _ = x.size()

        # Project to Q, K, V
        Q = self.linear_Q(x)  # (batch, seq_len, emb_dim)
        K = self.linear_K(x)
        V = self.linear_V(x)

        # Reshape for multi-head attention: (batch, seq_len, emb_dim) -> (batch, n_heads, seq_len, head_dim)
        Q = Q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE if configured
        if self.rope is not None:
            Q, K = self.rope(Q, K)

        # Compute attention scores: (batch, n_heads, seq_len, head_dim) @ (batch, n_heads, head_dim, seq_len)
        attn_scores = Q @ K.transpose(-2, -1)  # (batch, n_heads, seq_len, seq_len)
        attn_scores = attn_scores * (1.0 / math.sqrt(self.head_dim))

        # Apply causal mask
        attn_scores = attn_scores.masked_fill(self.causal_mask[:, :, :seq_len, :seq_len] == 0, float('-inf'))
    
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights_dropped = self.dropout(attn_weights)
        out = attn_weights_dropped @ V
        
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.emb_dim)
        out = self.output_projection(out)

        if return_attention:
            return out, attn_weights  # pre-dropout weights
        return out
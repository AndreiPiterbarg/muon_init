# impl trainable embeddings for positional encoding
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(self, emb_dim: int, max_len:int = 512):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, emb_dim)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        batch, seq_len, emb_dim = x.size()
        positions = torch.arange(seq_len, device=x.device)
        return x + self.pos_embedding(positions)
    
'''
RoPE or Rotary Position Embeddings: roatates query/vectors in complex spaces
 '''

class RotaryPositionalEmbeddings(nn.Module):

    def __init__(self, head_dim: int, max_len: int = 512):
        super().__init__()
        N = 10000
        inv_freq = 1. / (N ** (torch.arange(0, head_dim, 2).float() / head_dim))
        position = torch.arange(max_len).float()
        inv_freq = torch.cat((inv_freq, inv_freq), dim=-1)
        sinusoid_inp = torch.outer(position, inv_freq)
        self.register_buffer("cos", sinusoid_inp.cos())
        self.register_buffer("sin", sinusoid_inp.sin())

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to query and key tensors.

        Args:
            q: (batch, n_heads, seq_len, head_dim)
            k: (batch, n_heads, seq_len, head_dim)

        Returns:
            Tuple of rotated (q, k) tensors
        """
        seq_len = q.size(2)
        # Reshape cos/sin for broadcasting: (1, 1, seq_len, head_dim)
        cos = self.cos[:seq_len].view(1, 1, seq_len, -1)
        sin = self.sin[:seq_len].view(1, 1, seq_len, -1)
        q_rot = apply_rotary_pos_emb(q, cos, sin)
        k_rot = apply_rotary_pos_emb(k, cos, sin)
        return q_rot, k_rot
    
    
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)
 
def apply_rotary_pos_emb(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)
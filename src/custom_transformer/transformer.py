# custom GPT class that matches GPT2Model interface
import torch.nn as nn
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional, Tuple

from custom_transformer.block import TransformerBLock, get_norm
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
            TransformerBLock(config.n_embd, config.n_head, config=config)
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


class CustomTransformerModel(nn.Module):
    """
    Custom transformer model matching TransformerModel interface from model.py.
    Uses custom GPT backbone instead of HuggingFace GPT2Model.
    """

    def __init__(self, n_dims, n_positions, name, n_embd=128, n_layer=6, n_head=4,
                 config: TransformerConfig = None):
        super().__init__()
        self.name = name
        self.n_positions = n_positions
        self.n_dims = n_dims

        # Build config if not provided
        if config is None:
            config = TransformerConfig(
                n_embd=n_embd,
                n_layer=n_layer,
                n_head=n_head,
                n_positions=2 * n_positions
            )
        self.config = config

        self._read_in = nn.Linear(n_dims, config.n_embd)
        self._backbone = CustomGPTBackbone(config=config)
        self._read_out = nn.Linear(config.n_embd, 1)

    def forward(self, xs, ys, inds=None, return_attention: bool = False):
        if inds is None:
            inds = torch.arange(ys.shape[1], device=ys.device)
        else:
            inds = torch.tensor(inds, device=ys.device)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        zs = self._combine(xs, ys)
        embeds = self._read_in(zs)

        out = self._backbone(inputs_embeds=embeds, return_attention=return_attention)
        prediction = self._read_out(out.last_hidden_state)

        if return_attention:
            return prediction, out.attention_maps  # List[(B, H, T, T)]
        return prediction

    @staticmethod
    def _combine(xs_b, ys_b):
        """
        Interleaves x's and y's to create: x_0, y_0, x_1, y_1, ..., x_{N-2}, y_{N-2}, x_{N-1}, 0

        This creates n_points-1 complete (x,y) pairs, then a final x followed by 0.
        The model sees N-1 examples and predicts the final y at position 2N-1.

        Args:
            xs_b: (B, N, input_dim) - input features
            ys_b: (B, N, output_dim) - target values

        Returns:
            (B, 2N, D) - interleaved sequence where D = max(input_dim, output_dim)
        """
        B, N, input_dim = xs_b.shape
        output_dim = ys_b.shape[-1]
        D = max(input_dim, output_dim)

        # Pad xs to match max dimension if needed
        if input_dim < D:
            xs_b = F.pad(xs_b, (0, D - input_dim))

        # Create y sequence: [y_0, y_1, ..., y_{N-2}, 0]
        ys_in = torch.zeros(B, N, D, device=xs_b.device, dtype=xs_b.dtype)
        ys_in[:, :-1, :output_dim] = ys_b[:, :-1, :]

        # Interleave: x_0, y_0, x_1, y_1, ..., x_{N-1}, 0
        toks = []
        for i in range(N):
            toks.append(xs_b[:, i, :])
            toks.append(ys_in[:, i, :])

        return torch.stack(toks, dim=1)


class CustomNNTransformer(nn.Module):
    """
    Custom NN transformer model matching NNTransformer interface from model.py.
    Uses custom GPT backbone instead of HuggingFace GPT2Model.
    """

    def __init__(self, n_input_dims, n_output_dims, n_positions, name, n_embd=128, n_layer=6, n_head=4,
                 config: TransformerConfig = None):
        super().__init__()
        # Import here to avoid circular imports
        from config import nn_output_dim

        self.name = "nn"
        self.n_input_dims = n_input_dims
        self.n_output_dims = n_output_dims
        self.n_positions = n_positions

        max_dim = max(n_input_dims, n_output_dims)

        # Build config if not provided
        if config is None:
            config = TransformerConfig(
                n_embd=n_embd,
                n_layer=n_layer,
                n_head=n_head,
                n_positions=2 * n_positions
            )
        self.config = config

        self._read_in = nn.Linear(max_dim, config.n_embd)
        self._backbone = CustomGPTBackbone(config=config)
        self._read_out = nn.Linear(config.n_embd, nn_output_dim)

    def forward(self, xs, ys, inds=None, return_attention: bool = False):
        if inds is None:
            inds = torch.arange(ys.shape[1], device=ys.device)
        else:
            inds = torch.tensor(inds, device=ys.device)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        zs = self._combine(xs, ys)
        embeds = self._read_in(zs)

        out = self._backbone(inputs_embeds=embeds, return_attention=return_attention)
        prediction = self._read_out(out.last_hidden_state)

        if return_attention:
            return prediction, out.attention_maps
        return prediction

    @staticmethod
    def _combine(xs_b, ys_b):
        """
        Interleaves x's and y's to create: x_0, y_0, x_1, y_1, ..., x_{N-2}, y_{N-2}, x_{N-1}, 0

        Args:
            xs_b: (B, N, input_dim) - input features
            ys_b: (B, N, output_dim) - target values

        Returns:
            (B, 2N, D) - interleaved sequence where D = max(input_dim, output_dim)
        """
        B, N, input_dim = xs_b.shape
        output_dim = ys_b.shape[-1]
        D = max(input_dim, output_dim)

        # Pad xs to match max dimension if needed
        if input_dim < D:
            xs_b = F.pad(xs_b, (0, D - input_dim))

        # Create y sequence
        ys_in = torch.zeros(B, N, D, device=xs_b.device, dtype=xs_b.dtype)
        ys_in[:, :-1, :output_dim] = ys_b[:, :-1, :]

        # Interleave
        toks = []
        for i in range(N):
            toks.append(xs_b[:, i, :])
            toks.append(ys_in[:, i, :])

        return torch.stack(toks, dim=1)

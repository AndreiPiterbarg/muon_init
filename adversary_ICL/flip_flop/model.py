"""Models for FFLM: GPT-2-style Transformer and a 1-layer LSTM baseline.

Architecture choices match the canonical baseline in Liu et al. 2023 Section 4
(6 layers, d=512, 8 heads -> ~19M params), with dropout disabled by default
(the "baseline Transformer" column of Figures 4/13).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

from .data import VOCAB_SIZE


class FFLMTransformer(nn.Module):
    """Autoregressive Transformer FFLM.

    Thin wrapper around HuggingFace GPT-2 with a flip-flop vocabulary
    (5 tokens) and learned absolute position embeddings.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        n_positions: int = 512,
        n_embd: int = 512,
        n_layer: int = 6,
        n_head: int = 8,
        resid_pdrop: float = 0.0,
        embd_pdrop: float = 0.0,
        attn_pdrop: float = 0.0,
    ):
        super().__init__()
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=resid_pdrop,
            embd_pdrop=embd_pdrop,
            attn_pdrop=attn_pdrop,
            use_cache=False,
        )
        self.model = GPT2LMHeadModel(config)
        self.name = f"gpt2_L={n_layer}_d={n_embd}_H={n_head}"

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, T) long -> logits: (B, T, V)."""
        return self.model(input_ids=tokens).logits

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FFLMLSTM(nn.Module):
    """1-layer LSTM FFLM baseline.

    Per Section 4 (R2): a 1-layer, hidden-size 128 LSTM (~133K params)
    extrapolates perfectly on FFL.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, vocab_size)
        self.vocab_size = vocab_size
        self.name = f"lstm_L={num_layers}_d={hidden_size}"

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        h, _ = self.lstm(x)
        return self.head(h)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(cfg) -> nn.Module:
    """Construct a model from a config dict / object with `.family` field."""
    family = getattr(cfg, "family", None) if not isinstance(cfg, dict) else cfg.get("family")
    get = (lambda k, d=None: getattr(cfg, k, d)) if not isinstance(cfg, dict) else (lambda k, d=None: cfg.get(k, d))

    if family == "gpt2":
        return FFLMTransformer(
            vocab_size=get("vocab_size", VOCAB_SIZE),
            n_positions=get("n_positions", 512),
            n_embd=get("n_embd", 512),
            n_layer=get("n_layer", 6),
            n_head=get("n_head", 8),
            resid_pdrop=get("resid_pdrop", 0.0),
            embd_pdrop=get("embd_pdrop", 0.0),
            attn_pdrop=get("attn_pdrop", 0.0),
        )
    if family == "lstm":
        return FFLMLSTM(
            vocab_size=get("vocab_size", VOCAB_SIZE),
            hidden_size=get("hidden_size", 128),
            num_layers=get("num_layers", 1),
            dropout=get("dropout", 0.0),
        )
    raise ValueError(f"Unknown model family: {family!r}")

from dataclasses import dataclass


@dataclass
class TransformerConfig:
    n_embd: int = 128
    n_layer: int = 6
    n_head: int = 4
    n_positions: int = 40

    batch_size = 64
    n_dims = 4
    n_points = 10

    # Architecture options
    pos_encoding_type: str = "learned"  # "learned" or "rope"
    norm_type: str = "layernorm"  # "layernorm" or "rmsnorm"
    ffn_type: str = "mlp"  # "mlp" (future: "swiglu")
    activation: str = "gelu"  # "gelu", "relu", "silu"

    pre_norm: bool = True
    dropout: float = 0.0
    return_attention = True
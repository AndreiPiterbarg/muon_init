from custom_transformer.config import TransformerConfig
import torch
import torch.nn as nn
import math
config = TransformerConfig()


def init_weights(module: nn.Module):
    '''
    Initialize for linear layers, embeddings and LayerNorm
    '''
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=0.02)

    # implicitly do nothing for layernorm

def apply_scaled_init(model, config, init_std: float = 0.02):
    """Scale output projections by 1/sqrt(2*n_layer)."""
    scale = 1.0 / math.sqrt(2 * config.n_layer)

    for block in model.blocks:
        # Scale attention output projection (last linear in attn branch)
        nn.init.normal_(block.attention.output_projection.weight,
                        mean=0.0,
                        std=init_std * scale)

        # Scale FFN output projection (last linear in FFN branch)
        if hasattr(block.ffw, 'second_layer'):  # MLP
            nn.init.normal_(block.ffw.second_layer.weight,
                            mean=0.0,
                            std=init_std * scale)

def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.view(1, 1, seq_len, seq_len)
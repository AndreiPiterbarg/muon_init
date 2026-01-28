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
        nn.init.xavier_uniform(module.weight)
    elif isinstance(module, nn.Embedding):
        nn.init.normal(module.weight, std = 0.02)
    
    # implicitly do nothing for layernorm

def apply_scaled_init(model, config):
    """Scale output projections by 1/sqrt(2*n_layer)."""
    scale = 1.0 / math.sqrt(2 * config.n_layers)

    for block in model.blocks:
        # Scale attention output projection (last linear in attn branch)
        nn.init.normal_(block.attn.out_proj.weight,
                        mean=0.0,
                        std=config.init_std * scale)  # 0.02 * scale

        # Scale FFN output projection (last linear in FFN branch)
        if hasattr(block.ffn, 'fc2'):          # MLP
            nn.init.normal_(block.ffn.fc2.weight,
                            mean=0.0,
                            std=config.init_std * scale)
        elif hasattr(block.ffn, 'down_proj'):  # SwiGLU
            nn.init.normal_(block.ffn.down_proj.weight,
                            mean=0.0,
                            std=config.init_std * scale)

def create_casual_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.view(1, 1, seq_len, seq_len)
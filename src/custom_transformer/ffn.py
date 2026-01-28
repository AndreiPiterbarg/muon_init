# impl standard Linear -> Activation -> Linear MLP feed forwards
import torch.nn as nn
from custom_transformer.config import TransformerConfig
import torch
class FeedForward(nn.Module):

    def __init__(self, n_embd, config: TransformerConfig = None):
        super().__init__()
        if config is None:
            config = TransformerConfig()
        self.n_embd = n_embd
        self.first_layer = nn.Linear(n_embd, 4 * n_embd)
        self.activation = nn.GELU()
        self.second_layer = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.first_layer(x)
        x = self.activation(x)
        x = self.second_layer(x)
        x = self.dropout(x)
        return x


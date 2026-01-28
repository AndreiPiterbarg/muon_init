# impl standard Linear -> Activation -> Linear MLP feed forwards
import torch.nn as nn
from custom_transformer.config import TransformerConfig
import torch
class FeedForward(nn.Module):

    def __init__(self, n_embd, ):
        super().__init__()
        self.n_embd = n_embd
        self.first_layer = nn.Linear(n_embd, 4 * n_embd)
        self.activation = nn.GELU()
        self.second_layer = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(TransformerConfig.dropout)

    def forward(self, x):
        x = self.first_layer(x)
        x = self.activation(x)
        x = self.second_layer(x)
        x = self.dropout(x)
        return x



def ffTest():
    # Test config
    batch, seq_len, n_embd = 2, 10, 128

    # Create module
    ffn = FeedForward(n_embd)

    # Test forward pass
    x = torch.randn(batch, seq_len, n_embd)
    out = ffn(x)

    assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print("FeedForward test passed!")
if __name__ == "__main__":
    ffTest()

   

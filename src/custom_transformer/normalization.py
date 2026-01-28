import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, dim, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x_norm + self.bias
    
class RMSNorm(nn.Module):
    
    
    def __init__(self, dim, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.dim = dim
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        rms = torch.rsqrt(x.pow(2).mean(dim = -1, keepdim=True) + self.eps)
        out = self.weight * x * rms
        return out
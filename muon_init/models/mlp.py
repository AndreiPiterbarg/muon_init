"""Deep MLP for CIFAR-10 (flattened input).

8-layer MLP, hidden dim 256, ReLU, no BatchNorm, no skip connections.
~600K params, nearly 100% Muon-eligible 2D weight matrices.
"""

import torch.nn as nn


class DeepMLP(nn.Module):
    def __init__(self, input_dim=3072, hidden_dim=256, num_classes=10, num_layers=8):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.flatten(1))

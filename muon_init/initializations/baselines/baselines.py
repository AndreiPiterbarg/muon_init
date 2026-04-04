# Baseline initialization schemes for comparison against Muon-matched inits.

import torch.nn as nn


def _apply_init(model, weight_init_fn):
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            weight_init_fn(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def kaiming_normal(model):
    _apply_init(model, lambda w: nn.init.kaiming_normal_(w, nonlinearity="relu"))


def kaiming_uniform(model):
    _apply_init(model, lambda w: nn.init.kaiming_uniform_(w, nonlinearity="relu"))


def xavier_normal(model):
    _apply_init(model, nn.init.xavier_normal_)


def xavier_uniform(model):
    _apply_init(model, nn.init.xavier_uniform_)


def orthogonal(model):
    _apply_init(model, nn.init.orthogonal_)

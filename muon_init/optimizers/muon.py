"""
Muon optimizer — MomentUm Orthogonalized by Newton-schulz.

Minimal single-GPU implementation based on Keller Jordan's reference:
https://github.com/KellerJordan/Muon

The core idea: run SGD with momentum, then replace each 2D parameter's
update with its nearest orthogonal matrix (the polar factor from the
polar decomposition). The orthogonalization is done via a Newton-Schulz
iteration that runs stably in bfloat16.

Muon should only be applied to hidden weight matrices (2D params).
Embeddings, output heads, biases, and normalization parameters should
use a standard optimizer like AdamW. This file provides both the
standalone Muon optimizer and a combined MuonAdamW that handles both
parameter groups in a single optimizer.
"""

import torch
from torch.optim import Optimizer

# Newton-Schulz quintic iteration coefficients, selected to maximize
# the slope at zero for fast convergence in few steps.
_NS_COEFFS = (3.4445, -4.7750, 2.0315)


def newton_schulz_orthogonalize(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Approximate the polar factor of G via Newton-Schulz iteration.

    Given G with SVD = USV^T, this produces approximately US'V^T where
    S' has entries close to 1 (not exact, but close enough that it
    doesn't hurt training). Runs in bfloat16 for speed.

    Args:
        G: Tensor with ndim >= 2.
        steps: Number of Newton-Schulz iterations (default 5).

    Returns:
        Orthogonalized tensor, same shape as G.
    """
    assert G.ndim >= 2
    a, b, c = _NS_COEFFS
    X = G.bfloat16()

    # Work with the shorter dimension on the right for efficiency.
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT

    # Normalize so spectral norm <= 1 (NS iteration requires this).
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    # Quintic Newton-Schulz iterations.
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if transposed:
        X = X.mT
    return X


class Muon(Optimizer):
    """
    Muon optimizer for 2D weight matrices (single-GPU).

    Runs SGD with momentum, then orthogonalizes each update via
    Newton-Schulz iteration. The learning rate is in units of
    spectral norm per step.

    This optimizer should only receive 2D weight matrix parameters.
    Use AdamW (or MuonAdamW below) for everything else.

    Args:
        params: Iterable of parameters (should all be 2D weight matrices).
        lr: Learning rate in spectral-norm units (default: 0.02).
        momentum: Momentum coefficient (default: 0.95).
        nesterov: Use Nesterov momentum (default: True).
        ns_steps: Number of Newton-Schulz iterations (default: 5).
        weight_decay: AdamW-style weight decay (default: 0).
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 ns_steps=5, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # Initialize momentum buffer.
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]

                # SGD momentum update.
                buf.lerp_(grad, 1 - beta)

                # Nesterov: blend gradient with momentum buffer.
                if nesterov:
                    update = grad.lerp(buf, beta)
                else:
                    update = buf.clone()

                # Reshape conv filters (4D) to 2D for orthogonalization.
                orig_shape = update.shape
                if update.ndim == 4:
                    update = update.view(update.size(0), -1)

                # Newton-Schulz orthogonalization.
                update = newton_schulz_orthogonalize(update, steps=ns_steps)

                # Scale so that the update norm is independent of aspect ratio.
                update *= max(1, update.size(-2) / update.size(-1)) ** 0.5

                # Weight decay (AdamW-style: applied to weights, not gradient).
                if wd > 0:
                    p.mul_(1 - lr * wd)

                # Apply update.
                p.add_(update.reshape(orig_shape).to(p.dtype), alpha=-lr)

        return loss


class MuonAdamW(Optimizer):
    """
    Combined optimizer: Muon for 2D hidden weight matrices, AdamW for
    everything else — in a single optimizer.

    Usage:
        # Separate params by type.
        muon_params = [p for n, p in model.named_parameters()
                       if p.ndim >= 2 and "embed" not in n and "head" not in n]
        adam_params = [p for n, p in model.named_parameters()
                       if p.ndim < 2 or "embed" in n or "head" in n]

        optimizer = MuonAdamW(
            muon_params=muon_params,
            adam_params=adam_params,
        )

    Args:
        muon_params: Parameters for Muon (2D weight matrices).
        adam_params: Parameters for AdamW (biases, embeddings, etc.).
        lr_muon: Muon learning rate (default: 0.02).
        momentum: Muon momentum (default: 0.95).
        nesterov: Use Nesterov momentum for Muon (default: True).
        ns_steps: Newton-Schulz iterations (default: 5).
        lr_adam: AdamW learning rate (default: 3e-4).
        betas_adam: AdamW beta coefficients (default: (0.9, 0.95)).
        eps_adam: AdamW epsilon (default: 1e-8).
        weight_decay_muon: Weight decay for Muon params (default: 0).
        weight_decay_adam: Weight decay for AdamW params (default: 0).
    """

    def __init__(self, muon_params, adam_params, *,
                 lr_muon=0.02, momentum=0.95, nesterov=True, ns_steps=5,
                 lr_adam=3e-4, betas_adam=(0.9, 0.95), eps_adam=1e-8,
                 weight_decay_muon=0.0, weight_decay_adam=0.0):

        # We store all config in param_groups. Group 0 = muon, Group 1 = adam.
        muon_params = list(muon_params)
        adam_params = list(adam_params)

        param_groups = [
            dict(params=muon_params, lr=lr_muon, momentum=momentum,
                 nesterov=nesterov, ns_steps=ns_steps,
                 weight_decay=weight_decay_muon, use_muon=True),
            dict(params=adam_params, lr=lr_adam, betas=betas_adam,
                 eps=eps_adam, weight_decay=weight_decay_adam, use_muon=False),
        ]
        # Filter out empty groups.
        param_groups = [g for g in param_groups if len(g["params"]) > 0]
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group.get("use_muon", False):
                self._step_muon(group)
            else:
                self._step_adam(group)

        return loss

    def _step_muon(self, group):
        lr = group["lr"]
        beta = group["momentum"]
        nesterov = group["nesterov"]
        ns_steps = group["ns_steps"]
        wd = group["weight_decay"]

        for p in group["params"]:
            if p.grad is None:
                continue

            grad = p.grad
            state = self.state[p]
            if len(state) == 0:
                state["momentum_buffer"] = torch.zeros_like(p)
            buf = state["momentum_buffer"]

            buf.lerp_(grad, 1 - beta)

            if nesterov:
                update = grad.lerp(buf, beta)
            else:
                update = buf.clone()

            orig_shape = update.shape
            if update.ndim == 4:
                update = update.view(update.size(0), -1)

            update = newton_schulz_orthogonalize(update, steps=ns_steps)
            update *= max(1, update.size(-2) / update.size(-1)) ** 0.5

            if wd > 0:
                p.mul_(1 - lr * wd)

            p.add_(update.reshape(orig_shape).to(p.dtype), alpha=-lr)

    def _step_adam(self, group):
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        wd = group["weight_decay"]

        for p in group["params"]:
            if p.grad is None:
                continue

            grad = p.grad
            state = self.state[p]
            if len(state) == 0:
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] = 0

            state["step"] += 1
            step = state["step"]
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]

            # Adam momentum updates.
            exp_avg.lerp_(grad, 1 - beta1)
            exp_avg_sq.lerp_(grad.square(), 1 - beta2)

            # Bias correction.
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            update = (exp_avg / bias_correction1) / (
                (exp_avg_sq / bias_correction2).sqrt() + eps
            )

            # Weight decay (AdamW-style).
            if wd > 0:
                p.mul_(1 - lr * wd)

            p.add_(update, alpha=-lr)

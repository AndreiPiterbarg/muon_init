"""Evaluation utilities for FFLM: "clean" read-error rate (glitch rate).

The paper's primary metric is the fraction of read-answer tokens mispredicted
by the model. In causal LM terms, logits at position t predict token t+1;
we score only positions where tokens[t] == R.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .data import R


@dataclass
class EvalResult:
    error_rate: float
    num_errors: int
    num_predictions: int
    loss: float  # mean cross-entropy over scored positions

    def to_dict(self) -> dict:
        return {
            "error_rate": self.error_rate,
            "num_errors": self.num_errors,
            "num_predictions": self.num_predictions,
            "loss": self.loss,
        }


def clean_loss(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Cross-entropy on deterministic (post-read) positions only.

    Args:
        logits: (B, T, V) next-token logits.
        tokens: (B, T) input token ids.

    Returns:
        Scalar loss averaged over scored positions. If the batch contains no
        read instructions, returns 0 (keeps gradients well-defined).
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = tokens[:, 1:].contiguous()
    mask = tokens[:, :-1] == R

    ce = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_targets.view(-1),
        reduction="none",
    ).view_as(shift_targets)

    mask_f = mask.float()
    denom = mask_f.sum().clamp_min(1.0)
    return (ce * mask_f).sum() / denom


@torch.no_grad()
def evaluate_dataset(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    batch_size: int = 64,
    device: str = "cpu",
) -> EvalResult:
    """Compute read-error rate on a pre-sampled dataset.

    Uses argmax decoding (hard, noise-free) for the error rate and the
    per-position cross-entropy for the reported loss.
    """
    was_training = model.training
    model.eval()
    total_errors = 0
    total_preds = 0
    total_loss_sum = 0.0

    try:
        for i in range(0, tokens.size(0), batch_size):
            batch = tokens[i : i + batch_size].to(device)
            logits = model(batch)

            shift_logits = logits[:, :-1, :]
            shift_targets = batch[:, 1:]
            mask = batch[:, :-1] == R

            preds = shift_logits.argmax(dim=-1)
            errs = ((preds != shift_targets) & mask).sum().item()
            n = mask.sum().item()

            if n > 0:
                ce = F.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    shift_targets.reshape(-1),
                    reduction="none",
                ).view_as(shift_targets)
                total_loss_sum += (ce * mask.float()).sum().item()

            total_errors += errs
            total_preds += n
    finally:
        if was_training:
            model.train()

    denom = max(total_preds, 1)
    return EvalResult(
        error_rate=total_errors / denom,
        num_errors=total_errors,
        num_predictions=total_preds,
        loss=total_loss_sum / denom,
    )

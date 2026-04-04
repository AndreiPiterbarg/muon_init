"""Spectral Norm Ball Membership and Phase 1 Duration Tracking.

Mathematical Background
-----------------------
"Muon Optimizes Under Spectral Norm Constraints" (arXiv 2506.15054) proves
that Muon with decoupled weight decay λ implicitly constrains:

    ||W||_op  <=  1 / λ

where ||·||_op is the spectral norm (largest singular value).  Convergence
under Muon therefore proceeds in two phases:

  **Phase 1 (Constraint Satisfaction):**  Parameters enter the spectral norm
  ball at an exponential rate.  Training dynamics in this phase are dominated
  by norm reduction, not loss minimization.

  **Phase 2 (Optimization):**  Once all layers satisfy the constraint, Muon
  optimizes the loss within the constrained set.  This is where meaningful
  learning happens.

Why This Metric Matters for Initialization
------------------------------------------
If the initialization already places every weight matrix inside the spectral
norm ball (||W_init||_op <= 1/λ), Phase 1 is skipped entirely.  This:

  1. Eliminates wasted training steps spent on norm reduction.
  2. May remove the need for learning rate warmup (which empirically
     correlates with Phase 1 duration).
  3. Ensures the first Muon steps are immediately productive.

The **spectral excess** — how far outside the ball each layer is — directly
quantifies the initialization mismatch for a given weight decay setting.

Metrics Provided
----------------
- ``check_spectral_norm_ball``: Per-layer ball membership check.
- ``compute_spectral_excess``: Aggregate excess spectral norm across layers.
- ``Phase1Tracker``: Online tracker that logs spectral norms during training
  and detects when Phase 1 completes (all layers enter the ball).

References
----------
- "Muon Optimizes Under Spectral Norm Constraints", arXiv 2506.15054
- Turbo-Muon (arXiv 2512.04632): AOL preconditioner tightens initial SVD
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from evaluation.metrics.spectral._utils import iter_weight_matrices


# ---------------------------------------------------------------------------
# Per-layer status
# ---------------------------------------------------------------------------

@dataclass
class SpectralNormStatus:
    """Status of a single layer w.r.t. the spectral norm ball."""
    layer_name: str
    spectral_norm: float
    threshold: float        # 1 / weight_decay
    ratio: float            # spectral_norm / threshold
    is_inside: bool         # spectral_norm <= threshold


# ---------------------------------------------------------------------------
# Static checks
# ---------------------------------------------------------------------------

def check_spectral_norm_ball(
    model: nn.Module,
    weight_decay: float,
) -> Dict[str, SpectralNormStatus]:
    """Check whether each weight matrix is inside the spectral norm ball.

    The ball radius is ``1 / weight_decay`` as derived in arXiv 2506.15054.

    Parameters
    ----------
    model : nn.Module
        Any PyTorch model.
    weight_decay : float
        Muon's decoupled weight decay parameter λ.  Must be > 0.

    Returns
    -------
    Dict[str, SpectralNormStatus]
        Per-layer spectral norm status.

    Raises
    ------
    ValueError
        If ``weight_decay <= 0``.
    """
    if weight_decay <= 0:
        raise ValueError(f"weight_decay must be > 0, got {weight_decay}")

    threshold = 1.0 / weight_decay
    results: Dict[str, SpectralNormStatus] = {}

    for name, W in iter_weight_matrices(model):
        sn = torch.linalg.norm(W.float(), ord=2).item()
        results[name] = SpectralNormStatus(
            layer_name=name,
            spectral_norm=sn,
            threshold=threshold,
            ratio=sn / threshold,
            is_inside=sn <= threshold,
        )

    return results


def compute_spectral_excess(
    model: nn.Module,
    weight_decay: float,
) -> float:
    """Aggregate spectral norm excess across all weight matrices.

    Defined as::

        spectral_excess = sum_l max(0, ||W_l||_op - 1/λ)

    A value of 0 means the entire model is already inside the constraint set.

    Parameters
    ----------
    model : nn.Module
        Any PyTorch model.
    weight_decay : float
        Muon's decoupled weight decay λ.  Must be > 0.

    Returns
    -------
    float
        Total spectral excess (non-negative).
    """
    if weight_decay <= 0:
        raise ValueError(f"weight_decay must be > 0, got {weight_decay}")

    threshold = 1.0 / weight_decay
    total_excess = 0.0

    for _name, W in iter_weight_matrices(model):
        sn = torch.linalg.norm(W.float(), ord=2).item()
        total_excess += max(0.0, sn - threshold)

    return total_excess


# ---------------------------------------------------------------------------
# Phase 1 Tracker — online monitoring during training
# ---------------------------------------------------------------------------

class Phase1Tracker:
    """Track spectral norms during training to detect Phase 1 completion.

    Phase 1 ends when *all* weight matrices enter the spectral norm ball
    ``||W||_op <= 1/λ`` for the first time.

    Usage::

        tracker = Phase1Tracker(weight_decay=0.01)
        for step in range(num_steps):
            ...  # training step
            tracker.log_step(model, step)
            if tracker.phase1_complete() is not None:
                print(f"Phase 1 ended at step {tracker.phase1_complete()}")

    Parameters
    ----------
    weight_decay : float
        Muon's decoupled weight decay λ.
    """

    def __init__(self, weight_decay: float) -> None:
        if weight_decay <= 0:
            raise ValueError(f"weight_decay must be > 0, got {weight_decay}")
        self.weight_decay = weight_decay
        self.threshold = 1.0 / weight_decay
        self._trajectories: Dict[str, List[Tuple[int, float]]] = {}
        self._phase1_step: Optional[int] = None
        self._all_inside_ever: bool = False

    def log_step(self, model: nn.Module, step: int) -> None:
        """Record spectral norms for all weight matrices at the current step.

        Parameters
        ----------
        model : nn.Module
            The model being trained.
        step : int
            Current training step number.
        """
        all_inside = True
        for name, W in iter_weight_matrices(model):
            sn = torch.linalg.norm(W.float(), ord=2).item()
            if name not in self._trajectories:
                self._trajectories[name] = []
            self._trajectories[name].append((step, sn))
            if sn > self.threshold:
                all_inside = False

        if all_inside and not self._all_inside_ever:
            self._all_inside_ever = True
            self._phase1_step = step

    def phase1_complete(self) -> Optional[int]:
        """Return the step at which Phase 1 completed, or None.

        Phase 1 is complete when all layers first enter the spectral norm ball.

        Returns
        -------
        Optional[int]
            Step number, or ``None`` if Phase 1 has not yet completed.
        """
        return self._phase1_step

    def get_trajectories(self) -> Dict[str, List[Tuple[int, float]]]:
        """Return per-layer spectral norm trajectories.

        Returns
        -------
        Dict[str, List[Tuple[int, float]]]
            Mapping from layer name to list of ``(step, spectral_norm)`` pairs.
        """
        return dict(self._trajectories)

    def get_summary(self) -> Dict[str, object]:
        """Return a summary of the current tracking state.

        Returns
        -------
        dict
            Summary with keys: ``threshold``, ``phase1_step``, ``num_layers``,
            ``num_steps_logged``, ``layers_inside`` (count at last logged step).
        """
        layers_inside = 0
        total_layers = 0
        for name, traj in self._trajectories.items():
            if traj:
                total_layers += 1
                if traj[-1][1] <= self.threshold:
                    layers_inside += 1

        num_steps = max((len(t) for t in self._trajectories.values()), default=0)

        return {
            "threshold": self.threshold,
            "phase1_step": self._phase1_step,
            "num_layers": total_layers,
            "num_steps_logged": num_steps,
            "layers_inside": layers_inside,
        }

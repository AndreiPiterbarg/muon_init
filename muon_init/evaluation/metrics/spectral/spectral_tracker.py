"""Training-time spectral tracker.

Logs spectral metrics (effective rank, stable rank, condition number,
SVD entropy, spectral norm) for all weight matrices at periodic checkpoints
during training. Supports saving/loading trajectories to JSON for
post-hoc analysis and plotting.

Usage:
    tracker = SpectralTracker()
    for step in range(num_steps):
        train_step(model)
        if step % log_every == 0:
            tracker.log_step(model, step)
    tracker.save("spectral_trajectories.json")
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch.nn as nn

from .effective_rank import effective_rank
from .stable_rank import stable_rank
from .condition_number import condition_number
from .svd_entropy import svd_entropy
from ._utils import iter_weight_matrices, singular_values

# Metrics computed per layer at each step
_METRIC_FNS = {
    "effective_rank": effective_rank,
    "stable_rank": stable_rank,
    "condition_number": condition_number,
    "svd_entropy": svd_entropy,
}


class SpectralTracker:
    """Tracks spectral metrics across training steps for all weight matrices.

    Attributes:
        trajectories: Nested dict of {layer_name: {metric_name: [values]}}.
        steps: List of step indices at which metrics were logged.
    """

    def __init__(self, metrics: Optional[List[str]] = None) -> None:
        """Initialize the tracker.

        Args:
            metrics: List of metric names to track. If None, tracks all
                available metrics: effective_rank, stable_rank,
                condition_number, svd_entropy, spectral_norm.
        """
        all_metrics = list(_METRIC_FNS.keys()) + ["spectral_norm"]
        if metrics is None:
            self._metrics = all_metrics
        else:
            for m in metrics:
                if m not in all_metrics:
                    raise ValueError(
                        f"Unknown metric '{m}'. Available: {all_metrics}"
                    )
            self._metrics = metrics

        self.steps: List[int] = []
        # {layer_name: {metric_name: [values]}}
        self._trajectories: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

    def log_step(self, model: nn.Module, step: int) -> None:
        """Log spectral metrics for all weight matrices at the current step.

        Args:
            model: PyTorch model.
            step: Current training step index.
        """
        self.steps.append(step)
        for name, W in iter_weight_matrices(model):
            layer_data = self._trajectories[name]
            # Compute SVD once, reuse for metrics that need it
            sv = singular_values(W)
            for metric_name in self._metrics:
                if metric_name == "spectral_norm":
                    val = sv[0].item() if len(sv) > 0 else 0.0
                elif metric_name in _METRIC_FNS:
                    val = _METRIC_FNS[metric_name](W)
                else:
                    continue
                layer_data[metric_name].append(val)

    def get_trajectories(self) -> Dict[str, Dict[str, List[float]]]:
        """Return the full trajectories dict.

        Returns:
            Nested dict: {layer_name: {metric_name: [value_per_step]}}.
        """
        # Convert defaultdicts to regular dicts for cleaner output
        return {
            layer: dict(metrics)
            for layer, metrics in self._trajectories.items()
        }

    def get_steps(self) -> List[int]:
        """Return the list of logged step indices."""
        return list(self.steps)

    def save(self, path: Union[str, Path]) -> None:
        """Save trajectories to a JSON file.

        Args:
            path: Output file path.
        """
        data = {
            "steps": self.steps,
            "trajectories": self.get_trajectories(),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "SpectralTracker":
        """Load trajectories from a JSON file.

        Args:
            path: Input file path.

        Returns:
            A SpectralTracker instance with loaded data.
        """
        with open(path) as f:
            data = json.load(f)
        tracker = cls()
        tracker.steps = data["steps"]
        for layer, metrics in data["trajectories"].items():
            for metric_name, values in metrics.items():
                tracker._trajectories[layer][metric_name] = values
        return tracker

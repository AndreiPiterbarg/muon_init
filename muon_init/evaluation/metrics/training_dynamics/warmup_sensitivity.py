"""Warmup sensitivity analysis for evaluating initialization quality.

A good initialization should reduce or eliminate the need for learning rate
warmup.  This module sweeps warmup lengths and quantifies the sensitivity
of final performance to the warmup schedule.

References
----------
- Gilmer et al. (NeurIPS 2024). "Why Warmup the Learning Rate?"
  arXiv:2406.09405.
- Survey Section 7: "Warmup Sensitivity as a Proxy for Init Quality."
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class WarmupSensitivityResult:
    """Stores the full outcome of a warmup sensitivity sweep.

    Attributes
    ----------
    logs : dict
        ``{warmup_steps: {seed: [(step, value), ...]}}`` — raw training logs.
    final_values : dict
        ``{warmup_steps: [final_value_per_seed]}``.
    min_warmup_for_stability : Optional[int]
        Smallest warmup where no seed diverges (all final values finite and
        below ``divergence_threshold``).
    performance_at_zero_warmup : float
        Mean final metric with warmup=0.
    warmup_sensitivity_score : float
        Std of *mean* final performance across warmup values.  Lower is
        better — means the init is insensitive to warmup choice.
    optimal_warmup : int
        Warmup value achieving the best mean final performance.
    lower_is_better : bool
        Whether lower metric values are better (True for loss).
    """

    logs: Dict[int, Dict[int, List[Tuple[int, float]]]]
    final_values: Dict[int, List[float]]
    min_warmup_for_stability: Optional[int]
    performance_at_zero_warmup: float
    warmup_sensitivity_score: float
    optimal_warmup: int
    lower_is_better: bool = True


class WarmupSensitivityAnalyzer:
    """Sweep warmup lengths and measure their impact on training.

    Parameters
    ----------
    train_fn : callable
        Signature: ``(warmup_steps: int, seed: int, **kwargs)
        -> List[Tuple[int, float]]``.
        Returns a training log of ``(step, val_metric)`` pairs.
    warmup_steps_list : list of int
        Warmup durations to sweep.
    lower_is_better : bool
        If ``True`` (default), lower metric = better (e.g. loss).
    divergence_threshold : float
        A final metric above this value is considered "diverged."
    train_kwargs : dict
        Extra keyword arguments forwarded to *train_fn*.
    """

    def __init__(
        self,
        train_fn: Callable[..., List[Tuple[int, float]]],
        warmup_steps_list: Optional[List[int]] = None,
        lower_is_better: bool = True,
        divergence_threshold: float = 1e6,
        **train_kwargs: object,
    ) -> None:
        self.train_fn = train_fn
        self.warmup_steps_list = warmup_steps_list or [0, 100, 500, 1000, 5000]
        self.lower_is_better = lower_is_better
        self.divergence_threshold = divergence_threshold
        self.train_kwargs = train_kwargs

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, num_seeds: int = 3) -> WarmupSensitivityResult:
        """Execute the full warmup x seed grid and summarise results.

        Parameters
        ----------
        num_seeds : int
            Number of random seeds per warmup value.

        Returns
        -------
        WarmupSensitivityResult
        """
        logs: Dict[int, Dict[int, List[Tuple[int, float]]]] = {}
        final_values: Dict[int, List[float]] = {}

        for warmup in self.warmup_steps_list:
            logs[warmup] = {}
            finals: List[float] = []
            for seed in range(num_seeds):
                log = self.train_fn(
                    warmup_steps=warmup, seed=seed, **self.train_kwargs
                )
                logs[warmup][seed] = log
                finals.append(log[-1][1] if log else float("nan"))
            final_values[warmup] = finals

        # --- Summary statistics ---
        mean_finals = {
            w: float(np.nanmean(vs)) for w, vs in final_values.items()
        }

        # min_warmup_for_stability: smallest warmup where all seeds are
        # finite and below divergence threshold.
        min_warmup: Optional[int] = None
        for warmup in sorted(self.warmup_steps_list):
            vals = final_values[warmup]
            if all(math.isfinite(v) and v < self.divergence_threshold for v in vals):
                min_warmup = warmup
                break

        # performance_at_zero_warmup
        perf_zero = mean_finals.get(0, float("nan"))

        # warmup_sensitivity_score: std of mean finals across warmup values
        sensitivity = float(np.nanstd(list(mean_finals.values())))

        # optimal_warmup
        if self.lower_is_better:
            optimal = min(mean_finals, key=mean_finals.get)  # type: ignore[arg-type]
        else:
            optimal = max(mean_finals, key=mean_finals.get)  # type: ignore[arg-type]

        return WarmupSensitivityResult(
            logs=logs,
            final_values=final_values,
            min_warmup_for_stability=min_warmup,
            performance_at_zero_warmup=perf_zero,
            warmup_sensitivity_score=sensitivity,
            optimal_warmup=optimal,
            lower_is_better=self.lower_is_better,
        )

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot(self, result: WarmupSensitivityResult) -> plt.Figure:
        """Plot mean final performance vs. warmup length with error bars.

        Parameters
        ----------
        result : WarmupSensitivityResult
            Output of :meth:`run`.

        Returns
        -------
        matplotlib.figure.Figure
        """
        warmups = sorted(result.final_values.keys())
        means = [float(np.nanmean(result.final_values[w])) for w in warmups]
        stds = [float(np.nanstd(result.final_values[w])) for w in warmups]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.errorbar(warmups, means, yerr=stds, marker="o", capsize=4)
        ax.set_xlabel("Warmup steps")
        ax.set_ylabel("Final metric value")
        ax.set_title("Warmup Sensitivity")

        # Annotate optimal
        opt = result.optimal_warmup
        opt_mean = float(np.nanmean(result.final_values[opt]))
        ax.annotate(
            f"optimal={opt}",
            xy=(opt, opt_mean),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="green",
        )

        # Annotate sensitivity score
        ax.text(
            0.98,
            0.02,
            f"sensitivity={result.warmup_sensitivity_score:.4f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="gray",
        )

        fig.tight_layout()
        return fig

"""Hyperparameter robustness via 1D ablation sweeps.

For each hyperparameter, sweep its value while holding all others at their
defaults.  A robust initialisation widens the "valley" of near-optimal HP
values, reducing sensitivity to exact HP tuning.

References
----------
- Zhao et al. (2024). "Anything but SGD: Evaluating Optimizers for LLM
  Training."  Kempner Institute / Harvard.
- Survey Section 8: "Hyperparameter Robustness."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class HPSweepCurve:
    """Results for a single hyperparameter sweep.

    Attributes
    ----------
    hp_name : str
        Name of the swept hyperparameter.
    hp_values : list of float
        The grid of values tested.
    final_means : list of float
        Mean final metric across seeds for each HP value.
    final_stds : list of float
        Std of final metric across seeds.
    valley_width : float
        Range of HP values within ``tolerance_pct`` of optimal performance.
    optimal_value : float
        HP value achieving the best mean final metric.
    sensitivity_score : float
        Std of mean final performance across HP values (lower = more robust).
    """

    hp_name: str
    hp_values: List[float]
    final_means: List[float]
    final_stds: List[float]
    valley_width: float
    optimal_value: float
    sensitivity_score: float


@dataclass
class HPRobustnessResult:
    """Aggregated results across all HP sweeps.

    Attributes
    ----------
    sweeps : dict mapping hp_name -> HPSweepCurve
    valley_widths : dict mapping hp_name -> float
    optimal_values : dict mapping hp_name -> float
    sensitivity_scores : dict mapping hp_name -> float
    """

    sweeps: Dict[str, HPSweepCurve]
    valley_widths: Dict[str, float] = field(default_factory=dict)
    optimal_values: Dict[str, float] = field(default_factory=dict)
    sensitivity_scores: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.valley_widths:
            self.valley_widths = {n: s.valley_width for n, s in self.sweeps.items()}
        if not self.optimal_values:
            self.optimal_values = {n: s.optimal_value for n, s in self.sweeps.items()}
        if not self.sensitivity_scores:
            self.sensitivity_scores = {
                n: s.sensitivity_score for n, s in self.sweeps.items()
            }


class HPRobustnessAnalyzer:
    """Run 1D HP ablation sweeps and measure robustness.

    Parameters
    ----------
    train_fn : callable
        Signature: ``(**hp_kwargs, seed: int) -> List[Tuple[int, float]]``.
        Returns a training log of ``(step, val_metric)`` pairs.
    hp_grids : dict
        ``{hp_name: [value1, value2, ...]}`` — the grid for each HP.
    default_hps : dict
        ``{hp_name: default_value}`` — baseline HP values.  During each
        sweep only the swept HP varies; the rest stay at defaults.
    lower_is_better : bool
        If ``True`` (default), lower metric = better.
    tolerance_pct : float
        Percentage tolerance for computing valley width.  An HP value is
        "in the valley" if its mean final metric is within
        ``tolerance_pct`` percent of the optimal mean.
    """

    def __init__(
        self,
        train_fn: Callable[..., List[Tuple[int, float]]],
        hp_grids: Dict[str, List[float]],
        default_hps: Dict[str, float],
        lower_is_better: bool = True,
        tolerance_pct: float = 5.0,
    ) -> None:
        self.train_fn = train_fn
        self.hp_grids = hp_grids
        self.default_hps = default_hps
        self.lower_is_better = lower_is_better
        self.tolerance_pct = tolerance_pct

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, num_seeds: int = 3) -> HPRobustnessResult:
        """Execute all 1D sweeps and compute summary statistics.

        Parameters
        ----------
        num_seeds : int
            Seeds per HP value.

        Returns
        -------
        HPRobustnessResult
        """
        sweeps: Dict[str, HPSweepCurve] = {}

        for hp_name, values in self.hp_grids.items():
            means: List[float] = []
            stds: List[float] = []

            for val in values:
                hp_kwargs = {**self.default_hps, hp_name: val}
                finals: List[float] = []
                for seed in range(num_seeds):
                    log = self.train_fn(**hp_kwargs, seed=seed)
                    finals.append(log[-1][1] if log else float("nan"))
                means.append(float(np.nanmean(finals)))
                stds.append(float(np.nanstd(finals)))

            # Optimal value
            if self.lower_is_better:
                best_idx = int(np.nanargmin(means))
            else:
                best_idx = int(np.nanargmax(means))
            optimal_val = values[best_idx]
            best_mean = means[best_idx]

            # Valley width: range of HP values within tolerance_pct of best
            threshold = abs(best_mean) * (self.tolerance_pct / 100.0)
            in_valley = [
                v
                for v, m in zip(values, means)
                if abs(m - best_mean) <= threshold
            ]
            if len(in_valley) >= 2:
                valley_w = float(max(in_valley) - min(in_valley))
            elif in_valley:
                valley_w = 0.0
            else:
                valley_w = 0.0

            sensitivity = float(np.nanstd(means))

            sweeps[hp_name] = HPSweepCurve(
                hp_name=hp_name,
                hp_values=list(values),
                final_means=means,
                final_stds=stds,
                valley_width=valley_w,
                optimal_value=optimal_val,
                sensitivity_score=sensitivity,
            )

        return HPRobustnessResult(sweeps=sweeps)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot(self, result: HPRobustnessResult) -> plt.Figure:
        """Grid of 1D HP sweep plots.

        Parameters
        ----------
        result : HPRobustnessResult
            Output of :meth:`run`.

        Returns
        -------
        matplotlib.figure.Figure
        """
        n = len(result.sweeps)
        cols = min(n, 3)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

        for idx, (hp_name, curve) in enumerate(result.sweeps.items()):
            ax = axes[idx // cols][idx % cols]
            ax.errorbar(
                curve.hp_values,
                curve.final_means,
                yerr=curve.final_stds,
                marker="o",
                capsize=4,
            )
            ax.axvline(
                curve.optimal_value, color="green", linestyle="--", alpha=0.6,
                label=f"optimal={curve.optimal_value:.4g}",
            )
            ax.set_xlabel(hp_name)
            ax.set_ylabel("Final metric")
            ax.set_title(f"{hp_name}  (valley={curve.valley_width:.4g})")
            ax.legend(fontsize=8)

        # Hide unused axes
        for idx in range(n, rows * cols):
            axes[idx // cols][idx % cols].set_visible(False)

        fig.suptitle("1D Hyperparameter Ablations", fontsize=13)
        fig.tight_layout()
        return fig

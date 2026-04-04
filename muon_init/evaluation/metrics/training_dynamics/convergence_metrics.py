"""Convergence metrics: time-to-target, loss curve AUC, and multi-run comparison.

Follows the evaluation philosophy of AlgoPerf (Dahl et al., 2023): measure
wall-clock / step count to a *validation* metric threshold, and integrate
loss curves to capture consistent improvement rather than just endpoints.

References
----------
- Dahl et al. (2023). "Benchmarking Neural Network Training Algorithms."
  arXiv:2306.07179.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd


def time_to_target(
    log: List[Tuple[int, float]],
    threshold: float,
    lower_is_better: bool = True,
) -> Optional[int]:
    """Return the first step at which *metric_value* crosses *threshold*.

    Parameters
    ----------
    log : list of (step, metric_value)
        Training or validation log, assumed sorted by step.
    threshold : float
        Target metric value.
    lower_is_better : bool
        If ``True`` (default, e.g. loss), crossing means ``value <= threshold``.
        If ``False`` (e.g. accuracy), crossing means ``value >= threshold``.

    Returns
    -------
    int or None
        The step at which the threshold is first met, or ``None`` if it is
        never reached.
    """
    for step, value in log:
        if lower_is_better and value <= threshold:
            return step
        if not lower_is_better and value >= threshold:
            return step
    return None


def loss_curve_auc(
    log: List[Tuple[int, float]],
    normalize: bool = True,
) -> float:
    """Trapezoidal area under the loss (or metric) curve.

    Parameters
    ----------
    log : list of (step, value)
        Must contain at least two entries.
    normalize : bool
        If ``True``, divide the integral by the step span so the result is
        an average metric value rather than a raw area.

    Returns
    -------
    float
        The (optionally normalized) AUC.

    Raises
    ------
    ValueError
        If *log* has fewer than two entries.
    """
    if len(log) < 2:
        raise ValueError("Need at least two log entries to compute AUC.")

    auc = 0.0
    for i in range(1, len(log)):
        s0, v0 = log[i - 1]
        s1, v1 = log[i]
        auc += 0.5 * (v0 + v1) * (s1 - s0)

    if normalize:
        total_steps = log[-1][0] - log[0][0]
        if total_steps == 0:
            raise ValueError("Total step span is zero; cannot normalize.")
        auc /= total_steps

    return auc


def compare_runs(
    logs: Dict[str, List[Tuple[int, float]]],
    threshold: float,
    lower_is_better: bool = True,
) -> pd.DataFrame:
    """Compare multiple training runs on convergence metrics.

    For each named run, computes time-to-target, normalized AUC, and final
    metric value.

    Parameters
    ----------
    logs : dict mapping run_name -> list of (step, value)
    threshold : float
        Target metric value for time-to-target.
    lower_is_better : bool
        Passed through to :func:`time_to_target`.

    Returns
    -------
    pd.DataFrame
        One row per run with columns ``run``, ``time_to_target``, ``auc``,
        ``final_value``.
    """
    rows = []
    for name, log in logs.items():
        ttt = time_to_target(log, threshold, lower_is_better=lower_is_better)
        auc = loss_curve_auc(log, normalize=True) if len(log) >= 2 else float("nan")
        final = log[-1][1] if log else float("nan")
        rows.append(
            {
                "run": name,
                "time_to_target": ttt,
                "auc": auc,
                "final_value": final,
            }
        )
    return pd.DataFrame(rows).set_index("run")

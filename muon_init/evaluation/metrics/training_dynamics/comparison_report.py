"""Multi-initialization comparison report.

Given training logs from several initialization schemes (possibly with
multiple seeds and warmup sweeps), produce a comprehensive comparison
with convergence metrics, AUC, final performance, and statistical tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .convergence_metrics import loss_curve_auc, time_to_target
from .training_logger import TrainingLogger


@dataclass
class ComparisonReport:
    """Structured comparison across initialization schemes.

    Attributes
    ----------
    summary : pd.DataFrame
        One row per init scheme with columns: ``mean_ttt``, ``std_ttt``,
        ``mean_auc``, ``std_auc``, ``mean_final``, ``std_final``.
    pairwise_tests : pd.DataFrame
        Paired t-test results for every pair of init schemes on final
        metric values.  Columns: ``init_a``, ``init_b``, ``t_stat``,
        ``p_value``, ``significant`` (p < 0.05).
    per_run : pd.DataFrame
        Long-form table with one row per (init, seed) containing raw
        ``ttt``, ``auc``, ``final_value``.
    metric_name : str
        The metric used for comparison.
    targets : dict
        The threshold targets used.
    """

    summary: pd.DataFrame
    pairwise_tests: pd.DataFrame
    per_run: pd.DataFrame
    metric_name: str
    targets: Dict[str, float]


def compare_initializations(
    logs: Dict[str, List[TrainingLogger]],
    targets: Dict[str, float],
    metric_name: str = "val_loss",
    lower_is_better: bool = True,
) -> ComparisonReport:
    """Produce a comprehensive comparison of initialization schemes.

    Parameters
    ----------
    logs : dict
        ``{init_name: [TrainingLogger, ...]}`` — one logger per seed.
    targets : dict
        ``{metric_name: threshold}`` — used for time-to-target.
    metric_name : str
        Which logged metric to analyse.
    lower_is_better : bool
        Passed through to :func:`time_to_target`.

    Returns
    -------
    ComparisonReport
    """
    threshold = targets.get(metric_name, float("inf") if lower_is_better else 0.0)

    # --- Per-run statistics ---
    per_run_rows: List[Dict[str, object]] = []
    for init_name, loggers in logs.items():
        for i, logger in enumerate(loggers):
            log = logger.get_log(metric_name)
            ttt = time_to_target(log, threshold, lower_is_better=lower_is_better)
            auc = loss_curve_auc(log, normalize=True) if len(log) >= 2 else float("nan")
            final = log[-1][1] if log else float("nan")
            per_run_rows.append(
                {
                    "init": init_name,
                    "seed": i,
                    "ttt": ttt,
                    "auc": auc,
                    "final_value": final,
                }
            )
    per_run = pd.DataFrame(per_run_rows)

    # --- Summary table ---
    summary_rows = []
    for init_name in logs:
        subset = per_run[per_run["init"] == init_name]
        # ttt can be None; convert to NaN for aggregation
        ttt_vals = pd.to_numeric(subset["ttt"], errors="coerce")
        summary_rows.append(
            {
                "init": init_name,
                "mean_ttt": ttt_vals.mean(),
                "std_ttt": ttt_vals.std(),
                "mean_auc": subset["auc"].mean(),
                "std_auc": subset["auc"].std(),
                "mean_final": subset["final_value"].mean(),
                "std_final": subset["final_value"].std(),
                "num_seeds": len(subset),
            }
        )
    summary = pd.DataFrame(summary_rows).set_index("init")

    # --- Pairwise paired t-tests on final metric ---
    test_rows: List[Dict[str, object]] = []
    init_names = list(logs.keys())
    for a, b in combinations(init_names, 2):
        vals_a = per_run[per_run["init"] == a]["final_value"].values
        vals_b = per_run[per_run["init"] == b]["final_value"].values
        n = min(len(vals_a), len(vals_b))
        if n >= 2:
            t_stat, p_val = stats.ttest_rel(vals_a[:n], vals_b[:n])
        else:
            t_stat, p_val = float("nan"), float("nan")
        test_rows.append(
            {
                "init_a": a,
                "init_b": b,
                "t_stat": float(t_stat),
                "p_value": float(p_val),
                "significant": bool(p_val < 0.05) if np.isfinite(p_val) else False,
            }
        )
    pairwise = pd.DataFrame(test_rows)

    return ComparisonReport(
        summary=summary,
        pairwise_tests=pairwise,
        per_run=per_run,
        metric_name=metric_name,
        targets=targets,
    )


def plot_comparison(report: ComparisonReport) -> plt.Figure:
    """Multi-panel comparison figure.

    Panels:
    1. Final performance (bar + error bars)
    2. Time-to-target (bar, NaN shown as missing)
    3. Loss curve AUC (bar + error bars)

    Parameters
    ----------
    report : ComparisonReport

    Returns
    -------
    matplotlib.figure.Figure
    """
    summary = report.summary
    inits = summary.index.tolist()
    x = np.arange(len(inits))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: Final performance
    ax = axes[0]
    ax.bar(x, summary["mean_final"], yerr=summary["std_final"], capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(inits, rotation=30, ha="right")
    ax.set_ylabel(report.metric_name)
    ax.set_title("Final Performance")

    # Panel 2: Time-to-target
    ax = axes[1]
    ttt_means = summary["mean_ttt"].fillna(0)
    colors = ["gray" if np.isnan(summary["mean_ttt"].iloc[i]) else "C0" for i in range(len(inits))]
    ax.bar(x, ttt_means, color=colors, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(inits, rotation=30, ha="right")
    ax.set_ylabel("Steps")
    ax.set_title(f"Time to Target ({report.metric_name} = {report.targets.get(report.metric_name, '?')})")

    # Panel 3: AUC
    ax = axes[2]
    ax.bar(x, summary["mean_auc"], yerr=summary["std_auc"], capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(inits, rotation=30, ha="right")
    ax.set_ylabel("Normalized AUC")
    ax.set_title("Loss Curve AUC")

    fig.suptitle("Initialization Comparison", fontsize=13)
    fig.tight_layout()
    return fig

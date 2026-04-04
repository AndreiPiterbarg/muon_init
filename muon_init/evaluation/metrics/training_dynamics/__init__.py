"""Training dynamics metrics for evaluating initialization quality.

Tier 2 (short runs) and Tier 3 (full runs) metrics that require actual
training loops — warmup sensitivity, HP robustness, convergence speed,
and multi-init comparison reports.
"""

from .convergence_metrics import compare_runs, loss_curve_auc, time_to_target
from .comparison_report import (
    ComparisonReport,
    compare_initializations,
    plot_comparison,
)
from .hp_robustness import (
    HPRobustnessAnalyzer,
    HPRobustnessResult,
    HPSweepCurve,
)
from .training_logger import TrainingLogger
from .warmup_sensitivity import (
    WarmupSensitivityAnalyzer,
    WarmupSensitivityResult,
)

__all__ = [
    # convergence
    "time_to_target",
    "loss_curve_auc",
    "compare_runs",
    # warmup
    "WarmupSensitivityAnalyzer",
    "WarmupSensitivityResult",
    # hp robustness
    "HPRobustnessAnalyzer",
    "HPRobustnessResult",
    "HPSweepCurve",
    # logger
    "TrainingLogger",
    # comparison
    "ComparisonReport",
    "compare_initializations",
    "plot_comparison",
]

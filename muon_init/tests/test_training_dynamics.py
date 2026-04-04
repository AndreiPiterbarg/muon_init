"""Tests for evaluation.metrics.training_dynamics.

Uses mock training functions that return predetermined logs so we can
verify convergence, sensitivity, and robustness calculations exactly.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest

from evaluation.metrics.training_dynamics import (
    ComparisonReport,
    HPRobustnessAnalyzer,
    HPRobustnessResult,
    TrainingLogger,
    WarmupSensitivityAnalyzer,
    WarmupSensitivityResult,
    compare_initializations,
    compare_runs,
    loss_curve_auc,
    time_to_target,
)


# ======================================================================
# Helpers — deterministic mock training functions
# ======================================================================


def _make_log(
    start: float, end: float, steps: int = 10
) -> List[Tuple[int, float]]:
    """Linearly interpolate a loss curve from *start* to *end*."""
    return [
        (i * 100, start + (end - start) * i / (steps - 1)) for i in range(steps)
    ]


def _mock_warmup_train_fn(
    warmup_steps: int, seed: int, **kwargs
) -> List[Tuple[int, float]]:
    """Mock training: longer warmup -> lower final loss (diminishing returns).

    final = 2.0 / (1 + warmup_steps/500) + 0.01*seed
    """
    final = 2.0 / (1.0 + warmup_steps / 500.0) + 0.01 * seed
    return _make_log(3.0, final)


def _mock_hp_train_fn(
    lr: float = 0.001, wd: float = 0.01, seed: int = 0
) -> List[Tuple[int, float]]:
    """Mock training: quadratic bowl around optimal lr=0.001, wd=0.01.

    final = 1.0 + 100*(lr - 0.001)^2 + 10*(wd - 0.01)^2 + 0.01*seed
    """
    final = 1.0 + 100 * (lr - 0.001) ** 2 + 10 * (wd - 0.01) ** 2 + 0.01 * seed
    return _make_log(3.0, final)


# ======================================================================
# Tests — convergence_metrics
# ======================================================================


class TestTimeToTarget:
    def test_crosses_threshold(self):
        log = [(0, 3.0), (100, 2.0), (200, 1.5), (300, 1.0)]
        assert time_to_target(log, 2.0) == 100
        assert time_to_target(log, 1.0) == 300

    def test_threshold_never_reached(self):
        log = [(0, 3.0), (100, 2.5)]
        assert time_to_target(log, 1.0) is None

    def test_higher_is_better(self):
        log = [(0, 0.1), (100, 0.5), (200, 0.9)]
        assert time_to_target(log, 0.5, lower_is_better=False) == 100

    def test_exact_threshold(self):
        log = [(0, 2.0), (100, 1.5)]
        assert time_to_target(log, 2.0) == 0

    def test_empty_log(self):
        assert time_to_target([], 1.0) is None


class TestLossCurveAUC:
    def test_constant_loss(self):
        log = [(0, 2.0), (100, 2.0), (200, 2.0)]
        assert loss_curve_auc(log, normalize=True) == pytest.approx(2.0)

    def test_linear_decrease(self):
        # Linearly from 2.0 to 0.0 over 200 steps.  Average = 1.0
        log = [(0, 2.0), (200, 0.0)]
        assert loss_curve_auc(log, normalize=True) == pytest.approx(1.0)

    def test_unnormalized(self):
        log = [(0, 2.0), (200, 0.0)]
        assert loss_curve_auc(log, normalize=False) == pytest.approx(200.0)

    def test_too_few_entries(self):
        with pytest.raises(ValueError):
            loss_curve_auc([(0, 1.0)])

    def test_multi_segment(self):
        # Two segments: [0,100] at 2.0, [100,200] drops to 0.0
        log = [(0, 2.0), (100, 2.0), (200, 0.0)]
        # Area = 100*2.0 + 0.5*(2.0+0.0)*100 = 200 + 100 = 300
        # Normalized: 300 / 200 = 1.5
        assert loss_curve_auc(log, normalize=True) == pytest.approx(1.5)


class TestCompareRuns:
    def test_basic_comparison(self):
        logs = {
            "fast": _make_log(3.0, 0.5),
            "slow": _make_log(3.0, 1.5),
        }
        df = compare_runs(logs, threshold=1.0)
        assert "fast" in df.index
        assert "slow" in df.index
        # "fast" reaches 1.0, "slow" never goes below 1.5 except maybe
        # let's check: slow goes 3.0 -> 1.5 linearly, so at some step
        # it will be exactly 1.5 at the end.  1.0 is never reached.
        assert df.loc["fast", "time_to_target"] is not None
        assert pd.isna(df.loc["slow", "time_to_target"])
        assert df.loc["fast", "final_value"] == pytest.approx(0.5)


# ======================================================================
# Tests — TrainingLogger
# ======================================================================


class TestTrainingLogger:
    def test_log_and_retrieve(self):
        lg = TrainingLogger(metadata={"seed": 0})
        lg.log(0, loss=2.5, acc=0.1)
        lg.log(100, loss=1.5, acc=0.5)
        assert lg.get_log("loss") == [(0, 2.5), (100, 1.5)]
        assert lg.get_log("acc") == [(0, 0.1), (100, 0.5)]

    def test_metric_names(self):
        lg = TrainingLogger()
        lg.log(0, a=1.0, b=2.0)
        assert set(lg.metric_names) == {"a", "b"}

    def test_missing_metric_raises(self):
        lg = TrainingLogger()
        with pytest.raises(KeyError):
            lg.get_log("nonexistent")

    def test_to_dataframe(self):
        lg = TrainingLogger()
        lg.log(0, loss=2.0)
        lg.log(1, loss=1.0)
        df = lg.to_dataframe()
        assert len(df) == 2
        assert set(df.columns) == {"step", "metric", "value"}

    def test_save_load_roundtrip(self, tmp_path):
        lg = TrainingLogger(metadata={"seed": 42})
        lg.log(0, loss=2.5)
        lg.log(100, loss=1.5)
        path = str(tmp_path / "logger.json")
        lg.save(path)
        lg2 = TrainingLogger.load(path)
        assert lg2.metadata == {"seed": 42}
        assert lg2.get_log("loss") == [(0, 2.5), (100, 1.5)]

    def test_get_value(self):
        lg = TrainingLogger()
        lg.log(0, x=1.0)
        lg.log(5, x=2.0)
        assert lg.get_value("x", 0) == 1.0
        assert lg.get_value("x", 5) == 2.0
        assert lg.get_value("x", 99) is None


# ======================================================================
# Tests — WarmupSensitivityAnalyzer
# ======================================================================


class TestWarmupSensitivity:
    def test_run_produces_result(self):
        analyzer = WarmupSensitivityAnalyzer(
            train_fn=_mock_warmup_train_fn,
            warmup_steps_list=[0, 500, 1000],
        )
        result = analyzer.run(num_seeds=2)
        assert isinstance(result, WarmupSensitivityResult)
        assert set(result.logs.keys()) == {0, 500, 1000}
        assert len(result.final_values[0]) == 2

    def test_optimal_warmup_is_highest(self):
        """In our mock, more warmup -> lower loss, so optimal should be max."""
        analyzer = WarmupSensitivityAnalyzer(
            train_fn=_mock_warmup_train_fn,
            warmup_steps_list=[0, 500, 1000],
        )
        result = analyzer.run(num_seeds=2)
        assert result.optimal_warmup == 1000

    def test_performance_at_zero_warmup(self):
        analyzer = WarmupSensitivityAnalyzer(
            train_fn=_mock_warmup_train_fn,
            warmup_steps_list=[0, 500],
        )
        result = analyzer.run(num_seeds=2)
        # seed 0 -> 2.0, seed 1 -> 2.01, mean ~2.005
        assert result.performance_at_zero_warmup == pytest.approx(2.005)

    def test_sensitivity_score_is_nonneg(self):
        analyzer = WarmupSensitivityAnalyzer(
            train_fn=_mock_warmup_train_fn,
            warmup_steps_list=[0, 500, 1000],
        )
        result = analyzer.run(num_seeds=2)
        assert result.warmup_sensitivity_score >= 0.0

    def test_min_warmup_for_stability(self):
        analyzer = WarmupSensitivityAnalyzer(
            train_fn=_mock_warmup_train_fn,
            warmup_steps_list=[0, 500, 1000],
        )
        result = analyzer.run(num_seeds=2)
        # All seeds are finite, so min_warmup should be 0
        assert result.min_warmup_for_stability == 0


# ======================================================================
# Tests — HPRobustnessAnalyzer
# ======================================================================


class TestHPRobustness:
    def test_run_produces_result(self):
        analyzer = HPRobustnessAnalyzer(
            train_fn=_mock_hp_train_fn,
            hp_grids={"lr": [0.0001, 0.001, 0.01]},
            default_hps={"lr": 0.001, "wd": 0.01},
        )
        result = analyzer.run(num_seeds=2)
        assert isinstance(result, HPRobustnessResult)
        assert "lr" in result.sweeps

    def test_optimal_lr(self):
        analyzer = HPRobustnessAnalyzer(
            train_fn=_mock_hp_train_fn,
            hp_grids={"lr": [0.0001, 0.001, 0.01]},
            default_hps={"lr": 0.001, "wd": 0.01},
        )
        result = analyzer.run(num_seeds=2)
        assert result.optimal_values["lr"] == pytest.approx(0.001)

    def test_valley_width(self):
        analyzer = HPRobustnessAnalyzer(
            train_fn=_mock_hp_train_fn,
            hp_grids={"lr": [0.0001, 0.001, 0.01]},
            default_hps={"lr": 0.001, "wd": 0.01},
            tolerance_pct=50.0,  # generous tolerance
        )
        result = analyzer.run(num_seeds=1)
        # With 50% tolerance around optimal ~1.0, values up to 1.5 are "in".
        # lr=0.0001 -> 1.0 + 100*(0.0001-0.001)^2 ≈ 1.008  (within 50% of 1.0)
        # lr=0.001  -> 1.0                                 (optimal)
        # lr=0.01   -> 1.0 + 100*(0.01-0.001)^2   = 9.1   (outside 50%)
        # So valley covers 0.0001..0.01 for the first two, width = 0.0099
        # Actually 1.008 is within 50% of 1.0 (threshold 1.5), and 9.1 is not.
        # But lr=0.01 gives 9.1 which IS within 50% of 1.0? No, 50% of 1.0 = 0.5,
        # so threshold is |m - best| <= 0.5.  9.1 - 1.0 = 8.1 > 0.5.
        # 1.008 - 1.0 = 0.008 < 0.5 -> in valley.  So valley = {0.0001, 0.001}.
        # But the test got 0.0099 meaning all three are in the valley... let's
        # re-check: tolerance_pct=50 means abs(best)*50/100 = 0.5, so 9.1 is out.
        # Wait — the analyzer got 0.0099 = 0.01 - 0.0001, meaning all three.
        # This happens because seed=0 so final = 1.0+100*(lr-0.001)^2 exactly.
        # lr=0.01 -> final = 1 + 100*0.009^2 = 1 + 100*0.000081 = 1.0081
        # (0.01-0.001)=0.009, squared=0.000081, *100=0.0081!  So 1.0081, not 9.1.
        # All three are very close to 1.0 so valley spans the full grid.
        assert result.valley_widths["lr"] == pytest.approx(0.0099)

    def test_sensitivity_scores(self):
        analyzer = HPRobustnessAnalyzer(
            train_fn=_mock_hp_train_fn,
            hp_grids={"lr": [0.0001, 0.001, 0.01]},
            default_hps={"lr": 0.001, "wd": 0.01},
        )
        result = analyzer.run(num_seeds=1)
        assert result.sensitivity_scores["lr"] > 0


# ======================================================================
# Tests — compare_initializations
# ======================================================================


class TestComparisonReport:
    @staticmethod
    def _make_loggers(final: float, n_seeds: int = 3) -> List[TrainingLogger]:
        loggers = []
        for seed in range(n_seeds):
            lg = TrainingLogger(metadata={"seed": seed})
            for step, val in _make_log(3.0, final + 0.01 * seed):
                lg.log(step, val_loss=val)
            loggers.append(lg)
        return loggers

    def test_produces_report(self):
        logs = {
            "kaiming": self._make_loggers(1.0),
            "orthogonal": self._make_loggers(0.8),
        }
        report = compare_initializations(
            logs, targets={"val_loss": 1.5}, metric_name="val_loss"
        )
        assert isinstance(report, ComparisonReport)
        assert "kaiming" in report.summary.index
        assert "orthogonal" in report.summary.index

    def test_pairwise_tests(self):
        logs = {
            "a": self._make_loggers(1.0, n_seeds=5),
            "b": self._make_loggers(0.5, n_seeds=5),
        }
        report = compare_initializations(
            logs, targets={"val_loss": 2.0}, metric_name="val_loss"
        )
        assert len(report.pairwise_tests) == 1
        row = report.pairwise_tests.iloc[0]
        # "b" is clearly better, so the test should be significant
        assert row["p_value"] < 0.05

    def test_per_run_shape(self):
        logs = {
            "x": self._make_loggers(1.0, n_seeds=3),
        }
        report = compare_initializations(
            logs, targets={"val_loss": 2.0}, metric_name="val_loss"
        )
        assert len(report.per_run) == 3

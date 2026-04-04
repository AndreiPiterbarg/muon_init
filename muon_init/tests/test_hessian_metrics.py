"""Tests for Hessian and loss landscape metrics.

Uses a small MLP (2 hidden layers, 64 units) with synthetic data to verify
that all metrics produce sane values.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from evaluation.metrics.hessian import (
    EoSTracker,
    compute_hessian_trace,
    compute_lambda_max,
    compute_normalized_lambda_max,
    compute_normalized_trace,
    compute_spectral_density,
    compute_spikiness,
    compute_weight_norm_squared,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture()
def small_mlp():
    """2-hidden-layer MLP with 64 units, for testing."""
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(16, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 4),
    )
    return model


@pytest.fixture()
def synthetic_loader():
    """Synthetic classification data: 64 samples, 16 features, 4 classes."""
    torch.manual_seed(0)
    X = torch.randn(64, 16)
    y = torch.randint(0, 4, (64,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=32)


@pytest.fixture()
def loss_fn():
    return nn.CrossEntropyLoss()


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

class TestHessianTopEigenvalue:
    def test_lambda_max_positive(self, small_mlp, loss_fn, synthetic_loader):
        lam = compute_lambda_max(
            small_mlp, loss_fn, synthetic_loader, num_iterations=50, tol=1e-4
        )
        assert lam > 0, f"Expected lambda_max > 0, got {lam}"

    def test_lambda_max_finite(self, small_mlp, loss_fn, synthetic_loader):
        lam = compute_lambda_max(
            small_mlp, loss_fn, synthetic_loader, num_iterations=50
        )
        assert np.isfinite(lam), f"Expected finite lambda_max, got {lam}"


class TestHessianTrace:
    def test_trace_positive(self, small_mlp, loss_fn, synthetic_loader):
        tr = compute_hessian_trace(
            small_mlp, loss_fn, synthetic_loader, num_samples=30
        )
        assert tr > 0, f"Expected trace > 0, got {tr}"

    def test_trace_finite(self, small_mlp, loss_fn, synthetic_loader):
        tr = compute_hessian_trace(
            small_mlp, loss_fn, synthetic_loader, num_samples=30
        )
        assert np.isfinite(tr), f"Expected finite trace, got {tr}"

    def test_spikiness_positive(self):
        ratio = compute_spikiness(lambda_max=5.0, trace=100.0)
        assert ratio == pytest.approx(0.05)

    def test_spikiness_zero_trace(self):
        ratio = compute_spikiness(lambda_max=5.0, trace=0.0)
        assert ratio == float("inf")


class TestSpectralDensity:
    def test_density_shape(self, small_mlp, loss_fn, synthetic_loader):
        grid, density = compute_spectral_density(
            small_mlp,
            loss_fn,
            synthetic_loader,
            num_lanczos_steps=30,
            num_density_points=256,
        )
        assert grid.shape == (256,)
        assert density.shape == (256,)

    def test_density_non_negative(self, small_mlp, loss_fn, synthetic_loader):
        grid, density = compute_spectral_density(
            small_mlp,
            loss_fn,
            synthetic_loader,
            num_lanczos_steps=30,
        )
        assert np.all(density >= -1e-10), "Density should be non-negative"

    def test_density_integrates_roughly(self, small_mlp, loss_fn, synthetic_loader):
        """The integral of the spectral density ≈ 1 (normalized)."""
        grid, density = compute_spectral_density(
            small_mlp,
            loss_fn,
            synthetic_loader,
            num_lanczos_steps=30,
            num_density_points=1024,
        )
        dx = grid[1] - grid[0]
        integral = np.sum(density) * dx
        # Should be roughly 1.0 (it's a probability density)
        assert 0.5 < integral < 2.0, (
            f"Spectral density integral = {integral}, expected ~1.0"
        )


class TestNormalizedSharpness:
    def test_weight_norm_positive(self, small_mlp):
        w_norm_sq = compute_weight_norm_squared(small_mlp)
        assert w_norm_sq > 0

    def test_normalized_lambda_max_positive(self, small_mlp):
        val = compute_normalized_lambda_max(lambda_max=5.0, model=small_mlp)
        assert val > 0

    def test_normalized_trace_positive(self, small_mlp):
        val = compute_normalized_trace(trace=100.0, model=small_mlp)
        assert val > 0

    def test_normalized_values_smaller_than_raw(self, small_mlp):
        """Normalized values should be smaller (weight norm > 1 for this model)."""
        w_norm_sq = compute_weight_norm_squared(small_mlp)
        # For a freshly initialized MLP with ~5k params, ||w||^2 >> 1
        assert w_norm_sq > 1.0
        raw = 5.0
        normalized = compute_normalized_lambda_max(raw, small_mlp)
        assert normalized < raw


class TestEoSTracker:
    def test_no_eos_initially(self):
        tracker = EoSTracker(learning_rate=0.01)
        assert tracker.eos_onset_step is None
        assert not tracker.is_at_eos()

    def test_eos_detected(self):
        tracker = EoSTracker(learning_rate=0.01)
        # Progressive sharpening below threshold
        for s in range(10):
            tracker.step(s, lambda_max=100.0)  # 0.01 * 100 = 1.0 < 2.0
        assert tracker.eos_onset_step is None

        # Cross the threshold: 0.01 * 250 = 2.5 >= 2.0
        tracker.step(10, lambda_max=250.0)
        assert tracker.eos_onset_step == 10
        assert tracker.is_at_eos()

    def test_sharpening_rate_positive(self):
        tracker = EoSTracker(learning_rate=0.01)
        # Increasing lambda_max
        for s in range(20):
            tracker.step(s, lambda_max=float(10 + s * 5))
        rate = tracker.sharpening_rate()
        assert rate > 0, f"Expected positive sharpening rate, got {rate}"
        assert rate == pytest.approx(5.0, rel=0.01)

    def test_trajectory_as_dict(self):
        tracker = EoSTracker(learning_rate=0.01)
        tracker.step(0, 10.0)
        tracker.step(1, 20.0)
        d = tracker.trajectory_as_dict()
        assert d["steps"] == [0, 1]
        assert d["lambda_max"] == [10.0, 20.0]
        assert d["eta_lambda_max"] == [0.1, 0.2]

    def test_update_learning_rate(self):
        tracker = EoSTracker(learning_rate=0.01)
        tracker.step(0, 100.0)
        assert tracker.eta_lambda_maxs[-1] == pytest.approx(1.0)

        tracker.update_learning_rate(0.05)
        tracker.step(1, 100.0)
        assert tracker.eta_lambda_maxs[-1] == pytest.approx(5.0)
        assert tracker.eos_onset_step == 1

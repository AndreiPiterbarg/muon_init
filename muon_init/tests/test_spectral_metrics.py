"""Tests for spectral metrics on matrices with known properties.

Test strategy:
- Identity matrix: eRank=n, srank=n, kappa=1, max SVD entropy
- Rank-1 matrix: eRank≈1, srank=1, kappa→large
- Orthogonal matrix: kappa=1, srank=n, eRank=n
- Random Gaussian: eRank close to min(m,n), MP-distributed singular values
- Zero matrix: degenerate edge case
"""

from __future__ import annotations

import json
import math
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn

from evaluation.metrics.spectral import (
    effective_rank,
    effective_rank_all_layers,
    stable_rank,
    stable_rank_all_layers,
    condition_number,
    condition_number_all_layers,
    svd_entropy,
    svd_entropy_all_layers,
    compute_esd,
    marchenko_pastur_fit,
    check_spectral_norm_ball,
    spectral_norm_ratio,
    SpectralTracker,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def identity_matrix():
    return torch.eye(64)


@pytest.fixture
def rank1_matrix():
    u = torch.randn(64, 1)
    v = torch.randn(1, 32)
    return u @ v


@pytest.fixture
def orthogonal_matrix():
    """Random orthogonal matrix via QR decomposition."""
    A = torch.randn(64, 64)
    Q, _ = torch.linalg.qr(A)
    return Q


@pytest.fixture
def tall_orthogonal_matrix():
    """Partial isometry: (128, 64) with orthonormal columns."""
    A = torch.randn(128, 64)
    Q, _ = torch.linalg.qr(A)
    return Q


@pytest.fixture
def random_gaussian():
    """Large random Gaussian matrix."""
    torch.manual_seed(42)
    return torch.randn(256, 128) / math.sqrt(128)


@pytest.fixture
def simple_model():
    """Small model for testing all_layers functions."""
    model = nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 16),
    )
    return model


# ---------------------------------------------------------------------------
# Effective Rank
# ---------------------------------------------------------------------------

class TestEffectiveRank:
    def test_identity(self, identity_matrix):
        """Identity has uniform singular values → eRank = n."""
        er = effective_rank(identity_matrix)
        assert abs(er - 64.0) < 0.1

    def test_rank1(self, rank1_matrix):
        """Rank-1 matrix → eRank ≈ 1."""
        er = effective_rank(rank1_matrix)
        assert abs(er - 1.0) < 0.1

    def test_orthogonal(self, orthogonal_matrix):
        """Orthogonal matrix has uniform SVs → eRank = n."""
        er = effective_rank(orthogonal_matrix)
        assert abs(er - 64.0) < 0.1

    def test_all_layers(self, simple_model):
        result = effective_rank_all_layers(simple_model)
        assert "0.weight" in result
        assert "2.weight" in result
        assert all(v >= 1.0 for v in result.values())

    def test_3d_tensor(self):
        """Conv-like 4D tensor should be reshaped and work."""
        W = torch.randn(16, 3, 3, 3)
        er = effective_rank(W)
        assert er >= 1.0

    def test_zero_matrix(self):
        er = effective_rank(torch.zeros(10, 10))
        assert er == 0.0


# ---------------------------------------------------------------------------
# Stable Rank
# ---------------------------------------------------------------------------

class TestStableRank:
    def test_identity(self, identity_matrix):
        """Identity: srank = n (all SVs equal)."""
        sr = stable_rank(identity_matrix)
        assert abs(sr - 64.0) < 0.1

    def test_rank1(self, rank1_matrix):
        """Rank-1: srank = 1."""
        sr = stable_rank(rank1_matrix)
        assert abs(sr - 1.0) < 0.1

    def test_orthogonal(self, orthogonal_matrix):
        """Orthogonal: srank = n."""
        sr = stable_rank(orthogonal_matrix)
        assert abs(sr - 64.0) < 0.1

    def test_bounded_by_rank(self, random_gaussian):
        """Stable rank <= min(m, n) = 128."""
        sr = stable_rank(random_gaussian)
        assert 1.0 <= sr <= 128.0

    def test_all_layers(self, simple_model):
        result = stable_rank_all_layers(simple_model)
        assert len(result) == 2  # two Linear layers


# ---------------------------------------------------------------------------
# Condition Number
# ---------------------------------------------------------------------------

class TestConditionNumber:
    def test_identity(self, identity_matrix):
        """Identity: kappa = 1."""
        kappa = condition_number(identity_matrix)
        assert abs(kappa - 1.0) < 1e-4

    def test_orthogonal(self, orthogonal_matrix):
        """Orthogonal: kappa = 1."""
        kappa = condition_number(orthogonal_matrix)
        assert abs(kappa - 1.0) < 1e-4

    def test_rank1_large(self, rank1_matrix):
        """Rank-1 matrix should have very large condition number."""
        kappa = condition_number(rank1_matrix)
        assert kappa > 1e5

    def test_tall_orthogonal(self, tall_orthogonal_matrix):
        """Partial isometry: kappa = 1."""
        kappa = condition_number(tall_orthogonal_matrix)
        assert abs(kappa - 1.0) < 1e-3

    def test_all_layers(self, simple_model):
        result = condition_number_all_layers(simple_model)
        assert all(v >= 1.0 for v in result.values())


# ---------------------------------------------------------------------------
# SVD Entropy
# ---------------------------------------------------------------------------

class TestSVDEntropy:
    def test_identity(self, identity_matrix):
        """Identity: uniform SVs → max entropy = log(n)."""
        h = svd_entropy(identity_matrix)
        expected = math.log(64)
        assert abs(h - expected) < 0.01

    def test_rank1(self, rank1_matrix):
        """Rank-1: all energy in one SV → entropy ≈ 0."""
        h = svd_entropy(rank1_matrix)
        assert abs(h) < 0.01

    def test_orthogonal_max_entropy(self, orthogonal_matrix):
        """Orthogonal: max entropy."""
        h = svd_entropy(orthogonal_matrix)
        expected = math.log(64)
        assert abs(h - expected) < 0.01

    def test_all_layers(self, simple_model):
        result = svd_entropy_all_layers(simple_model)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Empirical Spectral Density
# ---------------------------------------------------------------------------

class TestESD:
    def test_compute_esd_shape(self, random_gaussian):
        centers, density = compute_esd(random_gaussian, num_bins=50)
        assert len(centers) == 50
        assert len(density) == 50
        assert np.all(density >= 0)

    def test_mp_fit_gaussian(self, random_gaussian):
        """Random Gaussian should be close to MP distribution."""
        kl, ks = marchenko_pastur_fit(random_gaussian)
        # KS should be relatively small for a well-matched distribution
        assert ks < 0.35
        assert kl >= 0

    def test_mp_fit_orthogonal_diverges(self, orthogonal_matrix):
        """Orthogonal matrix should diverge strongly from MP."""
        kl, ks = marchenko_pastur_fit(orthogonal_matrix)
        # Orthogonal has all SVs=1, MP is spread → should be different
        assert ks > 0.3

    def test_num_bins_respected(self, random_gaussian):
        centers, density = compute_esd(random_gaussian, num_bins=20)
        assert len(centers) == 20


# ---------------------------------------------------------------------------
# Spectral Norm Ball Membership
# ---------------------------------------------------------------------------

class TestSpectralNormBall:
    def test_identity_inside(self):
        """Identity has spectral norm 1; with wd=0.5, ratio = 0.5 < 1."""
        W = torch.eye(32)
        ratio = spectral_norm_ratio(W, weight_decay=0.5)
        assert abs(ratio - 0.5) < 1e-5

    def test_identity_outside(self):
        """Identity with wd=2.0: ratio = 2.0 > 1."""
        W = torch.eye(32)
        ratio = spectral_norm_ratio(W, weight_decay=2.0)
        assert abs(ratio - 2.0) < 1e-5

    def test_scaled_matrix(self):
        """Scaled identity: ||W||_op = 3, wd=0.5 → ratio = 1.5."""
        W = 3.0 * torch.eye(32)
        ratio = spectral_norm_ratio(W, weight_decay=0.5)
        assert abs(ratio - 1.5) < 1e-4

    def test_check_model(self, simple_model):
        result = check_spectral_norm_ball(simple_model, weight_decay=0.01)
        assert len(result) == 2
        for name, (inside, ratio) in result.items():
            assert isinstance(inside, bool)
            assert ratio > 0

    def test_invalid_weight_decay(self, simple_model):
        with pytest.raises(ValueError):
            check_spectral_norm_ball(simple_model, weight_decay=0.0)


# ---------------------------------------------------------------------------
# Spectral Tracker
# ---------------------------------------------------------------------------

class TestSpectralTracker:
    def test_log_and_retrieve(self, simple_model):
        tracker = SpectralTracker()
        tracker.log_step(simple_model, step=0)
        tracker.log_step(simple_model, step=100)

        assert tracker.get_steps() == [0, 100]
        traj = tracker.get_trajectories()
        assert len(traj) == 2  # two weight layers
        for layer_name, metrics in traj.items():
            assert "effective_rank" in metrics
            assert "stable_rank" in metrics
            assert "condition_number" in metrics
            assert "svd_entropy" in metrics
            assert "spectral_norm" in metrics
            assert len(metrics["effective_rank"]) == 2

    def test_save_load(self, simple_model, tmp_path):
        tracker = SpectralTracker()
        tracker.log_step(simple_model, step=0)
        tracker.log_step(simple_model, step=50)

        path = tmp_path / "traj.json"
        tracker.save(path)
        assert path.exists()

        loaded = SpectralTracker.load(path)
        assert loaded.get_steps() == [0, 50]
        assert loaded.get_trajectories().keys() == tracker.get_trajectories().keys()

    def test_selective_metrics(self, simple_model):
        tracker = SpectralTracker(metrics=["effective_rank", "spectral_norm"])
        tracker.log_step(simple_model, step=0)
        traj = tracker.get_trajectories()
        for layer_name, metrics in traj.items():
            assert set(metrics.keys()) == {"effective_rank", "spectral_norm"}

    def test_invalid_metric(self):
        with pytest.raises(ValueError):
            SpectralTracker(metrics=["nonexistent_metric"])

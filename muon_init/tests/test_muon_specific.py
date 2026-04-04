"""Tests for Muon-specific initialization quality metrics.

Test strategy:
- Polar error: orthogonal matrices should have error ~0; diagonal matrices
  with spread singular values should have error > 0.
- Newton-Schulz convergence: near-orthogonal matrices should converge in
  very few (0-1) iterations.
- Stiefel tangent projection: projected vectors must satisfy the tangent
  space condition W^T @ Delta + Delta^T @ W = 0.
- Spectral norm ball: correctly identify matrices inside/outside the ball.
- Phase1Tracker: detects when all layers enter the spectral norm ball.
- Angular update analysis: basic smoke test with a small model.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from evaluation.metrics.muon_specific.polar_error import (
    polar_factor,
    compute_polar_error,
    compute_polar_error_all_layers,
    newton_schulz_convergence_steps,
    muon_ns_approximation_quality,
    analyze_polar_error,
)
from evaluation.metrics.muon_specific.spectral_norm_ball import (
    check_spectral_norm_ball,
    compute_spectral_excess,
    Phase1Tracker,
)
from evaluation.metrics.muon_specific.stiefel_sharpness import (
    project_to_stiefel_tangent,
    random_stiefel_tangent_vector,
    verify_tangent_condition,
)
from evaluation.metrics.muon_specific.angular_update_analysis import (
    analyze_first_muon_step,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def orthogonal_matrix():
    """A random orthogonal matrix (all singular values = 1)."""
    Q, _ = torch.linalg.qr(torch.randn(64, 64))
    return Q


@pytest.fixture
def partial_isometry():
    """A tall partial isometry (64x32 with all singular values = 1)."""
    Q, _ = torch.linalg.qr(torch.randn(64, 64))
    return Q[:, :32]


@pytest.fixture
def diagonal_spread():
    """A diagonal matrix with singular values spread from 0.1 to 10."""
    sv = torch.linspace(0.1, 10.0, 64)
    return torch.diag(sv)


@pytest.fixture
def near_orthogonal():
    """An almost-orthogonal matrix (small perturbation from orthogonal)."""
    Q, _ = torch.linalg.qr(torch.randn(64, 64))
    return Q + 0.001 * torch.randn(64, 64)


@pytest.fixture
def simple_model():
    """A small 2-layer MLP for integration tests."""
    model = nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 8),
    )
    return model


@pytest.fixture
def dummy_data():
    """Dummy data loader for the simple model."""
    X = torch.randn(32, 16)
    y = torch.randint(0, 8, (32,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=32)


# ===================================================================
# Polar Error Tests
# ===================================================================

class TestPolarFactor:
    def test_orthogonal_is_itself(self, orthogonal_matrix):
        """Polar factor of an orthogonal matrix should be itself."""
        P = polar_factor(orthogonal_matrix)
        assert torch.allclose(P, orthogonal_matrix, atol=1e-5)

    def test_partial_isometry(self, partial_isometry):
        """Polar factor of a partial isometry should be itself."""
        P = polar_factor(partial_isometry)
        assert torch.allclose(P, partial_isometry, atol=1e-5)

    def test_result_is_isometry(self, diagonal_spread):
        """Polar factor should have all singular values = 1."""
        P = polar_factor(diagonal_spread)
        sv = torch.linalg.svdvals(P)
        assert torch.allclose(sv, torch.ones_like(sv), atol=1e-5)

    def test_rectangular(self):
        """Polar factor works on rectangular matrices."""
        M = torch.randn(100, 50)
        P = polar_factor(M)
        sv = torch.linalg.svdvals(P)
        assert torch.allclose(sv, torch.ones_like(sv), atol=1e-5)


class TestPolarError:
    def test_orthogonal_zero_error(self, orthogonal_matrix):
        """Polar error of an orthogonal matrix should be ~0."""
        err = compute_polar_error(orthogonal_matrix)
        assert err < 1e-5

    def test_partial_isometry_zero_error(self, partial_isometry):
        """Polar error of a partial isometry should be ~0."""
        err = compute_polar_error(partial_isometry)
        assert err < 1e-5

    def test_diagonal_positive_error(self, diagonal_spread):
        """Diagonal with spread singular values should have positive error."""
        err = compute_polar_error(diagonal_spread)
        assert err > 0.1  # non-trivial error for sv in [0.1, 10]

    def test_identity_zero_error(self):
        """Identity matrix (orthogonal) should have zero polar error."""
        I = torch.eye(32)
        err = compute_polar_error(I)
        assert err < 1e-6

    def test_scaled_identity(self):
        """Scaled identity (5*I) should have positive polar error."""
        M = 5.0 * torch.eye(32)
        err = compute_polar_error(M)
        assert err > 0  # it's not an isometry

    def test_zero_matrix(self):
        """Zero matrix should return 0 error (degenerate case)."""
        Z = torch.zeros(32, 32)
        err = compute_polar_error(Z)
        assert err == 0.0

    def test_all_layers(self, simple_model):
        """compute_polar_error_all_layers returns dict for all 2D params."""
        results = compute_polar_error_all_layers(simple_model)
        assert isinstance(results, dict)
        assert len(results) > 0
        for name, val in results.items():
            assert isinstance(val, float)
            assert val >= 0


class TestNewtonSchulzConvergence:
    """Tests for convergent cubic NS iteration step count.

    Uses the standard cubic NS iteration X_{k+1} = 3/2 X - 1/2 X(X^TX),
    which has a stable fixed point at σ = 1 and provably converges for
    starting singular values in (0, sqrt(3)).
    """

    def test_orthogonal_fast_convergence(self, orthogonal_matrix):
        """Orthogonal matrices (all sv=1) should converge in 1-2 steps."""
        steps = newton_schulz_convergence_steps(orthogonal_matrix, tol=1e-4)
        assert steps <= 2

    def test_near_orthogonal_fast(self, near_orthogonal):
        """Slightly perturbed orthogonal should converge quickly."""
        steps = newton_schulz_convergence_steps(near_orthogonal, tol=1e-4)
        assert steps <= 5

    def test_diagonal_spread_more_steps(self, diagonal_spread):
        """Spread singular values need more NS iterations."""
        steps = newton_schulz_convergence_steps(diagonal_spread, tol=1e-4)
        assert steps >= 2  # needs real work

    def test_identity_fast(self):
        """Identity matrix should converge in 1-2 iterations."""
        I = torch.eye(32)
        steps = newton_schulz_convergence_steps(I, tol=1e-6)
        assert steps <= 2

    def test_zero_returns_zero(self):
        """Zero matrix is degenerate — should return 0."""
        Z = torch.zeros(16, 16)
        steps = newton_schulz_convergence_steps(Z)
        assert steps == 0


class TestMuonNSApproximationQuality:
    """Tests for Muon's approximate (non-convergent) NS iteration.

    Muon intentionally uses coefficients that do NOT converge to the exact
    polar factor — the result is US'V^T where S' ~ Uniform(0.5, 1.5).
    These tests verify that the quality metric behaves sensibly.
    """

    def test_quality_nonnegative(self):
        """Quality metric should always be non-negative."""
        M = torch.randn(32, 32)
        q = muon_ns_approximation_quality(M, steps=10)
        assert q >= 0.0

    def test_quality_orthogonal_input(self, orthogonal_matrix):
        """Orthogonal input should give reasonable quality after NS."""
        q = muon_ns_approximation_quality(orthogonal_matrix, steps=10)
        # Muon's NS doesn't converge exactly, so quality won't be 0,
        # but should be bounded (typically < 0.5)
        assert q < 1.0

    def test_more_steps_doesnt_diverge(self):
        """Quality should stay bounded with more steps (not diverge)."""
        M = torch.randn(32, 32)
        q10 = muon_ns_approximation_quality(M, steps=10)
        q20 = muon_ns_approximation_quality(M, steps=20)
        # Both should be bounded (NS oscillates but doesn't blow up)
        assert q10 < 2.0
        assert q20 < 2.0


class TestPolarErrorReport:
    def test_analyze_polar_error(self, simple_model):
        """Full analysis should return a PolarErrorReport."""
        report = analyze_polar_error(simple_model)
        assert report.mean_polar_error >= 0
        assert report.max_polar_error >= report.mean_polar_error
        assert len(report.polar_errors) == len(report.ns_convergence_steps)


# ===================================================================
# Spectral Norm Ball Tests
# ===================================================================

class TestSpectralNormBall:
    def test_small_weights_inside_ball(self):
        """Small-norm weights should be inside the ball for moderate WD."""
        model = nn.Linear(32, 32)
        # Scale weights down so spectral norm << 1/0.01 = 100
        with torch.no_grad():
            model.weight.mul_(0.01)
        results = check_spectral_norm_ball(model, weight_decay=0.01)
        for status in results.values():
            assert status.is_inside
            assert status.spectral_norm <= status.threshold

    def test_large_weights_outside_ball(self):
        """Large-norm weights should be outside the ball for large WD."""
        model = nn.Linear(32, 32)
        with torch.no_grad():
            model.weight.mul_(100.0)
        results = check_spectral_norm_ball(model, weight_decay=1.0)
        # threshold = 1/1.0 = 1; spectral norm >> 1
        for status in results.values():
            assert not status.is_inside
            assert status.ratio > 1.0

    def test_invalid_weight_decay(self):
        """Should raise on non-positive weight decay."""
        model = nn.Linear(16, 16)
        with pytest.raises(ValueError):
            check_spectral_norm_ball(model, weight_decay=0.0)
        with pytest.raises(ValueError):
            check_spectral_norm_ball(model, weight_decay=-0.1)

    def test_spectral_excess_zero_inside(self):
        """Spectral excess should be 0 when all layers are inside."""
        model = nn.Linear(32, 32)
        with torch.no_grad():
            model.weight.mul_(0.001)
        excess = compute_spectral_excess(model, weight_decay=0.01)
        assert excess == pytest.approx(0.0, abs=1e-6)

    def test_spectral_excess_positive_outside(self):
        """Spectral excess should be > 0 when layers are outside."""
        model = nn.Linear(32, 32)
        with torch.no_grad():
            model.weight.mul_(100.0)
        excess = compute_spectral_excess(model, weight_decay=1.0)
        assert excess > 0


class TestPhase1Tracker:
    def test_detects_entry(self):
        """Phase1Tracker should detect when model enters the ball."""
        model = nn.Linear(32, 32, bias=False)
        tracker = Phase1Tracker(weight_decay=0.01)

        # Start outside the ball
        with torch.no_grad():
            model.weight.mul_(200.0)
        tracker.log_step(model, step=0)
        assert tracker.phase1_complete() is None

        # Shrink to be inside the ball (threshold = 100)
        with torch.no_grad():
            model.weight.mul_(0.001)  # now spectral norm ~ 0.2
        tracker.log_step(model, step=1)
        assert tracker.phase1_complete() == 1

    def test_trajectories_recorded(self):
        """Trajectories should contain entries for each logged step."""
        model = nn.Linear(16, 16, bias=False)
        tracker = Phase1Tracker(weight_decay=0.1)

        for step in range(5):
            tracker.log_step(model, step)

        trajectories = tracker.get_trajectories()
        assert len(trajectories) > 0
        for name, traj in trajectories.items():
            assert len(traj) == 5

    def test_already_inside(self):
        """If model starts inside, phase1 should complete at step 0."""
        model = nn.Linear(16, 16, bias=False)
        with torch.no_grad():
            model.weight.mul_(0.001)
        tracker = Phase1Tracker(weight_decay=0.01)
        tracker.log_step(model, step=0)
        assert tracker.phase1_complete() == 0


# ===================================================================
# Stiefel Tangent Space Tests
# ===================================================================

class TestStiefelTangent:
    def test_projection_satisfies_tangent_condition(self, orthogonal_matrix):
        """Projected vector should satisfy W^T Delta + Delta^T W = 0."""
        W = orthogonal_matrix
        Delta = torch.randn_like(W)
        proj = project_to_stiefel_tangent(W, Delta)
        violation = verify_tangent_condition(W, proj)
        assert violation < 1e-4

    def test_projection_on_partial_isometry(self, partial_isometry):
        """Tangent projection works on rectangular (tall) matrices."""
        W = partial_isometry
        Delta = torch.randn_like(W)
        proj = project_to_stiefel_tangent(W, Delta)
        violation = verify_tangent_condition(W, proj)
        assert violation < 1e-4

    def test_tangent_vector_already_tangent(self, orthogonal_matrix):
        """Projecting a tangent vector should not change it much."""
        W = orthogonal_matrix
        # W @ A where A is skew-symmetric is tangent
        A = torch.randn(64, 64)
        A = A - A.T  # make skew-symmetric
        tangent = W @ A
        proj = project_to_stiefel_tangent(W, tangent)
        assert torch.allclose(proj, tangent, atol=1e-4)

    def test_random_tangent_vector_is_tangent(self, orthogonal_matrix):
        """random_stiefel_tangent_vector should produce a valid tangent."""
        W = orthogonal_matrix
        v = random_stiefel_tangent_vector(W)
        violation = verify_tangent_condition(W, v)
        assert violation < 1e-4

    def test_random_tangent_unit_norm(self, orthogonal_matrix):
        """random_stiefel_tangent_vector should return unit Frobenius norm."""
        W = orthogonal_matrix
        v = random_stiefel_tangent_vector(W)
        norm = torch.linalg.norm(v, ord="fro").item()
        assert abs(norm - 1.0) < 1e-5

    def test_projection_idempotent(self, orthogonal_matrix):
        """Projecting twice should give the same result."""
        W = orthogonal_matrix
        Delta = torch.randn_like(W)
        proj1 = project_to_stiefel_tangent(W, Delta)
        proj2 = project_to_stiefel_tangent(W, proj1)
        assert torch.allclose(proj1, proj2, atol=1e-4)

    def test_rectangular_tall(self):
        """Tangent projection on tall matrices (more rows than cols).

        The Stiefel manifold St(n, p) requires n >= p (tall or square).
        For wide matrices, the transpose should be used.
        """
        M = torch.randn(64, 32)
        U, _, Vh = torch.linalg.svd(M, full_matrices=False)
        W = U @ Vh  # partial isometry (64x32), columns are orthonormal
        Delta = torch.randn_like(W)
        proj = project_to_stiefel_tangent(W, Delta)
        violation = verify_tangent_condition(W, proj)
        assert violation < 1e-3


# ===================================================================
# Angular Update Analysis Tests (smoke tests)
# ===================================================================

class TestAngularUpdateAnalysis:
    def test_basic_analysis(self, simple_model, dummy_data):
        """analyze_first_muon_step should run and return valid results."""
        loss_fn = nn.CrossEntropyLoss()
        result = analyze_first_muon_step(
            simple_model, loss_fn, dummy_data, lr=0.02
        )
        assert len(result.layers) > 0
        for name, diag in result.layers.items():
            assert diag.gradient_frobenius_norm >= 0
            assert diag.update_effective_rank >= 0
            # Cosine should be in [-1, 1]
            assert -1.0 <= diag.update_weight_cosine <= 1.0
            # Angle should be in [0, 180]
            assert 0.0 <= diag.update_weight_angle_deg <= 180.0
            assert len(diag.sv_before) > 0
            assert len(diag.sv_after) > 0

    def test_with_weight_decay(self, simple_model, dummy_data):
        """Analysis should work with weight decay applied."""
        loss_fn = nn.CrossEntropyLoss()
        result = analyze_first_muon_step(
            simple_model, loss_fn, dummy_data, lr=0.02, weight_decay=0.01
        )
        assert len(result.layers) > 0

    def test_update_full_rank(self, simple_model, dummy_data):
        """Muon's update (polar factor of gradient) should be ~full rank."""
        loss_fn = nn.CrossEntropyLoss()
        result = analyze_first_muon_step(
            simple_model, loss_fn, dummy_data, lr=0.02
        )
        for name, diag in result.layers.items():
            # Effective rank of the polar factor should be close to min(m, n)
            # (since all SVs = 1), but gradient might be low-rank at init
            assert diag.update_effective_rank >= 1.0

"""Tests for signal propagation metrics.

Uses a small MLP with identity-like initialization (orthogonal weights,
no bias, linear activation) to verify that all metrics report near-perfect
dynamical isometry.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

import sys, os

# Ensure the project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.metrics.signal_propagation import (
    ActivationStats,
    GradientFlowStats,
    JacobianStats,
    LayerJacobianInfo,
    compute_activation_stats,
    compute_angular_gsnr,
    compute_gradient_flow,
    compute_jacobian_singular_values,
    compute_layer_jacobians,
    dynamical_isometry_score,
    estimate_jacobian_singular_values,
    jacobian_stats,
    run_signal_propagation_diagnostics,
)


# ---------------------------------------------------------------------------
# Helper: identity-initialised MLP (perfect isometry for linear activations)
# ---------------------------------------------------------------------------

class IdentityMLP(nn.Module):
    """MLP with orthogonal (identity) weight matrices and no nonlinearity.

    For a square, identity-initialised deep linear network, the end-to-end
    Jacobian is the identity matrix — perfect dynamical isometry.
    """

    def __init__(self, width: int = 16, depth: int = 4):
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(depth):
            lin = nn.Linear(width, width, bias=False)
            nn.init.eye_(lin.weight)
            layers.append(lin)
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


def _make_batch(batch_size: int = 8, width: int = 16):
    torch.manual_seed(42)
    return torch.randn(batch_size, width)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestJacobianSpectrum:
    def test_identity_mlp_exact(self):
        model = IdentityMLP(width=16, depth=4)
        x = _make_batch()
        svs = compute_jacobian_singular_values(model, x)
        # All singular values should be ~1 for identity network
        assert svs.shape[0] == x.shape[0]
        assert svs.shape[1] == 16
        assert torch.allclose(svs, torch.ones_like(svs), atol=1e-4), (
            f"Expected all SVs ~1, got min={svs.min():.4f}, max={svs.max():.4f}"
        )

    def test_identity_mlp_stochastic(self):
        model = IdentityMLP(width=16, depth=4)
        x = _make_batch(batch_size=4)
        # Use more projections than dimensions for accurate randomized SVD
        svs = estimate_jacobian_singular_values(model, x, num_projections=32)
        # For identity Jacobian, all 16 singular values should be ~1
        assert svs.shape[1] == 16  # min(32, 16, 16)
        assert torch.allclose(svs, torch.ones_like(svs), atol=0.15), (
            f"Stochastic SVs too far from 1: min={svs.min():.4f}, max={svs.max():.4f}"
        )

    def test_dynamical_isometry_score_perfect(self):
        svs = torch.ones(8, 16)
        score = dynamical_isometry_score(svs)
        assert score > 0.99, f"Perfect isometry should score ~1, got {score:.4f}"

    def test_dynamical_isometry_score_poor(self):
        svs = torch.tensor([100.0, 0.01, 50.0, 0.001])
        score = dynamical_isometry_score(svs)
        assert score < 0.5, f"Poor isometry should score low, got {score:.4f}"

    def test_jacobian_stats_fields(self):
        svs = torch.ones(4, 16)
        stats = jacobian_stats(svs)
        assert isinstance(stats, JacobianStats)
        assert abs(stats.mean - 1.0) < 1e-4
        assert stats.condition_number < 1.1


class TestLayerJacobians:
    def test_identity_mlp(self):
        model = IdentityMLP(width=16, depth=4)
        x = _make_batch()
        results = compute_layer_jacobians(model, x)
        assert len(results) == 4  # 4 linear layers
        for info in results:
            assert isinstance(info, LayerJacobianInfo)
            # Each layer is identity -> all SVs = 1
            assert torch.allclose(
                info.singular_values,
                torch.ones_like(info.singular_values),
                atol=1e-3,
            ), f"Layer {info.layer_name}: SVs not ~1"
            assert info.condition_number < 1.1


class TestActivationStatistics:
    def test_identity_mlp(self):
        model = IdentityMLP(width=16, depth=4)
        x = _make_batch()
        stats = compute_activation_stats(model, x)
        assert len(stats) > 0
        # For identity network with Gaussian input, variance should stay ~1
        for name, s in stats.items():
            assert isinstance(s, ActivationStats)
            # Variance shouldn't explode or collapse
            assert 0.1 < s.variance < 10.0, (
                f"Layer {name}: variance={s.variance:.4f} out of range"
            )
            # All activations are non-zero (no ReLU killing)
            assert s.nonzero_fraction > 0.9


class TestGradientFlow:
    def test_identity_mlp(self):
        model = IdentityMLP(width=16, depth=4)
        x = _make_batch()
        targets = torch.randn_like(x)
        loss_fn = nn.MSELoss()
        stats = compute_gradient_flow(model, loss_fn, x, targets)
        assert isinstance(stats, GradientFlowStats)
        assert len(stats.gradient_norms) == 4  # 4 weight matrices
        # Gradient norms should be similar across layers (no vanishing/exploding)
        norms = torch.tensor(stats.gradient_norms)
        cv = norms.std() / norms.mean()
        assert cv < 0.5, f"Gradient CV too high: {cv:.4f}"
        # No dead neurons
        for pct in stats.dead_neuron_pct.values():
            assert pct < 0.01


class TestAngularGSNR:
    def test_runs(self):
        """Smoke test that angular GSNR computes without error."""
        model = IdentityMLP(width=16, depth=4)
        loss_fn = nn.MSELoss()

        def loader():
            for _ in range(4):
                yield torch.randn(8, 16), torch.randn(8, 16)

        result = compute_angular_gsnr(model, loss_fn, loader(), num_microbatches=4)
        assert len(result.per_layer) > 0
        for snr in result.per_layer.values():
            assert snr >= 0  # SNR should be non-negative


class TestFullReport:
    def test_runs(self):
        """Smoke test the combined diagnostic report."""
        model = IdentityMLP(width=16, depth=4)
        loss_fn = nn.MSELoss()

        def loader():
            for _ in range(6):
                yield torch.randn(8, 16), torch.randn(8, 16)

        report = run_signal_propagation_diagnostics(
            model, loss_fn, loader(), exact_jacobian=True
        )
        assert "activation_stats" in report
        assert "gradient_flow" in report
        assert "jacobian_stats" in report
        assert "dynamical_isometry_score" in report
        assert "layer_jacobians" in report
        assert "angular_gsnr" in report
        assert report["dynamical_isometry_score"] > 0.9


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])

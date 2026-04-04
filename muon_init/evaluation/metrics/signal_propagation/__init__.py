"""Signal propagation metrics for evaluating initialization quality.

Public API
----------
Jacobian spectrum:
    compute_jacobian_singular_values, estimate_jacobian_singular_values,
    jacobian_stats, dynamical_isometry_score, JacobianStats

Layer Jacobians:
    compute_layer_jacobians, LayerJacobianInfo

Activation statistics:
    compute_activation_stats, ActivationStats

Gradient flow:
    compute_gradient_flow, GradientFlowStats

Angular gradient SNR (Muon-specific):
    compute_angular_gsnr, AngularGSNRResult

Combined report:
    run_signal_propagation_diagnostics, plot_signal_propagation
"""

from .activation_statistics import ActivationStats, compute_activation_stats
from .angular_gradient_snr import AngularGSNRResult, compute_angular_gsnr
from .gradient_flow import GradientFlowStats, compute_gradient_flow
from .jacobian_spectrum import (
    JacobianStats,
    compute_jacobian_singular_values,
    dynamical_isometry_score,
    estimate_jacobian_singular_values,
    jacobian_stats,
)
from .layer_jacobians import LayerJacobianInfo, compute_layer_jacobians
from .signal_propagation_report import (
    plot_signal_propagation,
    run_signal_propagation_diagnostics,
)

__all__ = [
    # jacobian_spectrum
    "compute_jacobian_singular_values",
    "estimate_jacobian_singular_values",
    "jacobian_stats",
    "dynamical_isometry_score",
    "JacobianStats",
    # layer_jacobians
    "compute_layer_jacobians",
    "LayerJacobianInfo",
    # activation_statistics
    "compute_activation_stats",
    "ActivationStats",
    # gradient_flow
    "compute_gradient_flow",
    "GradientFlowStats",
    # angular_gradient_snr
    "compute_angular_gsnr",
    "AngularGSNRResult",
    # report
    "run_signal_propagation_diagnostics",
    "plot_signal_propagation",
]

"""Muon-specific initialization quality metrics.

Novel metrics designed to evaluate initialization quality in the geometry
that the Muon optimizer actually operates in.  Standard optimization metrics
assume Euclidean parameter space geometry, which is incorrect for Muon since
it orthogonalizes all updates via polar decomposition (Newton-Schulz
iterations), producing partial isometries with all singular values = 1.

Modules
-------
polar_error
    Measures how far weight matrices are from their polar factor, and how
    many Newton-Schulz iterations are needed to converge.
spectral_norm_ball
    Checks spectral norm ball membership (||W||_op <= 1/lambda_wd) and
    tracks Phase 1 duration during training.
stiefel_sharpness
    Sharpness projected onto the Stiefel manifold tangent space — curvature
    in directions Muon can actually move.
angular_update_analysis
    Simulates Muon's first step and analyzes update-weight alignment,
    effective rank, and spectral changes.
"""

# --- polar_error ---
from evaluation.metrics.muon_specific.polar_error import (
    polar_factor,
    compute_polar_error,
    compute_polar_error_all_layers,
    newton_schulz_convergence_steps,
    muon_ns_approximation_quality,
    analyze_polar_error,
    PolarErrorReport,
)

# --- spectral_norm_ball ---
from evaluation.metrics.muon_specific.spectral_norm_ball import (
    SpectralNormStatus,
    check_spectral_norm_ball,
    compute_spectral_excess,
    Phase1Tracker,
)

# --- stiefel_sharpness ---
from evaluation.metrics.muon_specific.stiefel_sharpness import (
    project_to_stiefel_tangent,
    random_stiefel_tangent_vector,
    verify_tangent_condition,
    stiefel_sharpness,
    SharpnessResult,
)

# --- angular_update_analysis ---
from evaluation.metrics.muon_specific.angular_update_analysis import (
    analyze_first_muon_step,
    FirstStepAnalysis,
    LayerFirstStepDiagnostics,
)

__all__ = [
    # polar_error
    "polar_factor",
    "compute_polar_error",
    "compute_polar_error_all_layers",
    "newton_schulz_convergence_steps",
    "muon_ns_approximation_quality",
    "analyze_polar_error",
    "PolarErrorReport",
    # spectral_norm_ball
    "SpectralNormStatus",
    "check_spectral_norm_ball",
    "compute_spectral_excess",
    "Phase1Tracker",
    # stiefel_sharpness
    "project_to_stiefel_tangent",
    "random_stiefel_tangent_vector",
    "verify_tangent_condition",
    "stiefel_sharpness",
    "SharpnessResult",
    # angular_update_analysis
    "analyze_first_muon_step",
    "FirstStepAnalysis",
    "LayerFirstStepDiagnostics",
]

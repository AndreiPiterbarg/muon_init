"""Hessian and loss landscape metrics for evaluating initialization quality.

Metrics implemented (Sections 2–3 of the optimization metrics survey):

- **lambda_max**: Hessian top eigenvalue via power iteration
- **Hessian trace**: Hutchinson's stochastic estimator + spikiness ratio
- **Spectral density**: Full eigenvalue density via Stochastic Lanczos Quadrature
- **Normalized sharpness**: Scale-invariant metrics (Neyshabur et al., Tsuzuku et al.)
- **Edge of Stability**: Tracker for eta * lambda_max dynamics (Cohen et al.)
"""

from .hessian_top_eigenvalue import compute_lambda_max
from .hessian_trace import compute_hessian_trace, compute_spikiness
from .hessian_spectral_density import compute_spectral_density
from .normalized_sharpness import (
    compute_normalized_lambda_max,
    compute_normalized_trace,
    compute_weight_norm_squared,
)
from .edge_of_stability import EoSTracker

__all__ = [
    "compute_lambda_max",
    "compute_hessian_trace",
    "compute_spikiness",
    "compute_spectral_density",
    "compute_normalized_lambda_max",
    "compute_normalized_trace",
    "compute_weight_norm_squared",
    "EoSTracker",
]

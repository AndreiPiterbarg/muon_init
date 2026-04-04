"""Spectral metrics for evaluating weight matrix properties.

This package provides metrics for analyzing the singular value structure
of neural network weight matrices, with a focus on properties relevant
to the Muon optimizer's orthogonalized update geometry.
"""

from .effective_rank import effective_rank, effective_rank_all_layers
from .stable_rank import stable_rank, stable_rank_all_layers
from .condition_number import condition_number, condition_number_all_layers
from .svd_entropy import svd_entropy, svd_entropy_all_layers
from .empirical_spectral_density import compute_esd, marchenko_pastur_fit
from .spectral_norm_membership import check_spectral_norm_ball, spectral_norm_ratio
from .spectral_tracker import SpectralTracker

__all__ = [
    # Effective rank
    "effective_rank",
    "effective_rank_all_layers",
    # Stable rank
    "stable_rank",
    "stable_rank_all_layers",
    # Condition number
    "condition_number",
    "condition_number_all_layers",
    # SVD entropy
    "svd_entropy",
    "svd_entropy_all_layers",
    # Empirical spectral density
    "compute_esd",
    "marchenko_pastur_fit",
    # Spectral norm ball
    "check_spectral_norm_ball",
    "spectral_norm_ratio",
    # Tracker
    "SpectralTracker",
]

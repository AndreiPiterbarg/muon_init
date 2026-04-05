from .baselines.baselines import (
    kaiming_normal,
    kaiming_uniform,
    xavier_normal,
    xavier_uniform,
    orthogonal,
)
from .implementations.scaled_orthogonal import (
    scaled_orthogonal,
    optimal_alpha,
    compute_activation_gain,
)

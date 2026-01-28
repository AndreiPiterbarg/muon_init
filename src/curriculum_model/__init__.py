"""
Curriculum Model Package - Minimal version for Self-Refine ICL experiments.
"""

from .embedders import (
    VectorEmbedder,
    MatrixEmbedder,
    ScalarEmbedder,
    ComponentEmbedders,
)

from .roles import (
    Role,
    NUM_ROLES,
    RoleEmbedding,
    compose_token,
)

from .special_tokens import SpecialTokens

from .tasks import (
    OutputType,
    ComponentSpec,
    TaskSpec,
)

from .sequence_builder import (
    COMPONENT_ORDER,
    SequenceOutput,
    SequenceBuilder,
    PositionalEncoder,
)

from .output_heads import (
    OutputHeadResult,
    VectorHead,
    ScalarHead,
    DualOutputHead,
    compute_task_loss,
)

from .component_model import (
    ComponentModelConfig,
    ComponentModelOutput,
    ComponentTransformerModel,
    create_model,
)

__all__ = [
    # Embedders
    "VectorEmbedder",
    "MatrixEmbedder",
    "ScalarEmbedder",
    "ComponentEmbedders",
    # Roles
    "Role",
    "NUM_ROLES",
    "RoleEmbedding",
    "compose_token",
    # Special Tokens
    "SpecialTokens",
    # Tasks
    "OutputType",
    "ComponentSpec",
    "TaskSpec",
    # Sequence Builder
    "COMPONENT_ORDER",
    "SequenceOutput",
    "SequenceBuilder",
    "PositionalEncoder",
    # Output Heads
    "OutputHeadResult",
    "VectorHead",
    "ScalarHead",
    "DualOutputHead",
    "compute_task_loss",
    # Component Model
    "ComponentModelConfig",
    "ComponentModelOutput",
    "ComponentTransformerModel",
    "create_model",
]

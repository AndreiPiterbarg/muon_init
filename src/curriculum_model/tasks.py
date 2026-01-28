"""
Task registry for the curriculum transformer.

This module defines the task specifications and registry pattern for
primitive tasks. Each task specifies:
- A name identifier
- Input component specifications (what components and their roles)
- Output type (vector or scalar)
- A reference to the generation function
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Callable, Any, Optional
import torch

from .roles import Role


class OutputType(Enum):
    """Type of output produced by a task."""
    VECTOR = "vector"
    SCALAR = "scalar"


@dataclass
class ComponentSpec:
    """Specification for a component in a task."""
    name: str  # Component name (e.g., "x", "A", "alpha")
    role: Role  # Semantic role
    component_type: str  # "vector", "matrix", or "scalar"


@dataclass
class TaskSpec:
    """Specification for a primitive task."""
    name: str
    components: List[ComponentSpec]
    output_type: OutputType
    description: str = ""

    def get_component_names(self) -> List[str]:
        """Get list of component names in order."""
        return [c.name for c in self.components]

    def get_component_by_name(self, name: str) -> Optional[ComponentSpec]:
        """Get component spec by name."""
        for c in self.components:
            if c.name == name:
                return c
        return None

    def has_matrix(self) -> bool:
        """Check if task involves a matrix component."""
        return any(c.component_type == "matrix" for c in self.components)


# ============================================================================
# Task Specifications
# ============================================================================

VECTOR_ADD_SPEC = TaskSpec(
    name="vector_add",
    components=[
        ComponentSpec("x", Role.VEC_PRIMARY, "vector"),
        ComponentSpec("z", Role.VEC_SECONDARY, "vector"),
        ComponentSpec("y", Role.OUTPUT, "vector"),
    ],
    output_type=OutputType.VECTOR,
    description="y = x + z"
)

SCALAR_MUL_SPEC = TaskSpec(
    name="scalar_mul",
    components=[
        ComponentSpec("alpha", Role.SCALAR, "scalar"),
        ComponentSpec("x", Role.VEC_PRIMARY, "vector"),
        ComponentSpec("y", Role.OUTPUT, "vector"),
    ],
    output_type=OutputType.VECTOR,
    description="y = alpha * x"
)

INNER_PRODUCT_SPEC = TaskSpec(
    name="inner_product",
    components=[
        ComponentSpec("x", Role.VEC_PRIMARY, "vector"),
        ComponentSpec("y", Role.VEC_SECONDARY, "vector"),
        ComponentSpec("s", Role.OUTPUT, "scalar"),
    ],
    output_type=OutputType.SCALAR,
    description="s = x^T y"
)

MAT_VEC_SPEC = TaskSpec(
    name="mat_vec",
    components=[
        ComponentSpec("A", Role.MATRIX, "matrix"),
        ComponentSpec("x", Role.VEC_PRIMARY, "vector"),
        ComponentSpec("y", Role.OUTPUT, "vector"),
    ],
    output_type=OutputType.VECTOR,
    description="y = A @ x"
)

GRADIENT_SPEC = TaskSpec(
    name="gradient",
    components=[
        ComponentSpec("A", Role.MATRIX, "matrix"),
        ComponentSpec("b", Role.VEC_BIAS, "vector"),
        ComponentSpec("x", Role.VEC_SECONDARY, "vector"),
        ComponentSpec("g", Role.OUTPUT, "vector"),
    ],
    output_type=OutputType.VECTOR,
    description="g = A @ x - b (gradient of quadratic)"
)

QUAD_VALUE_SPEC = TaskSpec(
    name="quad_value",
    components=[
        ComponentSpec("A", Role.MATRIX, "matrix"),
        ComponentSpec("b", Role.VEC_BIAS, "vector"),
        ComponentSpec("x", Role.VEC_SECONDARY, "vector"),
        ComponentSpec("f", Role.OUTPUT, "scalar"),
    ],
    output_type=OutputType.SCALAR,
    description="f = 0.5 * x^T A x - b^T x (quadratic value)"
)

ICL_SPEC = TaskSpec(
    name="icl",
    components=[
        ComponentSpec("A", Role.MATRIX, "matrix"),
        ComponentSpec("b", Role.VEC_BIAS, "vector"),
        ComponentSpec("x_star", Role.OUTPUT, "vector"),
    ],
    output_type=OutputType.VECTOR,
    description="x* = A^{-1} b (in-context learning of linear system)"
)

# Level 4: Composed operation tasks

GRAD_STEP_SPEC = TaskSpec(
    name="grad_step",
    components=[
        ComponentSpec("x", Role.VEC_PRIMARY, "vector"),
        ComponentSpec("g", Role.VEC_SECONDARY, "vector"),
        ComponentSpec("alpha", Role.SCALAR, "scalar"),
        ComponentSpec("x_next", Role.OUTPUT, "vector"),
    ],
    output_type=OutputType.VECTOR,
    description="x_next = x - alpha * g (gradient step)"
)

SINGLE_ITER_SPEC = TaskSpec(
    name="single_iter",
    components=[
        ComponentSpec("A", Role.MATRIX, "matrix"),
        ComponentSpec("b", Role.VEC_BIAS, "vector"),
        ComponentSpec("x", Role.VEC_SECONDARY, "vector"),
        ComponentSpec("alpha", Role.SCALAR, "scalar"),
        ComponentSpec("x_next", Role.OUTPUT, "vector"),
    ],
    output_type=OutputType.VECTOR,
    description="x_next = x - alpha * (A @ x - b) (single GD iteration)"
)

MULTI_STEP_TRAJECTORY_SPEC = TaskSpec(
    name="multi_step_trajectory",
    components=[
        ComponentSpec("A", Role.MATRIX, "matrix"),
        ComponentSpec("b", Role.VEC_BIAS, "vector"),
        ComponentSpec("x", Role.VEC_SECONDARY, "vector"),
        ComponentSpec("alpha", Role.SCALAR, "scalar"),
        ComponentSpec("x_next", Role.OUTPUT, "vector"),
    ],
    output_type=OutputType.VECTOR,
    description="Multi-step GD trajectory: predict x_{t+1} from x_t"
)


# ============================================================================
# Task Registry
# ============================================================================

class TaskRegistry:
    """Registry for primitive tasks."""

    def __init__(self):
        self._specs: Dict[str, TaskSpec] = {}
        self._generators: Dict[str, Callable] = {}

    def register(self, spec: TaskSpec, generator: Callable):
        """Register a task with its specification and generator."""
        self._specs[spec.name] = spec
        self._generators[spec.name] = generator

    def get_spec(self, name: str) -> TaskSpec:
        """Get task specification by name."""
        if name not in self._specs:
            raise KeyError(f"Unknown task: {name}. Available: {list(self._specs.keys())}")
        return self._specs[name]

    def get_generator(self, name: str) -> Callable:
        """Get task generator by name."""
        if name not in self._generators:
            raise KeyError(f"Unknown task: {name}. Available: {list(self._generators.keys())}")
        return self._generators[name]

    def list_tasks(self) -> List[str]:
        """List all registered task names."""
        return list(self._specs.keys())

    def list_vector_output_tasks(self) -> List[str]:
        """List tasks that output vectors."""
        return [name for name, spec in self._specs.items()
                if spec.output_type == OutputType.VECTOR]

    def list_scalar_output_tasks(self) -> List[str]:
        """List tasks that output scalars."""
        return [name for name, spec in self._specs.items()
                if spec.output_type == OutputType.SCALAR]

    def list_matrix_tasks(self) -> List[str]:
        """List tasks that involve matrices."""
        return [name for name, spec in self._specs.items()
                if spec.has_matrix()]


# Global registry instance
TASK_REGISTRY = TaskRegistry()


def register_task(spec: TaskSpec):
    """Decorator to register a task generator."""
    def decorator(generator: Callable):
        TASK_REGISTRY.register(spec, generator)
        return generator
    return decorator


def get_task_spec(name: str) -> TaskSpec:
    """Get task specification by name."""
    return TASK_REGISTRY.get_spec(name)


def get_task_generator(name: str) -> Callable:
    """Get task generator by name."""
    return TASK_REGISTRY.get_generator(name)


def list_all_tasks() -> List[str]:
    """List all registered tasks."""
    return TASK_REGISTRY.list_tasks()

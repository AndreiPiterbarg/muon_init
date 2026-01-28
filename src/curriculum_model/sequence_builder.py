"""
Sequence builder for the curriculum transformer.

This module converts task data into token sequences that can be processed
by the transformer. It handles:
- Embedding components with appropriate embedders
- Adding role embeddings
- Maintaining fixed component ordering
- Adding SEP tokens between examples
- Placing MASK token at query output position
- Example-level positional encoding
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn

from .embedders import ComponentEmbedders
from .roles import Role, RoleEmbedding, compose_token
from .special_tokens import SpecialTokens
from .tasks import TaskSpec, OutputType


# Component ordering: the fixed order in which components appear in sequences
COMPONENT_ORDER = [
    Role.MATRIX,      # A
    Role.VEC_BIAS,    # b
    Role.VEC_PRIMARY, # x (primary)
    Role.VEC_SECONDARY, # z (secondary)
    Role.SCALAR,      # alpha
    Role.OUTPUT,      # y/s/g/f/x*
]


@dataclass
class SequenceOutput:
    """Output from sequence building."""
    tokens: torch.Tensor  # (batch_size, seq_len, n_embd)
    mask_positions: torch.Tensor  # (batch_size,) indices of MASK tokens
    output_type: OutputType  # Whether output is vector or scalar
    seq_len: int  # Sequence length
    example_positions: torch.Tensor  # (batch_size, seq_len) example index for each token


class SequenceBuilder(nn.Module):
    """
    Builds token sequences from task data.

    Takes task inputs and converts them into embedded token sequences
    ready for the transformer backbone.
    """

    def __init__(
        self,
        d: int,
        n_embd: int,
        embedders: Optional[ComponentEmbedders] = None,
        role_embedding: Optional[RoleEmbedding] = None,
        special_tokens: Optional[SpecialTokens] = None,
    ):
        """
        Args:
            d: Dimension for vectors and matrices
            n_embd: Transformer hidden dimension
            embedders: Component embedders (created if not provided)
            role_embedding: Role embeddings (created if not provided)
            special_tokens: Special tokens (created if not provided)
        """
        super().__init__()
        self.d = d
        self.n_embd = n_embd

        # Create or use provided modules
        self.embedders = embedders if embedders is not None else ComponentEmbedders(d, n_embd)
        self.role_embedding = role_embedding if role_embedding is not None else RoleEmbedding(n_embd)
        self.special_tokens = special_tokens if special_tokens is not None else SpecialTokens(n_embd)

    def _get_component_order(self, task_spec: TaskSpec) -> List[Tuple[str, Role, str]]:
        """
        Get ordered list of components for a task.

        Returns list of (name, role, type) tuples in canonical order.
        """
        # Build lookup from role to component
        role_to_component = {}
        for comp in task_spec.components:
            role_to_component[comp.role] = (comp.name, comp.role, comp.component_type)

        # Return components in canonical order, skipping missing ones
        ordered = []
        for role in COMPONENT_ORDER:
            if role in role_to_component:
                ordered.append(role_to_component[role])

        return ordered

    def _embed_component(
        self,
        value: torch.Tensor,
        component_type: str,
        role: Role,
    ) -> torch.Tensor:
        """
        Embed a component and add its role embedding.

        Args:
            value: Component tensor
            component_type: "vector", "matrix", or "scalar"
            role: Semantic role

        Returns:
            Token embedding of shape (..., n_embd)
        """
        # Embed based on type
        if component_type == "vector":
            embedded = self.embedders.embed_vector(value)
        elif component_type == "matrix":
            embedded = self.embedders.embed_matrix(value)
        elif component_type == "scalar":
            embedded = self.embedders.embed_scalar(value)
        else:
            raise ValueError(f"Unknown component type: {component_type}")

        # Add role embedding
        role_emb = self.role_embedding.get_role(role)
        return compose_token(embedded, role_emb)

    def _infer_batch_size(self, inputs: Dict[str, torch.Tensor], task_spec: TaskSpec) -> int:
        """Infer batch size from inputs, handling shared components."""
        ordered_components = self._get_component_order(task_spec)
        for name, role, comp_type in ordered_components:
            if name not in inputs:
                continue
            tensor = inputs[name]
            # For vectors: batch shape is (B, d), non-batch is (d,)
            # For matrices: batch shape is (B, d, d), non-batch is (d, d)
            # For scalars: batch shape is (B, 1) or (B,), non-batch is (1,) or ()
            if comp_type == "vector" and tensor.dim() == 2:
                return tensor.shape[0]
            elif comp_type == "matrix" and tensor.dim() == 3:
                return tensor.shape[0]
            elif comp_type == "scalar" and tensor.dim() >= 1 and tensor.shape[0] > 1:
                return tensor.shape[0]
        # Default to 1 if we can't determine
        return 1

    def build_example_tokens(
        self,
        inputs: Dict[str, torch.Tensor],
        task_spec: TaskSpec,
        is_query: bool = False,
    ) -> torch.Tensor:
        """
        Build tokens for a single example.

        Args:
            inputs: Dict mapping component names to tensors, shape (batch_size, ...)
                   Some components may be shared (without batch dim) and will be expanded.
            task_spec: Task specification
            is_query: If True, replace output with MASK token

        Returns:
            Tokens of shape (batch_size, num_components, n_embd)
        """
        ordered_components = self._get_component_order(task_spec)
        batch_size = self._infer_batch_size(inputs, task_spec)

        tokens = []
        for name, role, comp_type in ordered_components:
            if role == Role.OUTPUT and is_query:
                # For query, use MASK token with OUTPUT role
                mask_token = self.special_tokens.get_mask_with_role(
                    self.role_embedding.get_output_role()
                )
                # Expand to batch
                token = mask_token.unsqueeze(0).expand(batch_size, -1)
            else:
                # Normal component embedding
                if name not in inputs:
                    raise KeyError(f"Missing component '{name}' in inputs")

                value = inputs[name]

                # Check if this is a shared component (no batch dimension)
                is_shared = False
                if comp_type == "vector" and value.dim() == 1:
                    is_shared = True
                    value = value.unsqueeze(0).expand(batch_size, -1)
                elif comp_type == "matrix" and value.dim() == 2:
                    is_shared = True
                    value = value.unsqueeze(0).expand(batch_size, -1, -1)
                elif comp_type == "scalar" and (value.dim() == 0 or (value.dim() == 1 and value.shape[0] == 1)):
                    is_shared = True
                    if value.dim() == 0:
                        value = value.unsqueeze(0)
                    value = value.expand(batch_size, -1)

                token = self._embed_component(value, comp_type, role)

                # Ensure batch dimension
                if token.dim() == 1:
                    token = token.unsqueeze(0).expand(batch_size, -1)

            tokens.append(token)

        # Stack: (batch_size, num_components, n_embd)
        return torch.stack(tokens, dim=1)

    def build_sequence(
        self,
        context_examples: List[Dict[str, torch.Tensor]],
        query_inputs: Dict[str, torch.Tensor],
        task_spec: TaskSpec,
    ) -> SequenceOutput:
        """
        Build a complete sequence with context examples and query.

        Args:
            context_examples: List of input dicts for context examples
            query_inputs: Input dict for query (without target)
            task_spec: Task specification

        Returns:
            SequenceOutput with tokens, mask positions, and metadata
        """
        if len(context_examples) == 0:
            raise ValueError("Must have at least one context example")

        batch_size = next(iter(query_inputs.values())).shape[0]
        device = next(iter(query_inputs.values())).device

        all_tokens = []
        example_positions = []
        current_pos = 0

        # Process context examples
        for ex_idx, example in enumerate(context_examples):
            # Build tokens for this example
            ex_tokens = self.build_example_tokens(example, task_spec, is_query=False)
            num_tokens = ex_tokens.shape[1]

            all_tokens.append(ex_tokens)
            example_positions.extend([ex_idx] * num_tokens)

            # Add SEP token
            sep = self.special_tokens.get_sep_batch(batch_size)
            all_tokens.append(sep.unsqueeze(1))
            example_positions.append(ex_idx)

            current_pos += num_tokens + 1

        # Process query
        query_idx = len(context_examples)
        query_tokens = self.build_example_tokens(query_inputs, task_spec, is_query=True)
        num_query_tokens = query_tokens.shape[1]

        all_tokens.append(query_tokens)
        example_positions.extend([query_idx] * num_query_tokens)

        # Find MASK position (last token in query, which is the OUTPUT position)
        ordered_components = self._get_component_order(task_spec)
        mask_offset = len(ordered_components) - 1  # OUTPUT is last in order

        mask_positions = torch.full((batch_size,), current_pos + mask_offset, dtype=torch.long, device=device)

        # Concatenate all tokens: (batch_size, seq_len, n_embd)
        tokens = torch.cat(all_tokens, dim=1)
        seq_len = tokens.shape[1]

        # Create example positions tensor
        example_pos_tensor = torch.tensor(example_positions, dtype=torch.long, device=device)
        example_pos_tensor = example_pos_tensor.unsqueeze(0).expand(batch_size, -1)

        return SequenceOutput(
            tokens=tokens,
            mask_positions=mask_positions,
            output_type=task_spec.output_type,
            seq_len=seq_len,
            example_positions=example_pos_tensor,
        )

    def build_from_batch(
        self,
        inputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        task_spec: TaskSpec,
        num_context: int,
    ) -> SequenceOutput:
        """
        Build sequence from a batch where examples are stacked.

        This is a convenience method for the common case where we have
        a batch of examples and want to use the first num_context as
        context and the last one as query.

        Args:
            inputs: Dict with tensors of shape (batch_size, num_examples, ...)
            targets: Target tensor of shape (batch_size, num_examples, ...)
            task_spec: Task specification
            num_context: Number of context examples to use

        Returns:
            SequenceOutput
        """
        batch_size = targets.shape[0]
        num_examples = targets.shape[1]

        if num_context >= num_examples:
            raise ValueError(f"num_context ({num_context}) must be < num_examples ({num_examples})")

        # Identify shared components (matrices that are not per-example)
        # Use the task spec to determine component types
        shared_names = set()
        for name in inputs:
            comp = task_spec.get_component_by_name(name)
            if comp is not None and comp.component_type == "matrix":
                # Matrix components are shared across examples: (B, d, d)
                shared_names.add(name)
            elif inputs[name].dim() == 2:
                # Scalars/vectors with no example dim: (B, d) or (B, 1)
                shared_names.add(name)

        # Split into context and query
        context_examples = []
        for i in range(num_context):
            example = {}
            for name, tensor in inputs.items():
                if name in shared_names:
                    example[name] = tensor
                else:
                    # Shape (batch_size, num_examples, ...) - indexed by example
                    example[name] = tensor[:, i]

            # Add target as output component
            output_name = self._get_output_name(task_spec)
            example[output_name] = targets[:, i]
            context_examples.append(example)

        # Query inputs (no target)
        query = {}
        for name, tensor in inputs.items():
            if name in shared_names:
                query[name] = tensor
            else:
                query[name] = tensor[:, num_context]

        return self.build_sequence(context_examples, query, task_spec)

    def _get_output_name(self, task_spec: TaskSpec) -> str:
        """Get the name of the output component."""
        for comp in task_spec.components:
            if comp.role == Role.OUTPUT:
                return comp.name
        raise ValueError(f"Task {task_spec.name} has no OUTPUT component")


class PositionalEncoder(nn.Module):
    """
    Adds example-level positional encoding to sequences.

    All tokens from the same example receive the same position embedding,
    indexed by example number rather than absolute token position.
    """

    def __init__(self, n_embd: int, max_examples: int = 64):
        """
        Args:
            n_embd: Embedding dimension
            max_examples: Maximum number of examples supported
        """
        super().__init__()
        self.n_embd = n_embd
        self.max_examples = max_examples
        self.position_embedding = nn.Embedding(max_examples, n_embd)

    def forward(
        self,
        tokens: torch.Tensor,
        example_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add positional encoding based on example positions.

        Args:
            tokens: Token embeddings of shape (batch_size, seq_len, n_embd)
            example_positions: Example index for each token, shape (batch_size, seq_len)

        Returns:
            Tokens with positional encoding added
        """
        pos_emb = self.position_embedding(example_positions)
        return tokens + pos_emb


def build_task_sequence(
    task_name: str,
    batch_size: int,
    d: int,
    n_embd: int,
    num_context: int,
    num_examples: int,
    embedders: Optional[ComponentEmbedders] = None,
    role_embedding: Optional[RoleEmbedding] = None,
    special_tokens: Optional[SpecialTokens] = None,
    seed: Optional[int] = None,
    **generator_kwargs,
) -> Tuple[SequenceOutput, torch.Tensor]:
    """
    Convenience function to generate task data and build sequence.

    Args:
        task_name: Name of the task
        batch_size: Number of sequences
        d: Vector/matrix dimension
        n_embd: Embedding dimension
        num_context: Number of context examples
        num_examples: Total examples per episode (context + query)
        embedders: Optional pre-existing embedders
        role_embedding: Optional pre-existing role embedding
        special_tokens: Optional pre-existing special tokens
        seed: Random seed
        **generator_kwargs: Additional arguments for the generator

    Returns:
        (SequenceOutput, targets) where targets is the query target
    """
    from .tasks import get_task_spec, get_task_generator

    spec = get_task_spec(task_name)
    generator = get_task_generator(task_name)

    # Build sequence builder
    builder = SequenceBuilder(
        d=d,
        n_embd=n_embd,
        embedders=embedders,
        role_embedding=role_embedding,
        special_tokens=special_tokens,
    )

    # Handle episode-based tasks (ICL, multi_step_trajectory) specially
    if task_name == "icl":
        inputs, targets, _ = generator(
            batch_size=batch_size,
            d=d,
            num_examples=num_examples,
            seed=seed,
            **generator_kwargs,
        )
        seq_output = builder.build_from_batch(inputs, targets, spec, num_context)
        query_targets = targets[:, num_context]
    elif task_name == "multi_step_trajectory":
        inputs, targets, _ = generator(
            batch_size=batch_size,
            d=d,
            num_steps=num_examples,
            seed=seed,
            **generator_kwargs,
        )
        seq_output = builder.build_from_batch(inputs, targets, spec, num_context)
        query_targets = targets[:, num_context]
    else:
        # For other tasks, generate examples one at a time
        # This is a simplification - in practice, you'd batch this differently
        all_inputs = {name: [] for name in spec.get_component_names() if name != builder._get_output_name(spec)}
        all_targets = []

        for _ in range(num_examples):
            inputs, target = generator(
                batch_size=batch_size,
                d=d,
                seed=seed,
                **generator_kwargs,
            )
            for name, value in inputs.items():
                if name in all_inputs:
                    all_inputs[name].append(value)
            all_targets.append(target)

            # Increment seed for variety
            if seed is not None:
                seed += 1

        # Stack examples
        stacked_inputs = {name: torch.stack(vals, dim=1) for name, vals in all_inputs.items()}
        stacked_targets = torch.stack(all_targets, dim=1)

        seq_output = builder.build_from_batch(stacked_inputs, stacked_targets, spec, num_context)
        query_targets = stacked_targets[:, num_context]

    return seq_output, query_targets

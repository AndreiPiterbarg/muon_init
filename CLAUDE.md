# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project implementing iterative self-refinement for transformers trained on in-context learning (ICL) with linear systems. The core contribution is demonstrating that naive self-refinement fails catastrophically and proposing role-based disambiguation as a solution.

## Research Structure

The project is organized by research contribution:

1. **Section 1 - Phenomenon**: Naive ICL self-refinement fails (1600x degradation)
2. **Section 2 - Solution**: Role-based disambiguation (Role-Disambiguated Residual)
3. **Section 3 - Analysis**: What algorithm does the model learn?
4. **Section 4 - Bonus**: Classical solver comparison

## Commands

```bash
# Section 1: Demonstrate the phenomenon
python experiments/section1_phenomenon/naive_refinement_failure.py --device cuda

# Section 2: Train Role-Disambiguated Residual
python experiments/section2_solution/role_disambiguated_residual.py --device cuda

# Section 2: Run full comparison (Baseline, Iterative Supervision, Role-Disambiguated Residual)
python experiments/section2_solution/run_comparison.py --device cuda

# Run tests
python -m pytest tests/ -v

# Install dependencies
pip install torch numpy scipy
```

## Architecture

### Module Structure

**`src/curriculum_model/`** - Model components:
- `component_model.py`: Main `ComponentTransformerModel` class
- `roles.py`: 6 semantic roles (MATRIX, VEC_PRIMARY, VEC_SECONDARY, VEC_BIAS, SCALAR, OUTPUT)
- `embedders.py`: Vector/Matrix/Scalar embedders
- `special_tokens.py`: SEP and MASK tokens
- `sequence_builder.py`: PositionalEncoder (example-level positional encoding)
- `output_heads.py`: Dual output head (vector + scalar)

**`src/custom_transformer/`** - GPT-style transformer:
- `transformer.py`: CustomGPTBackbone
- `block.py`: TransformerBlock (pre-norm architecture)
- `attention.py`: Multi-head causal attention
- `ffn.py`: Feed-forward network

**`src/data/`** - Data generation:
- `spd_sampler.py`: SPD matrix sampling with controlled condition numbers

**`experiments/`** - Organized by research section:
- `section1_phenomenon/`: Naive refinement failure
- `section2_solution/`: Role-Disambiguated Residual and ablations
- `section3_analysis/`: Algorithm hypothesis testing (???)
- `section4_bonus/`: Classical solver comparison (???)

### Token Composition
```
token = embed_component(component) + embed_role(role)
```

### Key Design: Role-Based Disambiguation
Current estimate uses VEC_SECONDARY role; ground-truth solutions use OUTPUT role:
```
[SEP, A, SEP, b_1, x_1*, ..., SEP, b_query, x_tilde, MASK]
                                           ^^^^^^^^
                                    VEC_SECONDARY role
```

### Refinement Algorithm
```
x_0 = f(context, query)                 # Initial prediction
x_{k+1} = x_k + f(context, query, x_k)  # Refinement iterations
```

## Key Configuration

```python
# experiments/section2_solution/role_disambiguated_residual.py
d = 4                    # Vector/matrix dimension
n_embd = 128            # Transformer hidden dimension
n_layer = 6, n_head = 4 # Transformer architecture
training_steps = 50000
residual_weight = 0.5   # Mix of direct/residual loss
noise_scale = 0.5       # Noise for estimate perturbation
num_context = 5         # Context examples per sample
kappa_min = 1.0         # Condition number range
kappa_max = 100.0
```

## Data Generation

SPD matrices sampled with controlled condition numbers (eigenvalues on log scale). Training uses dual loss:
- L_direct = ||f(C, b_query) - x*||²
- L_residual = ||f(C, b_query, x_tilde) - (x* - x_tilde)||², where x_tilde = pred_0 + noise

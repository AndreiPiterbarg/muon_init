# Self-Refine ICL: Iterative Self-Refinement in Transformer In-Context Learning

Research project investigating why naive ICL self-refinement fails catastrophically and proposing a solution using role-based disambiguation.

## Research Structure

This project is organized into four research contributions:

### 1. Phenomenon: Naive ICL Self-Refinement Fails Catastrophically
**Problem discovery contribution** - documenting something the field hasn't characterized.

```
Standard ICL:           MSE = 5.93e-05
Naive refinement (K=1): MSE = 0.095      ← 1600x worse!
Naive refinement (K=2): MSE = ???
...
```

### 2. Solution: Role-Based Disambiguation
**Architectural contribution** - a principled fix with potential generality.

| Configuration | Performance |
|--------------|-------------|
| No role embedding, no dual objective | 1600x degradation (naive baseline) |
| Role embedding only | ??? |
| Dual objective only | ??? |
| Both (Role-Disambiguated Residual) | ??? (needs optimization) |

### 3. Analysis: What Algorithm Does the Model Learn?
**Scientific contribution** - the part that makes the paper citable.

For each refinement step, observe:
- Current estimate: x_k
- Correction: δ_k = f(context, query, x_k)
- Next estimate: x_{k+1} = x_k + δ_k

What does δ_k approximate?
- Richardson iteration?
- Gradient descent?
- Newton's method?
- Something new?

```
κ ∈ [1, 10]:    Model learns ???
κ ∈ [50, 100]:  Model learns ???
κ ∈ [100, 200]: Model learns ???
```

### 4. Bonus: Extrapolation and Classical Solver Comparison
Compare learned refinement against classical iterative solvers:
- Jacobi iteration
- Gauss-Seidel
- Conjugate Gradient

## Project Structure

```
Self_Refine_ICL/
├── experiments/                          # Organized by research section
│   ├── section1_phenomenon/              # 1. Naive refinement fails
│   │   └── naive_refinement_failure.py   # Demonstrate 1600x degradation
│   │
│   ├── section2_solution/                # 2. Role-based disambiguation
│   │   ├── role_disambiguated_residual.py # Main approach (residual prediction)
│   │   ├── ablation_study.py             # Role embedding vs dual objective (???)
│   │   └── run_comparison.py             # Compare all approaches
│   │
│   ├── section3_analysis/                # 3. What algorithm is learned? (???)
│   │   ├── hypothesis_tests.py           # Richardson, GD, Newton tests
│   │   └── kappa_range_analysis.py       # Analysis by condition number
│   │
│   └── section4_bonus/                   # 4. Classical solver comparison (???)
│       ├── classical_solvers.py          # Jacobi, GS, CG implementations
│       └── extrapolation.py              # Test extrapolation capabilities
│
├── src/                                  # Core model implementation
│   ├── curriculum_model/                 # ICL model components
│   │   ├── component_model.py            # Main ComponentTransformerModel
│   │   ├── roles.py                      # 6 semantic roles (KEY!)
│   │   ├── embedders.py                  # Vector/Matrix/Scalar embedders
│   │   └── ...
│   │
│   ├── custom_transformer/               # GPT-style transformer
│   │   ├── transformer.py                # CustomGPTBackbone
│   │   └── ...
│   │
│   └── data/                             # Data generation utilities
│       └── spd_sampler.py                # SPD matrix sampling
│
├── tests/                                # Test suite
├── results/                              # Experiment results
│   ├── section1/
│   ├── section2/
│   ├── section3/
│   └── section4/
│
└── scripts/                              # (Legacy) Original experiment scripts
```

## Quick Start

```bash
# Install dependencies
pip install torch numpy scipy

# Run Section 1: Demonstrate naive refinement failure
python experiments/section1_phenomenon/naive_refinement_failure.py --device cuda

# Run Section 2: Train Role-Disambiguated Residual (residual prediction)
python experiments/section2_solution/role_disambiguated_residual.py --device cuda

# Run Section 2: Compare all approaches
python experiments/section2_solution/run_comparison.py --device cuda

# Run tests
python -m pytest tests/ -v
```

## Key Insight: Role-Based Disambiguation

The model uses **role embeddings** to distinguish:
- **OUTPUT role**: Ground-truth solutions in context (x*)
- **VEC_SECONDARY role**: Current estimates during refinement (x̃)

Token composition:
```
token = embed_component(data) + embed_role(role)
```

Query sequence with estimate:
```
[SEP, A, SEP, b_1, x_1*, ..., SEP, b_query, x̃, MASK]
                                              ↑
                                     VEC_SECONDARY role
```

## Refinement Algorithm

```python
x_0 = f(context, query)                 # Initial prediction (standard ICL)
x_{k+1} = x_k + f(context, query, x_k)  # Refinement iterations
```

## Configuration

Key hyperparameters in `experiments/section2_solution/role_disambiguated_residual.py`:
```python
d = 4                    # Vector/matrix dimension
n_embd = 128            # Transformer hidden dimension
n_layer = 6, n_head = 4 # Transformer architecture
training_steps = 50000
residual_weight = 0.5   # Mix of direct/residual loss
num_context = 5         # Context examples per sample
```

## TODO: Items Not Yet Implemented (???)

See `TODO.md` for the complete list of unimplemented features.

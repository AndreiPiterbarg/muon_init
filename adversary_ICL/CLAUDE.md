# Adversarial In-Context Learning (ICL) Experiments

## Project Goal

Find **structured, interesting failure modes** of transformer ICL by training an adversary to discover input distributions where ICL breaks down. The end goal is a publishable result — proofs or empirical findings showing *when and why* ICL fails on specific function classes.

### Core Idea

Standard ICL results (Garg et al., 2022) show transformers can learn linear regression, k-nearest neighbors, decision trees, etc. in-context. These models are typically trained on random Gaussian inputs (covariance = I). An adversarial agent searches over the space of input distributions / covariance structures / task parameters to find cases where the ICL model fails — ideally revealing structured, non-trivial failure modes (not just "garbage in, garbage out").

## Reference Implementation

The ICL model code is based on the **standard implementation** from:
- **Paper**: "What Can Transformers Learn In-Context? A Case Study of Simple Function Classes" (Garg et al., 2022)
- **Repo**: `https://github.com/dtsip/in-context-learning`
- Code in `src/icl/` should stay close to the original repo structure. Do not refactor the upstream ICL training/model code unless necessary for the adversary interface.

## Repository Structure

```
adversary_ICL/
├── CLAUDE.md
├── src/
│   ├── icl/              # Upstream ICL code (Garg et al.) — models, training, samplers
│   ├── adversary/        # Adversary agent: searches for breaking distributions
│   ├── tasks/            # Function class definitions (linear regression, KNN, etc.)
│   └── eval/             # Evaluation: metrics, comparison, failure detection
├── configs/              # YAML/JSON configs for experiments
├── experiments/
│   ├── scripts/          # Shell/Python scripts to launch experiment runs
│   └── configs/          # Per-experiment config overrides
├── results/
│   ├── plots/            # Generated figures
│   ├── logs/             # Training and experiment logs
│   └── checkpoints/      # Model checkpoints
└── notebooks/            # Analysis and visualization notebooks
```

## Key Concepts

- **ICL Model**: A pretrained transformer that does in-context learning on (x, y) pairs for a given function class (linear regression, KNN, etc.).
- **Adversary**: An agent/optimizer that proposes input distributions (covariance matrices, feature transforms, noise structures, task parameters) to maximize ICL model error.
- **Function Classes**: Linear regression, sparse linear, K-nearest neighbors, decision trees, 2-layer ReLU nets — as in the original paper.
- **Break / Failure Mode**: A structured input distribution where ICL performance degrades significantly vs. the optimal estimator for that function class.

## Conventions

- Python 3.10+, PyTorch.
- Use `configs/` for all hyperparameters — no magic numbers in code.
- Experiment reproducibility: all runs must log the full config + random seed.
- Results go in `results/` — never commit large checkpoints to git, use `.gitignore`.
- Notebooks are for analysis only, not for running experiments.

## What Makes a Good Result

We are looking for **structured** failures, not trivial ones. A good adversarial example:
1. Is a well-defined, natural distribution (not adversarial noise / pathological edge cases).
2. Shows a clear gap between ICL performance and the optimal estimator.
3. Has an explainable reason *why* ICL fails (e.g., covariance structure, distribution shift, symmetry breaking).
4. Ideally generalizes: the failure mode reveals something about the inductive bias of the transformer.

Trivial breaks (e.g., inputs scaled to 1e6, pure noise) are not interesting. The adversary should be guided/constrained toward structured search spaces.

## Development Notes

- When modifying `src/icl/`, keep changes minimal and well-commented so diffs against upstream are clear.
- The adversary in `src/adversary/` is the novel contribution — this is where most development happens.
- Always compare ICL performance against the **optimal baseline** for the function class (e.g., ridge regression for linear, actual KNN for KNN tasks).
- When adding a new experiment, create a config in `experiments/configs/` and a launch script in `experiments/scripts/`.

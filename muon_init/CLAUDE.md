# Muon Initialization Research

## What This Is

Research repository exploring initialization schemes specifically derived for the Muon optimizer. Muon orthogonalizes all updates (polar factor UV^T), which fundamentally violates the assumptions behind Kaiming and Xavier initialization. Nobody has asked what the initial weight spectrum should look like when the optimizer will orthogonalize every update. This repo exists to answer that question.

## Two Research Axes

### 1. New Initializations (`initializations/`)
Deriving and implementing initialization schemes matched to Muon's update geometry. Starting from first principles: if every update is a partial isometry (all singular values = 1), what spectral structure should the initial weights have?

### 2. Evaluation & Classification (`evaluation/`)
Building metrics and analysis tools to measure whether a given initialization actually improves training under Muon. The goal is not just "did loss go down" but understanding *how* and *why* — spectral dynamics, warmup sensitivity, signal propagation, convergence speed.

## Repository Structure

```
muon_init/
├── initializations/          # New init schemes
│   ├── theory/               # Derivations, proofs, mathematical notes
│   ├── implementations/      # Code implementing each init scheme
│   └── baselines/            # Kaiming, Xavier, orthogonal — reference impls
├── evaluation/               # Measuring and classifying improvements
│   ├── metrics/              # Spectral metrics, signal propagation, etc.
│   ├── benchmarks/           # Standardized training runs for comparison
│   └── analysis/             # Post-hoc analysis scripts and notebooks
├── experiments/              # Experiment configs and run scripts
│   ├── configs/              # Hyperparameter configs
│   ├── scripts/              # Launch scripts
│   └── results/              # Raw results (gitignored if large)
├── models/                   # Model definitions used in experiments
├── data/                     # Data loading and preprocessing
├── utils/                    # Shared utilities (logging, plotting, etc.)
├── notebooks/                # Exploratory Jupyter notebooks
└── tests/                    # Unit tests
```

## Key Context

- **Muon's update structure**: The update is the polar factor of the momentum matrix. All singular values are set to 1. This is an orthogonal matrix (square) or partial isometry (rectangular).
- **Current practice**: Standard Kaiming/Xavier init + warmup + weight decay. The warmup is a patch for initialization mismatch.
- **Success criterion**: An initialization that reduces or eliminates the need for learning rate warmup under Muon, with equivalent or better final performance.
- **Related but distinct work**: Muon++/µP (lr scaling), Turbo-Muon AOL (NS convergence), Moonlight (weight decay scaling) — none modify initial weight spectrum.

## Conventions

- Python 3.10+, PyTorch
- Experiments should be reproducible: always log random seeds, configs, and git hash
- Keep theory notes in markdown or LaTeX in `initializations/theory/`
- Notebooks are for exploration; anything reusable should be extracted to modules
- Results that are too large for git go in `experiments/results/` (gitignored) with a summary committed

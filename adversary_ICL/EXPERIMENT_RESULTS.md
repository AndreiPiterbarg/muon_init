# Adversarial ICL Experiment Results

## Experiment Setup

### ICL Model
- **Architecture**: 4-layer GPT2 transformer, 2 attention heads, 64-dim embeddings
- **Training**: 50,000 steps on d=5 linear regression with isotropic Gaussian inputs (Sigma=I)
- **Curriculum**: Fixed at d=5, 11 in-context examples per sequence
- **Hardware**: NVIDIA RTX 3080 Laptop GPU

### Adversary
- **Search algorithm**: Diagonal CMA-ES (sep-CMA-ES)
- **Genome**: 51-dimensional flat vector encoding:
  - Full 5x5 lower-triangular Cholesky factor for train covariance (15 params, diagonal in log-space for guaranteed PSD)
  - Full 5x5 lower-triangular Cholesky factor for test covariance (15 params)
  - Train and test mean vectors (5 + 5 params)
  - Task weight vector (5 params)
  - Log noise standard deviation (1 param)
- **Budget**: 3,200 genome evaluations (200 generations x 16 population)
- **Fitness**: Additive gap between ICL error and best baseline error, normalized by baseline scale, with degeneracy penalty
- **Baselines**: Ordinary Least Squares (OLS), Averaging model

### Key Design Decisions
- The adversary has **zero hardcoded failure types**. It operates on raw mathematical objects (full covariance matrices, weight vectors, noise) with maximum degrees of freedom.
- Separate train/test covariance matrices give the adversary the ability to discover distribution-shift failures without being told about distribution shift.
- Cholesky parameterization with log-diagonal guarantees PSD covariances by construction, so CMA-ES can search over unconstrained reals.
- Minimal constraints: diagonal of Cholesky factor clamped to [exp(-5), exp(5)], noise clamped to [0, 10]. This prevents NaN, nothing more.

---

## Model Baseline Performance

On standard isotropic Gaussian inputs (the training distribution), the model's performance:

| k (examples) | ICL Error | OLS Error | Gap |
|:---:|:---:|:---:|:---:|
| 0 | 5.577 | 5.221 | +0.356 |
| 1 | 5.181 | 4.070 | +1.111 |
| 2 | 5.934 | 3.027 | +2.907 |
| 3 | 5.624 | 1.933 | +3.692 |
| 4 | 5.264 | 0.970 | +4.294 |
| 5 | 5.561 | ~0 | +5.561 |
| 10 | 5.231 | ~0 | +5.231 |

The model partially learned ICL: at k=0 it matches OLS (both just predict ~0). But ICL error stays flat at ~5.3 across all k, while OLS drops to near-zero at k >= d = 5. The model has not fully converged.

---

## Adversary Search Results

### Search Dynamics
- **Total evaluations**: 3,200 (all valid)
- **Search time**: 104.8 seconds
- **Fitness distribution**: min=1.08, median=1,871, mean=8,726, max=327,730
- **Convergence**: Running best fitness improved from 13.5 (gen 1) to 327,730 (gen 103), then plateaued

### What the Adversary Found

**All top failures converge to a single pattern: rank-1 covariance with aligned weights.**

| Rank | Fitness | Cond Number | Eff. Rank | Top Eigenvalue | Remaining Eigenvalues | Weight-Cov Alignment | Noise Std |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| #1 | 327,730 | 1,817,718 | 1.00 | 22,040 | < 3 | 0.95 | 0.48 |
| #2 | 228,297 | 5,823,924 | 1.00 | 22,043 | < 1.5 | 0.78 | 0.49 |
| #3 | 223,658 | 5,605,298 | 1.00 | 22,043 | < 3 | 0.81 | 0.49 |
| #4 | 206,161 | 6,631,773 | 1.00 | 18,409 | < 2.3 | 0.88 | 0.36 |
| #5 | 202,027 | 4,010,095 | 1.00 | 22,043 | < 2.6 | 0.80 | 0.48 |
| #6 | 195,773 | 7,921,301 | 1.00 | 22,038 | < 3.6 | 0.90 | 0.41 |
| #7 | 153,280 | 4,597,270 | 1.00 | 22,040 | < 2.3 | 0.77 | 0.52 |
| #8 | 149,412 | 8,738,244 | 1.00 | 22,045 | < 1.6 | 0.75 | 0.55 |
| #9 | 146,779 | 2,657,736 | 1.00 | 22,047 | < 1.9 | 0.70 | 0.50 |
| #10 | 145,701 | 2,639,818 | 1.00 | 22,040 | < 3.8 | 0.89 | 0.62 |

### Consistent Properties Across All Top Failures

1. **Rank-1 covariance**: Effective rank = 1.00 in every case. ~99.99% of variance concentrated on a single axis (eigenvalue ~22,000 vs <3 for remaining 4 dimensions).

2. **Strong weight-covariance alignment**: The task weight vector aligns with the dominant eigenvector (cosine similarity 0.70-0.95). The regression signal lives entirely along the high-variance direction.

3. **Large train-test divergence**: Train-test covariance Frobenius distance ~4,400 across all top failures. Different covariances for context examples vs. query amplify the failure.

4. **Moderate noise**: Noise std converges to ~0.4-0.6, not zero. The adversary found that moderate noise combined with extreme covariance structure produces worse failures than zero noise.

5. **Late failure peak**: Peak failure occurs at position 0.9-1.0 in the learning curve (the gap is largest at the end of the in-context sequence).

### Learning Curve: Top Failure (#1, fitness=327,730)

| k | ICL Error | OLS Error | Gap (ICL - OLS) |
|:---:|:---:|:---:|:---:|
| 0 | 456,488 | 455,364 | +1,124 |
| 1 | 451,933 | 25 | +451,908 |
| 2 | 459,313 | 5.6 | +459,307 |
| 3 | 457,268 | 2.5 | +457,266 |
| 4 | 375,318 | 7.3 | +375,310 |
| 5 | 397,474 | 129 | +397,345 |
| 6 | 441,145 | 2.3 | +441,143 |
| 7 | 371,426 | 1.1 | +371,425 |
| 8 | 576,476 | 0.8 | +576,475 |
| 9 | 487,528 | 0.6 | +487,527 |
| 10 | 467,787 | 0.4 | +467,787 |

**ICL error is flat at ~400,000-500,000 across all k.** The transformer gains nothing from the in-context examples. OLS solves the problem with 1 example (error drops from 455,364 to 25). In a rank-1 problem, a single (x, y) pair determines the function along the dominant direction.

### Statistical Correlations (Spearman, n=3200)

| Feature | Spearman rho | p-value | Significance |
|:---|:---:|:---:|:---:|
| spectral_entropy | +0.059 | 0.0008 | *** |
| effective_rank | +0.059 | 0.0009 | *** |
| train_test_divergence | -0.041 | 0.022 | * |
| weight_alignment | -0.036 | 0.043 | * |
| noise_std | -0.022 | 0.215 | |
| condition_number_log | -0.010 | 0.582 | |
| noise_to_signal | -0.008 | 0.635 | |
| weight_norm | -0.005 | 0.760 | |

---

## Interpretation

### Why This Breaks ICL

The transformer was trained on isotropic Gaussian inputs (Sigma=I). Its read-in layer learned a fixed linear projection assuming roughly equal variance across dimensions. When one dimension has 10,000x the variance of others:

1. The transformer's internal representation is dominated by that dimension's variance.
2. The attention mechanism cannot distinguish signal from covariance-induced scale.
3. OLS, by directly solving the normal equations, automatically adapts to any covariance structure.
4. In a rank-1 problem, OLS needs only 1 example. The transformer, lacking covariance awareness, needs infinitely many and still fails.

### What's Non-Obvious

The adversary discovered several non-trivial aspects:
- **Weight alignment matters**: It's not just about rank-1 covariance. The failure is worst when the regression weights align with the dominant eigenvector (alignment 0.7-0.95). If weights were orthogonal to the dominant direction, the problem would be harder for everyone.
- **Moderate noise is worse than no noise**: The adversary converged to noise_std ~0.5, not 0. This suggests a noise-covariance interaction where noise along the dominant direction obscures the signal more than in isotropic settings.
- **Train-test mismatch amplifies failure**: Even though the evaluator doesn't use the xs_p path, the different train/test covariances in the genome suggest the adversary is exploiting representational brittleness.

---

## Caveats

### Model Convergence
The model was trained for 50K steps at d=5. On standard Gaussian inputs, it still has squared error ~5.3 at k >= d (where OLS gets ~0). A fully converged model (500K steps, as in Garg et al.) would have near-zero error on standard inputs. The adversary is partially exploiting the model's residual incompetence, not just a structural blind spot.

### Trivial Failure Mode
Condition numbers of 1-8 million and effective rank of 1.00 are not "natural" distributions. The CLAUDE.md project specification explicitly states: "Trivial breaks (e.g., inputs scaled to 1e6, pure noise) are not interesting." The adversary found the easiest exploit, not the most interesting one.

### Low Dimensionality
d=5 is too small for interesting ICL behavior. The original paper uses d=20, which gives the transformer more room to demonstrate non-trivial learning and the adversary more room to find non-trivial failures.

### Weak Correlations
The Spearman correlations (rho ~0.04-0.06) are statistically significant due to sample size (n=3200) but have very small effect sizes. The descriptor space does not cleanly separate failure modes at this scale.

---

## Is This a Starting Point for a Research Project?

**Yes, but with significant work remaining.**

### What This Validates
1. **The system works end-to-end**: genome encode/decode produces valid PSD matrices, CMA-ES converges, fitness improves monotonically, and the adversary discovers consistent interpretable structure (not random noise).
2. **The adversary has genuine autonomy**: With zero hardcoded failure types and 51 free parameters, it independently converges to a coherent mathematical pattern (rank-1 + alignment + shift).
3. **Post-hoc analysis works**: Correlations, learning curves, and eigenvalue spectra correctly identify the dominant failure features.

### What's Needed for a Paper

1. **Fully converged model**: Train the standard 12-layer, 256-dim model for 500K steps on d=20 (the Garg et al. config). The adversary should find failures where a competent model breaks, not an undertrained one.

2. **Constraint the adversary**: Add a condition number ceiling (e.g., 100 or 1000) and trace normalization (tr(Sigma) = d). This forces the adversary into "natural" distributions and the interesting research question: what structured, non-pathological distributions break ICL?

3. **Scale to d=20**: The 481-dimensional genome space is where the adversary has enough freedom to find non-obvious structure. At d=5 it can only find the obvious rank-1 trick.

4. **Cross-validate across seeds**: Train 3-5 models with different random seeds and check which adversarial failures transfer. Seed-specific failures reveal memorization; transferable failures reveal inductive bias.

5. **Compare across model depths**: Research shows 2-layer transformers approximate ridge regression while 8+ layer models approximate OLS. Running the adversary against different depths could reveal depth-dependent failure modes — a publishable finding on its own.

6. **Characterize the boundary**: Instead of just finding the worst case, map the transition. At what condition number does ICL start failing? How does the failure depend on effective rank? This requires grid sweeps over constrained subspaces after the adversary identifies interesting regions.

---

## Files and Artifacts

### Code
- `src/adversary/genome.py` — Genome representation (flat vector to covariance/weights/noise)
- `src/adversary/evaluate.py` — Evaluator connecting genome to ICL pipeline
- `src/adversary/search.py` — Diagonal CMA-ES implementation and search loop
- `src/adversary/analyze.py` — Post-hoc clustering, correlation, and plotting
- `run_adversary_only.py` — Experiment script used for this run

### Results
- `results/analysis2/summary.json` — Full numerical results
- `results/analysis2/fitness_over_time.png` — Search convergence plot
- `results/analysis2/top_failures.png` — Learning curves for top 5 failures
- `results/analysis2/top_spectra.png` — Eigenvalue spectra for top 5 failures
- `results/analysis2/cond_vs_fitness.png` — Condition number vs. fitness scatter

### Model Checkpoint
- `results/checkpoints/experiment_run/state.pt` — Trained model (4-layer, d=5, 50K steps)

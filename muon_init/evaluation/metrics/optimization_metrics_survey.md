# Metrics for Evaluating "Improved Optimization" and Initialization Quality

## A Literature Survey for Muon Initialization Research

*Compiled April 2026*

---

## Table of Contents

1. [Convergence and Speed Metrics](#1-convergence-and-speed-metrics)
2. [Sharpness and Loss Landscape Metrics](#2-sharpness-and-loss-landscape-metrics)
3. [Edge of Stability](#3-edge-of-stability)
4. [Spectral Dynamics of Weight Matrices](#4-spectral-dynamics-of-weight-matrices)
5. [Signal Propagation and Dynamical Isometry](#5-signal-propagation-and-dynamical-isometry)
6. [Gradient-Level Metrics](#6-gradient-level-metrics)
7. [Warmup Sensitivity as a Proxy for Init Quality](#7-warmup-sensitivity-as-a-proxy-for-init-quality)
8. [Hyperparameter Robustness](#8-hyperparameter-robustness)
9. [Activation Statistics](#9-activation-statistics)
10. [Muon-Specific Considerations](#10-muon-specific-considerations)
11. [Recommended Evaluation Protocol for Muon Init](#11-recommended-evaluation-protocol-for-muon-init)
12. [Full Reference List](#12-full-reference-list)

---

## 1. Convergence and Speed Metrics

**Methodological caveat:** Raw convergence speed (steps to a loss threshold) is noisy and often misleading. Optimizer rankings are fragile across tasks and epochs (Schmidt et al., ICML 2021), and the hyperparameter search space is the single most important factor in comparisons (Choi, Shallue et al., 2019). Always use **validation** metric thresholds, not training loss, and control the HP search space rigorously.

### Time-to-Target Quality
The gold standard, used by AlgoPerf (Dahl et al., 2023) and MLPerf. Measures wall-clock time (or step count) to reach a **validation** metric threshold.

- **Muon** (Jordan et al., 2024): Evaluated via NanoGPT speedrunning (time to 3.28 val loss on FineWeb) and CIFAR-10 (time to 94% accuracy). Reports 1.35x speedup over AdamW.
- **LAMB** (You et al., ICLR 2020): Time to train BERT to target F1 score.
- **AlgoPerf**: Measures fraction of runtime budget required to hit target performance across 8 workloads.

### Loss Curve Area (AUC)
Integrates performance over the entire training run rather than just the endpoint. Captures whether one method is consistently better or only wins at convergence. Limitation: cannot distinguish methods whose curves cross.

### Final Performance
Every paper reports this (accuracy, loss, perplexity, F1, BLEU). Necessary but not sufficient -- says nothing about efficiency. Two optimizers may reach the same final loss, but one in half the steps.

---

## 2. Sharpness and Loss Landscape Metrics

### Hessian Top Eigenvalue (lambda_max)
The most common sharpness metric. Computed cheaply via power iteration on Hessian-vector products (no explicit Hessian).

- Ghorbani, Krishnan & Xiao (ICML 2019): Showed isolated large eigenvalues appear rapidly during training, and gradients concentrate in these eigenspaces.
- **PyHessian** (Yao et al., NeurIPS 2020): Open-source tool computing lambda_max, Tr(H), and full spectral density.

### Hessian Trace
`Tr(H)` measures average curvature across all directions. Computed cheaply via Hutchinson's estimator: `Tr(H) = E[v^T H v]` for random vectors v. The ratio `lambda_max / Tr(H)` indicates how "spiky" the curvature is.

### Full Hessian Spectral Density
Beyond lambda_max, the full spectral density (computed via stochastic Lanczos quadrature) reveals the bulk-plus-outlier structure. A few large outliers with bulk near zero = ill-conditioning; a compact spectrum = well-conditioned.

### The Scale-Invariance Problem
**Critical caveat:** Dinh et al. (ICML 2017, "Sharp Minima Can Generalize") showed that for ReLU networks, you can reparameterize any flat minimum to be arbitrarily sharp without changing the function. Therefore, raw sharpness metrics are **not reparameterization-invariant** and cannot alone predict generalization.

**Resolution:** Use **normalized sharpness** metrics:
- `lambda_max / ||w||_F^2` (Neyshabur et al., 2017)
- `Tr(H) / ||w||_F^2` (normalized flatness, Tsuzuku et al., 2020)
- Jiang et al. (2020, "Fantastic Generalization Measures") confirmed PAC-Bayes/sharpness measures correlate best with generalization, but only when properly normalized.

---

## 3. Edge of Stability

### The Phenomenon
Cohen, Kaber & Singer (ICLR 2021): When training with full-batch GD at fixed LR eta, lambda_max initially increases until it hits `2/eta` -- the classical stability threshold for quadratics. Training then enters the **edge of stability (EoS)**: lambda_max oscillates around `2/eta`, loss is non-monotonic short-term but still decreases long-term.

### Key Diagnostic
The ratio `eta * lambda_max` is diagnostic:
- Near 2: at the edge of stability
- Well below 2: optimizer hasn't explored sharp enough regions (possibly under-training)

### Muon Implication
Since Muon orthogonalizes updates (all singular values = 1), the effective step size in each direction is uniform. This fundamentally changes EoS dynamics -- the optimizer cannot take larger steps in flat directions and smaller in sharp ones. A good Muon initialization might place initial lambda_max near the Muon-specific EoS threshold, avoiding progressive sharpening entirely.

---

## 4. Spectral Dynamics of Weight Matrices

### Singular Value Distribution Evolution
Yunis et al. (2024, "Approaching Deep Learning through the Spectral Dynamics of Weights") track SVD evolution during training. Key findings:
- Generalizing networks develop **low-rank structure** (few dominant singular values); memorizing networks do not.
- Weight decay enhances this low-rank bias beyond simple norm regularization.
- At initialization: singular values follow semicircle law (Wigner) and Marchenko-Pastur statistics.

### Effective Rank (eRank)
Continuous measure of "dimensionality" of a matrix's singular value spectrum:

```
eRank(W) = exp(H(p))  where  p_i = sigma_i / sum(sigma_j),  H(p) = -sum(p_i log(p_i))
```

Ranges from 1 (one dominant direction) to full algebraic rank (uniform spectrum). Related findings:
- **Huh et al.** ("The Low-Rank Simplicity Bias"): Deep networks are biased toward lower-rank weight matrices. Lower effective rank correlates with better generalization.
- **Feng et al.** (NeurIPS 2022, "Rank Diminishing"): Both sub-network rank and feature manifold intrinsic dimension decrease monotonically with depth.
- **Daneshmand et al.** (NeurIPS 2020): Without BatchNorm, intermediate representation rank collapses with depth.

### Stable Rank
A cheaper, more robust alternative:

```
srank(W) = ||W||_F^2 / ||W||_2^2 = sum(sigma_i^2) / sigma_max^2
```

Always <= algebraic rank. Appears directly in generalization bounds (Neyshabur et al., Bartlett et al.).

**Sanyal, Torr & Dokania** (ICLR 2020, "Stable Rank Normalization"): Minimizing stable rank during training yielded 11.3% improvement in generalization gap and significant improvement in GAN metrics.

### Condition Number of Weight Matrices
The Muon blog notes that SGD-momentum and Adam updates have very high condition number (nearly low-rank, dominated by a few directions). Muon's orthogonalization produces updates with condition number exactly 1. Tracking weight matrix condition numbers reveals how the optimizer reshapes the spectral structure.

### Empirical Spectral Density (ESD)
Martin & Mahoney (ICML 2019, JMLR 2021): Framework using random matrix theory to classify training dynamics:
- At init: weights follow **Marchenko-Pastur distribution**.
- During training: ESDs develop "bulk+tail" structure through **5+1 phases** from Random-like to Heavy-Tailed.
- **WeightWatcher** tool: practical diagnostic for spectral health of trained networks.

### SVD Entropy
Moonlight (Liu, Su et al., 2025) uses SVD entropy across weight matrices, finding "Muon achieves higher SVD entropy than AdamW, verifying more diverse optimization directions." This is the only spectral metric used in Muon papers to date.

---

## 5. Signal Propagation and Dynamical Isometry

### Mean Field Theory Depth Scales
Schoenholz, Gilmer, Ganguli & Sohl-Dickstein (ICLR 2017, "Deep Information Propagation") define two critical depth scales:
- **Correlation depth scale** (xi_c): how deep input signals remain distinguishable.
- **Gradient depth scale** (xi_g): how deep gradients flow without vanishing/exploding.

Networks are trainable precisely when both scales diverge, at the **edge of chaos** -- the boundary between ordered phase (vanishing gradients, inputs become indistinguishable) and chaotic phase (exploding gradients, similar inputs look very different).

### Dynamical Isometry
The gold standard for signal propagation. Defined as: all singular values of the input-output Jacobian concentrate near 1.

- **Saxe, McClelland & Ganguli** (2014): Exact solutions in deep linear networks; orthogonal init enables.
- **Pennington, Schoenholz & Ganguli** (NeurIPS 2017, "Resurrecting the Sigmoid"): ReLU networks are **incapable** of dynamical isometry. Sigmoidal networks can achieve it only with **orthogonal** weight initialization. Networks with dynamical isometry learn "orders of magnitude faster."
- **Xiao et al.** (ICML 2018, "Dynamical Isometry and Mean Field Theory of CNNs"): **Delta-Orthogonal initialization** for CNNs enables training 10,000-layer vanilla CNNs without batch norm or residual connections.

### Critical Initialization
The network must be initialized at the edge of chaos for signal propagation to work at depth:
- **Lyapunov exponent** of signal propagation should be ~0 at criticality.
- **Correlation length** should diverge at criticality.
- **Yang & Schoenholz** (NeurIPS 2017): Residual networks naturally live near the edge of chaos but require rescaling by 1/L.

### Provable Benefits of Orthogonal Init
**Hu et al.** (ICLR 2020): With orthogonal init, the width needed for efficient convergence is **independent of depth**, whereas with Gaussian init it scales linearly with depth.

---

## 6. Gradient-Level Metrics

### Per-Layer Gradient Norm Ratios
Ratio of gradient norms between adjacent layers. Should be ~1.0 for healthy gradient flow. The coefficient of variation of gradient norms across layers is a summary statistic.

### Dead Neuron Percentage
Fraction of neurons whose output is always zero (for ReLU). A high percentage at initialization indicates too many neurons in the "dead zone." Target: ~50% active for ReLU with proper init.

### Angular Gradient Signal-to-Noise Ratio
Since Muon discards gradient magnitudes (only keeps singular vectors), the *angular* signal-to-noise ratio (consistency of gradient direction across samples) matters more than magnitude-based GSNR. This is a novel metric worth defining for Muon-specific evaluation.

---

## 7. Warmup Sensitivity as a Proxy for Init Quality

### Theoretical Backing
Gilmer et al. (NeurIPS 2024, "Why Warmup the Learning Rate?"): Warmup's primary benefit is forcing the network into well-conditioned regions of the loss landscape, enabling larger target learning rates. The need for warmup depends on initialization and parameterization.

Key mechanisms:
- **Sharpness reduction:** lambda_max gradually decreases during warmup, moving toward flatter regions.
- **Two regimes:** Progressive sharpening vs. sharpness reduction during warmup, depending on init.
- **Loss catapult mechanism:** Can replace warmup by properly choosing initial learning rates -- directly connecting init quality to warmup necessity.
- **Critical finding:** If initialization is well-conditioned, warmup can be eliminated entirely.

### Practical Metric
**Minimum warmup steps for stable training:** Sweep warmup length (0, 100, 500, 1000, 5000 steps) at a given target LR. A better initialization should:
1. Require fewer warmup steps to reach peak performance.
2. Show smaller performance degradation when warmup is removed entirely.
3. Flatten the "performance vs. warmup length" curve.

### GPT-Specific Work
"Analyzing & Reducing the Need for Learning Rate Warmup in GPT Training" (arXiv 2410.23922, 2024): Studies warmup specifically for GPT-scale models.

---

## 8. Hyperparameter Robustness

### 1D Hyperparameter Ablations
Zhao et al. (2024, "Anything but SGD," Harvard/Kempner): For each hyperparameter (LR, weight decay, momentum, warmup steps, epsilon), sweep it while holding others at best values. Plot loss as a function of each HP. A robust method shows a wide "valley" (insensitivity to exact HP choice).

### Application to Init Evaluation
For Muon initialization specifically:
- "How many HP trials to reach X% of optimal performance?" with vs. without the proposed init.
- 1D ablation of warmup length, LR, and weight decay -- does the init widen the optimal region?

---

## 9. Activation Statistics

### Standard Metrics
- **Mean and variance of pre-activations per layer:** The fundamental signal propagation diagnostic. Xavier preserves var=1 (linear); He preserves var=1 (ReLU).
- **Kurtosis of activations:** Measures tailedness. Normal = kurtosis 3. Heavy tails = potential instability.
- **Fraction of non-zero activations (ReLU):** Target ~50% for proper init.

---

## 10. Muon-Specific Considerations

### What Muon Papers Actually Measure

| Paper | Metrics |
|-------|---------|
| **Muon** (Jordan et al., 2024) | Wall-clock time (A100-seconds), val loss, accuracy, FLOP overhead |
| **Moonlight** (Liu, Su et al., 2025) | MMLU, HumanEval, GSM8K, MATH, SVD entropy |
| **Essential AI** (2025) | Token ratio, critical batch size, compute Pareto frontier |
| **NorMuon** (2025) | Training efficiency, neuron norm uniformity |

### Spectral Norm Constraint Discovery
"Muon Optimizes Under Spectral Norm Constraints" (arXiv 2506.15054): Muon with decoupled weight decay **implicitly solves** a constrained optimization where `||X||_op <= 1/lambda`. Convergence has two phases:
1. **Constraint satisfaction:** Parameters rapidly enter the spectral norm ball at exponential rate.
2. **Optimization within the constrained region.**

**Key implication:** Initialization that already satisfies `||W||_op <= 1/lambda` would skip phase 1 entirely -- potentially eliminating the need for warmup.

### POET: Closest Existing Work on Init for Orthogonal Updates
"POET: Reparameterized LLM Training via Orthogonal Equivalence Transformation" (arXiv 2506.08001):
- Parameterizes `W = R * W_0 * P` where R, P are learned orthogonal matrices, W_0 is frozen.
- **Uniform spectrum initialization:** Apply SVD to standard init, set all singular values to 1.
- Preserves singular values throughout training (only singular vectors change).
- Finds Muon can better minimize **hyperspherical energy** than AdamW.

### Turbo-Muon AOL Preconditioner
(arXiv 2512.04632): The AOL preconditioner is essentially an initialization-time intervention that "tightens the singular value distribution of the initial state" and "substantially lowers polar error" before Newton-Schulz orthogonalization. Direct evidence that spectral conditioning at init matters for Muon.

### Information Geometry Warning
"Information Geometry of Orthogonal Initializations and Training" (arXiv 1810.03785, ICLR 2020):
- Highly isometric initializations have lower FIM condition number but can be **harder to train with SGD**.
- Benefits of orthogonality require **maintaining orthogonality throughout training** (manifold optimization).
- **Critical implication for Muon:** Since Muon enforces orthogonality on every update, it may actually benefit from orthogonal initialization in ways that SGD does not. The optimizer-init interaction is non-trivial.

### Novel Metrics Needed for Muon

1. **Stiefel-projected sharpness:** Maximum loss increase under perturbations constrained to the tangent space of the Stiefel manifold -- sharpness aligned with directions Muon can actually move.
2. **Angular gradient SNR:** Consistency of gradient *direction* across samples, not just magnitude (since Muon discards magnitudes).
3. **Polar error at init:** How far the initial momentum matrix is from its polar factor -- determines how much work Newton-Schulz must do on the first step.
4. **Spectral norm ball membership:** Whether `||W_init||_op <= 1/lambda_wd` -- determines whether phase 1 (constraint satisfaction) is needed.

### The Research Gap
No existing paper:
- Derives initialization specifically for Muon's update geometry
- Measures warmup sensitivity as a function of initial weight spectrum under Muon
- Defines sharpness metrics aligned with the Stiefel manifold
- Studies how the initial singular value distribution affects Muon's convergence trajectory

---

## 11. Recommended Evaluation Protocol for Muon Init

Based on the full literature survey, here is a concrete evaluation protocol. There is **no widely adopted standardized evaluation protocol for initialization methods** comparable to AlgoPerf for optimizers. The literature converges on best practices: track per-layer weight standard deviations at periodic checkpoints, monitor gradient norm excursions, measure optimization smoothness (std of per-epoch loss), use 10+ randomized runs with paired t-tests (p<0.05), and controlled comparison where identical data, architecture, optimizer, schedule are held fixed and only init varies.

### Tier 1: Cheap Diagnostics (Run First)

| Metric | What It Measures | Cost | How to Compute |
|--------|-----------------|------|----------------|
| Activation variance per layer | Signal propagation health | 1 forward pass | Record layer outputs on calibration batch |
| Dead neuron % | Init in dead zone? | 1 forward pass | Count zero activations (ReLU) |
| Jacobian singular values | Dynamical isometry | O(depth * width^2) | SVD of end-to-end Jacobian on a few inputs |
| Weight matrix SVD | Spectral structure | O(width^2 * layers) | Per-layer SVD |
| Effective rank & stable rank | Spectral concentration | Same as above | Compute from SVDs |
| Spectral norm ball membership | `\|\|W\|\|_op <= 1/lambda_wd`? | O(width^2) | Power iteration per layer |
| Gradient norms per layer | Gradient flow health | 1 forward-backward pass | Record per-layer grad norms |

### Tier 2: Training Dynamics (Short Runs)

| Metric | What It Measures | Cost | Protocol |
|--------|-----------------|------|----------|
| Warmup sensitivity | Init-optimizer match | 5x short runs | Sweep warmup {0, 100, 500, 1k, 5k} at fixed LR |
| lambda_max trajectory | Sharpness evolution | Track during training | Power iteration every N steps |
| Progressive sharpening rate | How fast curvature grows | Same as above | d(lambda_max)/dt in early training |
| Edge of stability ratio | eta * lambda_max dynamics | Same as above | Track ratio over training |
| Weight SVD evolution | Spectral dynamics | Track during training | Per-layer SVDs at checkpoints |
| SVD entropy | Diversity of optimization directions | Same as above | From Moonlight |
| Loss curve AUC | Consistent improvement? | 1 full run | Integrate val loss over steps |

### Tier 3: Rigorous Evaluation (Full Runs)

| Metric | What It Measures | Cost | Protocol |
|--------|-----------------|------|----------|
| Time-to-target quality | Efficiency | 5-10 seeds per config | Wall-clock to val metric threshold |
| 1D HP ablations | Robustness | 5-10 runs per HP value | Sweep LR, WD, warmup independently |
| Normalized sharpness at convergence | Generalization prediction | Post-training | lambda_max / \|\|w\|\|_F^2 |

### Tier 4: Muon-Specific Novelty (Publish-Worthy)

| Metric | What It Measures | Cost | Notes |
|--------|-----------------|------|-------|
| Stiefel-projected sharpness | Curvature in Muon's action space | Moderate | Novel metric to define |
| Angular gradient SNR | Directional signal quality | Low | Novel metric to define |
| Polar error at init | NS iteration difficulty | Low | SVD of momentum at step 0 |
| Phase 1 duration | Constraint satisfaction time | Track during training | Steps until \|\|W\|\|_op <= 1/lambda |

### Workload Diversity
Evaluate on at least 3-4 tasks at different scales:
- Small: CIFAR-10 with ResNet/ViT
- Medium: NanoGPT on FineWeb (the Muon community benchmark)
- Larger: GPT-2 scale on OpenWebText or similar
- Optional: A non-language/vision task for generality

### Statistical Rigor
- Minimum 5 seeds per configuration (10 preferred)
- Report mean +/- std
- Paired t-tests or Wilcoxon signed-rank tests (p<0.05)
- Performance profiles for multi-workload comparison (following Schmidt et al.)

---

## 12. Full Reference List

### Optimization Metrics and Benchmarks
- Dahl et al. (2023). "Benchmarking Neural Network Training Algorithms." arXiv:2306.07179. [AlgoPerf]
- Schmidt, Schneider & Hennig (ICML 2021). "Descending through a Crowded Valley." PMLR v139.
- Choi, Shallue et al. (2019). "On Empirical Comparisons of Optimizers for Deep Learning." arXiv:1910.05446.
- Zhao et al. (2024). "Anything but SGD: Evaluating Optimizers for LLM Training." Kempner Institute.

### Sharpness, Loss Landscape, and Edge of Stability
- Cohen, Kaber & Singer (ICLR 2021). "Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability."
- Ghorbani, Krishnan & Xiao (ICML 2019). "An Investigation into Neural Net Optimization via Hessian Eigenvalue Density." arXiv:1901.10159.
- Yao, Gholami, Keutzer & Mahoney (NeurIPS 2020). "PyHessian."
- Dinh, Pascanu, Bengio & Bengio (ICML 2017). "Sharp Minima Can Generalize for Deep Nets."
- Jiang et al. (2020). "Fantastic Generalization Measures and Where to Find Them."

### Spectral Dynamics
- Yunis et al. (2024). "Approaching Deep Learning through the Spectral Dynamics of Weights." arXiv:2408.11804.
- Martin & Mahoney (ICML 2019, JMLR 2021). "Implicit Self-Regularization in Deep Neural Networks." arXiv:1810.01075.
- Sanyal, Torr & Dokania (ICLR 2020). "Stable Rank Normalization." arXiv:1906.04659.

### Signal Propagation and Dynamical Isometry
- Schoenholz et al. (ICLR 2017). "Deep Information Propagation." arXiv:1611.01232.
- Pennington, Schoenholz & Ganguli (NeurIPS 2017). "Resurrecting the Sigmoid." arXiv:1711.04735.
- Xiao et al. (ICML 2018). "Dynamical Isometry and Mean Field Theory of CNNs." arXiv:1806.05393.
- Saxe, McClelland & Ganguli (2014). "Exact Solutions in Deep Linear Networks." arXiv:1312.6120.
- Yang & Schoenholz (NeurIPS 2017). "Mean Field Residual Networks: On the Edge of Chaos."

### Initialization Methods and Evaluation
- Glorot & Bengio (AISTATS 2010). "Understanding the Difficulty of Training Deep Feedforward Neural Networks."
- He et al. (ICCV 2015). "Delving Deep into Rectifiers." arXiv:1502.01852. [Kaiming]
- Hu et al. (ICLR 2020). "Provable Benefit of Orthogonal Initialization." arXiv:2001.05992.

### Warmup and Learning Rate Dynamics
- Gilmer et al. (NeurIPS 2024). "Why Warmup the Learning Rate?" arXiv:2406.09405.
- "Analyzing & Reducing the Need for LR Warmup in GPT Training." arXiv:2410.23922.

### Gradient Metrics
- Neyshabur et al. (NeurIPS 2017). "Exploring Generalization in Deep Learning."

### Muon and Related Optimizers
- Jordan et al. (2024). "Muon: An Optimizer for Hidden Layers." kellerjordan.github.io/posts/muon/
- Liu, Su et al. (2025). "Muon is Scalable for LLM Training." arXiv:2502.16982. [Moonlight]
- Essential AI (2025). "Practical Efficiency of Muon for Pretraining." arXiv:2505.02222.
- "Muon Optimizes Under Spectral Norm Constraints." arXiv:2506.15054.
- Turbo-Muon (2024). arXiv:2512.04632.
- POET (2025). arXiv:2506.08001.
- NorMuon (2025). arXiv:2510.05491.
- "Information Geometry of Orthogonal Initializations." arXiv:1810.03785.

### Other
- Feng et al. (NeurIPS 2022). "Rank Diminishing in Deep Neural Networks." arXiv:2206.06072.
- Daneshmand et al. (NeurIPS 2020). "Batch Normalization Provably Avoids Rank Collapse." arXiv:2003.01652.
- Huh et al. "The Low-Rank Simplicity Bias in Deep Networks."
- You et al. (ICLR 2020). "Large Batch Optimization for Deep Learning." arXiv:1904.00962. [LAMB]

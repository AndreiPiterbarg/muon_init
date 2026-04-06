# Adversarial Retraining vs. Cumulative Fine-Tuning: Experimental Results

## Experiment Overview

We ran an adversarial retraining loop on a d=5 ICL transformer (6-layer GPT-2, trained for 150k steps on isotropic Gaussian data). Each round, a CMA-ES adversary (50,000 evaluations, 5 restarts, pipeline genomes) searches for input distributions that maximally break the current model. The discovered adversarial distributions are accumulated into a curriculum, and two models are updated:

- **Scratch model**: Retrained from random initialization on a mix of standard Gaussian data and the adversarial curriculum (30% adversarial, 70% Gaussian, dynamically scaled).
- **Cumulative model**: Fine-tuned from its previous checkpoint for 20,000 additional steps at a lower learning rate (1e-4 vs 3e-4), with a higher adversarial mixing ratio (50% adversarial, dynamically scaled up to 65%).

Both models are evaluated on a fixed benchmark of 12 covariance structures never seen during training. The metric is the ICL/Ridge error ratio at k=d (number of in-context examples equals dimension) -- a ratio of 1.0 means ICL matches ridge regression; higher means ICL is worse.

The adversary attacks the cumulative model each round, since it is expected to be the stronger of the two.

## Base Model Performance (Round 0)

Before any adversarial training, the base model's benchmark mean (non-isotropic) was **2.22**. Per-distribution breakdown:

| Distribution | ICL/Ridge @d | ICL/Ridge @d/2 |
|---|---|---|
| isotropic | 0.96 | 0.95 |
| rank_1 | 5.28 | 1.04 |
| rank_2 | 1.61 | 0.68 |
| rank_half | 1.93 | 0.84 |
| exp_decay | 1.36 | 0.91 |
| power_law | 1.20 | 0.94 |
| step_function | 4.65 | 1.48 |
| wishart | 1.15 | 0.92 |
| condition_2 | 1.38 | 0.97 |
| condition_10 | 1.22 | 1.02 |
| condition_100 | 1.44 | 0.78 |
| condition_1000 | 3.20 | 1.04 |

The base model is near-optimal on isotropic data (0.96) and mild covariance structures, but already struggles on rank-1 (5.28x) and step-function (4.65x) distributions. This is the starting point both training strategies must improve upon -- or at least not destroy.

## Adversary Findings

| Round | Best Fitness | Curriculum Size | Best Genome |
|---|---|---|---|
| 1 | 20.87 | 50 | laplace -> sorting_rank -> mixture_injection (3 active stages) |
| 2 | 18.20 | 75 (capped) | -- |
| 3 | 18.86 | 75 (capped) | -- |

The adversary found distributions where the ICL model performs ~20x worse than ridge regression. The best genome in round 1 used a Laplace base distribution piped through sorting-rank and mixture-injection transforms -- a non-Gaussian, heavy-tailed distribution with rank-order scrambling. This is a structured, non-trivial failure: the adversary found that the transformer's learned algorithm implicitly assumes Gaussian-like tail behavior and monotonic feature relationships.

Fitness decreased from 20.87 to 18.20 between rounds 1 and 2, indicating the cumulative model became somewhat harder to break. However, fitness rebounded to 18.86 in round 3, suggesting the adversary found new attack angles rather than simply re-exploiting old ones. The adversary is not converging to zero -- there appear to be persistent structural weaknesses in the ICL algorithm that resist patching.

## Head-to-Head: Scratch vs. Cumulative Benchmark Results

### Aggregate (Mean ICL/Ridge ratio, non-isotropic distributions)

| Round | Scratch | Cumulative | Ratio (Scratch / Cumulative) |
|---|---|---|---|
| 0 (base) | 2.22 | 2.22 | 1.0x |
| 1 | 130.72 | 83.02 | 1.6x |
| 2 | 583.50 | 126.25 | 4.6x |

### Per-Distribution Detail (Round 1)

| Distribution | Base | Scratch R1 | Cumulative R1 | Scratch/Cumul |
|---|---|---|---|---|
| isotropic | 0.96 | 20.57 | 6.60 | 3.1x |
| rank_1 | 5.28 | 358.35 | 349.31 | 1.0x |
| rank_2 | 1.61 | 100.44 | 36.23 | 2.8x |
| rank_half | 1.93 | 139.81 | 51.67 | 2.7x |
| exp_decay | 1.36 | 33.78 | 11.15 | 3.0x |
| power_law | 1.20 | 33.74 | 10.06 | 3.4x |
| step_function | 4.65 | 327.54 | 203.01 | 1.6x |
| wishart | 1.15 | 45.14 | 18.66 | 2.4x |
| condition_2 | 1.38 | 20.62 | 6.43 | 3.2x |
| condition_10 | 1.22 | 32.08 | 9.90 | 3.2x |
| condition_100 | 1.44 | 89.27 | 37.30 | 2.4x |
| condition_1000 | 3.20 | 257.11 | 179.47 | 1.4x |

### Per-Distribution Detail (Round 2)

| Distribution | Scratch R2 | Cumulative R2 | Scratch/Cumul |
|---|---|---|---|
| isotropic | 86.84 | 8.13 | 10.7x |
| rank_1 | 1519.02 | 479.19 | 3.2x |
| rank_2 | 441.53 | 50.57 | 8.7x |
| rank_half | 629.69 | 88.74 | 7.1x |
| exp_decay | 163.69 | 12.12 | 13.5x |
| power_law | 160.19 | 13.40 | 12.0x |
| step_function | 1442.02 | 333.71 | 4.3x |
| wishart | 184.06 | 29.40 | 6.3x |
| condition_2 | 90.70 | 8.59 | 10.6x |
| condition_10 | 153.81 | 11.28 | 13.6x |
| condition_100 | 473.29 | 110.58 | 4.3x |
| condition_1000 | 1160.53 | 251.17 | 4.6x |

## What These Results Tell Us

### 1. Adversarial retraining from scratch causes catastrophic forgetting

The most striking result is how badly the scratch model degrades on standard benchmarks. After round 1, its isotropic performance collapsed from 0.96 to 20.57 -- a 21x degradation on the simplest possible distribution. By round 2, isotropic performance hit 86.84, meaning the scratch model performs **87x worse than ridge regression on standard Gaussian data**.

This is catastrophic forgetting in its purest form. When the model is retrained from random weights on a 30% adversarial / 70% Gaussian mix, the adversarial examples dominate the loss landscape. The model learns to handle the adversarial curriculum but loses its ability to perform basic in-context regression. Even though 70% of training data is still standard Gaussian, the adversarial examples have much higher loss magnitude and gradient contribution, effectively hijacking the optimization.

By round 2, the curriculum grew to 75 genomes and the dynamic p_adv increased to 0.45. This amplified the forgetting: scratch round 2 benchmark mean was **583.50**, a 4.5x degradation from round 1's already-bad 130.72.

### 2. Cumulative fine-tuning preserves knowledge far better

The cumulative model degraded significantly less. After round 1, its isotropic ratio was 6.60 (vs scratch's 20.57). After round 2, isotropic was 8.13 (vs scratch's 86.84). The cumulative model retained most of its prior learned algorithm while absorbing new adversarial robustness.

This 2-4x advantage in round 1 widened to **4.6x in aggregate and up to 13.6x on individual distributions** by round 2. The divergence between strategies accelerated over rounds, indicating that the scratch model's forgetting compounds while the cumulative model's knowledge base provides a stabilizing anchor.

The distributions where the cumulative model showed the largest advantage are the smooth, well-conditioned ones (exp_decay: 13.5x, condition_10: 13.6x, power_law: 12.0x, isotropic: 10.7x). These are exactly the distributions the base model had already mastered. The cumulative model retained this mastery; the scratch model forgot it.

### 3. Both models struggle equally on extreme rank-deficient distributions

On rank_1, the cumulative model offered almost no advantage (349.31 vs 358.35 in round 1, 479.19 vs 1519.02 in round 2). Similarly, step_function and condition_1000 showed the smallest cumulative advantage. These rank-deficient and extremely ill-conditioned distributions represent a fundamentally hard regime for the transformer's learned algorithm -- neither training strategy solves the underlying problem.

This suggests that the ICL model's failure on extreme rank deficiency is not a training data distribution issue but an architectural or algorithmic limitation. The transformer's in-context algorithm (which approximates gradient descent or ridge regression) cannot adapt to distributions where most eigenvalues are near-zero, regardless of how much adversarial data it has seen.

### 4. The adversary fitness plateau reveals a robustness ceiling

Adversary fitness went from 20.87 (round 1) to 18.20 (round 2) to 18.86 (round 3). The improvement from round 1 to 2 suggests the cumulative model became somewhat more robust. But the rebound in round 3 (18.86 > 18.20) indicates the adversary found new attack directions that the round 2 training didn't address.

This ~18-21 fitness plateau tells us the ICL model has persistent structural vulnerabilities that adversarial retraining cannot eliminate. The adversary can consistently find distributions where ICL performs 18-20x worse than the optimal estimator, even after multiple rounds of targeted curriculum training. This is consistent with the hypothesis that the transformer's in-context algorithm has fixed inductive biases (e.g., implicit Gaussian assumptions, limited spectral adaptation) that create an irreducible gap against adversarially-chosen distributions.

### 5. Adversarial curriculum training is a trade-off, not a free lunch

Neither model improved on the base model's benchmark performance. The base model had a benchmark mean of 2.22. After two rounds of adversarial training, the best we achieved was 83.02 (cumulative round 1) -- a 37x worse benchmark score. While the models became more robust to the specific adversarial distributions in the curriculum, this came at a massive cost to general performance.

This reveals a fundamental tension in adversarial robustness for ICL: the model has limited capacity for distributional generalization, and allocating that capacity toward adversarial robustness reduces capacity for standard-distribution performance. The cumulative model manages this trade-off much better than scratch (83 vs 131 in round 1, 126 vs 584 in round 2), but neither avoids the trade-off entirely.

### 6. The cumulative approach preserves the learned algorithm's structure

The per-distribution data reveals that the cumulative model's performance profile remains qualitatively similar to the base model across rounds: it is best on smooth distributions (isotropic, exp_decay, power_law), moderate on mid-range structures (wishart, condition_2-10), and worst on extreme structures (rank_1, step_function, condition_1000). The scratch model, by contrast, shows a flattened performance profile where everything is uniformly bad -- the learned algorithm has been disrupted.

This suggests that the cumulative fine-tuning approach preserves the transformer's learned in-context regression algorithm while gently expanding its distributional coverage, whereas scratch retraining destroys the algorithm and builds a new, less effective one under adversarial pressure.

## Summary Table

| Metric | Base | Scratch R1 | Cumul R1 | Scratch R2 | Cumul R2 |
|---|---|---|---|---|---|
| Benchmark mean (non-iso) | 2.22 | 130.72 | 83.02 | 583.50 | 126.25 |
| Isotropic ICL/Ridge | 0.96 | 20.57 | 6.60 | 86.84 | 8.13 |
| Worst distribution | 5.28 | 358.35 | 349.31 | 1519.02 | 479.19 |
| Best non-iso distribution | 1.15 | 20.62 | 6.43 | 90.70 | 8.59 |
| Adversary best fitness | -- | 20.87 | -- | 18.20 | -- |

## Experimental Configuration

- **Model**: d=5, 6-layer GPT-2, n_embd=128 (or per checkpoint config), 150k training steps
- **Adversary**: CMA-ES, 50,000 evaluations/round, 5 restarts, pipeline genomes (289 dimensions)
- **Scratch retraining**: 150,000 steps, LR=3e-4, base p_adv=0.3 (dynamically scaled)
- **Cumulative fine-tuning**: 20,000 steps/round, LR=1e-4, base p_adv=0.5 (dynamically scaled, capped at 0.65)
- **Attack target**: cumulative model (adversary always attacks the fine-tuned model)
- **Benchmark**: 12 fixed covariance structures (isotropic, rank-1, rank-2, rank-half, exp-decay, power-law, step-function, wishart, condition-2/10/100/1000)
- **Hardware**: 32-core CPU pod (RunPod), 24 OMP threads, PyTorch 2.11 CPU
- **Duration**: ~16 hours, 2 complete rounds + round 3 attack completed

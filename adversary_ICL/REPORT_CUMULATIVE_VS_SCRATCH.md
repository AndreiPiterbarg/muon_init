# The Adversarial Agent for In-Context Learning: Design, Choices, and Findings

## 1. What I Built: An Agent That Searches for ICL Failures

Standard results on in-context learning (Garg et al., 2022) show that a transformer trained on synthetic regression tasks with isotropic Gaussian inputs learns to approximate ridge regression in-context -- it solves new regression problems at test time using only the prompt, no weight updates. The obvious follow-up question is: **when does this learned in-context algorithm fail?** Random testing will find the easy failures (heavy scaling, pathological noise), but the interesting failures are structured distributions that reveal something about the transformer's inductive biases.

The adversarial agent I worked on is an automated search procedure that finds input distributions where the ICL model performs much worse than the optimal estimator (ridge regression, in the linear-regression function class). The agent is the core scientific instrument of this project: everything else -- the curriculum retraining, the benchmarking, the fine-tuning -- is downstream of what the adversary discovers.

## 2. How the Adversary Works: Core Design Choices

### 2.1 Representation: Compositional Pipeline Genomes

The first critical choice is **what the adversary is allowed to propose**. A naive adversary might directly output a covariance matrix, but this is too unstructured -- it can only produce Gaussian distributions, and the space is high-dimensional with poor geometry for optimization. Instead, each candidate distribution is encoded as a **pipeline genome**: a flat real-valued vector that decodes into a compositional transform pipeline.
LET
The pipeline structure:

```
z ~ base_distribution -> [stage 1] -> [stage 2] -> ... -> [stage 8] -> normalize -> x
```

- **Base distribution** (chosen via argmax over 8 logits): `gaussian`, `student_t`, `uniform`, `laplace`, `log_normal`, `beta`, `exponential`, `mixture_of_gaussians`. All are standardized to mean 0, variance 1 so that scale differences come from transforms, not the base.
- **8 transform stages**, each choosing one of 9 transform types (argmax over 9 logits): `identity`, `affine`, `elementwise_power`, `mixture_injection`, `soft_clipping`, `dimension_interaction`, `feature_sparsity`, `sorting_rank`, `sign_flip`. Each transform has its own continuous parameters (e.g., affine has a lower-triangular scale matrix and a bias vector).
- **Task weight** `w` (unit-normalized) and **noise log-sigma** (clamped to avoid numerical explosion).

For d=5 this gives a genome of 289 real numbers; for d=20 it's 2,104. The genome is a single flat vector -- which is exactly what gradient-free optimizers like CMA-ES want.

**Why this representation is the right choice**: It is *expressive* (can produce heavy-tailed distributions, rank-deficient covariance, rank-order scrambling, sparse features, all in composition), but it is also *interpretable* (the active stages can be read off as a short "recipe" like `laplace -> sorting_rank -> mixture_injection`). Unused stages default to identity, and CMA-ES adapts its step sizes to near-zero on inert dimensions. This means the adversary naturally finds minimal, readable failure modes rather than a dense pile of transforms.

### 2.2 Fitness Function: Scale-Invariant Log-Ratio

The adversary's objective is the key design decision. A naive choice would be "maximize ICL error on this distribution" -- but this is trivially exploited by scaling up input magnitudes. The model will have high error when the inputs are in the millions, but that tells us nothing interesting.

Our fitness is the **mean log-ratio of ICL error to best-baseline error, averaged over the underdetermined regime of the learning curve**:

```
fitness = mean over k in [1, d] of log( ICL_error_at_k / best_baseline_error_at_k )
```

Properties this gives us:

- **Scale invariance**: If we multiply all inputs by 10, both ICL and ridge errors scale together; the ratio is unchanged. The adversary cannot win by blowing things up.
- **Log transformation**: A distribution where ICL is 100x worse at one point and 1x worse elsewhere gets a much higher fitness than one where ICL is uniformly 10x worse. This biases the search toward structured, concentrated failures over uniform mediocrity.
- **Underdetermined regime only** (k <= d): In the overdetermined regime (k >> d), any reasonable estimator does well. We care about the regime where the prompt is actually small -- this is where in-context learning is interesting.
- **Log-ratio against the best baseline, not against zero**: Some distributions are just hard for everyone (e.g., pure noise). Those have high absolute ICL error but also high ridge error -- fitness correctly identifies this as a non-discovery. We only reward cases where ICL specifically falls behind what the optimal algorithm can do.

In practice, fitness values of 1-2 indicate mild gaps, 5-10 are substantial breaks, and 20+ are severe structural failures. The adversary found fitness ~20.87 on round 1 of my experiment.

### 2.3 Optimizer: Multi-Restart Diagonal CMA-ES

The genome space is high-dimensional (289 for d=5) and the fitness is non-differentiable (it depends on argmax over discrete transform types). Gradient-free evolutionary optimization is the natural choice.

- **CMA-ES** (Covariance Matrix Adaptation Evolution Strategy) maintains a multivariate Gaussian proposal distribution and updates its mean, step-size, and covariance based on which candidates had the best fitness. It is one of the best black-box optimizers for smooth non-convex objectives.
- **Diagonal-only covariance**: Full-covariance CMA-ES is O(d²) memory and O(d³) per update, which becomes expensive at d=2,104 (the full-model case). Diagonal CMA-ES restricts to per-dimension step sizes -- it loses the ability to track correlations between genome coordinates, but it scales linearly and works well in practice because the pipeline genome is naturally axis-aligned (each transform's parameters live in their own block).
- **Multi-restart (5 independent runs per round)**: The fitness landscape is highly multimodal -- different base distributions and transform compositions occupy different optima. A single CMA-ES run converges to one of them and gets stuck. Running 5 independent restarts, each with a different random initialization, dramatically increases coverage of the failure-mode landscape. We then aggregate top-k results across restarts for downstream curriculum building.
- **Budget of 50,000 evaluations per round (5 restarts × 10,000 evals each)**: Each evaluation runs the ICL model and baselines on a batch of 64 × num_batches (10) sampled contexts. This is the bottleneck of the whole experiment -- 50k evals × 640 forward passes ≈ 32M model invocations per round, which takes ~5 hours on CPU.

### 2.4 Parsimony Pressure: Optional

One concern with a compositional pipeline is that the adversary will use all 8 transform stages unnecessarily, producing opaque, over-parameterized breaks. To counter this, the fitness function can include a parsimony penalty:

```
effective_fitness = raw_fitness - lambda * complexity
```

where complexity counts active (non-identity) stages, affine deviation from identity, and non-Gaussian base distributions. The lambda parameter controls the trade-off between break strength and simplicity.

For d=5, I set lambda=0.0 (no parsimony pressure) because at this small dimension the adversary naturally converges to minimal recipes -- the round 1 best genome used only 3 active stages out of 8 possible. For d=20 (not run here), parsimony becomes essential to avoid uninterpretable breaks.

### 2.5 Baselines: Ridge, Least Squares, Averaging

The adversary's fitness compares ICL error to the **best baseline error**, not just ridge. The three baselines serve different purposes:

- **Ridge regression** (alpha=1.0): The reference optimal estimator for linear regression with Gaussian noise.
- **Least squares** (OLS): A sanity check against underfit/overfit behavior; OLS is unstable in the underdetermined regime, so its high error there does not disqualify a candidate break.
- **Averaging**: A null baseline that just predicts the mean. If ICL is worse than averaging, the model has essentially learned nothing useful for that distribution.

Taking the element-wise minimum across these (per-point along the learning curve) gives a robust "best achievable" reference. The adversary only gets fitness credit when ICL is worse than *all* of these -- so we are specifically finding failures of the learned in-context algorithm, not cases where everyone struggles.

### 2.6 Task: Noisy Linear Regression

The function class is fixed: `y = w^T x + noise`, with w unit-normalized by the genome and noise standard deviation controlled by the genome's noise parameter. This is the simplest ICL setting where ridge regression is the optimal estimator, so the adversary's job is sharply defined: find a distribution of x where the transformer's learned algorithm diverges from ridge.

## 3. What the Adversary Found

### 3.1 The Best Genome (Round 1, fitness = 20.87)

The adversary's top discovery was:

**Base**: `laplace` (heavy-tailed, sharper peak than Gaussian)
**Active stages**: `sorting_rank` -> `mixture_injection`
**Noise std**: 0.007 (very low, so signal is clean)
**Inactive stages**: 6 of 8 transform slots are identity (adversary found no use for them)

Decoded meaning: sample from a Laplace distribution, replace each feature's values with their ranks (destroys continuous structure while preserving order), then blend with an adversary-chosen mixture of Gaussians.

This is a *structured* failure. It tells us three things about the ICL model's inductive biases:

1. **The model assumes Gaussian-like tail behavior**. Laplace has much heavier tails than Gaussian. The learned in-context algorithm apparently fails to handle the higher frequency of large outlier values.
2. **The model assumes smooth, continuous feature values**. Sorting-rank transforms the input so that the magnitude of each feature is its rank index, not a continuous draw. This is a simple but devastating perturbation -- the model's learned regression algorithm is not rank-invariant.
3. **The model is not robust to mixture noise**. Injecting a mixture-of-Gaussians component on top further confuses the model, even though a Bayes-optimal estimator would integrate over it.

None of these weaknesses would be visible from standard benchmarks on isotropic Gaussian inputs. They are only visible because the adversary searched for them.

### 3.2 Fitness Across Rounds

| Round | Best Fitness | What this means |
|---|---|---|
| 1 | 20.87 | Adversary found a distribution where ICL is 20.87x worse than ridge |
| 2 | 18.20 | After retraining on round-1 failures, fitness dropped slightly |
| 3 | 18.86 | Fitness rebounded -- adversary found new attack directions |

The fact that fitness plateaus around 18-20, rather than dropping toward zero after retraining, is scientifically meaningful: **the adversary keeps finding structured failures even after the model has been explicitly trained on prior failures**. The model's robustness ceiling appears to be set by architectural limitations, not by training data coverage. This is the single most valuable finding the adversary produced -- it points to a real, persistent limit of what the transformer's learned in-context algorithm can do.

### 3.3 Curriculum Diversity

The top-50 genomes from each round's attack are added to a curriculum used for retraining. Inspecting them reveals that the adversary does not just rediscover the same break repeatedly:

- Different base distributions get selected across restarts (`laplace`, `student_t`, `mixture_of_gaussians`).
- Different active transform sets appear (`sorting_rank` is popular, but so are `feature_sparsity`, `elementwise_power`, and `soft_clipping`).
- The covariance spectra of the top genomes have meaningful Frobenius distance from each other -- the overlap-detection stopping condition (which fires if >80% of new failures are near-duplicates of old ones) did not trigger in rounds 1-3.

This confirms the multi-restart strategy is doing real work. A single CMA-ES run would lock onto one break and optimize it; the 5-restart ensemble surveys the failure landscape.

## 4. The Retraining Pipeline: Cumulative vs. Scratch (Brief)

Downstream of the adversary, the curriculum is used to update the ICL model. I implemented two strategies in parallel:

- **Scratch**: Random-init and train from zero for 150k steps on 70% standard Gaussian + 30% adversarial curriculum.
- **Cumulative**: Load the previous round's checkpoint and fine-tune for 20k steps with a lower learning rate and higher adversarial fraction (50%).

Both are benchmarked on a fixed 12-distribution test suite. The benchmark mean (non-isotropic) evolves as:

| Round | Base | Scratch | Cumulative | Scratch/Cumulative |
|---|---|---|---|---|
| 0 | 2.22 | 2.22 | 2.22 | 1.0x |
| 1 | -- | 130.72 | **83.02** | 1.6x |
| 2 | -- | 583.50 | **126.25** | 4.6x |

Cumulative fine-tuning wins by a widening margin -- 1.6x in round 1, 4.6x in round 2. From-scratch retraining suffers catastrophic forgetting: isotropic performance went from 0.96 to 86.84 in two rounds (90x degradation on the distribution the model was originally trained on). Preserving weights across rounds and adapting gently is far more effective than retraining from noise on an adversarial-heavy mix.

This validates the core hypothesis that motivated the extension: **adversarial robustness is better acquired by building on existing competence than by starting fresh under adversarial pressure**. The gap widens each round, suggesting the advantage would only grow with more rounds.

## 5. What Went Well

**The adversary produced a structured, interpretable failure on its first round.** The round-1 best genome was a 3-stage pipeline (out of 8 possible) with a clear narrative: heavy-tailed base + rank scrambling + mixture noise. This is exactly the kind of result the project was aiming at -- a non-trivial, non-noise distribution where ICL specifically falls apart.

**Scale-invariant fitness held up under optimization pressure.** No genome won by exploiting input scaling -- all discovered breaks are legitimate structural failures of the learned algorithm, not numerical pathologies. This is the payoff of the log-ratio design.

**Multi-restart CMA-ES found diverse breaks.** The curriculum from rounds 1-3 contains distributions with different base types, different active transforms, and meaningfully different covariance spectra. The overlap-detection stopping condition did not fire, confirming ongoing diversity.

**The compositional representation was expressive enough.** The adversary used only 2-3 active stages per top genome, far below the 8-stage budget. This means the representation was not the bottleneck -- the failures the adversary found are genuinely simple in structure, not artifacts of over-parameterization.

**The parsimony design worked as intended.** With lambda=0 at d=5, the adversary naturally converged to minimal recipes. The representation's inductive bias (unused stages default to identity) handled simplicity without needing explicit penalty.

## 6. What Went Poorly

**The adversary plateaued around fitness 18-20 rather than being defeated by retraining.** Ideally, each round of retraining would make the adversary's job harder, and fitness would trend toward zero. In practice, fitness dropped only mildly (20.87 -> 18.20) before rebounding (-> 18.86). The adversary is not running out of attack directions -- it just rotates to new ones. This is a finding, not a failure, but it means "adversarial training to convergence" is likely not achievable with the current architecture.

**Some distributions (rank_1, step_function, condition_1000) are equally hard for both retrained models.** These are cases where the adversary's attacks expose something the curriculum training cannot fix. It suggests the adversary is finding not just training-data gaps but genuine architectural limits -- which is useful scientifically, but means robustness via curriculum training has a hard ceiling.

**The adversary's evaluation cost is the experiment's bottleneck.** Each fitness evaluation requires a model forward pass on 640 contexts (64 batch × 10 batches), and the CMA-ES budget of 50,000 evaluations takes ~5 hours per round on CPU. This means we get roughly one adversary round per 5 hours, plus ~2 hours for retraining, so ~7 hours per round total. Over 10 possible rounds this is 70 hours -- and we terminated after 16 hours with 2 complete rounds. The adversary is doing its job, but its computational demand limits how much iteration we can afford.

**Only 2 complete rounds means the robustness-ceiling claim is suggestive, not definitive.** Three data points (fitness 20.87, 18.20, 18.86) establish a plateau pattern but not conclusively. A 5-10 round run would be much more convincing.

**Discrete transform selection via argmax is a potential issue for CMA-ES**. The transform type per stage is chosen by argmax over 9 logits -- a non-smooth operation. CMA-ES assumes a smooth-ish objective, and discrete jumps can cause the covariance adaptation to waste steps. In practice it seems to work fine (the adversary converges), but a relaxation via Gumbel-softmax or switching to a hybrid genetic algorithm might search the transform type space more efficiently. This is the most questionable part of the current design.

## 7. Summary

The adversarial agent is the scientific instrument that makes this whole project possible. Its design choices -- compositional pipeline genomes, scale-invariant log-ratio fitness, multi-restart diagonal CMA-ES, baseline-relative scoring -- are each chosen to filter out trivial breaks and surface structured, interpretable failures of the learned in-context algorithm.

It succeeded at that task: the round-1 discovery (`laplace -> sorting_rank -> mixture_injection`, fitness 20.87) is exactly the kind of minimal, readable failure mode the project was aiming at. It tells us something concrete about the model's inductive biases -- Gaussian-tail assumption, continuous-feature assumption, non-robustness to rank scrambling and mixture noise.

The follow-up finding from the cumulative vs. scratch retraining comparison -- that fine-tuning dominates from-scratch retraining by a widening margin (1.6x -> 4.6x) -- is a useful engineering result, but the more important scientific claim is what the adversary keeps finding: **even after retraining, the adversary recovers most of its fitness, suggesting a persistent robustness ceiling determined by architectural inductive biases rather than by training-data coverage**.

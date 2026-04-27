# flip_flop project — what we have done and what we have found

This file is the running summary. It captures the pipeline, the findings to
date, and how confident we should be in each one. It does not advocate for
any particular framing of the results.

## What the project does

We train a flip-flop language model (FFLM) per Liu et al. 2023, then use an
adversarial loop to find input distributions on which the trained model has
a high read-position error rate ("glitch rate"). The published-baseline
comparator is **R4**: a fresh GPT-2 (6 layers, 512 dim, 8 heads, ~19M
params) trained from scratch for 10000 steps on a per-sequence-uniform
mixture of FFL(0.1) ∪ FFL(0.9) ∪ FFL(0.98). A finding only "counts" if R4
also fails on it; otherwise we have just rederived R4.

## Pipeline

1. **Train** an FFLM on FFL(0.8) ("baseline").
2. **Search** with diagonal CMA-ES over a parametric distribution family,
   maximising fitness = `T_glitch − λ·max(0, lstm_glitch − τ)`. The LSTM
   penalty filters distributions where a 1-layer LSTM also fails (signal
   that the distribution is ill-posed, not adversarial).
3. **Extract families** from the search log: HDBSCAN on flattened config
   parameters, geometric-median per cluster, then α-pull-back so the
   representative is "as close to FFL(0.8) as possible while still
   transformer-hard and LSTM-clean."
4. **Retrain** the baseline on a 70/30 mixture of FFL(0.8) and the
   axis-floor families.
5. **Decisive eval**: on identical sequences, score baseline vs R4 vs
   retrained model. Classify each family as BREAKTHROUGH / WEAK /
   REDISCOVERY / NULL.

## Findings to date

| family                 | axis         | baseline | R4 (best seed) | retrain | round |
| ---------------------- | ------------ | -------: | -------------: | ------: | ----- |
| piecewise_c00_a1.00    | piecewise    |   11.88% |          1.66% |   0.04% | prior |
| piecewise_c01_a1.00    | piecewise    |   13.12% |          0.00% |   0.02% | prior |
| piecewise_c02_a1.00    | piecewise    |   12.99% |          0.00% |   0.02% | prior |
| stationary_c00_a1.00   | stationary   |    6.86% |          0.00% |   0.26% | prior |
| planted_decoy_a1.00_n2 | planted      |    0.00% |          0.00% |   0.00% | prior |
| **bit_markov_c00_a1.00** | bit_markov |    0.18% |     **96.12%** |   0.18% | this  |
| bit_markov_c01_a1.00   | bit_markov   |    0.08% |          0.35% |   0.10% | this  |
| **write_flip_c00_a1.00** | write_flip |    2.78% |     **37.94%** |   0.03% | this  |
| write_flip_c01_a1.00   | write_flip   |    3.69% |          0.00% |   0.01% | this  |

Two distinct families, on two different rounds, where R4 fails substantially
above its FFL-battery floor: **piecewise_c00** (~1.7%) and the new
**bit_markov_c00 / write_flip_c00** pair (96% / 38%). Magnitudes very
different, and the validation depth of the two rounds is also very different
— see below.

## Finding 1 — piecewise_c00 (prior round, validated four ways)

What it is: a length-512 sequence in 4 segments. The first three segments
are densely write-heavy (p_i ≈ 0); the fourth segment is a sparse tail
(p_i ≈ 0.984). R4 was trained on stationary FFL(0.1)/FFL(0.9)/FFL(0.98)
mixtures, so it has seen sparse tails *as a whole sequence* but never
preceded by a dense-write prefix.

Validation that survived (full report:
[REPORT_PIECEWISE_C00_VALIDATION.md](REPORT_PIECEWISE_C00_VALIDATION.md)):

1. **Mechanism**: 100% of R4's c00 errors live in segment 3 (the sparse
   tail). The first 75% of the sequence is glitch-free. The failure is the
   *transition* from a dense-write context to a sparse one, not sparse
   tails per se.
2. **Data-seed robustness**: 1.662% ± 0.015% across 5 disjoint eval seeds at
   N=50000. ~110× the noise floor; ~46× the retrained model at the same
   point.
3. **Model-seed robustness**: three independently trained R4 seeds (0/1/2)
   all glitch 1.41–2.74% on c00 while staying < 0.005% on the other
   piecewise families. Seed 2 is the worst — the failure is reproducible,
   not a near-edge artifact of seed 0.
4. **Neighborhood**: 59 of 60 valid jittered configs around c00 trip R4
   (≥ 0.5%) and 0 of 60 trip the retrained model. The failure is regional,
   not pointlike.

What is *not* yet shown for this finding: which attention heads / layers
drive it (mechanistic-interpretability work); whether the failure persists
under boundary or count perturbations of the segments; explicit LSTM
skyline on the c00 neighborhood; comparison against a hand-designed
mixture that includes a dense → sparse transition.

## Finding 2 — bit_markov_c00 and write_flip_c00 (this round)

We searched two new axes intended to add structural blind spots that R4's
stationary mixture cannot cover:

- **BitMarkov**: data bits drawn from a two-state Markov chain
  (`bit_stay` = P(b_t == b_{t−1}); iid recovers at 0.5).
- **WriteFlipRate**: each write overrides the stored bit with a controllable
  probability (`flip_rate`; iid at 0.5; 0.0 = always-repeat;
  1.0 = always-flip).

CMA-ES on each, 3000 evals, pop 16, 3 restarts. The search worked: peak
fitness 0.962 (bitmarkov) and 0.523 (writeflip). The decisive eval (N=10000,
eval_seed=12345) produced two BREAKTHROUGH families, but with caveats that
matter for how to interpret them.

### Headline numbers

| family               | baseline | R4_seed0 | retrain |
| -------------------- | -------: | -------: | ------: |
| bit_markov_c00_a1.00 |   0.179% |  96.115% |  0.179% |
| write_flip_c00_a1.00 |   2.782% |  37.937% |  0.027% |

`bit_markov_c00`: R4 fails 537× worse than the baseline FFLM, which never
saw any FFL distribution other than FFL(0.8). Training on a *broader*
distribution made the model dramatically *worse* on this region. That is
the surprise.

### Caveat 1 — both BREAKTHROUGH configs collapse to the same regime

The cluster-representative configs the retrain trained on:

| family               | p_w   | p_r        | bit_p1 |
| -------------------- | ----: | ---------: | -----: |
| bit_markov_c00_a1.00 | 0.982 | 1.66e-05   | 0.0013 |
| write_flip_c00_a1.00 | 0.981 | 3.72e-04   | 0.0326 |

Both are write-heavy (p_w ≈ 0.98, on the simplex boundary), with effectively
zero reads, and an extreme bit bias (≈ 0% ones). The two "axes" found the
same mechanism twice; this is one finding, not two.

### Caveat 2 — the axis-specific dynamics never reach the retrain

The existing family extractor's `_cluster_representative_config` flattens
both BitMarkov and WriteFlipRate configs to `(p_w, p_r, bit_p1)` and emits
a Stationary representative. So the `bit_stay` / `flip_rate` parameters
that defined the new axes are stripped before retraining. The retrain's
families are extreme-Stationary distributions, not BitMarkov /
WriteFlipRate ones. The headline numbers therefore credit `(p_w, p_r,
bit_p1)` falling outside R4's symmetric mixture, not the new axis dynamics
per se. Properly testing the new axes would require modifying
`_cluster_representative_config` to retain those parameters — out of scope
for this round (the task spec forbade modifying existing files).

### Caveat 3 — depth of validation

For piecewise_c00 we ran four independent robustness tests (mechanism,
data seed, model seed, neighborhood). For bit_markov_c00 / write_flip_c00
we have run **none** of those. The numbers are at:

- one R4 seed (seed 0)
- one adversary seed (seed 0)
- one eval seed (12345), N=10000
- one cluster representative each

What this round therefore *does not* yet establish:

- Whether the 96% R4 failure persists across multiple R4 training seeds.
- Whether it persists in a parametric neighborhood of the rep config or
  collapses to a single point.
- Whether the failure is a meaningful regime or an artifact of the
  simplex boundary (p_w → 1, p_r → 0).
- Where in the sequence the errors live (per-position glitch breakdown).

The piecewise finding had to clear those bars before we treated it as
real; the new finding has not.

## Length extrapolation (Tier B)

Originally planned for this round. Skipped: **mechanically blocked**.
`flip_flop/model.py::FFLMTransformer` instantiates a HuggingFace
GPT2LMHeadModel with absolute learned positional embeddings of size
`n_positions = 512`. Inference at T > 512 errors on `wpe`. The task spec
forbade modifying `model.py`, so no length-extrap eval was run.

## Honest summary

- The pipeline reliably finds distributions where R4 fails. We have two
  independent rounds of evidence.
- One of those rounds (piecewise) is fully validated. The failure mode
  (dense → sparse transition) is interpretable and survives all four
  robustness tests. This is the result that matches the original project
  goal: a structured, reproducible, mechanistically attributable failure
  mode of Transformer FFLMs that an automated loop discovers.
- The other round (bit_markov / write_flip) shows a more dramatic
  *number* (96% vs 1.7%) but is much earlier-stage:
  - both "axes" collapsed to the same write-heavy boundary regime;
  - the family extractor stripped the very parameters the new axes were
    supposed to test, so what got retrained on is Stationary, not the new
    axis;
  - none of the validation steps from round 1 have been repeated.
- Both findings together say the same thing in different ways: R4's
  stationary symmetric mixture has structurally blind spots, and an
  automated CMA-ES + cluster-family loop finds them. The right way to
  describe round 2 right now is "additional supporting evidence consistent
  with the round 1 phenomenon," not "two independent new failure modes."

## What it would take to make round-2 paper-grade

1. Repeat the **model-seed** test: train R4 with seeds 1 and 2 (already
   trained for round 1), eval all three on the new families. If all three
   glitch ≥ 0.5%, robustness clears. If only seed 0 fails, the result is
   an init artifact.
2. Repeat the **neighborhood** test on the bit_markov_c00 rep config.
3. Per-position breakdown of the 96% R4 failure (analog of round 1's
   step-1 plot).
4. Fix `_cluster_representative_config` to retain `bit_stay` /
   `flip_rate` so the new axes are actually represented in the families.
   Re-run retrain and decisive eval. Without this, the round-2
   contribution is "another way to find the boundary regime," not "two
   new axes."
5. Sanity-check the boundary configs: re-eval an R4-style mixture that
   *includes* `(p_w → 1, bit_p1 → 0)` to confirm a hand-designed mixture
   could close the gap. If yes, the contribution narrows from
   "automated discovery beats hand design" to "automated discovery is
   more efficient than hand design"; if no, the contribution holds.

## Reproducibility

| artifact                                                 | path                                                  |
| -------------------------------------------------------- | ----------------------------------------------------- |
| Round-1 baseline + R4 seed 0/1/2 checkpoints             | `results/flip_flop/{baseline,liu_r4,liu_r4_seed1,liu_r4_seed2}/` |
| Round-1 piecewise validation report                      | [REPORT_PIECEWISE_C00_VALIDATION.md](REPORT_PIECEWISE_C00_VALIDATION.md)            |
| Round-1 diagnostic outputs                               | `results/flip_flop/liu_r4/diagnostic_step{1,2a,2b,3}/` |
| Round-2 adversary search outputs                         | `results/flip_flop/adversary/{bitmarkov,writeflip}/`  |
| Round-2 retrained models                                 | `results/flip_flop/retrain/tierA_{bitmarkov,writeflip}/` |
| Round-2 decisive-eval JSON + log                         | `results/flip_flop/tierA_results/`                    |
| Round-2 report                                           | [results/flip_flop/tierA_results/REPORT.md](results/flip_flop/tierA_results/REPORT.md) |
| Round-2 entry-point scripts                              | `flip_flop/scripts/run_adversary_{bitmarkov,writeflip}.py`, `run_retrain_tierA.py`, `eval_tierA_vs_r4.py` |
| Round-2 configs                                          | `flip_flop/configs/{adversary,retrain_tierA}_{bitmarkov,writeflip}.yaml` |

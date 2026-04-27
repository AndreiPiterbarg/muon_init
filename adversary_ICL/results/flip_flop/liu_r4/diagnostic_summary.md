# Validation report: piecewise_c00 as a robust failure mode of R4

**Verdict: SURVIVES.** All three robustness checks (mechanistic, data-seed,
model-seed, neighborhood) passed.

---

## Step 1 — Diagnosis (mechanistic)

The four piecewise families have nearly identical structure for segments 0–2
(dense-write regions, p_i ≈ 0.001–0.07), but **only c00** has a sparse-tail
final segment:

| family | seg 0 | seg 1 | seg 2 | seg 3 |
|---|---|---|---|---|
| c00 | p_i=0.000 | p_i=0.000 | p_i=0.018 | **p_i=0.984**, bit_p1=0.066 |
| c01 | p_i=0.001 | p_i=0.013 | p_i=0.043 | p_i=0.016 |
| c02 | p_i=0.000 | p_i=0.010 | p_i=0.041 | p_i=0.066 |

R4 (seed=0) per-position glitch on n=10000 fresh seqs (eval_seed=8888,
weighted-per-position aggregate matches `evaluate_dataset` scalar exactly:
`abs_diff = 0.00e+00` for all three families):

| family | total reads | total errors | glitch |
|---|---|---|---|
| c00 | 153,641 | 2,523 | **1.642%** |
| c01 | 127,439 | 1 | 0.001% |
| c02 | 123,306 | 0 | 0.000% |

Per-segment glitch on c00 (R4 seed=0):

| seg | tok range | p_w | p_r | p_i | reads | errs | glitch |
|---|---|---|---|---|---|---|---|
| 0 | [0,128) | 0.783 | 0.217 | 0.000 | 136,242 | 0 | **0.0%** |
| 1 | [128,256) | 0.990 | 0.009 | 0.000 | 6,037 | 0 | 0.0% |
| 2 | [256,384) | 0.981 | 0.001 | 0.018 | 336 | 0 | 0.0% |
| 3 | [384,511) | 0.014 | 0.002 | **0.984** | 11,026 | 2,523 | **22.88%** |

**Diagnosis:** R4's errors on c00 are *entirely* concentrated in segment 3,
where p_i suddenly jumps from ~0.018 to 0.984 after three dense write-heavy
segments. R4 was trained on stationary FFL(0.98) — but it has never seen the
*transition* from a dense write regime to a sparse regime. The Transformer's
learned "attend to previous data tokens" heuristic fails when the relevant
write was hundreds of positions back, but recent positions are filled with
ignores rather than the dense-write context the model was conditioned on.

Files: [piecewise_configs.json](diagnostic_step1/piecewise_configs.json),
[per_position_glitch.png](diagnostic_step1/per_position_glitch.png),
[per_position_glitch_summary.txt](diagnostic_step1/per_position_glitch_summary.txt).

---

## Step 2A — Data-seed robustness

n=50000 sequences per (model, family), 5 disjoint eval seeds {7001..7005}.

| family | baseline | R2-redone-v2 | R4 |
|---|---|---|---|
| planted_decoy_a1.00_n2 | 0.000% ± 0.000% | 0.000% ± 0.000% | 0.000% ± 0.000% |
| piecewise_c02_a1.00 | 12.988% ± 0.030% | 0.024% ± 0.001% | 0.001% ± 0.001% |
| stationary_c00_a1.00 | 6.863% ± 0.074% | 0.259% ± 0.014% | 0.000% ± 0.000% |
| **piecewise_c00_a1.00** | **11.878% ± 0.067%** | **0.036% ± 0.001%** | **1.662% ± 0.015%** |
| piecewise_c01_a1.00 | 13.123% ± 0.068% | 0.019% ± 0.001% | 0.001% ± 0.000% |

**Pass criterion check (piecewise_c00):**
- R4 mean = 1.662%, std = 0.015% → mean / std ≈ 110× (≫ 5×) ✅
- R4 mean / R2-v2 mean = 1.662% / 0.036% ≈ **46.4×** (≫ 10×) ✅
- **PASS**

The 1.66% gap is ~110σ above the data-seed noise floor. Nothing about the
~1.7% number is a sampling artifact.

Files: [seed_sweep_results.json](diagnostic_step2a/seed_sweep_results.json),
[summary.txt](diagnostic_step2a/summary.txt).

---

## Step 2B — Model-seed robustness

Two new R4 models trained from scratch with the *same* hyperparameters as
liu_r4.yaml, only `cfg.seed` changed (seed=1, seed=2). Standard FFL battery
at step 10000 (paper R2 sanity check):

| model | FFL(0.8) | FFL(0.98) | FFL(0.1) |
|---|---|---|---|
| R4_seed0 | 0.000% | 0.000% | 0.000% |
| R4_seed1 | 0.000% | 0.000% | 0.000% |
| R4_seed2 | 0.000% | 0.000% | 0.000% |

All three R4 seeds are skyline-clean; the differences below are not training
artifacts.

Family eval (n=10000, eval_seed=9999, identical sequences across all 3
models):

| family | R4_seed0 | R4_seed1 | R4_seed2 |
|---|---|---|---|
| planted_decoy_a1.00_n2 | 0.000% | 0.000% | 0.000% |
| piecewise_c02_a1.00 | 0.002% | 0.001% | 0.000% |
| stationary_c00_a1.00 | 0.000% | 0.000% | 0.000% |
| **piecewise_c00_a1.00** | **1.722%** | **1.413%** | **2.742%** |
| piecewise_c01_a1.00 | 0.002% | 0.000% | 0.000% |

**Pass criterion check:**
- All three R4 seeds glitch ≥ 0.5% on piecewise_c00? **True** (1.41%, 1.72%, 2.74%) ✅
- All three R4 seeds < 0.5% on c01 and c02?            **True** (max 0.002%) ✅
- **PASS**

The 2.74% glitch on seed=2 is even *higher* than seed=0; the failure is not
shrinking with different inits. The mode of failure is reproducible across
training seeds.

Files: [r4_multi_seed_results.json](diagnostic_step2b/r4_multi_seed_results.json),
[summary.txt](diagnostic_step2b/summary.txt).

---

## Step 3 — Neighborhood escalation

60 configs sampled from a per-segment jitter grid around c00's representative
config (p_w × {0.8, 0.9, 1.0, 1.1, 1.2}; p_r × {0.8, 0.9, 1.0, 1.1, 1.2};
bit_p1 + {-0.1, 0, +0.1}; one segment jittered at a time). N=2000 sequences
per config (10× the noise-floor batch), eval_seed=6000+idx.

Validity check: 0/60 invalid configs (every jittered config produces valid
FFL strings; verified by the data.enforce_read_determinism semantics on the
first 100 sequences of each config).

| model | fraction with glitch ≥ 0.5% |
|---|---|
| baseline | **60/60 (100%)** |
| R2-redone-v2 | **0/60 (0%)** |
| **R4** | **59/60 (98.33%)** |

**Region verdict (R4 high, R2-v2 low): TRUE.**

R4 fails not at a single point but across essentially the entire structured
neighborhood around c00. R2-v2 stays clean across the same neighborhood —
this is *not* a "harder distribution makes everyone fail" effect.

Per-segment-jittered breakdown (mean R4 glitch over the configs that
jittered each segment):

| jittered segment | n | baseline | R2-v2 | R4 |
|---|---|---|---|---|
| seg 0 | 13 | 9.286% | 0.036% | 1.529% |
| seg 1 | 18 | 12.642% | 0.036% | 1.405% |
| seg 2 | 15 | 12.325% | 0.033% | 1.231% |
| seg 3 | 14 | 11.826% | 0.034% | 1.509% |

R4 is roughly uniformly bad (~1.2–1.5%) regardless of which segment we
perturb — the failure mode survives jitter to any of c00's four segments.
Importantly, jittering segment 3 (the sparse tail itself) does not ablate
the failure: R4 is *still* high there. This reinforces the Step 1 diagnosis:
the failure is driven by the dense → sparse *transition*, which all 60
jittered configs preserve.

Files: [neighborhood_results.json](diagnostic_step3/neighborhood_results.json),
[neighborhood_summary.txt](diagnostic_step3/neighborhood_summary.txt).

---

## Bottom-line verdict

**The contribution survives all three robustness checks.**

| check | criterion | outcome |
|---|---|---|
| Step 2A: data-seed | mean R4 ≥ 5σ AND ≥ 10× R2-v2 mean | mean/std ≈ 110×; ratio = 46.4× |
| Step 2B: model-seed | all three R4 seeds ≥ 0.5% on c00, < 0.5% on c01/c02 | seeds ranged 1.41–2.74% on c00; all <0.005% on c01/c02 |
| Step 3: neighborhood | R4 fails on a region (not a point), R2-v2 clean across same region | 59/60 R4 fail; 0/60 R2-v2 fail |

The piecewise_c00 failure is a real, reproducible, *region-level* failure
mode of stationary-mixture R4 training that our adversarial pipeline
discovered. The mechanism is a dense→sparse transition that R4's training
distribution never exposes the model to: R4 sees stationary FFL(0.1),
FFL(0.9), FFL(0.98) but never sees a sequence that begins dense-write and
ends in a long stretch of ignores — yet that exact regime is where the
attention-glitch shows up.

This is exactly the kind of "structured, non-trivial failure mode" the
project goal calls for, and is differentiable from R4: training on the
adversarial distribution (R2-v2) closes the gap (R4: 1.66% → R2-v2: 0.036%,
~46× reduction at the same eval point), without breaking the standard
battery (all three R2-v2 metrics remain skyline-clean — see
`results_tier1_v2/.../eval_battery.json`).

### Caveats / what's not yet shown

- Mechanistic understanding is at the *position* level (Step 1 segment
  attribution), not the *circuit* level. We don't yet know which heads or
  layers are implicated in the dense→sparse failure; that's a follow-up
  investigation.
- The 60 jittered configs only perturb (p_w, p_r, bit_p1) per segment,
  not segment boundaries (start_frac) or segment count (K). A more
  aggressive neighborhood test would also vary K and the sparse-tail
  position; the current evidence shows the failure survives parameter jitter
  but not yet structural jitter.
- We have not retrained R4 on a sparse-tail-aware mixture to confirm that
  c00 is actually closeable by a smarter mixture (vs. requiring our
  cluster-family extractor). That would be the natural follow-up to claim
  "automated method is doing something hand-design wouldn't."

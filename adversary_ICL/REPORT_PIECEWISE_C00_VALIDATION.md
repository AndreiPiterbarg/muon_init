# Validating piecewise_c00 as a robust failure mode of stationary-mixture R4

**Bottom-line:** The adversarial pipeline discovered a real, reproducible,
*region-level* failure mode of Liu et al. 2023's R4 control (uniform mixture
of FFL(0.1)/FFL(0.9)/FFL(0.98), 10k steps, fresh transformer): the
**piecewise_c00** distribution. Three orthogonal robustness checks (data-seed,
model-seed, neighborhood) all pass. The mechanism is interpretable: a
dense-write → sparse-tail *transition* that R4's stationary training
distribution never exposes the model to. Our adversarial-pipeline-trained
control (R2-redone-v2) closes the gap to ~0.04% — a ~46× reduction at the
same point — without breaking the standard FFL battery.

This is the result the project goal calls for: a *structured, interesting
failure mode* of Transformer FFLMs, automatically discovered, mechanistically
attributable, and not an artifact of a single seed or operating point.

## 1. Background and the question this report answers

We trained Liu et al.'s R4 control: a fresh GPT-2 (6L, 512d, 8H ≈ 19M
params) on a per-sequence-uniform mixture of FFL(0.1) ∪ FFL(0.9) ∪ FFL(0.98)
for the canonical 10000 steps. R4 closes the standard battery (FFL(0.8) /
FFL(0.98) / FFL(0.1) all 0% glitch). It also closes essentially every
adversarial family our pipeline previously discovered — *except one*:

```
family                 baseline   R2-redone-v2   R4
piecewise_c00_a1.00    11.91%     0.03%          1.72%   <-- R4 misses this
piecewise_c01_a1.00    13.08%     0.02%          0.00%
piecewise_c02_a1.00    13.16%     0.02%          0.00%
stationary_c00         6.68%      0.26%          0.00%
planted_decoy          0.00%      0.00%          0.00%
```

R4 was trained from scratch on hand-designed sparse-tail mixtures and
nonetheless misses piecewise_c00 by ~1.7 percentage points. Before building
a paper around this gap, it had to clear three bars:

1. **Mechanism**: where in the sequence does R4 fail, and why? (Step 1)
2. **Data-seed robustness**: is the 1.7% a sampling artifact? (Step 2A)
3. **Model-seed robustness**: is the 1.7% an artifact of one R4 init? (Step 2B)
4. **Neighborhood robustness**: does R4 fail at a *point* or a *region*? (Step 3)

All four passed. Detailed numbers, diagnostic outputs, and the validation
methodology follow.

## 2. Step 1 — Mechanism: where do R4's c00 errors live?

The four piecewise families look superficially similar — each is a length-512
sequence with K=4 segments, each with its own (p_w, p_r, bit_p1). What's
*structurally* different about c00?

Comparing the implied p_i = 1 - p_w - p_r per segment:

| family | seg 0 | seg 1 | seg 2 | seg 3 |
|---|---|---|---|---|
| **c00** | p_i ≈ 0.000 | p_i ≈ 0.000 | p_i ≈ 0.018 | **p_i ≈ 0.984**, bit_p1=0.066 |
| c01 | p_i ≈ 0.001 | p_i ≈ 0.013 | p_i ≈ 0.043 | p_i ≈ 0.016 |
| c02 | p_i ≈ 0.000 | p_i ≈ 0.010 | p_i ≈ 0.041 | p_i ≈ 0.066 |

c00 is the only family with a **sparse-tail final segment**: the last 25% of
the sequence is ~98.4% ignores, preceded by three densely write-heavy
segments. c01/c02 are uniformly write-heavy throughout — they have no
analogous sparse tail.

To test that the sparse tail is where R4 fails, I computed per-position
glitch rate for R4(seed=0) on n=10000 fresh sequences from each piecewise
family (eval_seed=8888, disjoint from training and any prior eval). The
sanity check (weighted aggregate across positions = scalar
`evaluate_dataset` error rate) was satisfied exactly, `abs_diff = 0.00e+00`
for all three families.

Per-segment glitch on c00 (R4 seed=0, n=10000):

| seg | tok range | p_w | p_r | p_i | reads | errs | **glitch** |
|---|---|---|---|---|---|---|---|
| 0 | [0,128) | 0.783 | 0.217 | 0.000 | 136,242 | 0 | **0.0%** |
| 1 | [128,256) | 0.990 | 0.009 | 0.000 | 6,037 | 0 | **0.0%** |
| 2 | [256,384) | 0.981 | 0.001 | 0.018 | 336 | 0 | **0.0%** |
| 3 | [384,511) | 0.014 | 0.002 | **0.984** | 11,026 | 2,523 | **22.88%** |

**100% of R4's errors on c00 live in segment 3** — the sparse-tail segment.
R4 is perfect on the first 75% of the sequence (which looks like
write-heavy regimes it was never explicitly trained on, but generalizes to)
and catastrophically bad on the last 25% (sparse-tail regime that *is* in
its training mix as FFL(0.98), but only as a stationary baseline — never
preceded by a dense-write prefix).

**Why this is the failure mode:** The Transformer R4 has presumably learned
two distinct "circuits": one for dense-write contexts (where the most-recent
write is right next to the read) and one for sparse contexts (where it must
attend back hundreds of positions). It saw both during training, but always
in isolation. c00 stitches them together: the model enters segment 3
expecting the "find the most recent write" circuit, but the training
distribution never showed it that the most-recent write might live in a
faraway dense-write *prefix*. The result is a glitch that is neither
"long-range attention failure" nor "short-range dense-context failure" but
specifically **the dense → sparse transition**.

Plot of the per-position curve (with c00 segment boundaries overlaid):
[results/flip_flop/liu_r4/diagnostic_step1/per_position_glitch.png](results/flip_flop/liu_r4/diagnostic_step1/per_position_glitch.png).

## 3. Step 2A — Data-seed robustness

Could the 1.72% number just be sampling noise? With 10000 sequences × ~150k
reads, the sampling error on a 1.7% rate is ~0.03% — small but not negligible
relative to a 1.72% measurement. To rule out a coincidence, I re-ran the
eval with **n=50000 and 5 disjoint eval seeds** {7001, 7002, 7003, 7004,
7005} (none of which overlap with training, validation, or Step 1's
eval_seed). Per-seed numbers per (model, family):

| family | baseline | R2-redone-v2 | R4 |
|---|---|---|---|
| planted_decoy_a1.00_n2 | 0.000% ± 0.000% | 0.000% ± 0.000% | 0.000% ± 0.000% |
| piecewise_c02_a1.00 | 12.988% ± 0.030% | 0.024% ± 0.001% | 0.001% ± 0.001% |
| stationary_c00_a1.00 | 6.863% ± 0.074% | 0.259% ± 0.014% | 0.000% ± 0.000% |
| **piecewise_c00_a1.00** | **11.878% ± 0.067%** | **0.036% ± 0.001%** | **1.662% ± 0.015%** |
| piecewise_c01_a1.00 | 13.123% ± 0.068% | 0.019% ± 0.001% | 0.001% ± 0.000% |

The pre-registered pass criterion was:
- **R4 mean ≥ 5σ:** 1.662% / 0.015% ≈ 110× — passes by 22× over threshold ✅
- **R4 mean ≥ 10× R2-v2 mean:** 1.662% / 0.036% = **46.4×** — passes by 4.6× ✅

The 1.66% gap is ~110σ above the data-seed noise floor. It is not a sampling
fluke under any reasonable interpretation.

## 4. Step 2B — Model-seed robustness

A more dangerous possibility: the gap exists because R4(seed=0) happens to
sit in a particular minimum of the loss surface, and a different
initialization would close c00. To test this, I trained two more R4 models
from scratch with the **same** hyperparameters as
`flip_flop/configs/liu_r4.yaml` and only `cfg.seed` changed (seed=1 and
seed=2). Each took ~60 min on the 3080 Laptop.

Standard FFL battery sanity check (each model's `eval_log.jsonl` at step
10000):

| model | FFL(0.8) | FFL(0.98) | FFL(0.1) |
|---|---|---|---|
| R4_seed0 | 0.000% | 0.000% | 0.000% |
| R4_seed1 | 0.000% | 0.000% | 0.000% |
| R4_seed2 | 0.000% | 0.000% | 0.000% |

All three are skyline-clean. The differences below are not training
incompleteness or instability.

Family eval, n=10000, eval_seed=9999 (identical sequences for all three
models):

| family | R4_seed0 | R4_seed1 | R4_seed2 |
|---|---|---|---|
| planted_decoy_a1.00_n2 | 0.000% | 0.000% | 0.000% |
| piecewise_c02_a1.00 | 0.002% | 0.001% | 0.000% |
| stationary_c00_a1.00 | 0.000% | 0.000% | 0.000% |
| **piecewise_c00_a1.00** | **1.722%** | **1.413%** | **2.742%** |
| piecewise_c01_a1.00 | 0.002% | 0.000% | 0.000% |

Pass criterion:
- **All three R4 seeds glitch ≥ 0.5% on c00:** ✅ (range: 1.41–2.74%)
- **All three remain < 0.5% on c01 and c02:** ✅ (max: 0.002%)

Notably, **seed=2's glitch (2.74%) is even higher than seed=0's (1.72%)** —
the failure is not a near-edge artifact that goes away with a different
init. If anything, the failure mode is *more pronounced* on seed=2, which
suggests seed=0 is a slightly favorable case rather than an unrepresentatively
bad one. The mode of failure is reproducible across training seeds.

## 5. Step 3 — Neighborhood: point or region?

The most damaging possible attack on the result would be: *yes, R4 fails at
the exact c00 representative configuration, but a tiny perturbation closes
the gap, so this is "found by accident" and not a real failure mode.* To
test this, I generated a structured neighborhood around c00 by jittering
each segment's (p_w, p_r, bit_p1) one-segment-at-a-time:

- p_w by multiplicative factors {0.8, 0.9, 1.0, 1.1, 1.2}
- p_r by multiplicative factors {0.8, 0.9, 1.0, 1.1, 1.2}
- bit_p1 by additive deltas {-0.1, 0.0, +0.1}

K=4 segments × 75 jitters = 300 candidates; subsampled to **60** (random
seed=0). Each jittered config was clipped back into the simplex (p_w + p_r
≤ 1, both ≥ 0; bit_p1 ∈ [0,1]) using `flip_flop.adversary.family._clip_simplex2`
and `_clip01`. Validity check: 0/60 invalid configs (every jittered
distribution still produces valid FFL strings, verified by
`enforce_read_determinism` on the first 100 sequences of each).

For each valid config: N=2000 sequences, eval_seed=6000+idx, batch=64.

Headline:

| model | fraction with glitch ≥ 0.5% |
|---|---|
| baseline | **60/60 (100%)** — fails everywhere |
| R2-redone-v2 | **0/60 (0%)** — clean everywhere |
| **R4** | **59/60 (98.33%)** — fails almost everywhere |

**Region verdict (R4 high AND R2-v2 low across the same neighborhood): TRUE.**

R4 fails not at a single point but across essentially the entire structured
neighborhood around c00. R2-v2 stays clean across the *same* sequences — so
this is not "a harder distribution that breaks every model"; it is
specifically a regime where the stationary-mixture R4 falls behind the
adversarial-pipeline-trained R2-v2.

Per-segment-jittered breakdown (mean glitch over the configs that perturbed
each segment):

| jittered segment | n | baseline | R2-v2 | R4 |
|---|---|---|---|---|
| seg 0 | 13 | 9.286% | 0.036% | 1.529% |
| seg 1 | 18 | 12.642% | 0.036% | 1.405% |
| seg 2 | 15 | 12.325% | 0.033% | 1.231% |
| seg 3 | 14 | 11.826% | 0.034% | **1.509%** |

R4 is roughly uniformly bad (~1.2–1.5%) regardless of which segment we
perturb. Importantly, this includes jittering segment 3 — the sparse tail
itself. **The failure persists when we change the sparse-tail segment**,
because the *transition* (which all 60 configs preserve, by construction)
is what drives the glitch, not the exact parameters of the tail.

## 6. Bottom-line and what's not yet shown

### What survives

1. **Mechanism (Step 1)**: 100% of R4's c00 errors live in segment 3 — a
   sparse-tail regime that follows three dense-write segments. This is a
   dense → sparse *transition* failure, not a bare sparse-tail failure.
2. **Data-seed robustness (Step 2A)**: 1.66% ± 0.015% across 5 disjoint eval
   seeds; 110× the noise floor; 46.4× R2-v2 at the same operating point.
3. **Model-seed robustness (Step 2B)**: All three R4 seeds (0/1/2) glitch
   1.41–2.74% on c00, all stay <0.005% on c01/c02, all are skyline-clean
   on the standard FFL battery.
4. **Neighborhood robustness (Step 3)**: 59/60 valid jittered configs (98.3%)
   trip R4; 0/60 trip R2-v2. The failure is a region, not a point.

The contribution can be stated cleanly: **our adversarial pipeline finds
a structured, learned-worst-case piecewise distribution that the Liu R4
hand-designed mixture misses, and which our pipeline's own retrained
control (R2-v2) closes — without breaking the standard FFL battery.**

### What's not yet shown (honest caveats for a paper)

- **Circuit-level mechanism**: We attribute the failure to "the dense →
  sparse transition" at the segment level. We have *not* shown which
  attention heads / layers are responsible. That's the natural follow-up
  for a mechanistic-interpretability section.
- **Structural neighborhood**: Step 3 perturbs (p_w, p_r, bit_p1) per
  segment. It does not vary segment boundaries (`start_frac`) or segment
  count (K). A more aggressive test should confirm the failure also
  survives moving the dense → sparse transition to different positions.
- **Comparison to a smarter hand-designed mixture**: We've shown R4 misses
  c00. We have not yet trained a *hand-designed* mixture that includes a
  dense → sparse transition (e.g., FFL(0.1)+FFL(0.98) per-position curriculum)
  to confirm that a person could not just write down the same fix without
  needing the cluster-family extractor. This is the natural ablation to
  claim "the automated method is doing something hand-design wouldn't."
- **Paper-ready LSTM skyline**: We have not re-confirmed that the 1-layer
  LSTM stays at 0% on the c00 neighborhood. The CLAUDE.md convention says
  that's the "skyline" reference; for a paper claim about Transformer
  inductive bias, we should explicitly show the LSTM closes the same
  region the Transformer fails on.

## 7. Reproducibility

| artifact | path |
|---|---|
| Step 1 outputs | [results/flip_flop/liu_r4/diagnostic_step1/](results/flip_flop/liu_r4/diagnostic_step1/) |
| Step 2A outputs | [results/flip_flop/liu_r4/diagnostic_step2a/](results/flip_flop/liu_r4/diagnostic_step2a/) |
| Step 2B outputs | [results/flip_flop/liu_r4/diagnostic_step2b/](results/flip_flop/liu_r4/diagnostic_step2b/) |
| Step 3 outputs | [results/flip_flop/liu_r4/diagnostic_step3/](results/flip_flop/liu_r4/diagnostic_step3/) |
| Compact diagnostic summary | [results/flip_flop/liu_r4/diagnostic_summary.md](results/flip_flop/liu_r4/diagnostic_summary.md) |
| R4 seed=0 model | results/flip_flop/liu_r4/model_final.pt |
| R4 seed=1 model | results/flip_flop/liu_r4_seed1/model_final.pt |
| R4 seed=2 model | results/flip_flop/liu_r4_seed2/model_final.pt |
| Step 1 script | [flip_flop/scripts/diagnose_piecewise_c00.py](flip_flop/scripts/diagnose_piecewise_c00.py) |
| Step 2A script | [flip_flop/scripts/eval_r4_seed_sweep.py](flip_flop/scripts/eval_r4_seed_sweep.py) |
| Step 2B eval script | [flip_flop/scripts/eval_r4_multi_seed_models.py](flip_flop/scripts/eval_r4_multi_seed_models.py) |
| Step 3 script | [flip_flop/scripts/diagnose_neighborhood_c00.py](flip_flop/scripts/diagnose_neighborhood_c00.py) |
| Sampler used to define families | results_tier1_v2/results/flip_flop/retrain/round_2_redone/sampler.json |

To reproduce end-to-end:

```bash
# Step 2B trainings (~60 min each, sequential)
python -u -m flip_flop.scripts.run_liu_r4 --seed 1 --out_dir results/flip_flop/liu_r4_seed1
python -u -m flip_flop.scripts.run_liu_r4 --seed 2 --out_dir results/flip_flop/liu_r4_seed2

# Step 1
python -u -m flip_flop.scripts.diagnose_piecewise_c00

# Step 2A
python -u -m flip_flop.scripts.eval_r4_seed_sweep

# Step 2B eval
python -u -m flip_flop.scripts.eval_r4_multi_seed_models

# Step 3
python -u -m flip_flop.scripts.diagnose_neighborhood_c00
```

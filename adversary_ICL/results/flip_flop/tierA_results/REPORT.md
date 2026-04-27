# Tier-A Adversarial Discovery Report — BitMarkov & WriteFlipRate

Date: 2026-04-27
Pod: e3613e0oauwgwm (NVIDIA H100 80GB HBM3, Ubuntu 22.04, Python 3.11.10)

## Bottom line

**2 new BREAKTHROUGH families found this round.** Both new search axes
(BitMarkov, WriteFlipRate) produced at least one distribution where the Liu R4
control catastrophically fails (5×+ retrain glitch, ≥0.5% absolute) and the
Tier-A retrained model closes it.

| Axis            | BREAKTHROUGH | WEAK | REDISCOVERY | NULL |
| --------------- | -----------: | ---: | ----------: | ---: |
| bitmarkov       |            1 |    0 |           0 |    1 |
| writeflip       |            1 |    0 |           1 |    0 |

## Per-axis results

Sample sizes / seeds (identical sequences across all three models per family):
N=10000 sequences per family; eval_seed=12345; batch_size=64.

### Axis A1 — BitMarkov

| family               | classification |   baseline |    R4_seed0 |       retrain |
| -------------------- | -------------- | ---------: | ----------: | ------------: |
| bit_markov_c00_a1.00 | **BREAKTHROUGH** |    0.1793% | **96.1152%** |       0.1793% |
| bit_markov_c01_a1.00 | NULL           |    0.0799% |     0.3496% |       0.0999% |

### Axis A2 — WriteFlipRate

| family               | classification |   baseline |    R4_seed0 |       retrain |
| -------------------- | -------------- | ---------: | ----------: | ------------: |
| write_flip_c00_a1.00 | **BREAKTHROUGH** |    2.7821% | **37.9367%** |       0.0274% |
| write_flip_c01_a1.00 | REDISCOVERY    |    3.6910% |     0.0000% |       0.0094% |

## Distribution configs of the BREAKTHROUGH families

Both families come back from the cluster representative extractor as
*Stationary* configs, even though the search ran over BitMarkov / WriteFlipRate
(see "Caveat" below). The representative configs are:

- **bit_markov_c00_a1.00 (BREAKTHROUGH, R4 = 96.1%)**
  - `dist`: `{name: stationary, T: 512, p_w: 0.9821, p_r: 1.66e-05, bit_p1: 0.00128}`
  - `alpha`: 1.0, `cluster_size`: 721, `cluster_mean_glitch`: 0.418
  - Interpretation: write-heavy, near-zero reads, nearly-all-zero data bits.
    R4's stationary mixture (FFL{0.1, 0.9, 0.98}, all bit_p1 = 0.5) is blind
    to extreme bit bias.

- **write_flip_c00_a1.00 (BREAKTHROUGH, R4 = 37.9%)**
  - `dist`: `{name: stationary, T: 512, p_w: 0.9807, p_r: 0.000372, bit_p1: 0.0326}`
  - `alpha`: 1.0, `cluster_size`: 2627, `cluster_mean_glitch`: 0.396
  - Interpretation: same write-heavy / near-zero-read regime as above, with a
    sharp 1-bit bias (~3% ones). R4 fails substantially on this neighborhood
    too, less catastrophically than on bit_markov_c00.

Search budget: 3000 fitness evaluations × 2 axes × pop_size 16 × 3 restarts.
Adversary best fitness reached: 0.962 (bitmarkov), 0.523 (writeflip).
Per-eval n=2000 sequences from candidate distribution; lambda_lstm=10,
lstm_tolerance=0.001 (LSTM stays clean → no penalty).

## Tier B — length extrapolation

**SKIPPED. Mechanically blocked.**

Reason: `flip_flop/model.py::FFLMTransformer` instantiates
`GPT2LMHeadModel(GPT2Config(... n_positions=512 ...))`. HuggingFace GPT-2 uses
absolute learned positional embeddings of size n_positions, so inference at
T > 512 raises `IndexError` on `wpe`. The task spec explicitly forbids
modifying `model.py` to enable extrapolation, so no length-extrap evaluation
was run; no `length_extrap_eval.py` was created.

## Caveat — family extractor reduces axis-specific params to Stationary

Worth flagging for the writeup: even though the adversary searched 5-dim
spaces over `BitMarkov(p_w, p_r, bit_p1, bit_stay)` and
`WriteFlipRate(p_w, p_r, bit_p1, flip_rate)`, the existing
`flip_flop/adversary/family.py::_cluster_representative_config` flattens
those configs to `(p_w, p_r, bit_p1)` and emits a `Stationary` config as the
cluster representative (see `_flatten_config_params` and `_cluster_representative_config`
for `name in ("stationary", "bit_markov", "write_flip")`).

So the families the retrain actually trains on are **Stationary**, with the
`bit_stay` / `flip_rate` axis dropped. The breakthroughs found are therefore
properly attributed to the (p_w, p_r, bit_p1) sub-space being outside R4's
mixture, not to the new bit-Markov / write-flip dynamics per se. To get
credit for the actual axis dynamics, `_cluster_representative_config` would
need to retain the per-axis param and emit a `BitMarkov` / `WriteFlipRate`
rep — out of scope for this run (the spec forbids modifying existing files).

## Reproducibility

- Adversary searches: `flip_flop/scripts/run_adversary_{bitmarkov,writeflip}.py`,
  configs `flip_flop/configs/adversary_{bitmarkov,writeflip}.yaml`, seed=0.
- Retrains: `flip_flop/scripts/run_retrain_tierA.py` parameterized by
  `flip_flop/configs/retrain_tierA_{bitmarkov,writeflip}.yaml`, seed=0,
  init_from_ckpt=results/flip_flop/baseline/model_final.pt, replay_frac=0.7,
  lr=3e-5, train_steps=2000 (both halted early by hard cap at step 500).
- Decisive eval: `flip_flop/scripts/eval_tierA_vs_r4.py`, N=10000,
  eval_seed=12345, batch_size=64.
- All artifacts persisted under `results/flip_flop/{adversary,retrain,tierA_results}/`.

## Phase-7 cleanup actions

Both axes had ≥1 BREAKTHROUGH family → both axes are SUCCESSFUL → per the
spec "For successful axes: keep everything." **No checkpoints were deleted.**

## Phase-8 pod-teardown status

**Could not tear down programmatically.** runpodctl is not installed locally
(Windows). It is installed on the pod (`/usr/bin/runpodctl`) but the pod
image lacks a configured API key, so `runpodctl stop pod e3613e0oauwgwm`
fails with `API key not found`.

> Action required: please stop / terminate pod **e3613e0oauwgwm** via the
> RunPod web dashboard NOW to stop billing.

## Number of new breakthroughs found this round

**2** (`bit_markov_c00_a1.00`, `write_flip_c00_a1.00`).

"""Sanity check: is the 100% glitch finding real?

Loads the top-1 piecewise config, samples sequences, checks:
  1. enforce_read_determinism produces correct target at the R position
  2. The last write before R has bit=1 (our claimed "correct answer")
  3. The forced R's data bit equals that last-write bit
  4. The model's argmax prediction is wrong (and we print what it predicts)
  5. Manual re-compute of error rate from raw logits matches evaluate_dataset
"""
import json
import numpy as np
import torch

from flip_flop.adversary.distribution import FFLDistribution
from flip_flop.adversary.io import load_frozen_model
from flip_flop.data import W, R, I, ZERO, ONE, decode
from flip_flop.eval import evaluate_dataset

LOG_PATH = "results_h100/results/flip_flop/adversary/piecewise/adversary_log.jsonl"
T_CKPT = "results/flip_flop/baseline/model_final.pt"
T_CFG = "flip_flop/configs/baseline.yaml"
L_CKPT = "results/flip_flop/lstm/model_final.pt"
L_CFG = "flip_flop/configs/lstm.yaml"

# Take the highest-fitness valid record from the raw adversary log.
with open(LOG_PATH) as f:
    recs = [json.loads(l) for l in f]
valid = [r for r in recs if r.get("is_valid", True)]
valid.sort(key=lambda r: r["fitness"], reverse=True)
top = valid[0]
print("=== TOP-1 CONFIG ===")
print(json.dumps(top["config"], indent=2))
print(f"  search fitness = {top.get('fitness', 'NA')}")
print()

# Build the distribution
dist = FFLDistribution.from_dict(top["config"])

# Sample some sequences for inspection
rng = np.random.default_rng(0)
tokens = dist.sample(8, rng)
print("=== 8 SAMPLE SEQUENCES (first 30 tokens + last 30 tokens) ===")
for i, row in enumerate(tokens.tolist()):
    prefix = row[:30]
    suffix = row[-30:]
    # Pretty-print
    sym = {W: "w", R: "r", I: "i", ZERO: "0", ONE: "1"}
    p_str = " ".join(sym[t] for t in prefix)
    s_str = " ".join(sym[t] for t in suffix)
    print(f"  seq {i}: {p_str} ... {s_str}")
print()

# Verify structural properties per-sequence
T = tokens.shape[1]
print("=== STRUCTURAL CHECKS PER SEQUENCE ===")
for i, row in enumerate(tokens.numpy()):
    inst = row[0::2]
    data = row[1::2]
    # Find last write position in instruction-space
    w_positions = np.where(inst == W)[0]
    r_positions = np.where(inst == R)[0]
    last_w = w_positions[-1] if len(w_positions) else None
    last_w_bit = data[last_w] - ZERO if last_w is not None else None
    # Find R positions and the target bit at each
    # In instruction-space, R at position k means the read token is at 2k in the
    # original sequence, and its "data bit" is at position 2k+1.
    print(f"  seq {i}: num_writes={len(w_positions)} num_reads={len(r_positions)} "
          f"last_w_pos={last_w} last_w_bit={last_w_bit} "
          f"target_at_final_R={data[r_positions[-1]] - ZERO if len(r_positions) else 'NA'}")
print()

# Distribution of instructions per segment
print("=== WRITE BIT IN SEG 2 (should be 1 for 64/64) ===")
n_inst = T // 2
for i, row in enumerate(tokens.numpy()[:3]):
    inst = row[0::2]
    data = row[1::2]
    seg2_start = int(0.5 * n_inst)  # 128
    seg2_end = int(0.75 * n_inst)  # 192
    seg2_w = np.sum(inst[seg2_start:seg2_end] == W)
    seg2_w_bits = data[seg2_start:seg2_end][inst[seg2_start:seg2_end] == W] - ZERO
    print(f"  seq {i}: seg2 writes = {seg2_w}/64, bits = {list(seg2_w_bits.tolist()[:10])}...")

# Now actually run the trained model on a bigger batch and verify
device = "cuda" if torch.cuda.is_available() else "cpu"
t_model = load_frozen_model(T_CKPT, T_CFG, device)
l_model = load_frozen_model(L_CKPT, L_CFG, device)

big_tokens = dist.sample(256, rng)
print()
print("=== RUNNING MODELS ON 256 SEQUENCES FROM TOP-1 CONFIG ===")
t_res = evaluate_dataset(t_model, big_tokens, batch_size=64, device=device)
l_res = evaluate_dataset(l_model, big_tokens, batch_size=64, device=device)
print(f"  Transformer: error_rate = {t_res['error_rate']:.4f}  "
      f"({t_res['num_errors']}/{t_res['num_predictions']})")
print(f"  LSTM:         error_rate = {l_res['error_rate']:.4f}  "
      f"({l_res['num_errors']}/{l_res['num_predictions']})")

# Manual re-compute of Transformer predictions + logit inspection
print()
print("=== MANUAL: what does the Transformer predict at R positions? ===")
with torch.no_grad():
    logits = t_model(big_tokens.to(device))
shift_logits = logits[:, :-1, :]
shift_targets = big_tokens[:, 1:].to(device)
mask = big_tokens[:, :-1].to(device) == R
preds = shift_logits.argmax(dim=-1)
# Where did the errors happen?
errors = (preds != shift_targets) & mask
print(f"  total R positions: {mask.sum().item()}")
print(f"  total errors:       {errors.sum().item()}")

# Predicted-token distribution at the scored R positions
pred_at_r = preds[mask].cpu().numpy()
target_at_r = shift_targets[mask].cpu().numpy()
sym = {W: "w", R: "r", I: "i", ZERO: "0", ONE: "1"}
from collections import Counter
print(f"  distribution of model predictions at R: {Counter([sym[x] for x in pred_at_r])}")
print(f"  distribution of correct targets at R:    {Counter([sym[x] for x in target_at_r])}")

# Logit probe: at the first R position, what are the raw logits?
print()
print("=== LOGIT PROBE at the first R position of sequence 0 ===")
inst0 = big_tokens[0].numpy()[0::2]
r_positions = np.where(inst0 == R)[0]
r_pos_tok = 2 * r_positions[-1]  # token index of last R
print(f"  seq 0: R at token position {r_pos_tok}, target token = {sym[int(shift_targets[0, r_pos_tok].item())]}")
print(f"  logits at that position (shape {logits[0, r_pos_tok].shape}):")
l = logits[0, r_pos_tok].cpu().numpy()
for k in range(5):
    print(f"    vocab[{k}={sym[k]}]  logit = {l[k]:+.4f}")
print(f"  argmax = {sym[int(l.argmax())]}")

print()
print("=== CONCLUSION ===")
if t_res['error_rate'] > 0.9 and l_res['error_rate'] < 0.01 and (target_at_r == ONE).all():
    print(" VERIFIED: Targets are all bit-1, LSTM solves, Transformer fails."
          " 100% glitch finding is real.")
elif (target_at_r == ONE).all():
    print(f" Targets OK (all 1). Transformer err = {t_res['error_rate']:.3f}, LSTM err = {l_res['error_rate']:.3f}")
else:
    print(f" WARN: targets at R are NOT all 1. "
          f"target distribution = {Counter([sym[x] for x in target_at_r])}")

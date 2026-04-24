"""Does the a=1 adversarial tip still break the round-1 retrained model?

Loads the top-25 original adversary configs (a=1), samples fresh batches, and
scores both the baseline and the retrained Transformer + LSTM on each.
"""
import json
import numpy as np
import torch
from flip_flop.adversary.distribution import FFLDistribution
from flip_flop.adversary.io import load_frozen_model
from flip_flop.eval import evaluate_dataset

import os
RETRAINED = (
    "results/flip_flop/retrain/round_1/model_final.pt"
    if os.path.exists("results/flip_flop/retrain/round_1/model_final.pt")
    else "results/flip_flop/retrain/results/flip_flop/retrain/round_1/model_final.pt"
)
BASELINE  = "results/flip_flop/baseline/model_final.pt"
LSTM      = "results/flip_flop/lstm/model_final.pt"
LOG       = (
    "results/flip_flop/adversary/piecewise/adversary_log.jsonl"
    if os.path.exists("results/flip_flop/adversary/piecewise/adversary_log.jsonl")
    else "results_h100_final/results/flip_flop/adversary/piecewise/adversary_log.jsonl"
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

base_t  = load_frozen_model(BASELINE,  "flip_flop/configs/baseline.yaml", device)
retr_t  = load_frozen_model(RETRAINED, "flip_flop/configs/baseline.yaml", device)
lstm    = load_frozen_model(LSTM,      "flip_flop/configs/lstm.yaml",    device)

# Load top-25 adversarial configs
with open(LOG) as f:
    recs = [json.loads(l) for l in f]
valid = [r for r in recs if r.get("is_valid", True) and r.get("T_glitch", 0) > 0.5 and r.get("lstm_glitch", 1) < 0.01]
valid.sort(key=lambda r: r["fitness"], reverse=True)
top = valid[:25]
print(f"Loaded top-{len(top)} adversarial configs (a=1 tip)")

rng = np.random.default_rng(0)
print()
print(f"{'rank':>4}  {'baseline':>10}  {'retrained':>10}  {'lstm':>7}")
print(f"{'':>4}  {'T_glitch':>10}  {'T_glitch':>10}  {'glitch':>7}")
for i, r in enumerate(top):
    dist = FFLDistribution.from_dict(r["config"])
    tokens = dist.sample(2000, rng)
    base_err = evaluate_dataset(base_t, tokens, batch_size=64, device=device)["error_rate"]
    retr_err = evaluate_dataset(retr_t, tokens, batch_size=64, device=device)["error_rate"]
    lstm_err = evaluate_dataset(lstm,   tokens, batch_size=64, device=device)["error_rate"]
    print(f"{i+1:>4}  {base_err:>10.4f}  {retr_err:>10.4f}  {lstm_err:>7.4f}")

print()
# Summary
b_errs, r_errs, l_errs = [], [], []
rng = np.random.default_rng(1)
for r in top:
    dist = FFLDistribution.from_dict(r["config"])
    tokens = dist.sample(5000, rng)
    b_errs.append(evaluate_dataset(base_t, tokens, batch_size=64, device=device)["error_rate"])
    r_errs.append(evaluate_dataset(retr_t, tokens, batch_size=64, device=device)["error_rate"])
    l_errs.append(evaluate_dataset(lstm,   tokens, batch_size=64, device=device)["error_rate"])
print(f"MEAN over 25 tips @ 5000 samples each:")
print(f"  baseline T_glitch:  {np.mean(b_errs):.4f}")
print(f"  retrained T_glitch: {np.mean(r_errs):.4f}")
print(f"  lstm glitch:        {np.mean(l_errs):.4f}")

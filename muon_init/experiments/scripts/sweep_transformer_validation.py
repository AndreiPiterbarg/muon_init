"""Transformer validation: does init matter for post-norm under Muon?

Post-norm transformers are genuinely init-sensitive (T-Fixup, DeepNorm, Admin
all exist because post-LN can diverge). If init matters anywhere under Muon,
it's here.

Key question: does alpha=0.14 (EoS-derived for this architecture) prevent
divergence at warmup=0?

Usage:
    python -m experiments.scripts.sweep_transformer_validation --config experiments/configs/deep_narrow_wikitext.yaml --dry_run
    python -m experiments.scripts.sweep_transformer_validation --config experiments/configs/deep_narrow_wikitext.yaml
"""

import argparse
import itertools
import os
import subprocess
import sys


INITS = [
    "kaiming_normal",       # baseline
    "orthogonal",           # alpha=1.0
    "scaled_orth_0.5",      # MLP experiment winner
    "sharpness_eos_vit",    # alpha=0.14, EoS-derived for transformer
]

WARMUP_STEPS = [0, 200, 500, 1000, 2000]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--save_dir", type=str,
                        default="experiments/results/transformer_validation")
    args = parser.parse_args()

    seeds = list(range(42, 42 + args.seeds))
    runs = list(itertools.product(INITS, WARMUP_STEPS, seeds))
    print(f"Transformer validation: {len(runs)} runs "
          f"({len(INITS)} inits x {len(WARMUP_STEPS)} warmups x {args.seeds} seeds)")

    os.makedirs(args.save_dir, exist_ok=True)
    skipped = 0

    for init, warmup, seed in runs:
        result_name = f"deep_narrow_gpt_{init}_warmup{warmup}_seed{seed}.json"
        if os.path.exists(os.path.join(args.save_dir, result_name)):
            skipped += 1
            continue

        cmd = [
            sys.executable, "-m", "experiments.train",
            "--config", args.config,
            "--init", init,
            "--warmup_steps", str(warmup),
            "--seed", str(seed),
            "--save_dir", args.save_dir,
        ]
        print(f"\n{'='*60}")
        print(f"RUN: init={init} warmup={warmup} seed={seed}")
        print(f"{'='*60}")

        if args.dry_run:
            print(f"  [dry run] {' '.join(cmd)}")
            continue

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"  WARNING: run failed with code {result.returncode}")

    if skipped:
        print(f"\nSkipped {skipped} already-completed runs.")
    print(f"\nDone. Results in {args.save_dir}/")


if __name__ == "__main__":
    main()

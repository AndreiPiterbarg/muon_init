"""Momentum decomposition experiment: 2x2 factorial.

Tests whether Muon's warmup need comes from LR warmup or momentum warmup.
Runs a 2x2 grid of {LR warmup on/off} x {momentum warmup on/off},
across 2 inits and 3 seeds.

This is the gatekeeper diagnostic: if momentum warmup alone is sufficient,
the entire init research direction pivots to optimizer schedule design.

Usage:
    python -m experiments.scripts.sweep_momentum_decomposition --config experiments/configs/mlp_cifar10.yaml
    python -m experiments.scripts.sweep_momentum_decomposition --config experiments/configs/mlp_cifar10.yaml --dry_run
"""

import argparse
import itertools
import os
import subprocess
import sys


# 2x2 factorial: (lr_warmup_steps, momentum_warmup_steps)
CONDITIONS = [
    (0, 0),      # No warmup at all
    (500, 0),    # LR warmup only
    (0, 300),    # Momentum warmup only
    (500, 300),  # Both warmups
]

INITS = ["kaiming_normal", "scaled_orth_0.5"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--save_dir", type=str, default="experiments/results/momentum_decomposition")
    args = parser.parse_args()

    seeds = list(range(42, 42 + args.seeds))
    runs = list(itertools.product(INITS, CONDITIONS, seeds))
    print(f"Momentum decomposition: {len(runs)} runs "
          f"({len(INITS)} inits x {len(CONDITIONS)} conditions x {args.seeds} seeds)")

    os.makedirs(args.save_dir, exist_ok=True)
    skipped = 0

    for init, (lr_warmup, mom_warmup), seed in runs:
        mom_suffix = f"_momwarmup{mom_warmup}" if mom_warmup > 0 else ""
        result_name = f"deep_mlp_{init}_warmup{lr_warmup}{mom_suffix}_seed{seed}.json"
        if os.path.exists(os.path.join(args.save_dir, result_name)):
            skipped += 1
            continue

        cmd = [
            sys.executable, "-m", "experiments.train",
            "--config", args.config,
            "--init", init,
            "--warmup_steps", str(lr_warmup),
            "--momentum_warmup_steps", str(mom_warmup),
            "--seed", str(seed),
            "--save_dir", args.save_dir,
        ]
        print(f"\n{'='*60}")
        print(f"RUN: init={init} lr_warmup={lr_warmup} mom_warmup={mom_warmup} seed={seed}")
        print(f"{'='*60}")

        if args.dry_run:
            print(f"  [dry run] {' '.join(cmd)}")
            continue

        env = os.environ.copy()
        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            print(f"  WARNING: run failed with code {result.returncode}")

    if skipped:
        print(f"\nSkipped {skipped} already-completed runs.")
    print(f"\nDone. Results in {args.save_dir}/")


if __name__ == "__main__":
    main()

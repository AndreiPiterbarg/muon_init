"""Warmup sensitivity sweep: the core experiment.

For each model-dataset pairing, runs all init schemes across a grid of warmup steps.
This produces the primary result: warmup sensitivity curves per init per architecture.

Usage:
    python -m experiments.scripts.sweep_warmup --config experiments/configs/mlp_cifar10.yaml
    python -m experiments.scripts.sweep_warmup --config experiments/configs/mlp_cifar10.yaml --seeds 3
"""

import argparse
import itertools
import os
import subprocess
import sys

INITS = [
    "kaiming_normal", "kaiming_uniform", "xavier_normal", "xavier_uniform",
    "orthogonal", "scaled_orthogonal",
]
WARMUP_STEPS = [0, 100, 200, 500, 1000, 2000]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--inits", nargs="+", default=INITS)
    parser.add_argument("--warmups", nargs="+", type=int, default=WARMUP_STEPS)
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    seeds = list(range(42, 42 + args.seeds))
    runs = list(itertools.product(args.inits, args.warmups, seeds))
    print(f"Sweep: {len(runs)} runs ({len(args.inits)} inits x "
          f"{len(args.warmups)} warmups x {args.seeds} seeds)")

    save_dir = "experiments/results"
    skipped = 0
    for init, warmup, seed in runs:
        # Skip runs that already have results.
        result_name = f"deep_mlp_{init}_warmup{warmup}_seed{seed}.json"
        if os.path.exists(os.path.join(save_dir, result_name)):
            skipped += 1
            continue

        cmd = [
            sys.executable, "-m", "experiments.train",
            "--config", args.config,
            "--init", init,
            "--warmup_steps", str(warmup),
            "--seed", str(seed),
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


if __name__ == "__main__":
    main()

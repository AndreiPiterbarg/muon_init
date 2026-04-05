"""Sweep scaled orthogonal initialization across alpha values and warmup steps.

Matches the baseline sweep (sweep_warmup.py) grid so results are directly
comparable. Includes warmup values from both the baseline sweep and the
task description to ensure full coverage.

Usage:
    python -m experiments.scripts.sweep_scaled_orthogonal --config experiments/configs/mlp_cifar10.yaml
    python -m experiments.scripts.sweep_scaled_orthogonal --config experiments/configs/mlp_cifar10.yaml --dry_run
"""

import argparse
import itertools
import os
import subprocess
import sys

# All alpha variants registered in train.py INIT_REGISTRY.
SCALED_ORTH_INITS = [
    "scaled_orth_0.5",
    "scaled_orth_0.75",
    "scaled_orth_1.0",
    "scaled_orth_sqrt2",
    "scaled_orth_1.5",
    "scaled_orth_2.0",
    "scaled_orth_2.5",
]

# Union of baseline sweep [0, 100, 500, 1000, 2000] and task spec [0, 100, 200, 500, 1000].
WARMUP_STEPS = [0, 100, 200, 500, 1000, 2000]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--inits", nargs="+", default=SCALED_ORTH_INITS)
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
        model_name = args.config.split("/")[-1].replace(".yaml", "")
        # Derive model name from config the same way train.py does.
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

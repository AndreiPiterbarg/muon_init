"""Launch the canonical baseline FFLM training run.

Usage (from repo root):
    python -m flip_flop.scripts.run_baseline
    python -m flip_flop.scripts.run_baseline --config flip_flop/configs/lstm.yaml
    python -m flip_flop.scripts.run_baseline --test_run
"""
from __future__ import annotations

import argparse
import os
import sys

# Make the repo root importable when invoked as a script.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from flip_flop.train import TrainConfig, train


def _default_config() -> str:
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs",
        "baseline.yaml",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=_default_config())
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--test_run", action="store_true")
    args = parser.parse_args()

    cfg = TrainConfig.from_yaml(args.config)
    if args.out_dir is not None:
        cfg.out_dir = args.out_dir
    if args.seed is not None:
        cfg.seed = args.seed
    if args.test_run:
        cfg.train_steps = 100
        cfg.decay_end_step = 101
        cfg.warmup_steps = 10
        cfg.eval_in_n = 64
        cfg.eval_sparse_n = 256
        cfg.eval_dense_n = 64
        cfg.eval_every = 50
        cfg.save_every = 0

    result = train(cfg)
    print("[done]", result)


if __name__ == "__main__":
    main()

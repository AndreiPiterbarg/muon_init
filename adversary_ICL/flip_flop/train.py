"""Training loop for FFLM.

Matches Section 4 / Appendix B.2 of Liu et al. 2023 for the baseline sweep:
  * AdamW, (beta1, beta2) = (0.9, 0.999), weight_decay = 0.1
  * learning rate 3e-4, 50-step linear warmup, linear decay to 0 at step 10001
  * batch size 16, 10000 steps (~81.9M training tokens at T = 512)
  * Train on FFL(0.8); evaluate in-distribution FFL(0.8), sparse FFL(0.98),
    dense FFL(0.1).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import yaml

from .data import make_eval_dataset, sample_ffl
from .eval import clean_loss, evaluate_dataset
from .model import build_model


@dataclass
class TrainConfig:
    # model
    family: str = "gpt2"
    vocab_size: int = 5
    n_positions: int = 512
    n_embd: int = 512
    n_layer: int = 6
    n_head: int = 8
    resid_pdrop: float = 0.0
    embd_pdrop: float = 0.0
    attn_pdrop: float = 0.0
    # lstm-only
    hidden_size: int = 128
    num_layers: int = 1
    # data
    seq_len: int = 512
    train_p_i: float = 0.8
    eval_in_p_i: float = 0.8
    eval_sparse_p_i: float = 0.98
    eval_dense_p_i: float = 0.1
    eval_in_n: int = 1000
    eval_sparse_n: int = 100_000
    eval_dense_n: int = 3000
    eval_seed: int = 1
    # optimization
    lr: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.999
    warmup_steps: int = 50
    train_steps: int = 10_000
    decay_end_step: int = 10_001  # LR reaches 0 here
    batch_size: int = 16
    # logging / i/o
    seed: int = 0
    eval_every: int = 500
    training_eval_subset: int = 1000  # per paper: ~1% subset during training
    log_every: int = 50
    save_every: int = 2000
    eval_batch_size: int = 64
    out_dir: str = "results/flip_flop/baseline"
    device: str = "auto"

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TrainConfig":
        flat: dict[str, Any] = {}
        for section in ("model", "data", "training", "output"):
            if section in raw and isinstance(raw[section], dict):
                flat.update(raw[section])
        # Allow top-level overrides too.
        for k, v in raw.items():
            if not isinstance(v, dict):
                flat[k] = v
        # Filter to known fields.
        known = {f for f in cls.__dataclass_fields__}
        flat = {k: v for k, v in flat.items() if k in known}
        return cls(**flat)


def _resolve_device(want: str) -> str:
    if want == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return want


def _make_lr_lambda(warmup: int, decay_end: int):
    """Linear warmup then linear decay; step `decay_end` maps to 0."""
    def lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step) / float(max(1, warmup))
        remaining = decay_end - step
        total = decay_end - warmup
        return max(0.0, remaining / max(1, total))
    return lr_lambda


def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(cfg: TrainConfig) -> dict:
    device = _resolve_device(cfg.device)
    os.makedirs(cfg.out_dir, exist_ok=True)
    _seed_all(cfg.seed)

    # Persist the resolved config for reproducibility.
    with open(os.path.join(cfg.out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg.__dict__, f, sort_keys=False)

    # Model
    model = build_model(cfg).to(device)
    n_params = model.num_parameters()
    print(f"[flip_flop] model={model.name} params={n_params:,}")

    # Optimizer + LR schedule (AdamW with uniform weight decay, per paper).
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _make_lr_lambda(cfg.warmup_steps, cfg.decay_end_step)
    )

    # Eval sets (fixed across training)
    print("[flip_flop] building eval datasets...")
    eval_sets = {
        "in_distribution": make_eval_dataset(
            cfg.eval_in_p_i, cfg.eval_in_n, cfg.seq_len, cfg.eval_seed
        ),
        "sparse_tail": make_eval_dataset(
            cfg.eval_sparse_p_i, cfg.eval_sparse_n, cfg.seq_len, cfg.eval_seed + 1
        ),
        "dense_tail": make_eval_dataset(
            cfg.eval_dense_p_i, cfg.eval_dense_n, cfg.seq_len, cfg.eval_seed + 2
        ),
    }

    # Separate RNG for training so eval seeds don't collide
    train_rng = np.random.default_rng(cfg.seed + 100)

    log_path = os.path.join(cfg.out_dir, "train_log.jsonl")
    eval_path = os.path.join(cfg.out_dir, "eval_log.jsonl")
    log_fh = open(log_path, "a")
    eval_fh = open(eval_path, "a")

    model.train()
    t0 = time.time()
    last_loss = float("nan")

    try:
        for step in range(cfg.train_steps):
            tokens = sample_ffl(
                cfg.seq_len, cfg.train_p_i, cfg.batch_size, train_rng
            ).to(device)
            logits = model(tokens)
            loss = clean_loss(logits, tokens)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            last_loss = float(loss.item())
            if step % cfg.log_every == 0:
                rec = {
                    "step": step,
                    "loss": last_loss,
                    "lr": float(scheduler.get_last_lr()[0]),
                    "elapsed": time.time() - t0,
                }
                log_fh.write(json.dumps(rec) + "\n")
                log_fh.flush()
                print(
                    f"[step {step:>6d}] loss={last_loss:.4f} "
                    f"lr={rec['lr']:.2e} t={rec['elapsed']:.1f}s"
                )

            if cfg.eval_every > 0 and (step + 1) % cfg.eval_every == 0:
                _run_eval(model, eval_sets, cfg, device, step + 1, eval_fh,
                          subset=cfg.training_eval_subset)
                model.train()

            if cfg.save_every > 0 and (step + 1) % cfg.save_every == 0:
                ckpt = {
                    "step": step + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                }
                torch.save(ckpt, os.path.join(cfg.out_dir, "state.pt"))

        # Final eval + checkpoint
        final = _run_eval(model, eval_sets, cfg, device, cfg.train_steps, eval_fh)
        torch.save(
            {
                "step": cfg.train_steps,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            },
            os.path.join(cfg.out_dir, "model_final.pt"),
        )
    finally:
        log_fh.close()
        eval_fh.close()

    return {"final_eval": final, "last_loss": last_loss, "num_params": n_params}


def _run_eval(model, eval_sets, cfg, device, step, fh, subset=None) -> dict:
    out = {"step": step, "subset": subset}
    for name, ds in eval_sets.items():
        view = ds if subset is None else ds[: min(subset, len(ds))]
        res = evaluate_dataset(model, view, batch_size=cfg.eval_batch_size, device=device)
        out[name] = res.to_dict()
        print(
            f"  eval@{step} {name}: err={res.error_rate:.3e} "
            f"({res.num_errors}/{res.num_predictions}) loss={res.loss:.4f}"
        )
    fh.write(json.dumps(out) + "\n")
    fh.flush()
    return out


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_dir", default=None, help="override out_dir in config")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--test_run", action="store_true",
                        help="short smoke-test (100 steps, small eval sets)")
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

    train(cfg)


if __name__ == "__main__":
    main()

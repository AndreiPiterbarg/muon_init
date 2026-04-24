"""Training loop for FFLM.

Matches Section 4 / Appendix B.2 of Liu et al. 2023 for the baseline sweep:
  * AdamW, (beta1, beta2) = (0.9, 0.999), weight_decay = 0.1
  * learning rate 3e-4, 50-step linear warmup, linear decay to 0 at step 10001
  * batch size 16, 10000 steps (~81.9M training tokens at T = 512)
  * Train on FFL(0.8); evaluate FFL(0.8) / FFL(0.98) / FFL(0.1).

Entry point: flip_flop/scripts/run_baseline.py.
"""
import json
import os
import time
from dataclasses import dataclass

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
    grad_clip: float = 1.0  # matches HF / x-transformers default; prevents divergence spikes
    # logging / i/o
    seed: int = 0
    eval_every: int = 500
    training_eval_subset: int = 1000  # paper: "first 1% of (ii)" during training
    log_every: int = 50
    save_every: int = 2000
    eval_batch_size: int = 64
    out_dir: str = "results/flip_flop/baseline"
    device: str = "auto"
    # optional: continue training from an existing checkpoint (used by retrain loop)
    init_from_ckpt: str = ""

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            raw = yaml.safe_load(f)
        flat = {}
        for section in raw.values():
            if isinstance(section, dict):
                flat.update(section)
        known = {f.name for f in cls.__dataclass_fields__.values()}
        flat = {k: v for k, v in flat.items() if k in known}
        return cls(**flat)


def _make_lr_lambda(warmup, decay_end):
    """Linear warmup then linear decay; step `decay_end` maps to 0."""
    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        return max(0.0, (decay_end - step) / max(1, decay_end - warmup))
    return lr_lambda


def _run_eval(model, eval_sets, cfg, device, step, fh, subset=None):
    out = {"step": step, "subset": subset}
    for name, ds in eval_sets.items():
        view = ds if subset is None else ds[: min(subset, len(ds))]
        res = evaluate_dataset(model, view, batch_size=cfg.eval_batch_size, device=device)
        out[name] = res
        print(
            f"  eval@{step} {name}: err={res['error_rate']:.3e} "
            f"({res['num_errors']}/{res['num_predictions']}) loss={res['loss']:.4f}"
        )
    fh.write(json.dumps(out) + "\n")
    fh.flush()
    return out


def train(cfg, sampler=None):
    """Train the model. If `sampler` is provided (a callable `(batch_size, rng) ->
    LongTensor`), it replaces the default FFL(cfg.train_p_i) sampler — used by
    the retrain loop to inject MixedSampler. If `cfg.init_from_ckpt` is set,
    the model weights are loaded from that path before training (continue-train).
    """
    device = ("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device
    os.makedirs(cfg.out_dir, exist_ok=True)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    with open(os.path.join(cfg.out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg.__dict__, f, sort_keys=False)

    model = build_model(cfg).to(device)
    if cfg.init_from_ckpt:
        sd = torch.load(cfg.init_from_ckpt, map_location=device)
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        model.load_state_dict(sd)
        print(f"[flip_flop] loaded weights from {cfg.init_from_ckpt}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[flip_flop] model={model.name} params={n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _make_lr_lambda(cfg.warmup_steps, cfg.decay_end_step)
    )

    print("[flip_flop] building eval datasets...")
    eval_sets = {
        "in_distribution": make_eval_dataset(cfg.eval_in_p_i, cfg.eval_in_n, cfg.seq_len, cfg.eval_seed),
        "sparse_tail": make_eval_dataset(cfg.eval_sparse_p_i, cfg.eval_sparse_n, cfg.seq_len, cfg.eval_seed + 1),
        "dense_tail": make_eval_dataset(cfg.eval_dense_p_i, cfg.eval_dense_n, cfg.seq_len, cfg.eval_seed + 2),
    }
    train_rng = np.random.default_rng(cfg.seed + 100)

    log_fh = open(os.path.join(cfg.out_dir, "train_log.jsonl"), "a")
    eval_fh = open(os.path.join(cfg.out_dir, "eval_log.jsonl"), "a")

    model.train()
    t0 = time.time()
    last_loss = float("nan")

    if sampler is None:
        def sampler(bs, rng):
            return sample_ffl(cfg.seq_len, cfg.train_p_i, bs, rng)

    try:
        for step in range(cfg.train_steps):
            tokens = sampler(cfg.batch_size, train_rng).to(device)
            loss = clean_loss(model(tokens), tokens)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            last_loss = float(loss.item())
            if step % cfg.log_every == 0:
                rec = {"step": step, "loss": last_loss,
                       "lr": float(scheduler.get_last_lr()[0]),
                       "elapsed": time.time() - t0}
                log_fh.write(json.dumps(rec) + "\n")
                log_fh.flush()
                print(f"[step {step:>6d}] loss={last_loss:.4f} lr={rec['lr']:.2e} t={rec['elapsed']:.1f}s")

            if cfg.eval_every > 0 and (step + 1) % cfg.eval_every == 0:
                _run_eval(model, eval_sets, cfg, device, step + 1, eval_fh,
                          subset=cfg.training_eval_subset)
                model.train()

            if cfg.save_every > 0 and (step + 1) % cfg.save_every == 0:
                torch.save(
                    {"step": step + 1, "model_state_dict": model.state_dict(),
                     "optimizer_state_dict": optimizer.state_dict(),
                     "scheduler_state_dict": scheduler.state_dict()},
                    os.path.join(cfg.out_dir, "state.pt"),
                )

        final = _run_eval(model, eval_sets, cfg, device, cfg.train_steps, eval_fh)
        torch.save({"step": cfg.train_steps, "model_state_dict": model.state_dict()},
                   os.path.join(cfg.out_dir, "model_final.pt"))
    finally:
        log_fh.close()
        eval_fh.close()

    return {"final_eval": final, "last_loss": last_loss, "num_params": n_params}

"""Training harness for Muon initialization experiments.

Supports all 5 model-dataset pairings with a single entry point.
Handles optimizer construction, warmup scheduling, logging, and evaluation.

Usage:
    python -m experiments.train --config experiments/configs/mlp_cifar10.yaml
    python -m experiments.train --config experiments/configs/mlp_cifar10.yaml --init orthogonal --warmup_steps 0
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
import yaml

from models import build_model
from data import build_dataloaders
from optimizers.muon import MuonAdamW
from initializations.baselines.baselines import (
    kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform, orthogonal,
)


INIT_REGISTRY = {
    "kaiming_normal": kaiming_normal,
    "kaiming_uniform": kaiming_uniform,
    "xavier_normal": xavier_normal,
    "xavier_uniform": xavier_uniform,
    "orthogonal": orthogonal,
    "default": lambda model: None,  # PyTorch defaults
}


def get_lr(step, warmup_steps, max_lr, total_steps, min_lr=0.0):
    """Linear warmup + cosine decay."""
    max_lr, min_lr = float(max_lr), float(min_lr)
    if step < warmup_steps:
        return max_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def split_params(model, model_name):
    """Split parameters into Muon-eligible (2D weights) and AdamW (rest).

    Muon handles: Linear weights, Conv weights (2D+).
    AdamW handles: biases, embeddings, LayerNorm, cls_token, pos_emb.
    """
    muon_params = []
    adam_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_embedding = "emb" in name or "tok_emb" in name or "pos_emb" in name
        is_head = name.endswith("head.weight") and model_name in ("nanogpt", "deep_narrow_gpt")
        is_norm = "norm" in name or "bn" in name or "ln" in name
        is_bias = name.endswith(".bias")
        is_cls = "cls_token" in name

        if is_embedding or is_head or is_norm or is_bias or is_cls:
            adam_params.append(param)
        elif param.ndim >= 2:
            muon_params.append(param)
        else:
            adam_params.append(param)

    return muon_params, adam_params


def compute_grad_norms(model):
    """Per-layer gradient L2 norms for diagnostics."""
    norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            norms[name] = param.grad.norm().item()
    return norms


@torch.no_grad()
def evaluate_classification(model, val_loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += criterion(logits, y).item() * y.size(0)
        correct += (logits.argmax(-1) == y).sum().item()
        total += y.size(0)
    model.train()
    return {"val_loss": loss_sum / total, "val_acc": correct / total}


@torch.no_grad()
def evaluate_lm(model, val_loader, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
    model.train()
    return {"val_loss": total_loss / total_tokens,
            "val_ppl": math.exp(total_loss / total_tokens)}


def train(config):
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = config["model"]["name"]

    # Build model and data.
    model = build_model(config["model"]).to(device)
    train_loader, val_loader = build_dataloaders(config["data"])

    # Apply initialization.
    init_name = config.get("init", "default")
    if init_name in INIT_REGISTRY:
        INIT_REGISTRY[init_name](model)
    else:
        raise ValueError(f"Unknown init: {init_name}")

    # Count params.
    total_params = sum(p.numel() for p in model.parameters())
    muon_params, adam_params = split_params(model, model_name)
    muon_count = sum(p.numel() for p in muon_params)
    print(f"Model: {model_name} | Total params: {total_params:,} | "
          f"Muon params: {muon_count:,} ({100*muon_count/total_params:.1f}%)")

    # Optimizer.
    train_cfg = config["training"]
    use_muon = config.get("use_muon", True)

    if use_muon and muon_params:
        optimizer = MuonAdamW(
            muon_params=muon_params,
            adam_params=adam_params,
            lr_muon=train_cfg.get("lr_muon", 0.02),
            lr_adam=train_cfg.get("lr_adam", 3e-4),
            momentum=train_cfg.get("momentum", 0.95),
            weight_decay_muon=train_cfg.get("weight_decay_muon", 0.0),
            weight_decay_adam=train_cfg.get("weight_decay_adam", 0.01),
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_cfg.get("lr_adam", 3e-4),
            weight_decay=train_cfg.get("weight_decay_adam", 0.01),
        )

    # Training loop.
    total_steps = train_cfg["total_steps"]
    warmup_steps = train_cfg.get("warmup_steps", 0)
    eval_every = train_cfg.get("eval_every", 500)
    log_every = train_cfg.get("log_every", 100)
    is_lm = model_name in ("nanogpt", "deep_narrow_gpt")

    criterion = nn.CrossEntropyLoss() if not is_lm else None

    # Results log.
    results = {
        "config": config,
        "total_params": total_params,
        "muon_params": muon_count,
        "log": [],
    }

    save_dir = config.get("save_dir", "experiments/results")
    os.makedirs(save_dir, exist_ok=True)

    model.train()
    step = 0
    data_iter = iter(train_loader)
    t0 = time.time()

    while step < total_steps:
        # Get batch (cycle through data).
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        # Set learning rate.
        lr_muon = get_lr(step, warmup_steps, train_cfg.get("lr_muon", 0.02),
                         total_steps)
        lr_adam = get_lr(step, warmup_steps, train_cfg.get("lr_adam", 3e-4),
                         total_steps)
        for group in optimizer.param_groups:
            if group.get("use_muon", False):
                group["lr"] = lr_muon
            else:
                group["lr"] = lr_adam

        # Forward + backward.
        if is_lm:
            x, y = batch[0].to(device), batch[1].to(device)
            _, loss = model(x, y)
        else:
            x, y = batch[0].to(device), batch[1].to(device)
            logits = model(x)
            loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1

        # Logging.
        if step % log_every == 0:
            elapsed = time.time() - t0
            entry = {"step": step, "train_loss": loss.item(),
                     "lr_muon": lr_muon, "lr_adam": lr_adam, "time": elapsed}
            results["log"].append(entry)
            print(f"[step {step}/{total_steps}] loss={loss.item():.4f} "
                  f"lr_muon={lr_muon:.6f} time={elapsed:.1f}s")

        # Evaluation.
        if step % eval_every == 0 or step == total_steps:
            if is_lm:
                val_metrics = evaluate_lm(model, val_loader, device)
            else:
                val_metrics = evaluate_classification(model, val_loader, device)

            val_metrics["step"] = step
            results["log"].append(val_metrics)
            print(f"  EVAL: {val_metrics}")

    # Save results.
    run_name = f"{model_name}_{init_name}_warmup{warmup_steps}_seed{seed}"
    results_path = os.path.join(save_dir, f"{run_name}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--init", type=str, default=None,
                        help="Override init scheme")
    parser.add_argument("--warmup_steps", type=int, default=None,
                        help="Override warmup steps")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed")
    parser.add_argument("--use_muon", action="store_true", default=None)
    parser.add_argument("--no_muon", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # CLI overrides.
    if args.init is not None:
        config["init"] = args.init
    if args.warmup_steps is not None:
        config["training"]["warmup_steps"] = args.warmup_steps
    if args.seed is not None:
        config["seed"] = args.seed
    if args.use_muon:
        config["use_muon"] = True
    if args.no_muon:
        config["use_muon"] = False

    train(config)


if __name__ == "__main__":
    main()

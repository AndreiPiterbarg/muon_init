"""Adversarial retraining loop.

Each round:
  1. ATTACK: run adversary against current model
  2. RETRAIN: train new model from scratch on mixed distribution
     (p_iso * standard Gaussian + p_adv * adversarial curriculum)
  3. EVALUATE: measure on fixed benchmark

Stopping conditions (from RESEARCH_PLAN.md):
  1. Fitness threshold: adversary can't find breaks above threshold
  2. Benchmark plateau: no improvement for 2 consecutive rounds
  3. Overlap (secondary): only if fitness is marginal AND spectra match previous
  4. Max rounds: hard cap

Curriculum management:
  - Genomes stored with their fitness scores for weighted sampling
  - Capped at MAX_CURRICULUM=75 (keep strongest, drop weakest)
  - Dynamic p_adv scales with curriculum size, capped at 0.65
  - Fitness-weighted sampling so harder failures get more exposure

NOTE: p_adv base=0.3 is a starting point. The dynamic scaling
(p_adv = base * curriculum_size / 50, capped at 0.65) maintains
per-genome exposure as curriculum grows. These values need tuning.

Usage:
    python experiments/scripts/retrain_loop.py --config configs/adversary_d5.yaml
    python experiments/scripts/retrain_loop.py --config configs/adversary_d5.yaml --max-rounds 5 --p-adv 0.3
"""

import argparse
import json
import os
import pickle
import shutil
import sys
import time

import numpy as np
import torch
import yaml

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.icl.eval import get_model_from_run
from src.icl.models import build_model
from src.icl.schema import load_config
from src.icl.train import train
from src.icl.samplers import MixedSampler
from src.adversary.evaluate import GenomeEvaluator
from src.adversary.search import cma_search
from src.adversary.pipeline_genome import PipelineGenome
from src.adversary.genome import Genome
from src.eval.benchmark import evaluate_benchmark, print_benchmark, print_benchmark_comparison

MAX_CURRICULUM = 75
MAX_P_ADV = 0.65
P_ADV_REFERENCE_SIZE = 50  # p_adv scales relative to this curriculum size


def collect_top_k(results, k=50):
    """Get top-k valid results by fitness. Returns list of (genome, fitness) pairs."""
    valid = [r for r in results if r.is_valid]
    valid.sort(key=lambda r: r.fitness, reverse=True)
    return [(r.genome.copy(), float(r.fitness)) for r in valid[:k]]


def check_overlap(new_genomes, previous_genomes, threshold=0.8):
    """Check if >threshold of new genomes overlap with previous ones.

    Uses covariance spectrum Frobenius distance as the similarity metric.
    """
    if not previous_genomes:
        return False

    new_spectra = []
    for g in new_genomes:
        try:
            new_spectra.append(g.eigenvalues())
        except Exception:
            new_spectra.append(np.ones(g.n_dims))

    prev_spectra = []
    for g in previous_genomes:
        try:
            prev_spectra.append(g.eigenvalues())
        except Exception:
            prev_spectra.append(np.ones(g.n_dims))

    overlap_count = 0
    eps_dist = 0.5

    for ns in new_spectra:
        for ps in prev_spectra:
            if np.linalg.norm(ns - ps) < eps_dist:
                overlap_count += 1
                break

    overlap_frac = overlap_count / max(len(new_spectra), 1)
    return overlap_frac > threshold


def _benchmark_mean(benchmark_results):
    """Compute mean ICL/ridge ratio over non-isotropic distributions."""
    if not benchmark_results:
        return float("inf")
    ratios = [
        r["icl_ridge_ratio_at_d"]
        for name, r in benchmark_results.items()
        if name != "isotropic"
    ]
    return float(np.mean(ratios)) if ratios else float("inf")


def train_model_with_curriculum(
    out_dir, curriculum, p_adv, n_dims,
    train_steps, n_layer, n_head, n_embd, n_points_end,
    learning_rate=3e-4, batch_size=64,
):
    """Train a model FROM SCRATCH using MixedSampler with fitness-weighted curriculum.

    Args:
        curriculum: list of (genome, fitness) pairs
        p_adv: adversarial fraction (already computed dynamically by caller)
    """
    os.makedirs(out_dir, exist_ok=True)

    # Remove any existing checkpoint so we train from scratch
    state_path = os.path.join(out_dir, "state.pt")
    if os.path.exists(state_path):
        os.remove(state_path)

    config = {
        "out_dir": out_dir,
        "test_run": True,
        "model": {
            "family": "gpt2",
            "n_dims": n_dims,
            "n_positions": n_points_end,
            "n_embd": n_embd,
            "n_layer": n_layer,
            "n_head": n_head,
        },
        "training": {
            "task": "noisy_linear_regression",
            "task_kwargs": {"noise_std": 0.1},
            "data": "gaussian",
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "train_steps": train_steps,
            "save_every_steps": train_steps,
            "keep_every_steps": -1,
            "resume_id": None,
            "num_tasks": None,
            "num_training_examples": None,
            "curriculum": {
                "dims": {"start": n_dims, "end": n_dims, "inc": 1, "interval": 2000},
                "points": {"start": n_points_end, "end": n_points_end, "inc": 2, "interval": 2000},
            },
        },
        "wandb": {
            "project": "test", "entity": "test", "notes": "", "name": "test",
            "log_every_steps": 500,
        },
    }

    args = load_config(config)
    args.test_run = False
    args.out_dir = out_dir

    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.dump(args.toDict(), f, default_flow_style=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args.model)
    model.to(device)
    model.train()

    # Build fitness-weighted sampler
    genomes = [g for g, f in curriculum]
    weights = [f for g, f in curriculum]
    sampler = MixedSampler(n_dims, genomes=genomes, weights=weights, p_adv=p_adv)

    print(f"  Training from scratch: {train_steps} steps, p_adv={p_adv:.3f}, "
          f"{len(curriculum)} genomes (fitness-weighted)")

    train(model, args, data_sampler=sampler)

    model.eval()
    return model


def finetune_model_with_curriculum(
    out_dir, curriculum, p_adv, n_dims,
    finetune_steps, n_layer, n_head, n_embd, n_points_end,
    learning_rate=1e-4, batch_size=64,
):
    """Fine-tune an existing model checkpoint using MixedSampler with fitness-weighted curriculum.

    Unlike train_model_with_curriculum, this does NOT delete state.pt.
    It resumes from the existing checkpoint and trains for finetune_steps
    additional iterations at a (typically lower) learning rate.

    Args:
        out_dir: directory containing state.pt to resume from
        curriculum: list of (genome, fitness) pairs
        p_adv: adversarial fraction (already computed dynamically by caller)
        finetune_steps: number of additional training steps
        learning_rate: fine-tuning LR (passed as override_lr to train())
    """
    state_path = os.path.join(out_dir, "state.pt")
    assert os.path.exists(state_path), f"No checkpoint at {state_path} for fine-tuning"

    state = torch.load(state_path, map_location="cpu")
    starting_step = state["train_step"]
    total_steps = starting_step + finetune_steps

    config = {
        "out_dir": out_dir,
        "test_run": True,
        "model": {
            "family": "gpt2",
            "n_dims": n_dims,
            "n_positions": n_points_end,
            "n_embd": n_embd,
            "n_layer": n_layer,
            "n_head": n_head,
        },
        "training": {
            "task": "noisy_linear_regression",
            "task_kwargs": {"noise_std": 0.1},
            "data": "gaussian",
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "train_steps": total_steps,
            "save_every_steps": finetune_steps,
            "keep_every_steps": -1,
            "resume_id": None,
            "num_tasks": None,
            "num_training_examples": None,
            "curriculum": {
                "dims": {"start": n_dims, "end": n_dims, "inc": 1, "interval": 2000},
                "points": {"start": n_points_end, "end": n_points_end, "inc": 2, "interval": 2000},
            },
        },
        "wandb": {
            "project": "test", "entity": "test", "notes": "", "name": "test",
            "log_every_steps": 500,
        },
    }

    args = load_config(config)
    args.test_run = False
    args.out_dir = out_dir

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args.model)
    model.to(device)
    model.train()

    genomes = [g for g, f in curriculum]
    weights = [f for g, f in curriculum]
    sampler = MixedSampler(n_dims, genomes=genomes, weights=weights, p_adv=p_adv)

    print(f"  Fine-tuning: {finetune_steps} steps (from step {starting_step}), "
          f"lr={learning_rate}, p_adv={p_adv:.3f}, {len(curriculum)} genomes")

    train(model, args, data_sampler=sampler, override_lr=learning_rate)

    model.eval()
    return model


def run_retraining_loop(
    config_path,
    max_rounds=10,
    fitness_threshold=1.0,
    overlap_threshold=0.8,
    top_k=50,
    p_adv=0.3,
    attack_budget=50000,
    attack_restarts=5,
    output_dir=None,
    finetune_steps=20000,
    finetune_lr=1e-4,
    finetune_p_adv=0.5,
    attack_target="cumulative",
):
    """Run the full adversarial retraining loop."""

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("results", "retrain_loop", timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # --- Load base model ---
    model_config = config["icl_model"]
    run_path = model_config["run_path"]
    print(f"Loading base model from {run_path}...")
    model, conf = get_model_from_run(run_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    n_dims = config.get("task", {}).get("n_dims", conf.model.n_dims)
    n_points = config.get("task", {}).get("n_points", 2 * n_dims + 1)
    task_name = config.get("task", {}).get("name", "noisy_linear_regression")

    # Model arch params (for retraining from scratch)
    n_layer = conf.model.n_layer
    n_head = conf.model.n_head
    n_embd = conf.model.n_embd
    n_positions = conf.model.n_positions
    train_steps = conf.training.train_steps
    learning_rate = conf.training.learning_rate
    batch_size = conf.training.batch_size

    # Genome class
    genome_config = config.get("genome", {})
    genome_type = genome_config.get("type", "pipeline")
    genome_cls = PipelineGenome if genome_type == "pipeline" else Genome

    # Search params
    pop_size = config.get("search", {}).get("pop_size", 32)
    sigma_init = config.get("search", {}).get("sigma_init", 0.3)

    # Eval params
    eval_config = config.get("eval", {})
    eval_batch_size = eval_config.get("batch_size", 64)
    eval_num_batches = eval_config.get("num_batches", 10)
    baselines = eval_config.get("baselines", ["ridge", "least_squares", "averaging"])

    # --- Initialize cumulative model ---
    cumulative_dir = os.path.join(output_dir, "cumulative_model")
    os.makedirs(cumulative_dir, exist_ok=True)
    base_state_path = os.path.join(run_path, "state.pt")
    if os.path.exists(base_state_path):
        shutil.copy2(base_state_path, os.path.join(cumulative_dir, "state.pt"))
    else:
        print(f"WARNING: No state.pt at {base_state_path}, cumulative model will start fresh")

    scratch_model = model
    cumulative_model = model  # both start as the same base model

    # --- Round 0: Benchmark the base model ---
    print("\n" + "=" * 60)
    print("  ROUND 0: Baseline evaluation")
    print("=" * 60)

    benchmark_round_0 = evaluate_benchmark(model, n_dims, n_points, eval_batch_size, eval_num_batches)
    print_benchmark(benchmark_round_0, "Base model (round 0)")

    all_benchmarks_scratch = {0: benchmark_round_0}
    all_benchmarks_cumulative = {0: benchmark_round_0}
    all_curriculum = []  # list of (genome, fitness) pairs
    all_round_results = {}

    # --- Retraining loop ---
    for round_num in range(1, max_rounds + 1):
        print(f"\n{'='*60}")
        print(f"  ROUND {round_num}: ATTACK")
        print(f"{'='*60}")

        # 1. SELECT ATTACK TARGET
        if attack_target == "scratch":
            attack_model = scratch_model
        elif attack_target == "cumulative":
            attack_model = cumulative_model
        elif attack_target == "better":
            s_mean = _benchmark_mean(all_benchmarks_scratch.get(round_num - 1, benchmark_round_0))
            c_mean = _benchmark_mean(all_benchmarks_cumulative.get(round_num - 1, benchmark_round_0))
            attack_model = scratch_model if s_mean < c_mean else cumulative_model
            print(f"  Attack target: {'scratch' if attack_model is scratch_model else 'cumulative'} "
                  f"(scratch={s_mean:.4f}, cumulative={c_mean:.4f})")
        else:
            attack_model = cumulative_model

        # 2. ATTACK
        evaluator = GenomeEvaluator(
            icl_model=attack_model,
            task_name=task_name,
            n_dims=n_dims,
            n_points=n_points,
            batch_size=eval_batch_size,
            num_batches=eval_num_batches,
            baseline_names=baselines,
        )

        attack_results = cma_search(
            evaluator=evaluator,
            n_dims=n_dims,
            budget=attack_budget,
            pop_size=pop_size,
            sigma_init=sigma_init,
            num_restarts=attack_restarts,
            save_dir=os.path.join(output_dir, f"round_{round_num}_attack"),
            seed=round_num * 100,
            genome_cls=genome_cls,
        )

        valid = [r for r in attack_results if r.is_valid]
        if not valid:
            print(f"  No valid results in round {round_num}. Stopping.")
            break

        best_fitness = max(r.fitness for r in valid)
        print(f"\n  Round {round_num} attack: best_fitness={best_fitness:.4f}")

        # 3. PRIMARY STOP: fitness threshold
        if best_fitness < fitness_threshold:
            print(f"  STOP: best_fitness {best_fitness:.4f} < threshold {fitness_threshold}")
            all_round_results[round_num] = {"best_fitness": best_fitness, "stopped": "fitness_threshold"}
            break

        # 4. ACCUMULATE CURRICULUM (fitness-weighted, capped)
        new_entries = collect_top_k(attack_results, top_k)
        all_curriculum.extend(new_entries)
        # Cap: keep top MAX_CURRICULUM by fitness
        all_curriculum.sort(key=lambda x: x[1], reverse=True)
        dropped = len(all_curriculum) - MAX_CURRICULUM
        all_curriculum = all_curriculum[:MAX_CURRICULUM]
        if dropped > 0:
            print(f"  Curriculum capped: dropped {dropped} weakest genomes")
        print(f"  Curriculum: {len(all_curriculum)} genomes "
              f"(fitness range: {all_curriculum[-1][1]:.2f} - {all_curriculum[0][1]:.2f})")

        # 5. RETRAIN / FINE-TUNE
        print(f"\n{'='*60}")
        print(f"  ROUND {round_num}: RETRAIN (scratch) + FINE-TUNE (cumulative)")
        print(f"{'='*60}")

        effective_p_adv = min(p_adv * (len(all_curriculum) / P_ADV_REFERENCE_SIZE), MAX_P_ADV)
        effective_ft_p_adv = min(finetune_p_adv * (len(all_curriculum) / P_ADV_REFERENCE_SIZE), MAX_P_ADV)

        # 5a. Retrain scratch from scratch (existing behavior)
        retrain_dir = os.path.join(output_dir, f"round_{round_num}_model")
        scratch_model = train_model_with_curriculum(
            out_dir=retrain_dir,
            curriculum=all_curriculum,
            p_adv=effective_p_adv,
            n_dims=n_dims,
            train_steps=train_steps,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            n_points_end=n_positions,
            learning_rate=learning_rate,
            batch_size=batch_size,
        )
        scratch_model = scratch_model.to(device).eval()

        # 5b. Fine-tune cumulative model
        cumulative_model = finetune_model_with_curriculum(
            out_dir=cumulative_dir,
            curriculum=all_curriculum,
            p_adv=effective_ft_p_adv,
            n_dims=n_dims,
            finetune_steps=finetune_steps,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            n_points_end=n_positions,
            learning_rate=finetune_lr,
            batch_size=batch_size,
        )
        cumulative_model = cumulative_model.to(device).eval()

        # 6. EVALUATE BOTH MODELS
        print(f"\n{'='*60}")
        print(f"  ROUND {round_num}: EVALUATE")
        print(f"{'='*60}")

        bench_scratch = evaluate_benchmark(scratch_model, n_dims, n_points, eval_batch_size, eval_num_batches)
        bench_cumulative = evaluate_benchmark(cumulative_model, n_dims, n_points, eval_batch_size, eval_num_batches)

        print_benchmark_comparison(benchmark_round_0, bench_scratch, "Base", f"Scratch R{round_num}")
        print_benchmark_comparison(benchmark_round_0, bench_cumulative, "Base", f"Cumul R{round_num}")
        print_benchmark_comparison(bench_scratch, bench_cumulative, f"Scratch R{round_num}", f"Cumul R{round_num}")

        all_benchmarks_scratch[round_num] = bench_scratch
        all_benchmarks_cumulative[round_num] = bench_cumulative

        scratch_mean = _benchmark_mean(bench_scratch)
        cumulative_mean = _benchmark_mean(bench_cumulative)
        print(f"  Benchmark mean (non-iso): scratch={scratch_mean:.4f}, cumulative={cumulative_mean:.4f}")

        all_round_results[round_num] = {
            "best_fitness": best_fitness,
            "curriculum_size": len(all_curriculum),
            "effective_p_adv": effective_p_adv,
            "effective_ft_p_adv": effective_ft_p_adv,
            "scratch_benchmark_mean": scratch_mean,
            "cumulative_benchmark_mean": cumulative_mean,
            "stopped": None,
        }

        # 7. STOPPING CONDITIONS (post-eval)
        # Use the attacked model's benchmarks for plateau detection
        if attack_target == "scratch":
            active_benchmarks = all_benchmarks_scratch
        else:
            active_benchmarks = all_benchmarks_cumulative
        curr_mean = _benchmark_mean(active_benchmarks.get(round_num, {}))

        # 7a. Benchmark plateau: no improvement for 2 consecutive rounds
        if round_num >= 3:
            prev_mean = _benchmark_mean(active_benchmarks.get(round_num - 1, {}))
            prev_prev_mean = _benchmark_mean(active_benchmarks.get(round_num - 2, {}))
            improved_this = curr_mean < prev_mean - 0.05
            improved_last = prev_mean < prev_prev_mean - 0.05
            if not improved_this and not improved_last:
                print(f"  STOP: benchmark plateau (no improvement for 2 consecutive rounds)")
                all_round_results[round_num]["stopped"] = "benchmark_plateau"
                _save_progress(output_dir, all_benchmarks_scratch, all_benchmarks_cumulative,
                               all_round_results, all_curriculum)
                break

        # 7b. Overlap (secondary): only if fitness is marginal
        if best_fitness < 2 * fitness_threshold:
            new_genomes = [g for g, f in new_entries]
            prev_genomes = [g for g, f in all_curriculum if (g, f) not in new_entries]
            if check_overlap(new_genomes, prev_genomes, overlap_threshold):
                print(f"  STOP: marginal fitness ({best_fitness:.2f} < {2*fitness_threshold:.2f}) + overlap")
                all_round_results[round_num]["stopped"] = "marginal_fitness_overlap"
                _save_progress(output_dir, all_benchmarks_scratch, all_benchmarks_cumulative,
                               all_round_results, all_curriculum)
                break

        # Save progress after each round
        _save_progress(output_dir, all_benchmarks_scratch, all_benchmarks_cumulative,
                       all_round_results, all_curriculum)

    # --- Final summary ---
    print(f"\n{'='*60}")
    print(f"  RETRAINING LOOP COMPLETE: {len(all_round_results)} rounds")
    print(f"{'='*60}")

    for r, info in sorted(all_round_results.items()):
        status = f"stopped={info['stopped']}" if info.get("stopped") else "completed"
        p = info.get("effective_p_adv", p_adv)
        s_bm = info.get("scratch_benchmark_mean", 0)
        c_bm = info.get("cumulative_benchmark_mean", 0)
        print(f"  Round {r}: fitness={info['best_fitness']:.4f}, p_adv={p:.3f}, "
              f"scratch_bm={s_bm:.4f}, cumul_bm={c_bm:.4f}, {status}")

    _save_progress(output_dir, all_benchmarks_scratch, all_benchmarks_cumulative,
                   all_round_results, all_curriculum)
    print(f"\nResults saved to {output_dir}")


def _save_progress(output_dir, benchmarks_scratch, benchmarks_cumulative, round_results, curriculum):
    """Save loop state to disk."""
    with open(os.path.join(output_dir, "benchmarks.json"), "w") as f:
        json.dump({
            "scratch": {str(k): v for k, v in benchmarks_scratch.items()},
            "cumulative": {str(k): v for k, v in benchmarks_cumulative.items()},
        }, f, indent=2)

    with open(os.path.join(output_dir, "round_results.json"), "w") as f:
        json.dump({str(k): v for k, v in round_results.items()}, f, indent=2)

    # Save curriculum as (genome, fitness) pairs
    with open(os.path.join(output_dir, "curriculum.pkl"), "wb") as f:
        pickle.dump(curriculum, f)


def main():
    parser = argparse.ArgumentParser(description="Adversarial retraining loop")
    parser.add_argument("--config", type=str, required=True, help="Path to adversary config YAML")
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--fitness-threshold", type=float, default=1.0,
                        help="Stop when adversary fitness drops below this")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Number of top failures to add to curriculum per round")
    parser.add_argument("--p-adv", type=float, default=0.3,
                        help="Base adversarial fraction (scales dynamically with curriculum size)")
    parser.add_argument("--attack-budget", type=int, default=50000,
                        help="CMA-ES evaluation budget per attack round")
    parser.add_argument("--attack-restarts", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--finetune-steps", type=int, default=20000,
                        help="Training steps per round for cumulative fine-tuning")
    parser.add_argument("--finetune-lr", type=float, default=1e-4,
                        help="Learning rate for cumulative fine-tuning")
    parser.add_argument("--finetune-p-adv", type=float, default=0.5,
                        help="Base adversarial fraction for fine-tuning (scales dynamically)")
    parser.add_argument("--attack-target", type=str, default="cumulative",
                        choices=["scratch", "cumulative", "better"],
                        help="Which model the adversary attacks each round")
    args = parser.parse_args()

    run_retraining_loop(
        config_path=args.config,
        max_rounds=args.max_rounds,
        fitness_threshold=args.fitness_threshold,
        top_k=args.top_k,
        p_adv=args.p_adv,
        attack_budget=args.attack_budget,
        attack_restarts=args.attack_restarts,
        output_dir=args.output_dir,
        finetune_steps=args.finetune_steps,
        finetune_lr=args.finetune_lr,
        finetune_p_adv=args.finetune_p_adv,
        attack_target=args.attack_target,
    )


if __name__ == "__main__":
    main()

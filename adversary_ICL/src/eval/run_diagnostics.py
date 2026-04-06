"""Run diagnostics on adversarial failures.

Usage:
    from src.eval.run_diagnostics import run_all_diagnostics

    results = run_all_diagnostics(
        model=model,
        genome=best_genome,
        evaluator=evaluator,
        output_dir="results/diagnostics/",
    )
"""

import json
import os

import torch

from .diagnostics import DIAGNOSTICS, DiagnosticResult


# Default diagnostics to run (retrained_model excluded — expensive)
DEFAULT_DIAGNOSTICS = [
    "learning_curve",
    "algorithm_id",
    "weight_recovery",
    "attention",
    "linear_probing",
    "gradient_attribution",
    "pipeline_ablation",
]


def run_all_diagnostics(
    model,
    genome,
    evaluator,
    output_dir: str,
    diagnostics: list[str] | None = None,
    **kwargs,
) -> dict[str, DiagnosticResult]:
    """Run selected diagnostics on an adversarial genome.

    Args:
        model: The ICL model.
        genome: The adversarial genome (Genome or PipelineGenome).
        evaluator: GenomeEvaluator instance.
        output_dir: Where to save plots and results.
        diagnostics: List of diagnostic names to run (default: all except retrained_model).
        **kwargs: Extra args passed to each diagnostic (e.g., train_fn for retrained_model).

    Returns:
        Dict mapping diagnostic name -> DiagnosticResult.
    """
    if diagnostics is None:
        diagnostics = DEFAULT_DIAGNOSTICS

    os.makedirs(output_dir, exist_ok=True)

    # Generate adversarial data once for diagnostics that need it
    n_points = evaluator.n_points
    batch_size = evaluator.batch_size

    from ..adversary.pipeline_genome import PipelineGenome
    if isinstance(genome, PipelineGenome):
        xs = genome.sample_xs(n_points, batch_size)
    else:
        from ..icl.samplers import GaussianSampler
        L = genome.decode_L_normalized()
        mu = genome.decode_mu()
        sampler = GaussianSampler(genome.n_dims, bias=mu, scale=L)
        xs = sampler.sample_xs(n_points, batch_size)

    w = genome.decode_weights()
    noise_std = genome.decode_noise_std()
    ys = (xs @ w) + noise_std * torch.randn(xs.shape[0], xs.shape[1])

    results = {}

    for diag_name in diagnostics:
        if diag_name not in DIAGNOSTICS:
            print(f"  Unknown diagnostic: {diag_name}, skipping")
            continue

        print(f"  Running diagnostic: {diag_name}...")
        diag_cls = DIAGNOSTICS[diag_name]
        diag = diag_cls()

        diag_output_dir = os.path.join(output_dir, diag_name)

        try:
            result = diag.run(
                model=model,
                xs=xs,
                ys=ys,
                genome=genome,
                output_dir=diag_output_dir,
                evaluator=evaluator,
                **kwargs,
            )
            results[diag_name] = result
            print(f"    {result.summary}")
        except Exception as e:
            print(f"    ERROR: {e}")
            results[diag_name] = DiagnosticResult(
                name=diag_name,
                summary=f"Error: {e}",
            )

    # Save summary JSON
    summary_path = os.path.join(output_dir, "diagnostics_summary.json")
    summary = {}
    for name, result in results.items():
        summary[name] = {
            "summary": result.summary,
            "data": _serialize(result.data),
            "plots": result.plots,
        }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDiagnostics summary saved to {summary_path}")

    return results


def _serialize(obj):
    """Make data JSON-serializable."""
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_serialize(v) for v in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)

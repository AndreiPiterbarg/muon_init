import os
import pickle
import time

import numpy as np
from tqdm import tqdm

from .genome import Genome
from .evaluate import GenomeEvaluator, EvalResult

cma = None


def cma_search(
    evaluator: GenomeEvaluator,
    n_dims: int,
    budget: int = 50000,
    pop_size: int = 32,
    sigma_init: float = 0.5,
    save_dir: str | None = None,
    save_interval: int = 50,
    seed: int = 0,
) -> list[EvalResult]:
    """Run CMA-ES adversarial search over the genome space.

    Args:
        evaluator: Evaluates genomes against ICL model + baselines.
        n_dims: Dimensionality of the input space (determines genome size).
        budget: Total number of genome evaluations.
        pop_size: Population size per CMA-ES generation.
        sigma_init: Initial step size for CMA-ES.
        save_dir: Directory to save checkpoints. None = no saving.
        save_interval: Save checkpoint every N generations.
        seed: Random seed.

    Returns:
        List of all EvalResults from the search.
    """
    global cma
    if cma is None:
        try:
            import cma as _cma
            cma = _cma
        except ImportError:
            raise ImportError("Install pycma: pip install cma")

    genome_size = Genome.flat_size(n_dims)

    # Initialize from a structured random genome
    rng = np.random.default_rng(seed)
    x0 = Genome.random_structured(n_dims, rng).raw

    # CMA-ES options
    opts = cma.CMAOptions()
    opts["seed"] = seed
    opts["popsize"] = pop_size
    opts["maxfevals"] = budget
    opts["verb_disp"] = 0  # quiet
    opts["verb_log"] = 0
    # Use separable CMA-ES for high-dimensional space (faster convergence)
    if genome_size > 200:
        opts["CMA_diagonal"] = True

    es = cma.CMAEvolutionStrategy(x0, sigma_init, opts)

    all_results: list[EvalResult] = []
    best_fitness = 0.0
    best_result: EvalResult | None = None
    generation = 0
    total_evals = 0

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    print(f"Starting CMA-ES search: genome_size={genome_size}, budget={budget}, pop_size={pop_size}")

    while not es.stop() and total_evals < budget:
        # Sample population
        solutions = es.ask()
        generation += 1

        # Evaluate each genome
        fitnesses = []
        gen_results = []

        for sol in solutions:
            genome = Genome(n_dims, np.array(sol))
            result = evaluator.evaluate(genome)
            gen_results.append(result)

            # CMA-ES minimizes, so negate fitness (we want to maximize gap)
            fitnesses.append(-result.fitness if result.is_valid else 0.0)

        es.tell(solutions, fitnesses)
        all_results.extend(gen_results)
        total_evals += len(solutions)

        # Track best
        for r in gen_results:
            if r.is_valid and r.fitness > best_fitness:
                best_fitness = r.fitness
                best_result = r

        # Progress report
        valid = [r for r in gen_results if r.is_valid]
        if valid:
            gen_best = max(r.fitness for r in valid)
            gen_mean = np.mean([r.fitness for r in valid])
            print(
                f"Gen {generation:4d} | evals {total_evals:6d}/{budget} | "
                f"gen_best={gen_best:.3f} gen_mean={gen_mean:.3f} | "
                f"overall_best={best_fitness:.3f}"
            )

        # Checkpoint
        if save_dir and generation % save_interval == 0:
            _save_checkpoint(save_dir, all_results, es, generation, best_result)

    # Final save
    if save_dir:
        _save_checkpoint(save_dir, all_results, es, generation, best_result)

    print(f"\nSearch complete. {total_evals} evaluations, best fitness={best_fitness:.4f}")
    if best_result:
        print(f"Best genome: {best_result.genome}")

    return all_results


def _save_checkpoint(save_dir, all_results, es, generation, best_result):
    """Save search state to disk."""
    checkpoint = {
        "results": all_results,
        "generation": generation,
        "best_result": best_result,
        "cma_mean": es.mean.copy(),
        "cma_sigma": es.sigma,
    }
    path = os.path.join(save_dir, "checkpoint.pkl")
    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)
    print(f"  [saved checkpoint: {len(all_results)} results, gen {generation}]")


def load_checkpoint(save_dir: str) -> dict:
    """Load a saved checkpoint."""
    path = os.path.join(save_dir, "checkpoint.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

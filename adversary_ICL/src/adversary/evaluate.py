import numpy as np
import torch
from dataclasses import dataclass, field

from .genome import Genome
from .pipeline_genome import PipelineGenome

from ..icl.samplers import GaussianSampler
from ..icl.tasks import get_task_sampler
from ..icl import models


@dataclass
class EvalResult:
    genome: object  # Genome or PipelineGenome
    fitness: float
    raw_fitness: float = 0.0
    complexity: float = 0.0
    icl_curve: np.ndarray = field(default_factory=lambda: np.zeros(0))
    baseline_curves: dict = field(default_factory=dict)  # name -> per-point error
    covariance_spectrum: np.ndarray = field(default_factory=lambda: np.array([]))
    descriptors: dict = field(default_factory=dict)
    is_valid: bool = True


def _eval_model_on_batch(model, xs, ys, device):
    """Run model forward pass, return predictions."""
    with torch.no_grad():
        if device != "cpu" and device != torch.device("cpu"):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred = model(xs.to(device), ys.to(device)).detach().cpu()
        else:
            pred = model(xs.to(device), ys.to(device)).detach().cpu()
    return pred


class GenomeEvaluator:
    """Evaluates a Genome against the ICL model and baselines.

    Fitness is the mean ratio of ICL error to best baseline error over the
    learning curve, computed only at points where the baseline has meaningful
    error. This is fully scale-invariant: the adversary cannot gain fitness
    by inflating input/output magnitudes.
    """

    def __init__(
        self,
        icl_model,
        task_name: str,
        n_dims: int,
        n_points: int,
        batch_size: int = 64,
        num_batches: int = 10,
        baseline_names: list[str] | None = None,
        parsimony_lambda: float = 0.0,
        c_base: float = 1.0,
        c_stage: float = 1.0,
        c_affine: float = 1.0,
    ):
        self.icl_model = icl_model
        self.task_name = task_name
        self.n_dims = n_dims
        self.n_points = n_points
        self.batch_size = batch_size
        self.num_batches = num_batches

        # Parsimony pressure
        self.parsimony_lambda = parsimony_lambda
        self.c_base = c_base
        self.c_stage = c_stage
        self.c_affine = c_affine

        if baseline_names is None:
            baseline_names = ["ridge", "least_squares", "averaging"]
        self.baselines = self._build_baselines(baseline_names)

        # Get device from model
        if hasattr(icl_model, 'parameters'):
            self.device = next(icl_model.parameters()).device
        else:
            self.device = "cpu"

    def _build_baselines(self, names: list[str]) -> list:
        name_to_cls = {
            "least_squares": (models.LeastSquaresModel, {}),
            "ridge": (models.RidgeRegressionModel, {"alpha": 1.0}),
            "averaging": (models.AveragingModel, {}),
            "nn_3": (models.NNModel, {"n_neighbors": 3}),
        }
        result = []
        for name in names:
            if name in name_to_cls:
                cls, kwargs = name_to_cls[name]
                result.append((name, cls(**kwargs)))
        return result

    def evaluate(self, genome) -> EvalResult:
        """Full evaluation: run ICL model + baselines on genome's distribution.

        Accepts both Genome (legacy) and PipelineGenome.
        """
        genome = genome.copy()
        genome.clamp_()

        try:
            return self._evaluate_inner(genome)
        except Exception:
            return EvalResult(
                genome=genome,
                fitness=0.0,
                icl_curve=np.zeros(self.n_points),
                is_valid=False,
            )

    def _evaluate_inner(self, genome) -> EvalResult:
        # Decode common genome fields
        w = genome.decode_weights()  # unit-normalized
        noise_std = genome.decode_noise_std()

        is_pipeline = isinstance(genome, PipelineGenome)

        # Build sampler (legacy path uses GaussianSampler)
        if not is_pipeline:
            L = genome.decode_L_normalized()
            mu = genome.decode_mu()
            sampler = GaussianSampler(self.n_dims, bias=mu, scale=L)

        # Build task with adversary-controlled weight direction
        # pool_dict needs shape (num_tasks, n_dims, 1)
        pool_dict = {"w": w.unsqueeze(0).unsqueeze(-1).expand(self.batch_size, -1, -1)}

        # Bayes-optimal oracle baseline (per-genome).
        # Likelihood: y|x,w ~ N(w^T x, sigma^2). Adversary forces ||w||=1, so under
        # the uniform-on-sphere prior the per-coordinate variance is 1/d, giving the
        # closed-form posterior mean w_hat = (X^T X + alpha* I)^-1 X^T y with
        # alpha* = sigma^2 / (1/d) = d * sigma^2. This is the tightest estimator
        # given the true generative model at test time.
        alpha_oracle = float(self.n_dims) * (float(noise_std) ** 2)
        oracle_baseline = models.RidgeRegressionModel(alpha=alpha_oracle)
        per_genome_baselines = list(self.baselines) + [("oracle", oracle_baseline)]

        # Collect metrics across batches
        all_icl_err = []
        all_baseline_err = {name: [] for name, _ in per_genome_baselines}

        for _ in range(self.num_batches):
            if is_pipeline:
                xs = genome.sample_xs(self.n_points, self.batch_size)
            else:
                xs = sampler.sample_xs(self.n_points, self.batch_size)

            if torch.isnan(xs).any() or torch.isinf(xs).any():
                return EvalResult(
                    genome=genome, fitness=0.0,
                    icl_curve=np.zeros(self.n_points), is_valid=False,
                )

            # Generate ys using the adversary's task parameters
            task = get_task_sampler(
                self.task_name, self.n_dims, self.batch_size,
                pool_dict=pool_dict, noise_std=noise_std,
            )()
            ys = task.evaluate(xs)

            if torch.isnan(ys).any() or torch.isinf(ys).any():
                return EvalResult(
                    genome=genome, fitness=0.0,
                    icl_curve=np.zeros(self.n_points), is_valid=False,
                )

            # ICL model: single forward pass
            pred_icl = _eval_model_on_batch(self.icl_model, xs, ys, self.device)
            icl_err = ((pred_icl - ys) ** 2).mean(dim=0)  # (n_points,)
            all_icl_err.append(icl_err)

            # Baselines (including per-genome Bayes-optimal oracle)
            for name, baseline in per_genome_baselines:
                pred_bl = baseline(xs, ys)
                bl_err = ((pred_bl - ys) ** 2).mean(dim=0)
                all_baseline_err[name].append(bl_err)

        # Aggregate: mean over batches
        icl_curve = torch.stack(all_icl_err).mean(dim=0).numpy()
        baseline_curves = {}
        for name, err_list in all_baseline_err.items():
            baseline_curves[name] = torch.stack(err_list).mean(dim=0).numpy()

        # Best baseline at each point (element-wise minimum across all baselines)
        best_baseline = np.minimum.reduce(list(baseline_curves.values()))

        # --- Scale-invariant fitness: mean log-ratio in the underdetermined regime ---
        # Use log(ICL/baseline) to prevent a single extreme-ratio point from
        # dominating the fitness. This makes the adversary optimize for
        # *consistently* worse ICL across the learning curve, not just one
        # catastrophic point. We use k=1..n_dims (underdetermined regime).
        eps = 1e-8
        k_max = min(self.n_dims, len(icl_curve) - 1)

        # Compute log-ratio at each point in [1, k_max]
        log_ratios = []
        for k in range(1, k_max + 1):
            denom = max(best_baseline[k], eps)
            ratio = icl_curve[k] / denom
            log_ratios.append(np.log(max(ratio, eps)))

        if log_ratios:
            # Fitness = mean log-ratio (0 = ICL matches baseline, >0 = ICL worse)
            raw_fitness = max(float(np.mean(log_ratios)), 0.0)
        else:
            raw_fitness = 0.0

        # Parsimony pressure (pipeline genomes only)
        if is_pipeline and self.parsimony_lambda > 0:
            complexity = genome.complexity(self.c_base, self.c_stage, self.c_affine)
            fitness = raw_fitness - self.parsimony_lambda * complexity
            fitness = max(fitness, 0.0)
        else:
            complexity = 0.0
            fitness = raw_fitness

        # Descriptors for post-hoc analysis
        spectrum = genome.eigenvalues()
        descriptors = self._compute_descriptors(genome, icl_curve, best_baseline, spectrum)

        # Add pipeline-specific descriptors
        if is_pipeline:
            descriptors["base_distribution"] = genome.base_distribution_name()
            descriptors["num_active_stages"] = genome.num_active_stages()
            descriptors["active_transforms"] = [
                name for _, name in genome.active_stages()
            ]

        return EvalResult(
            genome=genome,
            fitness=fitness,
            raw_fitness=raw_fitness,
            complexity=complexity,
            icl_curve=icl_curve,
            baseline_curves=baseline_curves,
            covariance_spectrum=spectrum,
            descriptors=descriptors,
            is_valid=True,
        )

    def _compute_descriptors(
        self, genome, icl_curve: np.ndarray,
        baseline_curve: np.ndarray, spectrum: np.ndarray,
    ) -> dict:
        eps = 1e-8

        # Effective rank
        eff_rank = float(np.sum(spectrum) ** 2 / (np.sum(spectrum ** 2) + eps))

        # Condition number (log10)
        cond = float(spectrum[0] / (spectrum[-1] + eps))
        cond_log = float(np.log10(cond + 1))

        # Peak failure position (where in the learning curve the ratio is largest)
        ratio_curve = icl_curve / (baseline_curve + eps)
        peak_pos = float(np.argmax(ratio_curve) / max(len(ratio_curve) - 1, 1))

        # Weight-covariance alignment
        w = genome.decode_weights().numpy()
        if isinstance(genome, PipelineGenome):
            # Compute empirical covariance from eigenvalues (already computed)
            # Use spectrum directly; eigvec alignment requires full covariance
            # Estimate from a sample batch
            with torch.no_grad():
                xs_sample = genome.sample_xs(200, 1)
            if not torch.isnan(xs_sample).any():
                Sigma = np.cov(xs_sample[0].numpy(), rowvar=False)
            else:
                Sigma = np.eye(len(w))
        else:
            Sigma = genome.decode_covariance().numpy()
        eigvals_s, eigvecs = np.linalg.eigh(Sigma)
        top_eigvec = eigvecs[:, -1]
        alignment = float(np.abs(np.dot(w, top_eigvec)))

        # Spectral entropy
        p = spectrum / (spectrum.sum() + eps)
        spectral_entropy = float(-np.sum(p * np.log(p + eps)))

        # Noise-to-signal (w is unit, so this is just noise_std)
        noise_std = genome.decode_noise_std()

        return {
            "effective_rank": eff_rank,
            "condition_number_log": cond_log,
            "peak_failure_position": peak_pos,
            "weight_alignment": alignment,
            "spectral_entropy": spectral_entropy,
            "noise_std": noise_std,
        }

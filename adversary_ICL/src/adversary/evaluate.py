import numpy as np
import torch
from dataclasses import dataclass, field

from .genome import Genome

from ..icl.samplers import GaussianSampler
from ..icl.tasks import get_task_sampler
from ..icl import models


@dataclass
class EvalResult:
    genome: Genome
    fitness: float
    icl_curve: np.ndarray  # per-point ICL squared error, shape (n_points,)
    baseline_curves: dict = field(default_factory=dict)  # name -> per-point error
    covariance_spectrum: np.ndarray = field(default_factory=lambda: np.array([]))
    descriptors: dict = field(default_factory=dict)
    is_valid: bool = True


def _eval_model_on_batch(model, xs, ys, device):
    """Run model forward pass, return predictions."""
    with torch.no_grad():
        pred = model(xs.to(device), ys.to(device)).detach().cpu()
    return pred


class GenomeEvaluator:
    """Evaluates a Genome against the ICL model and baselines.

    Uses the train distribution for in-context examples. Evaluates predictions
    at each position using the standard forward pass (no xs_p loop, fast).
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
    ):
        self.icl_model = icl_model
        self.task_name = task_name
        self.n_dims = n_dims
        self.n_points = n_points
        self.batch_size = batch_size
        self.num_batches = num_batches

        if baseline_names is None:
            baseline_names = ["least_squares", "averaging"]
        self.baselines = self._build_baselines(baseline_names)

        # Get device from model
        if hasattr(icl_model, 'parameters'):
            self.device = next(icl_model.parameters()).device
        else:
            self.device = "cpu"

    def _build_baselines(self, names: list[str]) -> list:
        name_to_cls = {
            "least_squares": (models.LeastSquaresModel, {}),
            "averaging": (models.AveragingModel, {}),
            "nn_3": (models.NNModel, {"n_neighbors": 3}),
        }
        result = []
        for name in names:
            if name in name_to_cls:
                cls, kwargs = name_to_cls[name]
                result.append((name, cls(**kwargs)))
        return result

    def evaluate(self, genome: Genome) -> EvalResult:
        """Full evaluation: run ICL model + baselines on genome's distribution."""
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

    def _evaluate_inner(self, genome: Genome) -> EvalResult:
        # Decode genome
        L_train = genome.decode_L("L_train")
        mu_train = genome.decode_mu("mu_train")
        w = genome.decode_weights()
        noise_std = genome.decode_noise_std()

        # Build sampler from adversary's covariance
        sampler = GaussianSampler(self.n_dims, bias=mu_train, scale=L_train)

        # Build task with adversary-controlled weights
        # pool_dict needs shape (num_tasks, n_dims, 1)
        pool_dict = {"w": w.unsqueeze(0).unsqueeze(-1).expand(self.batch_size, -1, -1)}

        # Collect metrics across batches
        all_icl_err = []
        all_baseline_err = {name: [] for name, _ in self.baselines}

        for _ in range(self.num_batches):
            xs = sampler.sample_xs(self.n_points, self.batch_size)

            if torch.isnan(xs).any() or torch.isinf(xs).any():
                return EvalResult(
                    genome=genome, fitness=0.0,
                    icl_curve=np.zeros(self.n_points), is_valid=False,
                )

            # Generate ys using the adversary's task parameters
            task = get_task_sampler(
                "noisy_linear_regression", self.n_dims, self.batch_size,
                pool_dict=pool_dict, noise_std=noise_std,
            )()
            ys = task.evaluate(xs)

            if torch.isnan(ys).any() or torch.isinf(ys).any():
                return EvalResult(
                    genome=genome, fitness=0.0,
                    icl_curve=np.zeros(self.n_points), is_valid=False,
                )

            # ICL model: single forward pass (fast!)
            pred_icl = _eval_model_on_batch(self.icl_model, xs, ys, self.device)
            icl_err = ((pred_icl - ys) ** 2).mean(dim=0)  # (n_points,)
            all_icl_err.append(icl_err)

            # Baselines
            for name, baseline in self.baselines:
                pred_bl = baseline(xs, ys)
                bl_err = ((pred_bl - ys) ** 2).mean(dim=0)
                all_baseline_err[name].append(bl_err)

        # Aggregate: mean over batches
        icl_curve = torch.stack(all_icl_err).mean(dim=0).numpy()
        baseline_curves = {}
        for name, err_list in all_baseline_err.items():
            baseline_curves[name] = torch.stack(err_list).mean(dim=0).numpy()

        # Compute fitness: additive gap between ICL and best baseline
        # Use the gap at points where baseline has meaningful error (k < n_dims)
        # to avoid dividing by near-zero OLS error when the system is determined
        best_baseline = np.minimum.reduce(list(baseline_curves.values()))

        # Additive gap: how much worse ICL is in absolute terms
        gap_curve = icl_curve - best_baseline

        # Normalize by the scale of the problem (baseline error at k=1)
        scale = max(best_baseline[1] if len(best_baseline) > 1 else 1.0, 1e-6)
        normalized_gap = gap_curve / scale

        # Take mean gap over all points (captures sustained failure, not just spikes)
        mean_gap = float(normalized_gap.mean())

        # Also compute ratio but only where baseline has meaningful error
        meaningful_mask = best_baseline > 0.01 * scale
        if meaningful_mask.any():
            ratio_where_meaningful = (icl_curve[meaningful_mask] / best_baseline[meaningful_mask]).mean()
        else:
            ratio_where_meaningful = 1.0

        # Fitness = combination of normalized gap and ratio
        fitness = max(mean_gap, 0.0) + max(float(ratio_where_meaningful) - 1.0, 0.0)

        # Degeneracy penalty: penalize when the task is trivial or impossible
        baseline_at_1 = best_baseline[1] if len(best_baseline) > 1 else 1.0
        if baseline_at_1 < 1e-6:
            fitness *= 0.01  # trivial task
        elif baseline_at_1 > 1e4:
            fitness *= 1e4 / baseline_at_1  # impossible task

        # Descriptors
        spectrum = genome.eigenvalues("L_train")
        descriptors = self._compute_descriptors(genome, icl_curve, best_baseline, spectrum)

        return EvalResult(
            genome=genome,
            fitness=fitness,
            icl_curve=icl_curve,
            baseline_curves=baseline_curves,
            covariance_spectrum=spectrum,
            descriptors=descriptors,
            is_valid=True,
        )

    def _compute_descriptors(
        self, genome: Genome, icl_curve: np.ndarray,
        baseline_curve: np.ndarray, spectrum: np.ndarray,
    ) -> dict:
        eps = 1e-8

        eff_rank = float(np.sum(spectrum) ** 2 / (np.sum(spectrum ** 2) + eps))
        cond = float(spectrum[0] / (spectrum[-1] + eps))
        cond_log = float(np.log10(cond + 1))

        Sigma_train = genome.decode_covariance("L_train").numpy()
        Sigma_test = genome.decode_covariance("L_test").numpy()
        train_test_div = float(np.linalg.norm(Sigma_train - Sigma_test, "fro") / self.n_dims)

        gap = icl_curve / (baseline_curve + eps)
        peak_pos = float(np.argmax(gap) / max(len(gap) - 1, 1))

        w = genome.decode_weights().numpy()
        w_norm = w / (np.linalg.norm(w) + eps)
        eigvecs = np.linalg.eigh(Sigma_train)[1]
        top_eigvec = eigvecs[:, -1]
        alignment = float(np.abs(np.dot(w_norm, top_eigvec)))

        p = spectrum / (spectrum.sum() + eps)
        spectral_entropy = float(-np.sum(p * np.log(p + eps)))

        noise_std = genome.decode_noise_std()
        w_norm_val = float(np.linalg.norm(w))
        nsr = noise_std / (w_norm_val + eps)

        return {
            "effective_rank": eff_rank,
            "condition_number_log": cond_log,
            "train_test_divergence": train_test_div,
            "peak_failure_position": peak_pos,
            "weight_alignment": alignment,
            "spectral_entropy": spectral_entropy,
            "noise_to_signal": nsr,
        }

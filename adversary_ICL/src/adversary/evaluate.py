import numpy as np
import torch
from dataclasses import dataclass, field

from .genome import Genome

from ..icl.samplers import GaussianSampler
from ..icl.tasks import get_task_sampler
from ..icl.eval import eval_batch
from ..icl import models


@dataclass
class EvalResult:
    genome: Genome
    fitness: float
    icl_curve: np.ndarray  # per-point ICL squared error, shape (n_points,)
    baseline_curves: dict = field(default_factory=dict)  # name -> per-point error
    covariance_spectrum: np.ndarray = field(default_factory=lambda: np.array([]))
    descriptors: dict = field(default_factory=dict)  # named descriptor values
    is_valid: bool = True


class GenomeEvaluator:
    """Evaluates a Genome against the ICL model and baselines.

    Produces fitness = how much worse ICL is than the best baseline,
    plus per-point learning curves for analysis.
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

        # Set up baselines
        if baseline_names is None:
            baseline_names = ["least_squares", "averaging"]
        self.baselines = self._build_baselines(baseline_names)

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
        except Exception as e:
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
        L_test = genome.decode_L("L_test")
        mu_test = genome.decode_mu("mu_test")
        w = genome.decode_weights()
        noise_std = genome.decode_noise_std()

        # Build samplers
        train_sampler = GaussianSampler(self.n_dims, bias=mu_train, scale=L_train)
        test_sampler = GaussianSampler(self.n_dims, bias=mu_test, scale=L_test)

        # Build task sampler with adversary-controlled weights
        pool_dict = {"w": w.unsqueeze(0).unsqueeze(-1)}  # shape (1, d, 1)
        task_kwargs = {
            "noise_std": noise_std,
            "renormalize_ys": False,
        }

        def make_task_sampler():
            return get_task_sampler(
                "noisy_linear_regression",
                self.n_dims,
                self.batch_size,
                pool_dict=pool_dict,
                **task_kwargs,
            )

        # Collect metrics across batches
        all_icl_metrics = []
        all_baseline_metrics = {name: [] for name, _ in self.baselines}

        for _ in range(self.num_batches):
            xs_train = train_sampler.sample_xs(self.n_points, self.batch_size)
            xs_test = test_sampler.sample_xs(self.n_points, self.batch_size)

            # Check for NaN/Inf
            if torch.isnan(xs_train).any() or torch.isnan(xs_test).any():
                return EvalResult(
                    genome=genome, fitness=0.0,
                    icl_curve=np.zeros(self.n_points), is_valid=False,
                )

            task_sampler = make_task_sampler()

            # Evaluate ICL model (with separate train/test distributions)
            icl_metrics = eval_batch(self.icl_model, task_sampler, xs_train, xs_test)
            all_icl_metrics.append(icl_metrics)

            # Evaluate baselines on the same data
            # We need to replicate what eval_batch does for baselines
            for name, baseline_model in self.baselines:
                task_sampler_bl = make_task_sampler()
                bl_metrics = eval_batch(baseline_model, task_sampler_bl, xs_train, xs_test)
                all_baseline_metrics[name].append(bl_metrics)

        # Aggregate: mean over batches and batch items -> per-point curve
        icl_curve = torch.cat(all_icl_metrics, dim=0).mean(dim=0).numpy()
        baseline_curves = {}
        for name, metrics_list in all_baseline_metrics.items():
            baseline_curves[name] = torch.cat(metrics_list, dim=0).mean(dim=0).numpy()

        # Compute fitness: max ratio of ICL error to best baseline error
        best_baseline = np.minimum.reduce(list(baseline_curves.values()))
        eps = 1e-6
        ratio_curve = icl_curve / (best_baseline + eps)

        # Degeneracy penalty: penalize if baseline also fails badly
        # (meaning the task itself is too hard/trivial, not an ICL-specific failure)
        baseline_mean_err = best_baseline.mean()
        trivial_err = 1.0  # rough scale for "trivial" error
        # Penalty is low when baseline error is huge (task impossible) or tiny (task trivial)
        degeneracy_penalty = float(np.clip(
            np.minimum(baseline_mean_err / trivial_err, trivial_err / (baseline_mean_err + eps)),
            0.0, 1.0,
        ))

        fitness = float(ratio_curve.max()) * degeneracy_penalty

        # Compute descriptors for post-hoc analysis
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
        """Compute behavioral descriptors for post-hoc analysis."""
        eps = 1e-8

        # Effective rank
        eff_rank = float(np.sum(spectrum) ** 2 / (np.sum(spectrum ** 2) + eps))

        # Condition number
        cond = float(spectrum[0] / (spectrum[-1] + eps))
        cond_log = float(np.log10(cond + 1))

        # Train-test divergence (Frobenius distance between covariances)
        Sigma_train = genome.decode_covariance("L_train").numpy()
        Sigma_test = genome.decode_covariance("L_test").numpy()
        train_test_div = float(np.linalg.norm(Sigma_train - Sigma_test, "fro") / self.n_dims)

        # Peak failure position (where in the learning curve is the gap largest)
        gap = icl_curve / (baseline_curve + eps)
        peak_pos = float(np.argmax(gap) / max(len(gap) - 1, 1))

        # Weight-covariance alignment
        w = genome.decode_weights().numpy()
        w_norm = w / (np.linalg.norm(w) + eps)
        eigvecs = np.linalg.eigh(Sigma_train)[1]
        top_eigvec = eigvecs[:, -1]  # largest eigenvalue's eigenvector
        alignment = float(np.abs(np.dot(w_norm, top_eigvec)))

        # Spectral entropy
        p = spectrum / (spectrum.sum() + eps)
        spectral_entropy = float(-np.sum(p * np.log(p + eps)))

        # Noise-to-signal
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

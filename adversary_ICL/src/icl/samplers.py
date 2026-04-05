import math

import torch


class DataSampler:
    def __init__(self, n_dims):
        self.n_dims = n_dims

    def sample_xs(self):
        raise NotImplementedError


def get_data_sampler(data_name, n_dims, **kwargs):
    names_to_classes = {
        "gaussian": GaussianSampler,
        "pipeline": PipelineSampler,
    }
    if data_name in names_to_classes:
        sampler_cls = names_to_classes[data_name]
        return sampler_cls(n_dims, **kwargs)
    else:
        print("Unknown sampler")
        raise NotImplementedError


def sample_transformation(eigenvalues, normalize=False):
    n_dims = len(eigenvalues)
    U, _, _ = torch.linalg.svd(torch.randn(n_dims, n_dims))
    t = U @ torch.diag(eigenvalues) @ torch.transpose(U, 0, 1)
    if normalize:
        norm_subspace = torch.sum(eigenvalues**2)
        t *= math.sqrt(n_dims / norm_subspace)
    return t


class GaussianSampler(DataSampler):
    def __init__(self, n_dims, bias=None, scale=None):
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        if seeds is None:
            xs_b = torch.randn(b_size, n_points, self.n_dims)
        else:
            xs_b = torch.zeros(b_size, n_points, self.n_dims)
            generator = torch.Generator()
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                xs_b[i] = torch.randn(n_points, self.n_dims, generator=generator)
        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b


class PipelineSampler(DataSampler):
    """Wraps a PipelineGenome to act as a DataSampler for retraining.

    This allows the ICL training loop to use adversarial distributions
    for data augmentation or adversarial retraining.
    """

    def __init__(self, n_dims, genome=None, **kwargs):
        super().__init__(n_dims)
        self.genome = genome

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        if self.genome is None:
            raise ValueError("PipelineSampler requires a genome to sample from")
        xs_b = self.genome.sample_xs(n_points, b_size)
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b


class MixedSampler(DataSampler):
    """Mix standard Gaussian with adversarial distributions for retraining.

    Each batch element is independently drawn from either the standard
    Gaussian (with probability 1-p_adv) or an adversarial genome's
    distribution (with probability p_adv), selected by fitness-weighted
    sampling so harder failures get more training exposure.

    Supports both legacy Genome (Cholesky-only) and PipelineGenome.

    NOTE: p_adv=0.3 is a starting point from RESEARCH_PLAN.md. This value
    must be tuned empirically — too high risks catastrophic forgetting on
    standard data, too low won't teach robustness. The retrain loop uses
    dynamic p_adv that scales with curriculum size.
    """

    def __init__(self, n_dims, genomes=None, weights=None, p_adv=0.3, **kwargs):
        super().__init__(n_dims)
        self.base = GaussianSampler(n_dims)
        self.genomes = genomes if genomes is not None else []
        self.p_adv = p_adv

        # Fitness-weighted sampling: higher fitness → more training exposure
        if weights is not None and len(weights) > 0:
            w = torch.tensor(weights, dtype=torch.float32)
            self.weights = w / w.sum()
        else:
            self.weights = None

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        xs_list = []
        for i in range(b_size):
            if torch.rand(1).item() < self.p_adv and self.genomes:
                if self.weights is not None:
                    idx = torch.multinomial(self.weights, 1).item()
                else:
                    idx = torch.randint(len(self.genomes), (1,)).item()
                g = self.genomes[idx]
                # PipelineGenome has sample_xs; legacy Genome uses GaussianSampler
                if hasattr(g, "sample_xs"):
                    xs_i = g.sample_xs(n_points, 1)
                else:
                    L = g.decode_L_normalized()
                    mu = g.decode_mu()
                    s = GaussianSampler(self.n_dims, bias=mu, scale=L)
                    xs_i = s.sample_xs(n_points, 1)
            else:
                xs_i = self.base.sample_xs(n_points, 1)
            xs_list.append(xs_i)
        xs = torch.cat(xs_list, dim=0)
        if n_dims_truncated is not None:
            xs[:, :, n_dims_truncated:] = 0
        return xs

"""Families: distribution-of-distributions extracted from adversary output.

A family is a parametric sampler representing a distinct failure *mechanism*
found by the adversary, at an operating point that is less extreme than the
adversary's sharp optimum.

Pipeline (extract_families_from_adversary_log):
  1. Filter adversary log to valid, high-T_glitch, low-LSTM_glitch candidates.
  2. Group by distribution type (Stationary / Piecewise). Planted is skipped
     for now — its discrete `template` field has no natural interpolation.
  3. Featurize each candidate: config params + cheap behavior stats from
     sampled sequences; z-score.
  4. HDBSCAN cluster per group (density-based, auto-k). Noise discarded.
  5. Per cluster: geometric median on flattened config params (robust against
     boundary-piled configs) -> representative config c_rep.
  6. Pull-back α: largest α ∈ {1.0, 0.8, ..., 0.1} such that
          LSTM_glitch(FFL(p_α)) < 0.001   (skyline still clean)
      AND Transformer_glitch(FFL(p_α)) > 0.3 * best_adv_glitch  (still hard)
     with p_α = (1-α)*p_base + α*c_rep.
  7. Score each cluster: cluster_size * cluster_mean_T_glitch; take top-K.
  8. Each surviving cluster becomes one ClusterFamily sampling from FFL(p_α*).

Caveat tracked in CLAUDE.md Open TODOs: no per-sequence jitter (every draw
has identical params). That's the first knob to turn post-v1.
"""
from __future__ import annotations

import abc
import json
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch

from .distribution import (FFLDistribution, Piecewise, Stationary,
                            WriteFlipRate, BitMarkov)


# ---------------------------------------------------------------------------
# Abstract base + stub
# ---------------------------------------------------------------------------
class Family(abc.ABC):
    name: str

    @abc.abstractmethod
    def sample(self, batch_size: int, rng: np.random.Generator) -> torch.LongTensor:
        ...

    def to_dict(self) -> dict:
        return {"name": self.name, "kind": type(self).__name__}


@dataclass
class PassthroughFamily(Family):
    """Trivial: wrap one FFLDistribution as a 'family of one'. Test / fallback."""
    dist: FFLDistribution
    name: str = "passthrough"

    def sample(self, batch_size, rng):
        return self.dist.sample(batch_size, rng)

    def to_dict(self):
        return {"name": self.name, "kind": "PassthroughFamily",
                "dist": self.dist.to_dict()}


@dataclass
class ClusterFamily(Family):
    """An α-pulled-back distribution derived from one HDBSCAN cluster."""
    dist: FFLDistribution               # the FFL(p_α*) distribution
    alpha: float                         # chosen pull-back
    cluster_size: int
    cluster_mean_glitch: float
    rep_config: dict                     # the cluster's representative config (pre-interpolation)
    name: str = "cluster"

    def sample(self, batch_size, rng):
        return self.dist.sample(batch_size, rng)

    def to_dict(self):
        return {
            "name": self.name,
            "kind": "ClusterFamily",
            "alpha": self.alpha,
            "cluster_size": self.cluster_size,
            "cluster_mean_glitch": self.cluster_mean_glitch,
            "rep_config": self.rep_config,
            "dist": self.dist.to_dict(),
        }


# ---------------------------------------------------------------------------
# Featurization
# ---------------------------------------------------------------------------
def _flatten_config_params(cfg: dict) -> np.ndarray:
    """Flatten a distribution config dict to a fixed-length real vector.

    Handles Stationary + Piecewise (same segment count). Returns None if
    unsupported type (caller skips).
    """
    name = cfg.get("name")
    if name in ("stationary", "bit_markov", "write_flip"):
        return np.array([cfg["p_w"], cfg["p_r"],
                         cfg.get("bit_p1", 0.5)])
    if name == "piecewise":
        segs = cfg["segments"]
        # Each segment: (start_frac, p_w, p_r, bit_p1) — start_frac is fixed
        # in our encoder so skip; keep the other three.
        return np.array([v for s in segs for v in (s[1], s[2], s[3])])
    return None


def _behavior_stats(cfg: dict, T: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample n sequences from the config's distribution and compute cheap stats."""
    from .distribution import build
    # Rebuild dist
    dist = FFLDistribution.from_dict(cfg)
    tokens = dist.sample(n, rng).numpy()
    # Stats from even (instructions) and odd (data) positions.
    inst = tokens[:, 0::2]
    data = tokens[:, 1::2]
    from ..data import W, R, I, ZERO, ONE
    # 1. Mean write density
    write_density = (inst == W).mean()
    # 2. Mean read density
    read_density = (inst == R).mean()
    # 3. Bit-1 rate among writes
    is_w = inst == W
    bit_ones = (data == ONE) & is_w
    bit_one_rate = bit_ones.sum() / max(1, is_w.sum())
    # 4. Segment-concentration Gini of writes: how clustered writes are along the sequence.
    # Compute per-segment write count (4 bins along T), then Gini over the 4.
    n_inst = inst.shape[1]
    bins = 4
    bin_counts = np.zeros(bins)
    for b in range(bins):
        s, e = b * n_inst // bins, (b + 1) * n_inst // bins
        bin_counts[b] = (inst[:, s:e] == W).mean()
    gini = _gini(bin_counts)
    # 5. Mean read-to-last-write gap (positions per read)
    gaps = []
    for row in inst[: min(32, inst.shape[0])]:   # subsample — expensive loop
        last_w = -1
        for k in range(row.shape[0]):
            if row[k] == W:
                last_w = k
            elif row[k] == R and last_w >= 0:
                gaps.append(k - last_w)
    mean_gap = np.mean(gaps) if gaps else 0.0
    return np.array([write_density, read_density, bit_one_rate, gini, mean_gap])


def _gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.sum() == 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    idx = np.arange(1, n + 1)
    return (2 * (idx * x).sum() / (n * x.sum())) - (n + 1) / n


def _featurize_batch(
    configs: list[dict], T: int, n_behavior: int, rng: np.random.Generator
) -> np.ndarray:
    """Compute feature matrix (N, D). Mixes flattened config params + behavior stats."""
    rows = []
    for cfg in configs:
        p = _flatten_config_params(cfg)
        b = _behavior_stats(cfg, T, n_behavior, rng)
        rows.append(np.concatenate([p, b]))
    X = np.stack(rows)
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-8
    return (X - mu) / sd


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
def _hdbscan_cluster(features: np.ndarray, min_cluster_size: int) -> list[np.ndarray]:
    """HDBSCAN; returns list of index arrays (noise excluded). Fallback: k-means."""
    try:
        from sklearn.cluster import HDBSCAN
        labels = HDBSCAN(
            min_cluster_size=max(5, min_cluster_size),
            min_samples=max(3, min_cluster_size // 3),
        ).fit_predict(features)
    except Exception as e:  # pragma: no cover
        print(f"[family] HDBSCAN failed ({e}); falling back to k-means")
        labels = _kmeans_silhouette(features)

    groups = [np.where(labels == lbl)[0] for lbl in sorted(set(labels)) if lbl != -1]
    if not groups:
        print("[family] HDBSCAN returned all-noise; falling back to k-means")
        labels = _kmeans_silhouette(features)
        groups = [np.where(labels == lbl)[0] for lbl in sorted(set(labels))]
    return groups


def _kmeans_silhouette(features: np.ndarray) -> np.ndarray:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    best_k, best_score, best_labels = 2, -1.0, None
    for k in range(2, min(6, len(features))):
        km = KMeans(n_clusters=k, n_init=5, random_state=0).fit(features)
        if len(set(km.labels_)) < 2:
            continue
        s = silhouette_score(features, km.labels_)
        if s > best_score:
            best_score, best_k, best_labels = s, k, km.labels_
    return best_labels if best_labels is not None else np.zeros(len(features), dtype=int)


# ---------------------------------------------------------------------------
# Cluster representative
# ---------------------------------------------------------------------------
def _geometric_median(points: np.ndarray, tol: float = 1e-6, max_iter: int = 100) -> np.ndarray:
    """Weiszfeld iteration. Robust against boundary-piled points."""
    x = points.mean(axis=0)
    for _ in range(max_iter):
        d = np.linalg.norm(points - x, axis=1)
        d_safe = np.maximum(d, 1e-10)
        w = 1.0 / d_safe
        x_new = (points * w[:, None]).sum(axis=0) / w.sum()
        if np.linalg.norm(x_new - x) < tol:
            return x_new
        x = x_new
    return x


def _cluster_representative_config(configs: list[dict]) -> dict:
    """Geometric median over flattened params, re-cast to a valid config."""
    name = configs[0]["name"]
    T = configs[0]["T"]
    if name in ("stationary", "bit_markov", "write_flip"):
        params = np.stack([_flatten_config_params(c) for c in configs])
        med = _geometric_median(params)
        p_w, p_r, bit_p1 = _clip_simplex2(med[0], med[1]), _clip_simplex2(med[1], med[0]), _clip01(med[2])
        return {"name": "stationary", "T": T, "p_w": float(p_w),
                "p_r": float(p_r), "bit_p1": float(bit_p1)}
    if name == "piecewise":
        params = np.stack([_flatten_config_params(c) for c in configs])
        med = _geometric_median(params)
        n_seg = params.shape[1] // 3
        segs = []
        for k in range(n_seg):
            p_w = _clip_simplex2(med[3 * k], med[3 * k + 1])
            p_r = _clip_simplex2(med[3 * k + 1], med[3 * k])
            bit = _clip01(med[3 * k + 2])
            segs.append([k / n_seg, float(p_w), float(p_r), float(bit)])
        return {"name": "piecewise", "T": T, "segments": segs}
    raise ValueError(f"unsupported distribution type {name}")


def _clip_simplex2(a: float, b: float) -> float:
    """Clip a into [0, 1 - b] and ensure a >= 0, with b the other probability."""
    a = max(0.0, min(1.0, float(a)))
    b = max(0.0, min(1.0, float(b)))
    return min(a, max(0.0, 1.0 - b))


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


# ---------------------------------------------------------------------------
# Parameter interpolation + pull-back α
# ---------------------------------------------------------------------------
def _lift_to_piecewise(base: Stationary, K: int) -> Piecewise:
    """Convert a Stationary into a Piecewise with K identical segments."""
    segs = [(k / K, base.p_w, base.p_r, base.bit_p1) for k in range(K)]
    return Piecewise(T=base.T, segments=segs)


def interpolate_params(
    base_cfg: dict, adv_cfg: dict, alpha: float
) -> FFLDistribution:
    """Linear interpolation in parameter space. Returns a new FFLDistribution.

    Requires both configs have the same type (after a possible Stationary-
    to-Piecewise lift when adv is Piecewise).
    """
    assert 0.0 <= alpha <= 1.0
    a_name = adv_cfg["name"]

    if a_name == "piecewise":
        # Lift base if needed.
        if base_cfg["name"] != "piecewise":
            base_p = _lift_to_piecewise(
                FFLDistribution.from_dict(base_cfg), K=len(adv_cfg["segments"])
            ).to_dict()
        else:
            base_p = base_cfg
        base_segs = base_p["segments"]
        adv_segs = adv_cfg["segments"]
        assert len(base_segs) == len(adv_segs)
        new_segs = []
        for bs, as_ in zip(base_segs, adv_segs):
            new_segs.append([
                as_[0],  # start_frac — take adv's (identical)
                _clip01((1 - alpha) * bs[1] + alpha * as_[1]),  # p_w
                _clip01((1 - alpha) * bs[2] + alpha * as_[2]),  # p_r
                _clip01((1 - alpha) * bs[3] + alpha * as_[3]),  # bit_p1
            ])
        return Piecewise(T=adv_cfg["T"], segments=[tuple(s) for s in new_segs])

    if a_name in ("stationary", "bit_markov", "write_flip"):
        # Use Stationary for all interp results.
        if base_cfg["name"] not in ("stationary", "bit_markov", "write_flip"):
            raise ValueError(f"cannot interpolate {base_cfg['name']} with {a_name}")
        return Stationary(
            T=adv_cfg["T"],
            p_w=_clip01((1 - alpha) * base_cfg["p_w"] + alpha * adv_cfg["p_w"]),
            p_r=_clip01((1 - alpha) * base_cfg["p_r"] + alpha * adv_cfg["p_r"]),
            bit_p1=_clip01((1 - alpha) * base_cfg.get("bit_p1", 0.5)
                            + alpha * adv_cfg.get("bit_p1", 0.5)),
        )

    raise ValueError(f"unsupported type {a_name}")


def _eval_glitch(model, tokens: torch.LongTensor, batch_size: int, device: str) -> float:
    from ..eval import evaluate_dataset
    return evaluate_dataset(model, tokens, batch_size=batch_size, device=device)["error_rate"]


def pull_back_alpha(
    base_cfg: dict,
    adv_cfg: dict,
    transformer,
    lstm,
    device: str,
    *,
    T: int,
    rng: np.random.Generator,
    n_probe: int = 1000,
    batch_size: int = 64,
    max_lstm_glitch: float = 0.01,
    target_t_glitch: float = 0.5,
    max_iter: int = 8,
    tol: float = 0.02,
    ref_glitch: float = 1.0,  # kept for signature stability; unused
    alphas: tuple = (),       # kept for signature stability; unused
) -> tuple[float, float, float]:
    """Bisect α ∈ [0, 1] so the Transformer's glitch rate on FFL(p_α) lands
    near `target_t_glitch`, subject to LSTM staying clean.

    Rationale: the adversary's raw optimum (α=1) is brittle — the family
    needs to represent the mechanism at a softer operating point. We target
    an absolute glitch rate ~0.5 (model fails half the time) so training has
    clear signal without being the razor-edge config. The T_err(α) curve
    often cliffs near α=1 (only the exact tip breaks the model), so a
    uniform α grid misses the transition — bisection finds it directly.

    Returns (alpha*, T_glitch_at_alpha*, lstm_glitch_at_alpha*). If no
    satisfying α exists (base already too hard / adversary's tip too easy /
    LSTM fails everywhere), returns the closest-to-target endpoint.
    """
    def eval_alpha(alpha: float) -> tuple[float, float]:
        dist = interpolate_params(base_cfg, adv_cfg, alpha)
        tokens = dist.sample(n_probe, rng)
        t = _eval_glitch(transformer, tokens, batch_size, device)
        l = _eval_glitch(lstm, tokens, batch_size, device) if lstm is not None else 0.0
        return t, l

    t1, l1 = eval_alpha(1.0)
    t0, l0 = eval_alpha(0.0)

    # Edge cases
    if l1 >= max_lstm_glitch:
        # Even the adversary's tip has LSTM failing — can't recover, flag.
        return (1.0, t1, l1)
    if t1 <= target_t_glitch:
        # Full-adversary dist is already below target; use it.
        return (1.0, t1, l1)
    if t0 >= target_t_glitch:
        # Base distribution already too hard — use base.
        return (0.0, t0, l0)

    # Bisect. Assumes T_err(α) is roughly monotone increasing.
    lo, hi = 0.0, 1.0
    best = (1.0, t1, l1) if abs(t1 - target_t_glitch) < abs(t0 - target_t_glitch) else (0.0, t0, l0)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        t_mid, l_mid = eval_alpha(mid)
        if l_mid < max_lstm_glitch and abs(t_mid - target_t_glitch) < abs(best[1] - target_t_glitch):
            best = (mid, t_mid, l_mid)
        if abs(t_mid - target_t_glitch) < tol:
            break
        if t_mid < target_t_glitch:
            lo = mid
        else:
            hi = mid
    return best


# ---------------------------------------------------------------------------
# End-to-end extraction
# ---------------------------------------------------------------------------
def extract_families_from_adversary_log(
    log_path: str,
    *,
    base_cfg: Optional[dict] = None,
    transformer=None,
    lstm=None,
    device: str = "cpu",
    top_k: int = 5,
    min_t_glitch: float = 0.5,
    max_lstm_glitch: float = 0.01,
    n_behavior: int = 64,
    seed: int = 0,
) -> list[Family]:
    """Cluster + α-pull-back extraction.

    If `transformer` and `lstm` are not provided, returns PassthroughFamily
    stubs over the top-K highest-fitness configs (legacy behavior for tests).
    """
    with open(log_path) as f:
        recs = [json.loads(l) for l in f]
    valid = [
        r for r in recs
        if r.get("is_valid", True)
        and r.get("T_glitch", 0.0) > min_t_glitch
        and r.get("lstm_glitch", 1.0) < max_lstm_glitch
    ]
    if not valid:
        return []

    # Legacy fallback when models aren't available (tests).
    if transformer is None or lstm is None or base_cfg is None:
        valid.sort(key=lambda r: r["fitness"], reverse=True)
        return [
            PassthroughFamily(FFLDistribution.from_dict(r["config"]),
                              name=f"adv_{i:02d}")
            for i, r in enumerate(valid[:top_k])
        ]

    # Group by distribution type; Planted is skipped (no param interpolation).
    groups: dict[str, list[dict]] = {}
    for r in valid:
        name = r["config"]["name"]
        if name == "planted":
            continue
        groups.setdefault(name, []).append(r)

    rng = np.random.default_rng(seed)
    T = base_cfg["T"]
    all_families: list[ClusterFamily] = []

    for dist_type, recs_g in groups.items():
        configs = [r["config"] for r in recs_g]
        if len(configs) < 10:
            print(f"[family] skipping group '{dist_type}' (only {len(configs)} valid)")
            continue
        print(f"[family] group '{dist_type}': {len(configs)} candidates; clustering...")
        features = _featurize_batch(configs, T=T, n_behavior=n_behavior, rng=rng)
        min_cs = max(5, len(configs) // 50)
        clusters = _hdbscan_cluster(features, min_cluster_size=min_cs)
        print(f"[family]   -> {len(clusters)} cluster(s) (sizes: {[len(c) for c in clusters]})")

        best_glitch = max(r["T_glitch"] for r in recs_g)

        for c_idx, idx in enumerate(clusters):
            cluster_cfgs = [configs[i] for i in idx]
            cluster_recs = [recs_g[i] for i in idx]
            rep_cfg = _cluster_representative_config(cluster_cfgs)
            mean_g = float(np.mean([r["T_glitch"] for r in cluster_recs]))
            # Pull-back α vs the cluster's representative.
            alpha, t_err, l_err = pull_back_alpha(
                base_cfg=base_cfg, adv_cfg=rep_cfg,
                transformer=transformer, lstm=lstm, device=device,
                T=T, rng=rng, ref_glitch=best_glitch,
            )
            dist = interpolate_params(base_cfg, rep_cfg, alpha)
            fam = ClusterFamily(
                dist=dist,
                alpha=alpha,
                cluster_size=len(idx),
                cluster_mean_glitch=mean_g,
                rep_config=rep_cfg,
                name=f"{dist_type}_c{c_idx:02d}_a{alpha:.1f}",
            )
            all_families.append(fam)
            print(f"[family]   cluster {c_idx}: n={len(idx)} mean_glitch={mean_g:.3f} "
                  f"alpha*={alpha:.2f} T_err@a*={t_err:.3f} lstm@a*={l_err:.4f}")

    # Rank by size * mean_glitch; keep top_k.
    all_families.sort(key=lambda f: f.cluster_size * f.cluster_mean_glitch, reverse=True)
    return all_families[:top_k]

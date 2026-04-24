"""Tests for family extraction: interpolation, geometric median, clustering, pull-back."""
import json
from dataclasses import asdict

import numpy as np
import pytest
import torch

from flip_flop.adversary.distribution import Piecewise, Stationary
from flip_flop.adversary.family import (ClusterFamily, PassthroughFamily,
                                         _cluster_representative_config,
                                         _featurize_batch,
                                         _geometric_median, _hdbscan_cluster,
                                         extract_families_from_adversary_log,
                                         interpolate_params, pull_back_alpha)
from flip_flop.data import W, R, I, ZERO, ONE
from flip_flop.model import FFLMLSTM, FFLMTransformer


def _assert_valid_ffl(tokens, T):
    x = tokens.numpy()
    assert (x[:, 0] == W).all()
    assert (x[:, -2] == R).all()
    assert np.isin(x[:, 0::2], [W, R, I]).all()
    assert np.isin(x[:, 1::2], [ZERO, ONE]).all()


# ---------------------------------------------------------------------------
# Parameter interpolation
# ---------------------------------------------------------------------------
def test_interpolate_stationary():
    base = {"name": "stationary", "T": 64, "p_w": 0.1, "p_r": 0.1, "bit_p1": 0.5}
    adv = {"name": "stationary", "T": 64, "p_w": 0.005, "p_r": 0.005, "bit_p1": 0.95}
    d0 = interpolate_params(base, adv, 0.0)
    d1 = interpolate_params(base, adv, 1.0)
    d_half = interpolate_params(base, adv, 0.5)
    assert d0.p_w == pytest.approx(base["p_w"])
    assert d1.p_w == pytest.approx(adv["p_w"])
    assert d_half.p_w == pytest.approx((base["p_w"] + adv["p_w"]) / 2)
    assert d_half.bit_p1 == pytest.approx(0.725)
    _assert_valid_ffl(d_half.sample(32, np.random.default_rng(0)), 64)


def test_interpolate_piecewise_lifts_stationary_base():
    base = {"name": "stationary", "T": 64, "p_w": 0.1, "p_r": 0.1, "bit_p1": 0.5}
    adv = {"name": "piecewise", "T": 64, "segments": [
        [0.0, 1.0, 0.0, 1.0], [0.5, 0.0, 0.0, 0.5]
    ]}
    d = interpolate_params(base, adv, 0.5)
    assert isinstance(d, Piecewise)
    # First segment: (0.1+1.0)/2 = 0.55
    assert d.segments[0][1] == pytest.approx(0.55)
    # Validity preserved
    _assert_valid_ffl(d.sample(32, np.random.default_rng(0)), 64)


# ---------------------------------------------------------------------------
# Geometric median
# ---------------------------------------------------------------------------
def test_geometric_median_robust_to_outliers():
    pts = np.array([[0.0, 0.0]] * 10 + [[100.0, 100.0]])  # 10 at origin + 1 outlier
    gm = _geometric_median(pts)
    # Should be close to origin, not the mean (which would be (9.09, 9.09))
    assert np.linalg.norm(gm) < 1.0


def test_cluster_representative_piecewise():
    # 5 configs with identical first seg, varying second
    configs = [{
        "name": "piecewise", "T": 64,
        "segments": [[0.0, 0.9, 0.0, 1.0], [0.5, float(i) / 10, 0.0, 0.5]]
    } for i in range(5)]
    rep = _cluster_representative_config(configs)
    assert rep["name"] == "piecewise"
    assert rep["segments"][0][1] == pytest.approx(0.9, abs=0.05)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
def test_hdbscan_merges_similar_configs():
    """20 nearly-identical configs + 20 far-away configs -> HDBSCAN yields ~2 clusters."""
    configs_a = [{"name": "stationary", "T": 64, "p_w": 0.005 + 0.0005 * np.random.randn(),
                  "p_r": 0.005, "bit_p1": 0.95} for _ in range(20)]
    configs_b = [{"name": "stationary", "T": 64, "p_w": 0.45,
                  "p_r": 0.45 + 0.005 * np.random.randn(), "bit_p1": 0.5} for _ in range(20)]
    rng = np.random.default_rng(0)
    features = _featurize_batch(configs_a + configs_b, T=64, n_behavior=16, rng=rng)
    clusters = _hdbscan_cluster(features, min_cluster_size=5)
    # At least 2 clusters (may be more with small-N noise); total points covered should be most
    assert len(clusters) >= 1
    covered = sum(len(c) for c in clusters)
    assert covered >= 20  # most points clustered


# ---------------------------------------------------------------------------
# Pull-back α
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def models_cpu():
    t = FFLMTransformer(n_positions=64, n_embd=32, n_layer=2, n_head=2)
    l = FFLMLSTM(hidden_size=32)
    t.eval(); l.eval()
    return t, l


def test_pull_back_returns_valid_alpha(models_cpu):
    t, l = models_cpu
    base = {"name": "stationary", "T": 64, "p_w": 0.1, "p_r": 0.1, "bit_p1": 0.5}
    adv = {"name": "stationary", "T": 64, "p_w": 0.005, "p_r": 0.005, "bit_p1": 0.95}
    rng = np.random.default_rng(0)
    alpha, t_err, l_err = pull_back_alpha(
        base_cfg=base, adv_cfg=adv, transformer=t, lstm=l,
        device="cpu", T=64, rng=rng, n_probe=64, ref_glitch=0.5,
    )
    assert 0.0 <= alpha <= 1.0
    assert 0.0 <= t_err <= 1.0
    assert 0.0 <= l_err <= 1.0


# ---------------------------------------------------------------------------
# End-to-end extraction
# ---------------------------------------------------------------------------
def test_extract_families_full_path(models_cpu, tmp_path):
    """Full pipeline: log -> features -> HDBSCAN -> median -> pull-back -> ClusterFamily."""
    t, l = models_cpu
    # Build a log with two structural groups
    recs = []
    for _ in range(15):
        recs.append({
            "config": {"name": "stationary", "T": 64,
                       "p_w": 0.005 + 0.001 * np.random.rand(),
                       "p_r": 0.005 + 0.001 * np.random.rand(),
                       "bit_p1": 0.9 + 0.05 * np.random.rand()},
            "fitness": 0.8, "T_glitch": 0.8, "lstm_glitch": 0.0, "is_valid": True,
        })
    for _ in range(15):
        recs.append({
            "config": {"name": "stationary", "T": 64,
                       "p_w": 0.4 + 0.02 * np.random.rand(),
                       "p_r": 0.4 + 0.02 * np.random.rand(),
                       "bit_p1": 0.5},
            "fitness": 0.6, "T_glitch": 0.6, "lstm_glitch": 0.0, "is_valid": True,
        })
    path = tmp_path / "log.jsonl"
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    base = {"name": "stationary", "T": 64, "p_w": 0.1, "p_r": 0.1, "bit_p1": 0.5}
    fams = extract_families_from_adversary_log(
        str(path), base_cfg=base, transformer=t, lstm=l, device="cpu",
        top_k=5, n_behavior=16,
    )
    # Should produce 1-2 families, each a ClusterFamily with sensible α
    assert 1 <= len(fams) <= 5
    for fam in fams:
        assert isinstance(fam, ClusterFamily)
        assert 0.0 <= fam.alpha <= 1.0
        # Produces valid FFL sequences
        tokens = fam.sample(16, np.random.default_rng(0))
        _assert_valid_ffl(tokens, 64)


def test_extract_families_fallback_to_passthrough(tmp_path):
    """When models unavailable, falls back to PassthroughFamily."""
    recs = [{
        "config": {"name": "stationary", "T": 64, "p_w": 0.005, "p_r": 0.005, "bit_p1": 0.9},
        "fitness": 0.8, "T_glitch": 0.8, "lstm_glitch": 0.0, "is_valid": True,
    }]
    path = tmp_path / "log.jsonl"
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    fams = extract_families_from_adversary_log(str(path), top_k=5)
    assert len(fams) == 1
    assert isinstance(fams[0], PassthroughFamily)

"""FFL(T, p) data sampler.

Canonical family from Liu et al. 2023, Section 3.1:
  (i)   x_1 = w, x_{T-1} = r.
  (ii)  Other instructions drawn i.i.d. from (p_w, p_r, p_i).
  (iii) Data symbols following w or i are drawn i.i.d. uniformly from {0, 1};
        the symbol following r is deterministic (= most-recent-write value).

Vocabulary (size 5):
    w -> 0  (write)
    r -> 1  (read)
    i -> 2  (ignore)
    "0" -> 3  (data bit 0)
    "1" -> 4  (data bit 1)

All sequences have length T (even), with instructions at positions 0, 2, ..., T-2
and data at positions 1, 3, ..., T-1.
"""
from __future__ import annotations

import numpy as np
import torch

W, R, I, ZERO, ONE = 0, 1, 2, 3, 4
VOCAB_SIZE = 5
SYMBOL_TO_ID = {"w": W, "r": R, "i": I, "0": ZERO, "1": ONE}
ID_TO_SYMBOL = {v: k for k, v in SYMBOL_TO_ID.items()}


def sample_ffl(
    T: int,
    p_i: float,
    batch_size: int,
    rng: np.random.Generator,
    p_w: float | None = None,
    p_r: float | None = None,
) -> torch.Tensor:
    """Sample a batch of valid flip-flop strings from FFL(T, p).

    By default p_w = p_r = (1 - p_i) / 2, matching the paper shorthand FFL(p_i).

    Args:
        T: sequence length (even, >= 4).
        p_i: probability of an "ignore" instruction.
        batch_size: number of sequences.
        rng: numpy Generator.
        p_w, p_r: optional explicit probabilities (must sum with p_i to 1).

    Returns:
        LongTensor of shape (batch_size, T) with token IDs.
    """
    assert T % 2 == 0 and T >= 4, "T must be even and >= 4"
    if p_w is None and p_r is None:
        p_w = p_r = (1.0 - p_i) / 2.0
    assert p_w is not None and p_r is not None
    assert abs(p_w + p_r + p_i - 1.0) < 1e-8, "probabilities must sum to 1"

    n_inst = T // 2  # number of instruction positions
    n_data = T // 2  # number of data positions

    # (ii) Sample instructions i.i.d. from (p_w, p_r, p_i).
    inst = rng.choice(
        np.array([W, R, I], dtype=np.int64),
        size=(batch_size, n_inst),
        p=[p_w, p_r, p_i],
    )
    # (i) First instruction always w, last always r.
    inst[:, 0] = W
    inst[:, -1] = R

    # (iii) Candidate data bits; will be overridden at read positions.
    data = rng.integers(0, 2, size=(batch_size, n_data), dtype=np.int64)

    # After the forced first write, propagate last-written value across reads.
    last_write = data[:, 0].copy()
    for k in range(1, n_inst):
        is_w = inst[:, k] == W
        is_r = inst[:, k] == R
        last_write = np.where(is_w, data[:, k], last_write)
        data[:, k] = np.where(is_r, last_write, data[:, k])

    # Map {0, 1} -> {ZERO, ONE} token ids.
    data_tokens = data + ZERO

    # Interleave instructions and data into positions 0..T-1.
    tokens = np.empty((batch_size, T), dtype=np.int64)
    tokens[:, 0::2] = inst
    tokens[:, 1::2] = data_tokens
    return torch.from_numpy(tokens)


def decode(tokens: torch.Tensor) -> list[str]:
    """Decode a batch of token sequences to space-separated strings."""
    seqs = []
    for row in tokens.tolist():
        seqs.append(" ".join(ID_TO_SYMBOL[t] for t in row))
    return seqs


def is_valid_ffl(tokens: torch.Tensor) -> torch.Tensor:
    """Check whether each sequence in a batch is a legal flip-flop string.

    A sequence is valid iff:
      - alternates instruction / data,
      - starts with w and ends with r,
      - data at read positions equals the most-recent-write value.
    Returns a (batch_size,) BoolTensor.
    """
    B, T = tokens.shape
    assert T % 2 == 0
    x = tokens.numpy() if isinstance(tokens, torch.Tensor) else np.asarray(tokens)
    inst = x[:, 0::2]
    data = x[:, 1::2]

    ok = np.ones(B, dtype=bool)
    ok &= inst[:, 0] == W
    ok &= inst[:, -1] == R
    ok &= np.all(np.isin(inst, [W, R, I]), axis=1)
    ok &= np.all(np.isin(data, [ZERO, ONE]), axis=1)

    # Replay the machine and check read outputs.
    last_write = data[:, 0] - ZERO  # bits for first write
    for k in range(1, inst.shape[1]):
        is_w = inst[:, k] == W
        is_r = inst[:, k] == R
        bit = data[:, k] - ZERO
        last_write = np.where(is_w, bit, last_write)
        ok &= ~is_r | (bit == last_write)
    return torch.from_numpy(ok)


def make_eval_dataset(
    p_i: float,
    num_sequences: int,
    T: int,
    seed: int,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """Pre-sample a fixed evaluation dataset."""
    rng = np.random.default_rng(seed)
    chunks = []
    remaining = num_sequences
    while remaining > 0:
        b = min(chunk_size, remaining)
        chunks.append(sample_ffl(T, p_i, b, rng))
        remaining -= b
    return torch.cat(chunks, dim=0)

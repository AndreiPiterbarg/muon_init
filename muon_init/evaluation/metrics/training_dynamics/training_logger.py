"""Standardized training logger for recording metrics during training runs.

Provides the glue between training code and analysis metrics (convergence,
warmup sensitivity, HP robustness).  Training loops call ``log()`` at each
step; analysis code consumes the result via ``get_log()`` or
``to_dataframe()``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class TrainingLogger:
    """Accumulates (step, value) pairs for arbitrary named metrics.

    Parameters
    ----------
    metadata : dict, optional
        Arbitrary key/value pairs stored alongside the logs (e.g. seed,
        init scheme, warmup steps).

    Examples
    --------
    >>> logger = TrainingLogger(metadata={"seed": 42})
    >>> logger.log(0, train_loss=2.3, val_loss=2.5)
    >>> logger.log(100, train_loss=1.8, val_loss=2.0)
    >>> logger.get_log("val_loss")
    [(0, 2.5), (100, 2.0)]
    """

    metadata: Dict[str, object] = field(default_factory=dict)
    _logs: Dict[str, List[Tuple[int, float]]] = field(
        default_factory=lambda: defaultdict(list), repr=False
    )

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def log(self, step: int, **metrics: float) -> None:
        """Record one or more metric values at *step*.

        Parameters
        ----------
        step : int
            Training step (or epoch) index.
        **metrics : float
            Named metric values, e.g. ``train_loss=1.5, val_loss=1.7``.
        """
        for name, value in metrics.items():
            self._logs[name].append((step, float(value)))

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    @property
    def metric_names(self) -> List[str]:
        """Return the names of all recorded metrics."""
        return list(self._logs.keys())

    def get_log(self, metric_name: str) -> List[Tuple[int, float]]:
        """Return the full ``(step, value)`` series for *metric_name*.

        Raises ``KeyError`` if the metric was never logged.
        """
        if metric_name not in self._logs:
            raise KeyError(
                f"Metric '{metric_name}' not found. "
                f"Available: {self.metric_names}"
            )
        return list(self._logs[metric_name])

    def get_value(self, metric_name: str, step: int) -> Optional[float]:
        """Return the value of *metric_name* at *step*, or ``None``."""
        for s, v in self._logs.get(metric_name, []):
            if s == step:
                return v
        return None

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all logs into a single long-form DataFrame.

        Columns: ``step``, ``metric``, ``value``.
        """
        rows = []
        for metric, pairs in self._logs.items():
            for step, value in pairs:
                rows.append({"step": step, "metric": metric, "value": value})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist to JSON."""
        payload = {
            "metadata": self.metadata,
            "logs": {k: list(v) for k, v in self._logs.items()},
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "TrainingLogger":
        """Load a previously saved logger from JSON."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        logger = cls(metadata=raw.get("metadata", {}))
        for metric, pairs in raw.get("logs", {}).items():
            logger._logs[metric] = [(int(s), float(v)) for s, v in pairs]
        return logger

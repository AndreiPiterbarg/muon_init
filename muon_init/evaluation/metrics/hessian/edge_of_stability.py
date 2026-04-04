"""Edge of Stability (EoS) tracker.

Tracks the product ``eta * lambda_max`` over training to detect the edge-of-
stability regime identified by Cohen, Kaplan & Singer (ICLR 2021).

In full-batch gradient descent at a fixed learning rate ``eta``, lambda_max
initially increases (progressive sharpening) until ``eta * lambda_max``
reaches 2 — the stability threshold for a quadratic.  Training then enters
the *edge of stability*: lambda_max oscillates around ``2 / eta``, loss is
non-monotonic short-term but still decreases long-term.

Usage::

    tracker = EoSTracker(learning_rate=0.01)

    for step, (x, y) in enumerate(train_loader):
        ...
        if step % log_every == 0:
            lam = compute_lambda_max(model, loss_fn, eval_loader)
            tracker.step(step, lam)

    print(tracker.eos_onset_step)
    print(tracker.sharpening_rate())

Reference:
    Cohen, J., Kaplan, S., & Singer, Y. (ICLR 2021).
    "Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability."
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EoSTracker:
    """Track ``eta * lambda_max`` over training and detect EoS onset.

    Parameters
    ----------
    learning_rate : float
        The (fixed or current) optimizer learning rate ``eta``.
    eos_threshold : float
        Value of ``eta * lambda_max`` considered to be at the edge of
        stability.  Default 2.0 (classical quadratic threshold).
    """

    learning_rate: float
    eos_threshold: float = 2.0

    # Trajectory storage
    steps: list[int] = field(default_factory=list)
    lambda_maxs: list[float] = field(default_factory=list)
    eta_lambda_maxs: list[float] = field(default_factory=list)

    # Detected onset step (None until detected)
    eos_onset_step: int | None = field(default=None, init=False)

    def step(self, training_step: int, lambda_max: float) -> None:
        """Record a measurement.

        Parameters
        ----------
        training_step : int
            Current training step number.
        lambda_max : float
            Hessian top eigenvalue at this step.
        """
        ratio = self.learning_rate * lambda_max
        self.steps.append(training_step)
        self.lambda_maxs.append(lambda_max)
        self.eta_lambda_maxs.append(ratio)

        # Detect first crossing of the EoS threshold
        if self.eos_onset_step is None and ratio >= self.eos_threshold:
            self.eos_onset_step = training_step

    def update_learning_rate(self, lr: float) -> None:
        """Update the learning rate (for schedules / warmup)."""
        self.learning_rate = lr

    def sharpening_rate(self, window: int | None = None) -> float:
        """Estimate progressive sharpening rate d(lambda_max)/dt.

        Fits a linear regression to the lambda_max trajectory in early
        training (before EoS onset, or over the first ``window`` entries).

        Parameters
        ----------
        window : int, optional
            Number of initial measurements to use.  Defaults to all
            measurements before EoS onset, or all measurements if EoS
            was never reached.

        Returns
        -------
        float
            Slope of lambda_max vs. training step (units: curvature / step).
        """
        if len(self.steps) < 2:
            return 0.0

        if window is not None:
            n = min(window, len(self.steps))
        elif self.eos_onset_step is not None:
            # Use everything before onset
            n = next(
                (i for i, s in enumerate(self.steps) if s >= self.eos_onset_step),
                len(self.steps),
            )
            n = max(n, 2)  # need at least 2 points
        else:
            n = len(self.steps)

        t = np.array(self.steps[:n], dtype=np.float64)
        lam = np.array(self.lambda_maxs[:n], dtype=np.float64)

        # Simple linear regression: slope = Cov(t, lam) / Var(t)
        t_mean = t.mean()
        lam_mean = lam.mean()
        var_t = ((t - t_mean) ** 2).sum()
        if var_t < 1e-12:
            return 0.0
        slope = ((t - t_mean) * (lam - lam_mean)).sum() / var_t
        return float(slope)

    def is_at_eos(self) -> bool:
        """Whether the most recent measurement is at or above the EoS threshold."""
        if not self.eta_lambda_maxs:
            return False
        return self.eta_lambda_maxs[-1] >= self.eos_threshold

    def trajectory_as_dict(self) -> dict:
        """Return the full trajectory as a dict (easy to serialize / plot)."""
        return {
            "steps": list(self.steps),
            "lambda_max": list(self.lambda_maxs),
            "eta_lambda_max": list(self.eta_lambda_maxs),
            "eos_onset_step": self.eos_onset_step,
            "sharpening_rate": self.sharpening_rate(),
        }

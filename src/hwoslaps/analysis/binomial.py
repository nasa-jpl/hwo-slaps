"""Exact two-sided binomial intervals for independent trials at one probability.

Directions of the same system are clustered. Their pooled interval is a nominal
summary; a population statement counts one outcome per independently drawn system.
"""

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from scipy.stats import beta

__all__ = ["BinomialCount", "clopper_pearson"]


@dataclass(frozen=True)
class BinomialCount:
    """Successes among independent trials with one success probability."""

    count: int
    trials: int

    def __post_init__(self) -> None:
        for name in ("count", "trials"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer, not boolean, got {value!r}")
        if self.trials < 1 or not 0 <= self.count <= self.trials:
            raise ValueError(f"count must lie in [0, trials] and trials >= 1, got {self.count} of {self.trials}")


def clopper_pearson(count: int, trials: int, *, confidence: float) -> tuple[float, float]:
    """Return the exact two-sided interval under the independent binomial model.

    Pooled directions from one system do not provide independent population trials.
    ``confidence`` is required and lies strictly between zero and one.
    """
    sample = BinomialCount(count, trials)
    if isinstance(confidence, (bool, np.bool_)) or not isinstance(confidence, Real) \
            or not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError(f"confidence must be a finite number in (0, 1), not boolean, got {confidence!r}")
    alpha = 1.0 - float(confidence)
    lower = 0.0 if sample.count == 0 else float(beta.ppf(alpha / 2, sample.count, sample.trials - sample.count + 1))
    upper = 1.0 if sample.count == sample.trials else float(beta.ppf(1 - alpha / 2, sample.count + 1,
                                                                sample.trials - sample.count))
    return lower, upper

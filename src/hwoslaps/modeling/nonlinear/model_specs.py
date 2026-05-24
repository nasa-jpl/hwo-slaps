"""PyAutoLens-neutral model specifications for validation tests."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class PriorSpec:
    """Source-neutral prior description.

    Parameters
    ----------
    kind : `str`
        Prior kind: ``"fixed"``, ``"uniform"``, or ``"log_uniform"``.
    value : `object`, optional
        Fixed value.
    lower : `float`, optional
        Lower limit for ranged priors.
    upper : `float`, optional
        Upper limit for ranged priors.
    """

    kind: str
    value: Any = None
    lower: Optional[float] = None
    upper: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the prior specification to a dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class ProfileSpec:
    """Source-neutral mass or light profile description."""

    class_name: str
    parameters: Dict[str, PriorSpec]

    def to_dict(self) -> Dict[str, Any]:
        """Convert the profile specification to a dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class GalaxySpec:
    """Source-neutral galaxy description."""

    name: str
    redshift: PriorSpec
    components: Dict[str, ProfileSpec]

    def to_dict(self) -> Dict[str, Any]:
        """Convert the galaxy specification to a dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class ModelSpec:
    """Source-neutral PyAutoFit model description."""

    model_type: str
    galaxies: Dict[str, GalaxySpec]
    fit_mode: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the model specification to a dictionary."""
        return asdict(self)


def fixed(value: Any) -> PriorSpec:
    """Return a fixed-value prior specification."""
    return PriorSpec(kind="fixed", value=value)


def uniform(lower: float, upper: float) -> PriorSpec:
    """Return a uniform-prior specification."""
    if lower >= upper:
        raise ValueError("uniform prior lower limit must be less than upper limit")
    return PriorSpec(kind="uniform", lower=float(lower), upper=float(upper))


def log_uniform(lower: float, upper: float) -> PriorSpec:
    """Return a log-uniform prior specification."""
    if lower <= 0.0 or lower >= upper:
        raise ValueError("log_uniform prior limits must satisfy 0 < lower < upper")
    return PriorSpec(kind="log_uniform", lower=float(lower), upper=float(upper))

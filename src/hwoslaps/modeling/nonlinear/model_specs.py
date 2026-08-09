"""PyAutoLens-neutral model specifications for validation tests."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class PriorSpec:
    """Source-neutral prior description.

    Parameters
    ----------
    kind : `str`
        Prior kind: ``"fixed"``, ``"uniform"``, ``"log_uniform"``, or
        ``"linked"``.
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
        """Convert the prior specification to a dictionary.

        Returns
        -------
        data : `dict`
            JSON-safe prior specification.
        """
        return _spec_value(self)


@dataclass(frozen=True)
class ProfileSpec:
    """Source-neutral mass or light profile description.

    Parameters
    ----------
    class_name : `str`
        Supported fit-profile class name.
    parameters : `dict`
        Constructor parameter prior specifications.
    """

    class_name: str
    parameters: Dict[str, PriorSpec]

    def to_dict(self) -> Dict[str, Any]:
        """Convert the profile specification to a dictionary.

        Returns
        -------
        data : `dict`
            JSON-safe profile specification.
        """
        return _spec_value(self)


@dataclass(frozen=True)
class GalaxySpec:
    """Source-neutral galaxy description.

    Parameters
    ----------
    name : `str`
        Galaxy component name.
    redshift : `PriorSpec`
        Fixed or free galaxy redshift specification.
    components : `dict`
        Named mass and light profile specifications.
    """

    name: str
    redshift: PriorSpec
    components: Dict[str, ProfileSpec]

    def to_dict(self) -> Dict[str, Any]:
        """Convert the galaxy specification to a dictionary.

        Returns
        -------
        data : `dict`
            JSON-safe galaxy specification.
        """
        return _spec_value(self)


@dataclass(frozen=True)
class ModelSpec:
    """Source-neutral PyAutoFit model description.

    Parameters
    ----------
    model_type : `str`
        Model role, such as ``"smooth"`` or ``"subhalo"``.
    galaxies : `dict`
        Named galaxy specifications.
    fit_mode : `str`
        Nonlinear fitting mode.
    metadata : `dict`, optional
        Model identity and profile provenance.
    """

    model_type: str
    galaxies: Dict[str, GalaxySpec]
    fit_mode: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the model specification to a dictionary.

        Fixed objects carrying a provenance hash are represented by a short
        string so array-backed assets and interpolation tables never enter
        serialized model payloads.

        Returns
        -------
        data : `dict`
            JSON-safe model specification.
        """
        return _spec_value(self)


def _spec_value(value: Any) -> Any:
    """Convert specification values while abbreviating hashed contexts."""
    if hasattr(value, "sha256_16"):
        return (
            f"{value.__class__.__name__}(sha256_16="
            f"{value.sha256_16})"
        )
    if hasattr(value, "context_hash"):
        return (
            f"{value.__class__.__name__}(context_hash="
            f"{value.context_hash})"
        )
    if is_dataclass(value):
        return {
            item.name: _spec_value(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, dict):
        return {key: _spec_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_spec_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_spec_value(item) for item in value)
    return value


def fixed(value: Any) -> PriorSpec:
    """Return a fixed-value prior specification.

    Parameters
    ----------
    value : `object`
        Fixed constructor value.

    Returns
    -------
    prior : `PriorSpec`
        Fixed prior specification.
    """
    return PriorSpec(kind="fixed", value=value)


def uniform(lower: float, upper: float) -> PriorSpec:
    """Return a uniform-prior specification.

    Parameters
    ----------
    lower : `float`
        Lower prior limit.
    upper : `float`
        Upper prior limit.

    Returns
    -------
    prior : `PriorSpec`
        Uniform prior specification.
    """
    if lower >= upper:
        raise ValueError("uniform prior lower limit must be less than upper limit")
    return PriorSpec(kind="uniform", lower=float(lower), upper=float(upper))


def log_uniform(lower: float, upper: float) -> PriorSpec:
    """Return a log-uniform prior specification.

    Parameters
    ----------
    lower : `float`
        Positive lower prior limit.
    upper : `float`
        Upper prior limit.

    Returns
    -------
    prior : `PriorSpec`
        Log-uniform prior specification.
    """
    if lower <= 0.0 or lower >= upper:
        raise ValueError("log_uniform prior limits must satisfy 0 < lower < upper")
    return PriorSpec(kind="log_uniform", lower=float(lower), upper=float(upper))


def linked(component: str, parameter: str) -> PriorSpec:
    """Return a same-galaxy linked-prior specification.

    Parameters
    ----------
    component : `str`
        Referenced component name in the same galaxy.
    parameter : `str`
        Referenced parameter name.

    Returns
    -------
    prior : `PriorSpec`
        Linked prior specification.
    """
    if not component or not parameter:
        raise ValueError("linked prior references must be non-empty")
    if "." in component:
        raise ValueError("linked priors cannot reference another galaxy")
    return PriorSpec(kind="linked", value=(component, parameter))

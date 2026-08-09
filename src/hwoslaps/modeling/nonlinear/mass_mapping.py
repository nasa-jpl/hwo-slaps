"""Mass-to-profile mappings for freed nonlinear subhalo fits."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np
from scipy.interpolate import PchipInterpolator

from ...lensing.mass_models import (
    concentration_mass_relation,
    einstein_radius_point_mass,
    einstein_radius_sis_m200,
    nfw_lensing_parameters,
)


_PROFILE_CLASS_NAMES = {
    "NFWMCRSubhaloSph",
    "SISMCRSubhalo",
    "PointMassMCRSubhalo",
}
_INTERPOLATOR_CACHE: Dict[
    str,
    Tuple[Any, Tuple[PchipInterpolator, ...]],
] = {}

if TYPE_CHECKING:
    NFWMCRSubhaloSph: Any
    PointMassMCRSubhalo: Any
    SISMCRSubhalo: Any


@dataclass(frozen=True)
class MassMappingContext:
    """Immutable mass-to-profile conversion context.

    Parameters
    ----------
    subhalo_model : `str`
        HWO-SLAPS subhalo model label.
    concentration_model : `str`, optional
        NFW concentration relation.
    x_sub : `float`, optional
        Host-centric radius for the Moline relation.
    h : `float`, optional
        Resolved reduced Hubble parameter for the Moline relation.
    z_lens : `float`
        Lens-plane redshift.
    z_source : `float`
        Source-plane redshift.
    cosmology_name : `str`
        Configured cosmology label.
    log10_m200_lower : `float`
        Lower log-mass table boundary.
    log10_m200_upper : `float`
        Upper log-mass table boundary.
    table_log10_m200 : `tuple` [`float`, ...]
        Uniform log-mass interpolation nodes.
    table_kappa_s : `tuple` [`float`, ...], optional
        NFW scale-convergence table.
    table_scale_radius_arcsec : `tuple` [`float`, ...], optional
        NFW angular scale-radius table.
    table_einstein_radius_arcsec : `tuple` [`float`, ...], optional
        SIS or point-mass Einstein-radius table.
    context_hash : `str`
        First 16 hexadecimal characters of the canonical context SHA-256.
    """

    subhalo_model: str
    concentration_model: Optional[str]
    x_sub: Optional[float]
    h: Optional[float]
    z_lens: float
    z_source: float
    cosmology_name: str
    log10_m200_lower: float
    log10_m200_upper: float
    table_log10_m200: Tuple[float, ...]
    table_kappa_s: Optional[Tuple[float, ...]]
    table_scale_radius_arcsec: Optional[Tuple[float, ...]]
    table_einstein_radius_arcsec: Optional[Tuple[float, ...]]
    context_hash: str


def _cosmology_from_name(cosmology_name: str) -> Any:
    """Return the supported PyAutoLens cosmology object."""
    import autolens as al

    if cosmology_name != "Planck15":
        raise ValueError(f"Unsupported cosmology: {cosmology_name}")
    return al.cosmo.Planck15()


def _infer_reduced_h(cosmology: Any) -> float:
    """Infer the reduced Hubble parameter from a cosmology object."""
    if hasattr(cosmology, "H"):
        value = cosmology.H(0.0)
        value = value.value if hasattr(value, "value") else value
        value = float(value)
        if np.isfinite(value) and value > 0.0:
            return value / 100.0
    if hasattr(cosmology, "H0"):
        value = cosmology.H0
        value = value.value if hasattr(value, "value") else value
        value = float(value)
        if np.isfinite(value) and value > 0.0:
            return value / 100.0
    return 0.6774


def _concentration(
    mass_msun: float,
    concentration_model: str,
    x_sub: Optional[float],
    h: Optional[float],
    z_lens: float,
) -> float:
    """Evaluate the configured concentration relation."""
    if concentration_model == "moline2017_eq7":
        return concentration_mass_relation(
            mass_msun,
            model=concentration_model,
            x_sub=x_sub,
            h=h,
        )
    return concentration_mass_relation(
        mass_msun,
        model=concentration_model,
        z=z_lens,
    )


def _direct_mapping_values(
    *,
    subhalo_model: str,
    concentration_model: Optional[str],
    x_sub: Optional[float],
    h: Optional[float],
    z_lens: float,
    z_source: float,
    cosmology: Any,
    log10_m200: np.ndarray,
) -> Tuple[np.ndarray, ...]:
    """Evaluate direct mass conversions on a log-mass grid."""
    outputs = []
    for log_mass in log10_m200:
        mass_msun = 10.0 ** float(log_mass)
        if subhalo_model == "NFW":
            c200 = _concentration(
                mass_msun,
                concentration_model,
                x_sub,
                h,
                z_lens,
            )
            outputs.append(
                nfw_lensing_parameters(
                    mass_msun,
                    c200,
                    z_lens,
                    z_source,
                    cosmology,
                )
            )
        elif subhalo_model == "SIS":
            outputs.append(
                (
                    einstein_radius_sis_m200(
                        mass_msun,
                        z_lens,
                        z_source,
                        cosmology,
                    ),
                )
            )
        else:
            outputs.append(
                (
                    einstein_radius_point_mass(
                        mass_msun,
                        z_lens,
                        z_source,
                        cosmology,
                    ),
                )
            )
    return tuple(
        np.asarray(values, dtype=float)
        for values in zip(*outputs)
    )


def _canonical_context_hash(fields: Tuple[Any, ...]) -> str:
    """Return the short SHA-256 of canonical context fields."""
    return hashlib.sha256(repr(fields).encode("utf-8")).hexdigest()[:16]


@lru_cache(maxsize=None)
def _build_context(
    subhalo_model: str,
    concentration_model: Optional[str],
    x_sub: Optional[float],
    h: Optional[float],
    z_lens: float,
    z_source: float,
    cosmology_name: str,
    log10_m200_lower: float,
    log10_m200_upper: float,
) -> MassMappingContext:
    """Construct and validate one cached mapping context."""
    cosmology = _cosmology_from_name(cosmology_name)
    node_count = 2049
    maximum_nodes = 32769
    tolerance = 1.0e-11

    while True:
        nodes = np.linspace(
            log10_m200_lower,
            log10_m200_upper,
            node_count,
        )
        table_outputs = _direct_mapping_values(
            subhalo_model=subhalo_model,
            concentration_model=concentration_model,
            x_sub=x_sub,
            h=h,
            z_lens=z_lens,
            z_source=z_source,
            cosmology=cosmology,
            log10_m200=nodes,
        )
        probes = np.linspace(
            log10_m200_lower,
            log10_m200_upper,
            4 * node_count,
        )
        direct_outputs = _direct_mapping_values(
            subhalo_model=subhalo_model,
            concentration_model=concentration_model,
            x_sub=x_sub,
            h=h,
            z_lens=z_lens,
            z_source=z_source,
            cosmology=cosmology,
            log10_m200=probes,
        )
        errors = []
        for table, direct in zip(table_outputs, direct_outputs):
            interpolated = PchipInterpolator(nodes, table)(probes)
            errors.append(
                float(np.max(np.abs(interpolated - direct) / np.abs(direct)))
            )
        if max(errors) <= tolerance:
            break
        if node_count == maximum_nodes:
            raise ValueError(
                "Mass-mapping interpolation could not meet the 1e-11 "
                "relative-error contract"
            )
        node_count = min(2 * node_count, maximum_nodes)

    table_log10_m200 = tuple(float(value) for value in nodes)
    if subhalo_model == "NFW":
        table_kappa_s = tuple(float(value) for value in table_outputs[0])
        table_scale_radius_arcsec = tuple(
            float(value) for value in table_outputs[1]
        )
        table_einstein_radius_arcsec = None
    else:
        table_kappa_s = None
        table_scale_radius_arcsec = None
        table_einstein_radius_arcsec = tuple(
            float(value) for value in table_outputs[0]
        )

    fields = (
        subhalo_model,
        concentration_model,
        x_sub,
        h,
        z_lens,
        z_source,
        cosmology_name,
        log10_m200_lower,
        log10_m200_upper,
        table_log10_m200,
        table_kappa_s,
        table_scale_radius_arcsec,
        table_einstein_radius_arcsec,
    )
    return MassMappingContext(*fields, context_hash=_canonical_context_hash(fields))


def build_mass_mapping_context(
    full_config: Dict[str, Any],
    log10_m200_range: Tuple[float, float] = (6.0, 8.5),
) -> MassMappingContext:
    """Build a mass-mapping context from a truth configuration.

    Parameters
    ----------
    full_config : `dict`
        Full HWO-SLAPS configuration with an enabled subhalo block.
    log10_m200_range : `tuple` [`float`, `float`], optional
        Closed log10 M200 interpolation range.

    Returns
    -------
    context : `MassMappingContext`
        Resolved mass conversion context.

    Raises
    ------
    ValueError
        Raised when no enabled truth subhalo is available.
    """
    lensing = full_config.get("lensing", {})
    subhalo = lensing.get("subhalo")
    if not isinstance(subhalo, dict) or not subhalo.get("enabled"):
        raise ValueError(
            "An enabled lensing.subhalo block is required; use "
            "build_mass_mapping_context_explicit for control data"
        )
    concentration = subhalo.get("concentration") or {}
    return build_mass_mapping_context_explicit(
        subhalo_model=subhalo.get("model"),
        concentration_model=concentration.get("model"),
        x_sub=concentration.get("x_sub"),
        h=concentration.get("h"),
        z_lens=lensing["lens_galaxy"]["redshift"],
        z_source=lensing["source_galaxy"]["redshift"],
        cosmology_name=lensing["cosmology"],
        log10_m200_range=log10_m200_range,
    )


def build_mass_mapping_context_explicit(
    *,
    subhalo_model: str,
    concentration_model: Optional[str] = None,
    x_sub: Optional[float] = None,
    h: Optional[float] = None,
    z_lens: float,
    z_source: float,
    cosmology_name: str,
    log10_m200_range: Tuple[float, float] = (6.0, 8.5),
) -> MassMappingContext:
    """Build a mass-mapping context from explicit physical inputs.

    Parameters
    ----------
    subhalo_model : `str`
        One of ``"NFW"``, ``"SIS"``, or ``"PointMass"``.
    concentration_model : `str`, optional
        NFW concentration relation.
    x_sub : `float`, optional
        Host-centric radius for the Moline relation.
    h : `float`, optional
        Reduced Hubble parameter. A null Moline value is inferred.
    z_lens : `float`
        Lens-plane redshift.
    z_source : `float`
        Source-plane redshift.
    cosmology_name : `str`
        Supported cosmology label.
    log10_m200_range : `tuple` [`float`, `float`], optional
        Closed log10 M200 interpolation range.

    Returns
    -------
    context : `MassMappingContext`
        Resolved mass conversion context.
    """
    if subhalo_model not in {"NFW", "SIS", "PointMass"}:
        raise ValueError("subhalo_model must be 'NFW', 'SIS', or 'PointMass'")
    lower, upper = (float(value) for value in log10_m200_range)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("log10_m200_range must contain finite ordered bounds")
    z_lens = float(z_lens)
    z_source = float(z_source)
    if z_lens <= 0.0 or z_source <= z_lens:
        raise ValueError("z_source must be greater than positive z_lens")

    if subhalo_model == "NFW":
        if concentration_model not in {"moline2017_eq7", "power_law"}:
            raise ValueError(
                "NFW concentration_model must be 'moline2017_eq7' or "
                "'power_law'"
            )
        if concentration_model == "moline2017_eq7":
            if x_sub is None:
                raise ValueError("x_sub is required for moline2017_eq7")
            x_sub = float(x_sub)
            cosmology = _cosmology_from_name(str(cosmology_name))
            h = _infer_reduced_h(cosmology) if h is None else float(h)
        else:
            x_sub = None
            h = None
    else:
        concentration_model = None
        x_sub = None
        h = None

    return _build_context(
        subhalo_model,
        concentration_model,
        x_sub,
        h,
        z_lens,
        z_source,
        str(cosmology_name),
        lower,
        upper,
    )


def _interpolators(context: MassMappingContext) -> Tuple[PchipInterpolator, ...]:
    """Return cached PCHIP interpolators for a context."""
    cached = _INTERPOLATOR_CACHE.get(context.context_hash)
    if cached is not None:
        cached_context, interpolators = cached
        if cached_context is context or cached_context == context:
            return interpolators
        raise ValueError(
            "mass-mapping context hash matches a different context"
        )
    nodes = np.asarray(context.table_log10_m200, dtype=float)
    if context.subhalo_model == "NFW":
        interpolators = (
            PchipInterpolator(nodes, context.table_kappa_s),
            PchipInterpolator(nodes, context.table_scale_radius_arcsec),
        )
    else:
        interpolators = (
            PchipInterpolator(nodes, context.table_einstein_radius_arcsec),
        )
    _INTERPOLATOR_CACHE[context.context_hash] = (context, interpolators)
    return interpolators


def evaluate_mass_mapping(
    context: MassMappingContext,
    log10_m200: float,
) -> Dict[str, float]:
    """Evaluate a mass-mapping context without extrapolation.

    Parameters
    ----------
    context : `MassMappingContext`
        Resolved conversion context.
    log10_m200 : `float`
        Log10 M200 in solar masses.

    Returns
    -------
    parameters : `dict`
        Derived profile parameters, including ``c200`` for NFW.

    Raises
    ------
    ValueError
        Raised when the requested mass lies outside the context range.
    """
    log_mass = float(log10_m200)
    if not (
        context.log10_m200_lower
        <= log_mass
        <= context.log10_m200_upper
    ):
        raise ValueError(
            "log10_m200 lies outside the mass-mapping context range"
        )
    values = _interpolators(context)
    if context.subhalo_model == "NFW":
        c200 = _concentration(
            10.0**log_mass,
            context.concentration_model,
            context.x_sub,
            context.h,
            context.z_lens,
        )
        return {
            "c200": float(c200),
            "kappa_s": float(values[0](log_mass)),
            "scale_radius_arcsec": float(values[1](log_mass)),
        }
    return {"einstein_radius_arcsec": float(values[0](log_mass))}


def _build_profile_classes() -> None:
    """Build and publish the lazy PyAutoLens adapter classes."""
    if "NFWMCRSubhaloSph" in globals():
        return

    import autolens as al

    class NFWMCRSubhaloSph(al.mp.NFWSph):
        """NFW profile whose lensing scales are derived from log10 M200.

        Parameters
        ----------
        centre : `tuple` [`float`, `float`], optional
            Profile centre in ``(y, x)`` coordinates.
        log10_m200 : `float`, optional
            Log10 M200 in solar masses.
        mapping_context : `MassMappingContext`, optional
            Fixed NFW conversion context.
        """

        def __init__(
            self,
            centre=(0.0, 0.0),
            log10_m200=7.0,
            mapping_context=None,
        ):
            if mapping_context is None:
                raise ValueError("mapping_context is required for freed NFW fits")
            if mapping_context.subhalo_model != "NFW":
                raise ValueError("NFWMCRSubhaloSph requires an NFW context")
            derived = evaluate_mass_mapping(mapping_context, log10_m200)
            super().__init__(
                centre=centre,
                kappa_s=derived["kappa_s"],
                scale_radius=derived["scale_radius_arcsec"],
            )
            self.log10_m200 = float(log10_m200)
            self.mapping_context = mapping_context
            self.concentration = float(derived["c200"])
            self.kappa_s_derived = float(derived["kappa_s"])
            self.scale_radius_arcsec_derived = float(
                derived["scale_radius_arcsec"]
            )

    class SISMCRSubhalo(al.mp.IsothermalSph):
        """SIS profile whose Einstein radius is derived from log10 M200.

        Parameters
        ----------
        centre : `tuple` [`float`, `float`], optional
            Profile centre in ``(y, x)`` coordinates.
        log10_m200 : `float`, optional
            Log10 M200 in solar masses.
        mapping_context : `MassMappingContext`, optional
            Fixed SIS conversion context.
        """

        def __init__(
            self,
            centre=(0.0, 0.0),
            log10_m200=7.0,
            mapping_context=None,
        ):
            if mapping_context is None:
                raise ValueError("mapping_context is required for freed SIS fits")
            if mapping_context.subhalo_model != "SIS":
                raise ValueError("SISMCRSubhalo requires an SIS context")
            derived = evaluate_mass_mapping(mapping_context, log10_m200)
            super().__init__(
                centre=centre,
                einstein_radius=derived["einstein_radius_arcsec"],
            )
            self.log10_m200 = float(log10_m200)
            self.mapping_context = mapping_context
            self.einstein_radius_arcsec_derived = float(
                derived["einstein_radius_arcsec"]
            )

    class PointMassMCRSubhalo(al.mp.PointMass):
        """Point mass whose Einstein radius is derived from log10 M200.

        Parameters
        ----------
        centre : `tuple` [`float`, `float`], optional
            Profile centre in ``(y, x)`` coordinates.
        log10_m200 : `float`, optional
            Log10 M200 in solar masses.
        mapping_context : `MassMappingContext`, optional
            Fixed point-mass conversion context.
        """

        def __init__(
            self,
            centre=(0.0, 0.0),
            log10_m200=7.0,
            mapping_context=None,
        ):
            if mapping_context is None:
                raise ValueError(
                    "mapping_context is required for freed PointMass fits"
                )
            if mapping_context.subhalo_model != "PointMass":
                raise ValueError(
                    "PointMassMCRSubhalo requires a PointMass context"
                )
            derived = evaluate_mass_mapping(mapping_context, log10_m200)
            super().__init__(
                centre=centre,
                einstein_radius=derived["einstein_radius_arcsec"],
            )
            self.log10_m200 = float(log10_m200)
            self.mapping_context = mapping_context
            self.einstein_radius_arcsec_derived = float(
                derived["einstein_radius_arcsec"]
            )

    for profile_class in (
        NFWMCRSubhaloSph,
        SISMCRSubhalo,
        PointMassMCRSubhalo,
    ):
        profile_class.__module__ = __name__
        profile_class.__qualname__ = profile_class.__name__
        globals()[profile_class.__name__] = profile_class


def __getattr__(name: str) -> Any:
    """Resolve a lazy module-level adapter class by name."""
    if name in _PROFILE_CLASS_NAMES:
        _build_profile_classes()
        return globals()[name]
    raise AttributeError(name)


__all__ = [
    "MassMappingContext",
    "NFWMCRSubhaloSph",
    "PointMassMCRSubhalo",
    "SISMCRSubhalo",
    "build_mass_mapping_context",
    "build_mass_mapping_context_explicit",
    "evaluate_mass_mapping",
]

"""Mass-to-profile mappings for freed nonlinear subhalo fits."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np

from ...lensing.mass_models import (
    LensingGeometryScalars,
    MOLINE_EQ7_MAX_MASS_MSUN,
    MOLINE_EQ7_MAX_X_SUB,
    MOLINE_EQ7_MIN_MASS_MSUN,
    concentration_moline2017_eq7_xp,
    concentration_power_law_xp,
    einstein_radius_point_mass_xp,
    einstein_radius_sis_m200_xp,
    lensing_geometry_scalars,
    nfw_lensing_parameters_xp,
)


_PROFILE_CLASS_NAMES = {
    "NFWMCRSubhaloSph",
    "SISMCRSubhalo",
    "PointMassMCRSubhalo",
}
MASS_MAPPING_CONTEXT_SCHEMA_VERSION = 2

if TYPE_CHECKING:
    NFWMCRSubhaloSph: Any
    PointMassMCRSubhalo: Any
    SISMCRSubhalo: Any


def _as_float(value: Any) -> Any:
    """Convert concrete scalars to float while preserving traced values."""
    if isinstance(value, (int, float, np.generic)):
        return float(value)
    return value


def _xp_for(*values: Any) -> Any:
    """Select JAX for concrete arrays or tracers nested in input values."""
    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        return np

    jax_types = tuple(
        value_type
        for value_type in (
            getattr(jax, "Array", None),
            getattr(jax.core, "Tracer", None),
        )
        if isinstance(value_type, type)
    )
    pending = list(values)
    while pending:
        value = pending.pop()
        if jax_types and isinstance(value, jax_types):
            return jnp
        if isinstance(value, dict):
            pending.extend(value.keys())
            pending.extend(value.values())
        elif isinstance(value, (tuple, list)):
            pending.extend(value)
    return np


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
        Lower closed-form mapping boundary.
    log10_m200_upper : `float`
        Upper closed-form mapping boundary.
    geometry : `LensingGeometryScalars`
        Eagerly resolved scalar lensing geometry and physical constants.
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
    geometry: LensingGeometryScalars
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


def _canonical_context_hash(fields: Tuple[Any, ...]) -> str:
    """Return the short SHA-256 of canonical context fields."""
    return hashlib.sha256(repr(fields).encode("utf-8")).hexdigest()[:16]


def _validate_geometry(geometry: LensingGeometryScalars) -> None:
    """Reject nonfinite or nonpositive resolved geometry before tracing."""
    values = (
        geometry.z_lens,
        geometry.z_source,
        geometry.d_l_m,
        geometry.d_s_m,
        geometry.d_ls_m,
        geometry.rho_crit_z_lens_kg_m3,
        geometry.sigma_crit_kg_m2,
        geometry.msun_kg,
        geometry.g_si,
        geometry.c_si,
    )
    if not all(np.isfinite(value) for value in values):
        raise ValueError("mass-mapping geometry values must all be finite")
    if not all(value > 0.0 for value in values):
        raise ValueError("mass-mapping geometry values must all be positive")
    if geometry.z_source <= geometry.z_lens:
        raise ValueError("mass-mapping geometry source must be behind lens")


def _validate_context_inputs(
    subhalo_model: str,
    concentration_model: Optional[str],
    x_sub: Optional[float],
    h: Optional[float],
    z_lens: float,
    z_source: float,
    log10_m200_lower: float,
    log10_m200_upper: float,
) -> None:
    """Validate every static mass-mapping condition before tracing."""
    if subhalo_model not in {"NFW", "SIS", "PointMass"}:
        raise ValueError("subhalo_model must be 'NFW', 'SIS', or 'PointMass'")
    scalar_values = (
        z_lens,
        z_source,
        log10_m200_lower,
        log10_m200_upper,
    )
    if not all(np.isfinite(value) for value in scalar_values):
        raise ValueError("mass-mapping scalar inputs must all be finite")
    if z_lens <= 0.0 or z_source <= z_lens:
        raise ValueError("z_source must be greater than positive z_lens")
    if log10_m200_lower >= log10_m200_upper:
        raise ValueError("log10_m200_range must contain finite ordered bounds")

    if subhalo_model != "NFW":
        if concentration_model is not None or x_sub is not None or h is not None:
            raise ValueError(
                "SIS and PointMass contexts do not accept concentration inputs"
            )
        return
    if concentration_model not in {"moline2017_eq7", "power_law"}:
        raise ValueError(
            "NFW concentration_model must be 'moline2017_eq7' or "
            "'power_law'"
        )
    if concentration_model == "power_law":
        if x_sub is not None or h is not None:
            raise ValueError("power_law contexts do not accept x_sub or h")
        return

    if x_sub is None or not np.isfinite(x_sub) or not 0.0 < x_sub <= MOLINE_EQ7_MAX_X_SUB:
        raise ValueError(
            f"x_sub must satisfy 0 < x_sub <= {MOLINE_EQ7_MAX_X_SUB:g}"
        )
    if h is None or not np.isfinite(h) or h <= 0.0:
        raise ValueError("h must be a finite positive number")
    supported_lower = np.log10(MOLINE_EQ7_MIN_MASS_MSUN)
    supported_upper = np.log10(MOLINE_EQ7_MAX_MASS_MSUN)
    if (
        log10_m200_lower < supported_lower
        or log10_m200_upper > supported_upper
    ):
        raise ValueError(
            "moline2017_eq7 mass range lies outside its supported domain"
        )


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
    """Construct and validate one cached closed-form mapping context."""
    _validate_context_inputs(
        subhalo_model,
        concentration_model,
        x_sub,
        h,
        z_lens,
        z_source,
        log10_m200_lower,
        log10_m200_upper,
    )
    cosmology = _cosmology_from_name(cosmology_name)
    geometry = lensing_geometry_scalars(z_lens, z_source, cosmology)
    _validate_geometry(geometry)
    context_fields = (
        MASS_MAPPING_CONTEXT_SCHEMA_VERSION,
        subhalo_model,
        concentration_model,
        x_sub,
        h,
        z_lens,
        z_source,
        cosmology_name,
        log10_m200_lower,
        log10_m200_upper,
        geometry.z_lens,
        geometry.z_source,
        geometry.d_l_m,
        geometry.d_s_m,
        geometry.d_ls_m,
        geometry.rho_crit_z_lens_kg_m3,
        geometry.sigma_crit_kg_m2,
        geometry.msun_kg,
        geometry.g_si,
        geometry.c_si,
    )
    return MassMappingContext(
        subhalo_model=subhalo_model,
        concentration_model=concentration_model,
        x_sub=x_sub,
        h=h,
        z_lens=z_lens,
        z_source=z_source,
        cosmology_name=cosmology_name,
        log10_m200_lower=log10_m200_lower,
        log10_m200_upper=log10_m200_upper,
        geometry=geometry,
        context_hash=_canonical_context_hash(context_fields),
    )


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
        Closed log10 M200 mapping range.

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
        Closed log10 M200 mapping range.

    Returns
    -------
    context : `MassMappingContext`
        Resolved mass conversion context.
    """
    lower, upper = (float(value) for value in log10_m200_range)
    z_lens = float(z_lens)
    z_source = float(z_source)
    x_sub = None if x_sub is None else float(x_sub)
    if subhalo_model == "NFW" and concentration_model == "moline2017_eq7":
        cosmology = _cosmology_from_name(str(cosmology_name))
        h = _infer_reduced_h(cosmology) if h is None else float(h)
    else:
        h = None if h is None else float(h)

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
    mass_msun = 10.0**log_mass
    if context.subhalo_model == "NFW":
        if context.concentration_model == "moline2017_eq7":
            c200 = concentration_moline2017_eq7_xp(
                mass_msun,
                context.x_sub,
                context.h,
                np,
            )
        else:
            c200 = concentration_power_law_xp(
                mass_msun,
                context.z_lens,
                np,
            )
        kappa_s, scale_radius = nfw_lensing_parameters_xp(
            mass_msun,
            c200,
            context.geometry,
            np,
        )
        return {
            "c200": float(c200),
            "kappa_s": float(kappa_s),
            "scale_radius_arcsec": float(scale_radius),
        }
    if context.subhalo_model == "SIS":
        einstein_radius = einstein_radius_sis_m200_xp(
            mass_msun,
            context.geometry,
            np,
        )
    else:
        einstein_radius = einstein_radius_point_mass_xp(
            mass_msun,
            context.geometry,
            np,
        )
    return {"einstein_radius_arcsec": float(einstein_radius)}


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
            xp = _xp_for(centre, log10_m200)
            mass_msun = 10.0**log10_m200
            if mapping_context.concentration_model == "moline2017_eq7":
                concentration = concentration_moline2017_eq7_xp(
                    mass_msun,
                    mapping_context.x_sub,
                    mapping_context.h,
                    xp,
                )
            elif mapping_context.concentration_model == "power_law":
                concentration = concentration_power_law_xp(
                    mass_msun,
                    mapping_context.z_lens,
                    xp,
                )
            else:
                raise ValueError("NFWMCRSubhaloSph requires a supported relation")
            kappa_s, scale_radius = nfw_lensing_parameters_xp(
                mass_msun,
                concentration,
                mapping_context.geometry,
                xp,
            )
            super().__init__(
                centre=centre,
                kappa_s=kappa_s,
                scale_radius=scale_radius,
            )
            self.log10_m200 = _as_float(log10_m200)
            self.mapping_context = mapping_context
            self.concentration = _as_float(concentration)
            self.kappa_s_derived = _as_float(kappa_s)
            self.scale_radius_arcsec_derived = _as_float(scale_radius)

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
            xp = _xp_for(centre, log10_m200)
            einstein_radius = einstein_radius_sis_m200_xp(
                10.0**log10_m200,
                mapping_context.geometry,
                xp,
            )
            super().__init__(
                centre=centre,
                einstein_radius=einstein_radius,
            )
            self.log10_m200 = _as_float(log10_m200)
            self.mapping_context = mapping_context
            self.einstein_radius_arcsec_derived = _as_float(einstein_radius)

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
            xp = _xp_for(centre, log10_m200)
            einstein_radius = einstein_radius_point_mass_xp(
                10.0**log10_m200,
                mapping_context.geometry,
                xp,
            )
            super().__init__(
                centre=centre,
                einstein_radius=einstein_radius,
            )
            self.log10_m200 = _as_float(log10_m200)
            self.mapping_context = mapping_context
            self.einstein_radius_arcsec_derived = _as_float(einstein_radius)

        def _cartesian_grid_via_radial_from(
            self,
            grid,
            radius,
            xp=np,
            **kwargs,
        ):
            """Unwrap AutoArray radius values only on the JAX path."""
            if xp is np:
                return super()._cartesian_grid_via_radial_from(
                    grid=grid,
                    radius=radius,
                    xp=xp,
                    **kwargs,
                )
            grid_values = grid.array if hasattr(grid, "array") else grid
            radius_values = (
                radius.array if hasattr(radius, "array") else radius
            )
            angles = xp.arctan2(grid_values[:, 0], grid_values[:, 1])
            directions = xp.stack(
                (xp.sin(angles), xp.cos(angles)),
                axis=-1,
            )
            return xp.multiply(radius_values[:, None], directions)

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

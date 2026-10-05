"""Halo physics: halo models, concentration relations and the lensing scales of a halo.

A ``Halo`` is one realized halo (the subhalo hypothesis, a listed perturber or a population
member): its model, mass, position in its own plane, redshift and cosmology. Lensing
normalizations are reduced deflections relative to the final source plane, so a halo at
``z_h`` uses ``LensingGeometry(z_h, z_s)``.

Mass definitions: ``M200c`` (SIS and NFW, 200 rho_crit at the halo redshift, rho_crit in the
convention of ``scene.cosmology``) and ``point_mass`` (the total mass of a point lens).

The lensing scales are evaluated in two operation orders, each a pinned convention of the
paper code:

- ``halo_lensing`` (concrete masses: truth scenes, forecast engines, fixed-template fits,
  perturbers) is the 8fa6209 scalar order, including the kpc round trip of the NFW scale
  radius and the km/s round trip of the SIS velocity dispersion. PARITY P1-P4 and the
  401-mass sweep fixture pin it.
- ``halo_lensing_traced`` (freed fits, numpy or JAX arrays) is the order of the 8fa6209
  array-namespace twins of those functions, without the round trips. PARITY N1 pins it.

The two agree to one or two ulp.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real as RealNumber
from typing import Any, Literal

import numpy as np

from ..config.checks import Key, Nullable, Real, Table, Variants
from ..constants import ARCSEC_PER_RAD, C_M_S, G_SI, KPC_TO_M, M_TO_KPC, MPC_TO_M, MSUN_KG
from .cosmology import Cosmology, LensingGeometry

__all__ = [
    "CONCENTRATION_TABLE", "ConcentrationSpec", "FixedConcentration", "HALO_MODEL_TABLE", "Halo",
    "HaloLensing", "HaloModel", "MOLINE2017_MASS_RANGE_MSUN", "Moline2017", "PowerLawConcentration",
    "concentration", "halo_lensing", "halo_lensing_traced", "halo_model_from_values", "halo_model_table",
    "make_halo",
]

MOLINE2017_MASS_RANGE_MSUN = (1.0e6, 1.0e12)
"""Calibrated M200 range of the Moline et al. (2017) relation, in Msun."""

_MOLINE_C0 = 19.9
_MOLINE_A1, _MOLINE_A2, _MOLINE_A3 = -0.195, 0.089, 0.089
_MOLINE_B = -0.54
_MOLINE_MAX_X_SUB = 1.5

_PROFILE_CLASSES = {"PointMass": "PointMass", "SIS": "IsothermalSph", "NFW": "NFWSph"}


def _finite(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, RealNumber) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return float(value)


def _positive(value: Any, name: str) -> float:
    number = _finite(value, name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return number


@dataclass(frozen=True)
class Moline2017:
    """Moline et al. (2017), eq. 7 with table 2: c200 of a subhalo at host radius ``x_sub``.

    ``h`` is the reduced Hubble constant of the relation's mass unit (1e8 / h Msun); None
    takes the cosmology's H0 / 100. Calibrated for subhalos inside hosts at z = 0, so the
    scene allows it only for halos at the lens redshift.
    """

    x_sub: float
    h: float | None

    def __post_init__(self) -> None:
        if not 0.0 < _finite(self.x_sub, "x_sub") <= _MOLINE_MAX_X_SUB:
            raise ValueError(f"x_sub must lie in (0, {_MOLINE_MAX_X_SUB}], got {self.x_sub!r}")
        if self.h is not None:
            _positive(self.h, "h")


@dataclass(frozen=True)
class PowerLawConcentration:
    """``c200 = c0 (M200 / mass_pivot_msun)^mass_slope (1 + z)^redshift_slope``."""

    c0: float
    mass_pivot_msun: float
    mass_slope: float
    redshift_slope: float

    def __post_init__(self) -> None:
        _positive(self.c0, "c0")
        _positive(self.mass_pivot_msun, "mass_pivot_msun")
        _finite(self.mass_slope, "mass_slope")
        _finite(self.redshift_slope, "redshift_slope")


@dataclass(frozen=True)
class FixedConcentration:
    """One concentration for every mass and redshift."""

    value: float

    def __post_init__(self) -> None:
        _positive(self.value, "value")


ConcentrationSpec = Moline2017 | PowerLawConcentration | FixedConcentration


@dataclass(frozen=True)
class HaloModel:
    """A halo family: ``PointMass``, ``SIS`` or ``NFW`` with its concentration relation.

    ``truncation`` belongs to the truncated NFW of W2-COSMO and is None for every type here.
    """

    type: Literal["PointMass", "SIS", "NFW"]
    concentration: ConcentrationSpec | None
    truncation: None

    def __post_init__(self) -> None:
        if self.type not in _PROFILE_CLASSES:
            raise ValueError(f"halo type must be one of {', '.join(_PROFILE_CLASSES)}, got {self.type!r}")
        if (self.type == "NFW") != (self.concentration is not None):
            raise ValueError(f"an NFW halo needs a concentration relation and other types none; got "
                             f"{self.type} with {self.concentration!r}")
        if self.truncation is not None:
            raise ValueError(f"halo type {self.type} has no truncation")

    @property
    def mass_definition(self) -> Literal["point_mass", "M200c"]:
        return "point_mass" if self.type == "PointMass" else "M200c"

    @property
    def profile_class(self) -> str:
        """The AutoLens mass profile class: PointMass, IsothermalSph or NFWSph."""
        return _PROFILE_CLASSES[self.type]


@dataclass(frozen=True)
class HaloLensing:
    """AutoLens keyword arguments of a halo's profile (centre excluded) and derived scalars.

    ``derived`` holds ``reduced_h`` for every type: the H0 / 100 of the cosmology, which a
    ``moline2017_eq7`` relation without its own ``h`` uses.
    """

    parameters: Mapping[str, float]
    derived: Mapping[str, float]


# ------------------------------------------------------------------ key tables

CONCENTRATION_TABLE = Variants(
    "kind",
    {
        "moline2017_eq7": Table((
            Key("x_sub", Real(min=0.0, min_open=True, max=_MOLINE_MAX_X_SUB),
                "host-centric radius of the subhalo in units of the host virial radius"),
            Key("h", Nullable(Real(min=0.0, min_open=True)),
                "reduced Hubble constant of the relation's mass unit 1e8 / h Msun; null takes H0 / 100 "
                "of the cosmology", None),
        ), doc="Moline et al. (2017), eq. 7: subhalos at the lens redshift, M200 in [1e6, 1e12] Msun."),
        "power_law": Table((
            Key("c0", Real(min=0.0, min_open=True), "concentration at the pivot mass and z = 0"),
            Key("mass_pivot_msun", Real(min=0.0, min_open=True), "pivot mass", unit="Msun"),
            Key("mass_slope", Real(), "exponent of M200 / mass_pivot_msun"),
            Key("redshift_slope", Real(), "exponent of 1 + z"),
        ), doc="c200 = c0 (M200 / mass_pivot_msun)^mass_slope (1 + z)^redshift_slope."),
        "fixed": Table((Key("value", Real(min=0.0, min_open=True), "concentration c200"),)),
    },
    doc="Concentration-mass relation of an NFW halo.",
)

_TYPE_KEYS: Mapping[str, tuple[Key, ...]] = {
    "PointMass": (),
    "SIS": (),
    "NFW": (Key("concentration", CONCENTRATION_TABLE, "concentration-mass relation"),),
}


def halo_model_table(extra_keys: Sequence[Key] = ()) -> Variants:
    """The halo-model variants (selected by ``type``), each extended by ``extra_keys``."""
    return Variants("type", {name: Table(keys + tuple(extra_keys)) for name, keys in _TYPE_KEYS.items()},
                    doc="PointMass (point_mass), SIS (M200c) or NFW (M200c) halo.")


HALO_MODEL_TABLE = halo_model_table()


def halo_model_from_values(values: Mapping[str, Any]) -> HaloModel:
    """The ``HaloModel`` of values read by a halo-model table (extra keys are ignored)."""
    relation = values.get("concentration")
    return HaloModel(type=values["type"], concentration=None if relation is None else _relation(relation),
                     truncation=None)


def _relation(values: Mapping[str, Any]) -> ConcentrationSpec:
    kind = values["kind"]
    if kind == "moline2017_eq7":
        return Moline2017(x_sub=values["x_sub"], h=values["h"])
    if kind == "power_law":
        return PowerLawConcentration(c0=values["c0"], mass_pivot_msun=values["mass_pivot_msun"],
                                     mass_slope=values["mass_slope"], redshift_slope=values["redshift_slope"])
    return FixedConcentration(value=values["value"])


def _model_mapping(model: HaloModel) -> dict[str, Any]:
    record: dict[str, Any] = {"type": model.type}
    relation = model.concentration
    if isinstance(relation, Moline2017):
        record["concentration"] = {"kind": "moline2017_eq7", "x_sub": relation.x_sub, "h": relation.h}
    elif isinstance(relation, PowerLawConcentration):
        record["concentration"] = {"kind": "power_law", "c0": relation.c0, "mass_pivot_msun": relation.mass_pivot_msun,
                                   "mass_slope": relation.mass_slope, "redshift_slope": relation.redshift_slope}
    elif isinstance(relation, FixedConcentration):
        record["concentration"] = {"kind": "fixed", "value": relation.value}
    return record


# ------------------------------------------------------------------ physics


def concentration(spec: ConcentrationSpec, m200_msun: Any, z_halo: float, reduced_h: float, *, xp: Any = np) -> Any:
    """c200 of a halo of mass ``m200_msun`` (Msun) at ``z_halo``; arrays of any namespace ``xp``."""
    if isinstance(spec, Moline2017):
        h = reduced_h if spec.h is None else spec.h
        log_mass_term = xp.log10((m200_msun * h) / 1.0e8)
        polynomial = (1.0 + (_MOLINE_A1 * log_mass_term) + (_MOLINE_A2 * log_mass_term) ** 2
                      + (_MOLINE_A3 * log_mass_term) ** 3)
        radial_factor = 1.0 + _MOLINE_B * xp.log10(spec.x_sub)
        return _MOLINE_C0 * polynomial * radial_factor
    if isinstance(spec, PowerLawConcentration):
        return spec.c0 * (m200_msun / spec.mass_pivot_msun) ** spec.mass_slope * (1.0 + z_halo) ** spec.redshift_slope
    if isinstance(spec, FixedConcentration):
        return spec.value
    raise TypeError(f"unknown concentration relation {spec!r}")


def _check_mass_domain(model: HaloModel, mass_msun: float) -> None:
    low, high = MOLINE2017_MASS_RANGE_MSUN
    if isinstance(model.concentration, Moline2017) and not low <= mass_msun <= high:
        raise ValueError(f"the moline2017_eq7 relation is calibrated for M200 in [{low:g}, {high:g}] Msun, "
                         f"got {mass_msun:g}")


def halo_lensing(model: HaloModel, mass_msun: float, geometry: LensingGeometry, *, reduced_h: float) -> HaloLensing:
    """Lensing scales of a halo of concrete mass, in the 8fa6209 scalar operation order (a pinned convention)."""
    mass = _positive(mass_msun, "mass_msun")
    _check_mass_domain(model, mass)
    if model.type == "PointMass":
        theta_squared = (4 * G_SI * (mass * MSUN_KG) * (geometry.d_deflector_source_mpc * MPC_TO_M)) / (
            C_M_S**2 * (geometry.d_deflector_mpc * MPC_TO_M) * (geometry.d_source_mpc * MPC_TO_M))
        return HaloLensing({"einstein_radius": float(np.sqrt(theta_squared) * ARCSEC_PER_RAD)},
                           {"reduced_h": float(reduced_h)})
    m200_kg = mass * MSUN_KG
    r200_m = ((3 * m200_kg) / (4 * np.pi * 200 * geometry.rho_crit_kg_m3)) ** (1 / 3)
    if model.type == "SIS":
        velocity_dispersion_km_s = float(np.sqrt(G_SI * m200_kg / (2 * r200_m)) / 1000.0)
        theta_rad = 4.0 * np.pi * ((velocity_dispersion_km_s * 1000.0) / C_M_S) ** 2 * (
            geometry.d_deflector_source_mpc / geometry.d_source_mpc)
        return HaloLensing({"einstein_radius": float(theta_rad) * ARCSEC_PER_RAD},
                           {"reduced_h": float(reduced_h), "r200_kpc": float(r200_m * M_TO_KPC),
                            "velocity_dispersion_km_s": velocity_dispersion_km_s})
    c200 = float(concentration(model.concentration, mass, geometry.z_deflector, reduced_h))
    scale_radius_kpc = float((r200_m / c200) * M_TO_KPC)
    f_c = np.log(1 + c200) - c200 / (1 + c200)
    rho_s = geometry.rho_crit_kg_m3 * (200.0 / 3.0) * c200**3 / f_c
    scale_radius_m = scale_radius_kpc * KPC_TO_M
    kappa_s = (rho_s * scale_radius_m) / geometry.sigma_crit_kg_m2
    scale_radius_arcsec = (scale_radius_m / (geometry.d_deflector_mpc * MPC_TO_M)) * ARCSEC_PER_RAD
    return HaloLensing(
        {"kappa_s": float(kappa_s), "scale_radius": float(scale_radius_arcsec)},
        {"reduced_h": float(reduced_h), "concentration": c200, "r200_kpc": float(r200_m * M_TO_KPC),
         "scale_radius_kpc": scale_radius_kpc, "rho_s_kg_m3": float(rho_s)})


def halo_lensing_traced(model: HaloModel, mass_msun: Any, geometry: LensingGeometry, *, reduced_h: float,
                        xp: Any) -> Mapping[str, Any]:
    """AutoLens keyword arguments (centre excluded) in the operation order of the 8fa6209 freed fits (pinned).

    ``mass_msun`` may be a numpy or JAX array or tracer; nothing is range-checked here (freed
    fits check their mass support before tracing).
    """
    m200_kg = mass_msun * MSUN_KG
    d_deflector_m = geometry.d_deflector_mpc * MPC_TO_M
    d_source_m = geometry.d_source_mpc * MPC_TO_M
    d_deflector_source_m = geometry.d_deflector_source_mpc * MPC_TO_M
    if model.type == "PointMass":
        theta_squared = (4 * G_SI * m200_kg * d_deflector_source_m) / (C_M_S**2 * d_deflector_m * d_source_m)
        return {"einstein_radius": xp.sqrt(theta_squared) * ARCSEC_PER_RAD}
    r200_m = ((3 * m200_kg) / (4 * xp.pi * 200 * geometry.rho_crit_kg_m3)) ** (1 / 3)
    if model.type == "SIS":
        velocity_dispersion_m_s = xp.sqrt(G_SI * m200_kg / (2 * r200_m))
        theta_rad = 4.0 * xp.pi * (velocity_dispersion_m_s / C_M_S) ** 2 * (d_deflector_source_m / d_source_m)
        return {"einstein_radius": theta_rad * ARCSEC_PER_RAD}
    c200 = concentration(model.concentration, mass_msun, geometry.z_deflector, reduced_h, xp=xp)
    scale_radius_m = r200_m / c200
    f_c = xp.log(1 + c200) - c200 / (1 + c200)
    rho_s = geometry.rho_crit_kg_m3 * (200.0 / 3.0) * c200**3 / f_c
    kappa_s = (rho_s * scale_radius_m) / geometry.sigma_crit_kg_m2
    return {"kappa_s": kappa_s, "scale_radius": (scale_radius_m / d_deflector_m) * ARCSEC_PER_RAD}


# ------------------------------------------------------------------ realized halos

_HALO_RECORD_KEYS = ("model", "mass_msun", "mass_definition", "position_yx_arcsec", "redshift", "source_redshift",
                     "cosmology", "lensing")


@dataclass(frozen=True)
class Halo:
    """One realized halo. ``position_yx_arcsec`` is its angular position in its own plane.

    The geometry and reduced Hubble constant are read from ``cosmology``, so
    ``dataclasses.replace(halo, mass_msun=..., position_yx_arcsec=..., redshift=...)`` is
    always valid.
    """

    model: HaloModel
    mass_msun: float
    position_yx_arcsec: tuple[float, float]
    redshift: float
    source_redshift: float
    cosmology: Cosmology

    def __post_init__(self) -> None:
        if not isinstance(self.model, HaloModel):
            raise TypeError(f"model must be a HaloModel, got {type(self.model).__name__}")
        if not isinstance(self.cosmology, Cosmology):
            raise TypeError(f"cosmology must be a Cosmology, got {type(self.cosmology).__name__}")
        mass = _positive(self.mass_msun, "mass_msun")
        _check_mass_domain(self.model, mass)
        position = tuple(self.position_yx_arcsec)
        if len(position) != 2:
            raise ValueError(f"position_yx_arcsec must be a (y, x) pair, got {self.position_yx_arcsec!r}")
        redshift = _positive(self.redshift, "redshift")
        source_redshift = _finite(self.source_redshift, "source_redshift")
        if not redshift < source_redshift:
            raise ValueError(f"a halo at z = {redshift} cannot lens a source at z = {source_redshift}")
        object.__setattr__(self, "mass_msun", mass)
        object.__setattr__(self, "position_yx_arcsec",
                           (_finite(position[0], "position_yx_arcsec[0]"), _finite(position[1], "position_yx_arcsec[1]")))
        object.__setattr__(self, "redshift", redshift)
        object.__setattr__(self, "source_redshift", source_redshift)

    @property
    def geometry(self) -> LensingGeometry:
        return self.cosmology.geometry(self.redshift, self.source_redshift)

    @property
    def reduced_h(self) -> float:
        return self.cosmology.reduced_h

    def lensing(self) -> HaloLensing:
        """Lensing scales in the concrete-mass operation order (``halo_lensing``)."""
        return halo_lensing(self.model, self.mass_msun, self.geometry, reduced_h=self.reduced_h)

    def autolens_profile(self, *, centre: tuple[float, float] | None = None) -> Any:
        """The AutoLens mass profile of this halo, at ``centre`` when given (radial tables use the origin)."""
        import autolens as al

        profile_class = getattr(al.mp, self.model.profile_class)
        return profile_class(centre=self.position_yx_arcsec if centre is None else tuple(centre),
                             **self.lensing().parameters)

    def to_mapping(self) -> dict[str, Any]:
        lensing = self.lensing()
        return {
            "model": _model_mapping(self.model),
            "mass_msun": self.mass_msun,
            "mass_definition": self.model.mass_definition,
            "position_yx_arcsec": list(self.position_yx_arcsec),
            "redshift": self.redshift,
            "source_redshift": self.source_redshift,
            "cosmology": self.cosmology.to_mapping(),
            "lensing": {"parameters": dict(lensing.parameters), "derived": dict(lensing.derived)},
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> Halo:
        """The halo of a ``to_mapping`` record; its recorded lensing scales must be this engine's."""
        if sorted(mapping) != sorted(_HALO_RECORD_KEYS):
            raise ValueError(f"halo record keys must be {list(_HALO_RECORD_KEYS)}, got {sorted(map(str, mapping))}")
        halo = make_halo(halo_model_from_values(HALO_MODEL_TABLE.read(mapping["model"], "model")),
                         mapping["mass_msun"], tuple(mapping["position_yx_arcsec"]), redshift=mapping["redshift"],
                         source_redshift=mapping["source_redshift"],
                         cosmology=Cosmology.from_mapping(mapping["cosmology"]))
        expected = halo.to_mapping()
        for key in ("mass_definition", "lensing"):
            if mapping[key] != expected[key]:
                raise ValueError(f"halo record {key} {mapping[key]!r} differs from this engine's {expected[key]!r}")
        return halo


def make_halo(model: HaloModel, mass_msun: float, position_yx: tuple[float, float], *, redshift: float,
              source_redshift: float, cosmology: Cosmology) -> Halo:
    """A validated ``Halo`` (mass, position, redshifts and the relation's calibrated mass range)."""
    return Halo(model=model, mass_msun=mass_msun, position_yx_arcsec=tuple(position_yx), redshift=redshift,
                source_redshift=source_redshift, cosmology=cosmology)

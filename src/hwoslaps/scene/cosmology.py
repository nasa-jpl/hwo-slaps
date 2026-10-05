"""Cosmology of a scene: the realization, its AutoGalaxy object and the lensing geometry of halos.

One ``Cosmology`` value serves a preparation; every consumer receives the same object, so the
AutoGalaxy distance cache and the per-redshift-pair geometries are computed once.

``LensingGeometry`` follows the scalar operation order of the RASTI-26-183 paper code. Its Hubble
rate omits radiation and massive neutrinos, ``H(z) = H0 sqrt(Om0 (1 + z)^3 + (1 - Om0))``, while the
distances of the same AutoGalaxy object include them (E(0.2) is low by 4.5e-4 for Planck15).
This is the convention of the paper path; ``Cosmology.to_mapping`` records it as
``rho_crit_convention: "matter_lambda"``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ..config.checks import ConfigError, Key, ListOf, Nullable, Real, Rule, Table, Text
from ..constants import C_M_S, G_SI, KM_TO_M, MPC_TO_M

__all__ = [
    "COSMOLOGY_TABLE", "Cosmology", "CosmologySpec", "FlatLCDMParameters", "LensingGeometry",
    "parse_cosmology",
]

RHO_CRIT_CONVENTION = "matter_lambda"


def _check_baryon_density(values: Mapping[str, Any], path: str) -> None:
    if values["Ob0"] > values["Om0"]:
        raise ConfigError(f"{path}.Ob0", "must not exceed Om0")


_FLAT_LCDM_TABLE = Table((
    Key("H0", Real(min=0.0, min_open=True), "Hubble constant", unit="km/s/Mpc"),
    Key("Om0", Real(min=0.0, min_open=True, max=1.0, max_open=True), "matter density fraction"),
    Key("Ob0", Real(min=0.0), "baryon density fraction", 0.0),
    Key("Tcmb0", Real(min=0.0), "CMB temperature", 0.0, unit="K"),
    Key("Neff", Real(min=0.0), "effective number of neutrino species", 3.046),
    Key("m_nu_eV", ListOf(Real(min=0.0), length=3), "three neutrino masses", [0.0, 0.0, 0.0], unit="eV"),
), rules=(Rule("Ob0 must not exceed Om0", _check_baryon_density),))

COSMOLOGY_TABLE = Table(
    keys=(Key("name", Nullable(Text()),
              "named astropy flat Lambda-CDM realization (Planck15 preserves the paper backend)", None),
          Key("flat_lcdm", Nullable(_FLAT_LCDM_TABLE), "custom flat Lambda-CDM parameters", None)),
    exactly_one=(("name", "flat_lcdm"),),
    doc="The cosmology of distances, critical densities and halo scales.",
)


@dataclass(frozen=True)
class FlatLCDMParameters:
    """Parameters of a flat Lambda-CDM cosmology (H0 in km/s/Mpc, Tcmb0 in K, neutrino masses in eV)."""

    H0: float
    Om0: float
    Ob0: float = 0.0
    Tcmb0: float = 0.0
    Neff: float = 3.046
    m_nu_eV: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class CosmologySpec:
    """A named realization or custom flat Lambda-CDM parameters."""

    name: str | None
    flat_lcdm: FlatLCDMParameters | None


@dataclass(frozen=True)
class LensingGeometry:
    """Distances and critical densities of a deflector at ``z_deflector`` lensing the source plane.

    ``rho_crit_kg_m3`` is the critical density at the deflector redshift and
    ``sigma_crit_kg_m2`` the critical surface density for the deflector-source pair.
    """

    z_deflector: float
    z_source: float
    d_deflector_mpc: float
    d_source_mpc: float
    d_deflector_source_mpc: float
    hubble_km_s_mpc: float
    rho_crit_kg_m3: float
    sigma_crit_kg_m2: float


def parse_cosmology(mapping: Mapping[str, Any], path: str = "cosmology") -> CosmologySpec:
    """Read the ``cosmology`` section strictly."""
    values = COSMOLOGY_TABLE.read(mapping, path)
    if values["name"] is not None:
        try:
            _flat_realization(values["name"])
        except ValueError as error:
            raise ConfigError(f"{path}.name", str(error)) from error
    custom = values["flat_lcdm"]
    return CosmologySpec(name=values["name"], flat_lcdm=None if custom is None else _parameters(custom))


def _parameters(values: Mapping[str, Any]) -> FlatLCDMParameters:
    return FlatLCDMParameters(**{**values, "m_nu_eV": tuple(values["m_nu_eV"])})


def _flat_realization(name: str) -> Any:
    from astropy.cosmology import FlatLambdaCDM, realizations

    realization = getattr(realizations, name, None)
    if not isinstance(realization, FlatLambdaCDM):
        raise ValueError(f"cosmology {name!r} is not an astropy flat Lambda-CDM realization")
    return realization


def _realization_parameters(name: str) -> FlatLCDMParameters:
    realization = _flat_realization(name)
    m_nu = realization.m_nu
    return FlatLCDMParameters(
        H0=float(realization.H0.value), Om0=float(realization.Om0), Ob0=float(realization.Ob0 or 0.0),
        Tcmb0=float(realization.Tcmb0.value), Neff=float(realization.Neff),
        m_nu_eV=(0.0, 0.0, 0.0) if m_nu is None else tuple(float(v) for v in m_nu.to_value("eV")))


class Cosmology:
    """An immutable cosmology value; equality and hash follow its spec.

    The AutoGalaxy object is built on first use and reused, and ``geometry`` is memoized per
    redshift pair. Pickling keeps only the spec, so a spawned worker rebuilds the backend
    object on first use.
    """

    __slots__ = ("_spec", "_parameters", "_backend", "_geometries")

    def __init__(self, spec: CosmologySpec) -> None:
        if not isinstance(spec, CosmologySpec):
            raise TypeError(f"Cosmology needs a CosmologySpec, got {type(spec).__name__}")
        if (spec.name is None) == (spec.flat_lcdm is None):
            raise ValueError("cosmology needs exactly one of name and flat_lcdm")
        parameters = (_realization_parameters(spec.name) if spec.name is not None else
                      _parameters(_FLAT_LCDM_TABLE.read(asdict(spec.flat_lcdm), "cosmology.flat_lcdm")))
        object.__setattr__(self, "_spec", spec)
        object.__setattr__(self, "_parameters", parameters)
        object.__setattr__(self, "_backend", None)
        object.__setattr__(self, "_geometries", {})

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Cosmology is immutable")

    @property
    def spec(self) -> CosmologySpec:
        return self._spec

    @property
    def parameters(self) -> FlatLCDMParameters:
        return self._parameters

    @property
    def reduced_h(self) -> float:
        """H0 / 100."""
        return self._parameters.H0 / 100.0

    def autogalaxy(self) -> Any:
        """The AutoGalaxy cosmology object, the same instance on every call."""
        if self._backend is None:
            import autogalaxy as ag

            parameters = self._parameters
            backend = (ag.cosmo.Planck15() if self._spec.name == "Planck15" else
                       ag.cosmo.FlatLambdaCDM(H0=parameters.H0, Om0=parameters.Om0, Ob0=parameters.Ob0,
                                             Tcmb0=parameters.Tcmb0, Neff=parameters.Neff,
                                             m_nu=list(parameters.m_nu_eV)))
            object.__setattr__(self, "_backend", backend)
        return self._backend

    def geometry(self, z_deflector: float, z_source: float) -> LensingGeometry:
        """Lensing geometry of a deflector at ``z_deflector`` for the source plane at ``z_source``."""
        key = (float(z_deflector), float(z_source))
        if not 0.0 < key[0] < key[1]:
            raise ValueError(f"lensing geometry needs 0 < z_deflector < z_source, got {key}")
        if key not in self._geometries:
            self._geometries[key] = self._compute_geometry(*key)
        return self._geometries[key]

    def _compute_geometry(self, z_deflector: float, z_source: float) -> LensingGeometry:
        backend = self.autogalaxy()
        d_deflector = float(backend.angular_diameter_distance_to_earth_in_kpc_from(z_deflector)) / 1000.0
        d_source = float(backend.angular_diameter_distance_to_earth_in_kpc_from(z_source)) / 1000.0
        d_deflector_source = float(
            backend.angular_diameter_distance_between_redshifts_in_kpc_from(z_deflector, z_source)) / 1000.0
        om0 = self._parameters.Om0
        hubble = self._parameters.H0 * np.sqrt(om0 * (1.0 + z_deflector) ** 3 + (1.0 - om0))
        hubble_si = hubble * KM_TO_M / MPC_TO_M
        rho_crit = 3 * hubble_si**2 / (8 * np.pi * G_SI)
        sigma_crit = (C_M_S**2 / (4 * np.pi * G_SI)) * (
            (d_source * MPC_TO_M) / ((d_deflector * MPC_TO_M) * (d_deflector_source * MPC_TO_M)))
        return LensingGeometry(
            z_deflector=z_deflector, z_source=z_source, d_deflector_mpc=d_deflector, d_source_mpc=d_source,
            d_deflector_source_mpc=d_deflector_source, hubble_km_s_mpc=float(hubble),
            rho_crit_kg_m3=float(rho_crit), sigma_crit_kg_m2=float(sigma_crit))

    def to_mapping(self) -> dict[str, Any]:
        parameters = asdict(self._parameters)
        parameters["m_nu_eV"] = list(parameters["m_nu_eV"])
        offset = None
        if self._spec.name is not None and self._spec.name != "Planck15":
            distance = self.autogalaxy().angular_diameter_distance_to_earth_in_kpc_from(0.5) / 1000.0
            reference = _flat_realization(self._spec.name).angular_diameter_distance(0.5).value
            offset = float(distance / reference - 1.0)
        return {"name": self._spec.name, "parameters": parameters, "rho_crit_convention": RHO_CRIT_CONVENTION,
                "autogalaxy_vs_astropy_d_a_rel_z0p5": offset}

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> Cosmology:
        """The cosmology a ``to_mapping`` record names; its recorded parameters must be this engine's."""
        values = ({"name": mapping["name"]} if mapping["name"] is not None else
                  {"flat_lcdm": mapping["parameters"]})
        cosmology = cls(parse_cosmology(values, "cosmology"))
        if cosmology.to_mapping() != dict(mapping):
            raise ValueError(f"cosmology record {dict(mapping)!r} differs from {cosmology.to_mapping()!r}")
        return cosmology

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Cosmology) and other._spec == self._spec

    def __hash__(self) -> int:
        return hash(self._spec)

    def __repr__(self) -> str:
        return f"Cosmology({self._spec!r})"

    def __getstate__(self) -> dict[str, Any]:
        return {"spec": self._spec}

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        Cosmology.__init__(self, state["spec"])

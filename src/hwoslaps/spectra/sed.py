"""Observed spectral shapes, including redshifted tabulated spectra."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real as _Number
from types import MappingProxyType
from typing import Any, Literal, Mapping

import numpy as np
from numpy.typing import ArrayLike

from ..config.checks import Key, Real, Table, Text, Variants
from ..identity import mapping_digest
from .tables import TABLE_TABLE, SpectralTable, TableSpec, read_table

__all__ = ["SED_TABLE", "FlatFlambda", "FlatFnu", "PowerLawSED", "SED", "SEDSpec", "TableSED", "build_sed", "parse_sed"]

_REFERENCE_WAVELENGTH_M = 1.0e-6


@dataclass(frozen=True)
class FlatFnu:
    pass


@dataclass(frozen=True)
class FlatFlambda:
    pass


@dataclass(frozen=True)
class PowerLawSED:
    index: float


@dataclass(frozen=True)
class TableSED:
    table: TableSpec
    quantity: Literal["fnu", "flambda"]
    frame: Literal["observed", "rest"] = "observed"


SEDSpec = FlatFnu | FlatFlambda | PowerLawSED | TableSED
SED_TABLE = Variants("kind", {
    "flat_fnu": Table(()),
    "flat_flambda": Table(()),
    "power_law": Table((Key("index", Real(), "index of f_nu proportional to nu**index"),)),
    "table": TABLE_TABLE.extend((Key("quantity", Text(choices=("fnu", "flambda")), "spectral density convention"),
                                  Key("frame", Text(choices=("observed", "rest")), "wavelength frame", "observed"))),
})


def parse_sed(mapping: Mapping[str, Any], path: str) -> SEDSpec:
    values = SED_TABLE.read(mapping, path)
    if values["kind"] == "flat_fnu":
        return FlatFnu()
    if values["kind"] == "flat_flambda":
        return FlatFlambda()
    if values["kind"] == "power_law":
        return PowerLawSED(values["index"])
    return TableSED(TableSpec.from_values(values), values["quantity"], values["frame"])


@dataclass(frozen=True, eq=False)
class SED:
    spec: SEDSpec
    redshift: float
    table: SpectralTable | None = None

    def __post_init__(self) -> None:
        if isinstance(self.redshift, (bool, np.bool_)) or not isinstance(self.redshift, _Number) or not np.isfinite(self.redshift) or self.redshift < 0.0:
            raise ValueError("SED redshift must be finite and non-negative")
        if isinstance(self.spec, PowerLawSED) and (isinstance(self.spec.index, (bool, np.bool_))
                or not isinstance(self.spec.index, _Number) or not np.isfinite(self.spec.index)):
            raise ValueError("power-law index must be a finite real number")
        if isinstance(self.spec, TableSED) != (self.table is not None):
            raise ValueError("a tabulated SED owns its loaded spectral table")
        if self.table is not None and np.any(self.table.values < 0.0):
            raise ValueError(f"{self.spec.table.path}: SED values must be non-negative")

    @property
    def file_digests(self) -> Mapping[str, str]:
        values = {} if self.table is None else {str(self.spec.table.path): self.table.digest}
        return MappingProxyType(values)

    def fnu(self, wavelengths_m: ArrayLike) -> np.ndarray:
        wavelengths = np.asarray(wavelengths_m, dtype=float)
        if not np.all(np.isfinite(wavelengths)) or np.any(wavelengths <= 0.0):
            raise ValueError("SED wavelengths must be positive and finite")
        if isinstance(self.spec, FlatFnu):
            result = np.ones_like(wavelengths)
        elif isinstance(self.spec, FlatFlambda):
            result = (wavelengths / _REFERENCE_WAVELENGTH_M) ** 2
        elif isinstance(self.spec, PowerLawSED):
            with np.errstate(over="ignore", invalid="ignore"):
                result = (wavelengths / _REFERENCE_WAVELENGTH_M) ** (-self.spec.index)
        else:
            evaluated = wavelengths / (1.0 + self.redshift) if self.spec.frame == "rest" else wavelengths
            if np.any(evaluated < self.table.wavelengths_m[0]) or np.any(evaluated > self.table.wavelengths_m[-1]):
                raise ValueError(f"{self.spec.table.path}: SED wavelengths exceed the table support")
            result = np.interp(evaluated, self.table.wavelengths_m, self.table.values)
            if self.spec.quantity == "flambda":
                result = result * evaluated**2
        if not np.all(np.isfinite(result)) or np.any(result < 0.0):
            raise ValueError("SED produced a non-finite or negative spectral shape")
        return np.asarray(result)

    def knots_m(self) -> np.ndarray:
        if self.table is None:
            knots = np.empty(0, dtype=float)
        else:
            factor = 1.0 + self.redshift if self.spec.frame == "rest" else 1.0
            knots = self.table.wavelengths_m * factor
        knots.setflags(write=False)
        return knots

    def digest(self) -> str:
        if isinstance(self.spec, FlatFnu):
            record = {"kind": "flat_fnu"}
        elif isinstance(self.spec, FlatFlambda):
            record = {"kind": "flat_flambda"}
        elif isinstance(self.spec, PowerLawSED):
            record = {"kind": "power_law", "index": self.spec.index}
        else:
            record = {"kind": "table", "quantity": self.spec.quantity, "frame": self.spec.frame,
                      "file_sha256": self.table.digest, "wavelength_key": self.spec.table.wavelength_key,
                      "value_key": self.spec.table.value_key, "wavelength_unit": self.spec.table.wavelength_unit}
            if self.spec.frame == "rest":
                record["redshift"] = self.redshift
        return mapping_digest(record)


def build_sed(spec: SEDSpec, *, redshift: float) -> SED:
    return SED(spec, redshift, read_table(spec.table) if isinstance(spec, TableSED) else None)

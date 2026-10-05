"""Tabulated spectral curves and the identities of the exact bytes decoded."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np

from ..config.checks import FilePath, Key, Table, Text
from ..config.loading import parse_yaml_value
from ..identity import read_file_snapshot

__all__ = ["TABLE_TABLE", "SpectralTable", "TableSpec", "parse_table", "read_table"]

WavelengthUnit = Literal["nm", "angstrom", "um", "m"]
TABLE_TABLE = Table((
    Key("path", FilePath((".yaml", ".yml", ".csv", ".npz")), "spectral table"),
    Key("wavelength_key", Text(), "wavelength column or array"),
    Key("value_key", Text(), "value column or array"),
    Key("wavelength_unit", Text(choices=("nm", "angstrom", "um", "m")), "wavelength unit"),
))


@dataclass(frozen=True)
class TableSpec:
    path: Path
    wavelength_key: str
    value_key: str
    wavelength_unit: WavelengthUnit

    @classmethod
    def from_values(cls, values: Mapping[str, Any]) -> TableSpec:
        return cls(Path(values["path"]), values["wavelength_key"], values["value_key"], values["wavelength_unit"])


def parse_table(mapping: Mapping[str, Any], path: str) -> TableSpec:
    return TableSpec.from_values(TABLE_TABLE.read(mapping, path))


@dataclass(frozen=True, eq=False)
class SpectralTable:
    wavelengths_m: np.ndarray
    values: np.ndarray
    digest: str

    def __post_init__(self) -> None:
        wavelengths = np.array(self.wavelengths_m, dtype=float, copy=True)
        values = np.array(self.values, dtype=float, copy=True)
        if wavelengths.ndim != 1 or values.shape != wavelengths.shape or wavelengths.size < 2:
            raise ValueError("a spectral table needs two or more equal-length one-dimensional arrays")
        if not np.all(np.isfinite(wavelengths)) or np.any(wavelengths <= 0.0) or np.any(np.diff(wavelengths) <= 0.0):
            raise ValueError("spectral wavelengths must be positive, finite and strictly increasing")
        if not np.all(np.isfinite(values)):
            raise ValueError("spectral values must be finite")
        wavelengths.setflags(write=False)
        values.setflags(write=False)
        object.__setattr__(self, "wavelengths_m", wavelengths)
        object.__setattr__(self, "values", values)


def read_table(spec: TableSpec) -> SpectralTable:
    """Read one named curve; the digest describes the same buffer that is decoded."""
    content, digest = read_file_snapshot(spec.path)
    suffix = spec.path.suffix.lower()
    try:
        if suffix in (".yaml", ".yml"):
            data = parse_yaml_value(content.decode("utf-8"))
            if not isinstance(data, Mapping):
                raise ValueError("the YAML table must be a mapping of columns")
        elif suffix == ".csv":
            rows = np.genfromtxt(io.StringIO(content.decode("utf-8")), delimiter=",", names=True,
                                 dtype=float, deletechars="", replace_space=" ")
            data = {} if rows.dtype.names is None else {key: np.atleast_1d(rows[key]) for key in rows.dtype.names}
        elif suffix == ".npz":
            with np.load(io.BytesIO(content), allow_pickle=False) as members:
                data = {key: members[key] for key in members.files}
        else:
            raise ValueError(f"unsupported spectral table suffix {suffix!r}")
        for key in (spec.wavelength_key, spec.value_key):
            if key not in data:
                raise ValueError(f"missing column {key!r}; columns are {list(data)}")
        wavelengths = np.asarray(data[spec.wavelength_key], dtype=float)
        values = np.asarray(data[spec.value_key], dtype=float)
        divisors = {"nm": 1.0e9, "angstrom": 1.0e10, "um": 1.0e6, "m": 1.0}
        if spec.wavelength_unit not in divisors:
            raise ValueError(f"unsupported wavelength unit {spec.wavelength_unit!r}")
        return SpectralTable(wavelengths / divisors[spec.wavelength_unit], values, digest)
    except (ValueError, TypeError, UnicodeDecodeError) as error:
        raise ValueError(f"{spec.path} ({spec.wavelength_key!r}, {spec.value_key!r}): {error}") from error

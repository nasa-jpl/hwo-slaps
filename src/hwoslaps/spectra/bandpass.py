"""Throughput curves and a shared logarithmic wavelength integration measure."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import trapezoid

from ..config.checks import ConfigError, Integer, Key, ListOf, Nullable, Pair, Real, Rule, Table, Text, Variants
from ..identity import array_digest, mapping_digest
from .tables import TABLE_TABLE, TableSpec, read_table

if TYPE_CHECKING:
    from .sed import SED

__all__ = ["BANDPASS_TABLE", "Bandpass", "BandpassSpec", "ConstantFactor", "ProductBand", "TableBand", "TableFactor",
           "TopHatBand", "bin_integrals", "build_bandpass", "integrate_dlnlambda", "parse_bandpass"]

_POSITIVE = Real(min=0.0, min_open=True)
_LABEL = Key("label", Nullable(Text()), "bandpass label", None)
_SUPPORT = Key("support_nm", Pair(_POSITIVE), "band support (low, high)", unit="nm")


def _ordered(values: Mapping[str, Any], path: str) -> None:
    low, high = (values["min_nm"], values["max_nm"]) if "min_nm" in values else values["support_nm"]
    if not high > low:
        raise ConfigError(path, "the upper wavelength must exceed the lower wavelength")


_TOP_HAT = Table((Key("min_nm", _POSITIVE, "lower wavelength", unit="nm"),
                  Key("max_nm", _POSITIVE, "upper wavelength", unit="nm"),
                  Key("throughput", Real(min=0.0, min_open=True, max=1.0), "electrons per entrance-pupil photon"),
                  _LABEL), rules=(Rule("wavelengths are ordered", _ordered),))
_TABLE_FACTOR = TABLE_TABLE.extend((Key("power", Integer(min=1), "number of identical surfaces", 1), _LABEL))
_FACTORS = Variants("kind", {"table": _TABLE_FACTOR, "top_hat": _TOP_HAT,
    "constant": Table((Key("value", Real(min=0.0, min_open=True, max=1.0), "constant factor"),))})


def _product_support(values: Mapping[str, Any], path: str) -> None:
    _ordered(values, path)
    low, high = values["support_nm"]
    for index, factor in enumerate(values["factors"]):
        if factor["kind"] == "top_hat" and (factor["min_nm"] > low or factor["max_nm"] < high):
            raise ConfigError(f"{path}.factors[{index}]", "a top-hat factor must contain the product support")


BANDPASS_TABLE = Variants("kind", {
    "top_hat": _TOP_HAT,
    "table": _TABLE_FACTOR.extend((_SUPPORT,), rules=(Rule("support is ordered", _ordered),)),
    "product": Table((_SUPPORT, Key("factors", ListOf(_FACTORS, min_length=1), "response factors"), _LABEL),
                     rules=(Rule("support is ordered and contained in each top-hat factor", _product_support),)),
})


@dataclass(frozen=True)
class TopHatBand:
    min_nm: float
    max_nm: float
    throughput: float
    label: str | None = None


@dataclass(frozen=True)
class TableFactor:
    table: TableSpec
    power: int = 1
    label: str | None = None


@dataclass(frozen=True)
class ConstantFactor:
    value: float


@dataclass(frozen=True)
class TableBand:
    table: TableSpec
    support_nm: tuple[float, float]
    power: int = 1
    label: str | None = None


@dataclass(frozen=True)
class ProductBand:
    support_nm: tuple[float, float]
    factors: tuple[TopHatBand | TableFactor | ConstantFactor, ...]
    label: str | None = None


BandpassSpec = TopHatBand | TableBand | ProductBand


def _factor(values: Mapping[str, Any]) -> TopHatBand | TableFactor | ConstantFactor:
    if values["kind"] == "top_hat":
        return TopHatBand(values["min_nm"], values["max_nm"], values["throughput"], values["label"])
    if values["kind"] == "table":
        return TableFactor(TableSpec.from_values(values), values["power"], values["label"])
    return ConstantFactor(values["value"])


def parse_bandpass(mapping: Mapping[str, Any], path: str) -> BandpassSpec:
    values = BANDPASS_TABLE.read(mapping, path)
    if values["kind"] == "top_hat":
        return _factor(values)
    if values["kind"] == "table":
        return TableBand(TableSpec.from_values(values), tuple(values["support_nm"]), values["power"], values["label"])
    return ProductBand(tuple(values["support_nm"]), tuple(_factor(item) for item in values["factors"]), values["label"])


def _integrand(values: ArrayLike, wavelengths_m: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    wavelengths = np.asarray(wavelengths_m, dtype=float)
    integrand = np.asarray(values, dtype=float)
    if wavelengths.ndim != 1 or wavelengths.size < 2 or integrand.shape != wavelengths.shape:
        raise ValueError("spectral integrands need equal one-dimensional arrays of at least two samples")
    if not np.all(np.isfinite(wavelengths)) or np.any(wavelengths <= 0.0) or np.any(np.diff(wavelengths) <= 0.0):
        raise ValueError("integration wavelengths must be positive, finite and strictly increasing")
    if not np.all(np.isfinite(integrand)):
        raise ValueError("spectral integrands must be finite")
    return integrand, wavelengths


def integrate_dlnlambda(values: np.ndarray, wavelengths_m: np.ndarray) -> float:
    integrand, wavelengths = _integrand(values, wavelengths_m)
    return float(trapezoid(integrand, np.log(wavelengths)))


def bin_integrals(values: np.ndarray, wavelengths_m: np.ndarray, edges_m: ArrayLike) -> np.ndarray:
    """Split one piecewise-linear log-wavelength integrand, without re-evaluating its spectrum."""
    integrand, wavelengths = _integrand(values, wavelengths_m)
    edges = np.asarray(edges_m, dtype=float)
    if edges.ndim != 1 or edges.size < 2 or not np.all(np.isfinite(edges)) or np.any(np.diff(edges) < 0.0):
        raise ValueError("bin edges must be a finite non-decreasing vector")
    if edges[0] < wavelengths[0] or edges[-1] > wavelengths[-1]:
        raise ValueError("bin edges must lie inside the integration support")
    log_wavelengths, log_edges = np.log(wavelengths), np.log(edges)
    edge_values = np.interp(log_edges, log_wavelengths, integrand)
    result = []
    for index, (low, high) in enumerate(zip(log_edges[:-1], log_edges[1:])):
        inside = (log_wavelengths > low) & (log_wavelengths < high)
        coordinates = np.concatenate(([low], log_wavelengths[inside], [high]))
        samples = np.concatenate(([edge_values[index]], integrand[inside], [edge_values[index + 1]]))
        result.append(float(trapezoid(samples, coordinates)))
    return np.asarray(result)


@dataclass(frozen=True, eq=False)
class Bandpass:
    support_m: tuple[float, float]
    wavelengths_m: np.ndarray
    throughput: np.ndarray
    label: str | None = None
    file_digests: Mapping[str, str] = field(default_factory=dict)
    source_spec: BandpassSpec | None = None

    def __post_init__(self) -> None:
        wavelengths = np.array(self.wavelengths_m, dtype=float, copy=True)
        throughput = np.array(self.throughput, dtype=float, copy=True)
        _integrand(throughput, wavelengths)
        if tuple(self.support_m) != (wavelengths[0], wavelengths[-1]):
            raise ValueError("bandpass grid endpoints must equal its support")
        if np.any(throughput < 0.0) or np.any(throughput > 1.0) or not np.any(throughput > 0.0):
            raise ValueError("bandpass throughput must be in [0, 1] with nonzero response")
        wavelengths.setflags(write=False)
        throughput.setflags(write=False)
        object.__setattr__(self, "wavelengths_m", wavelengths)
        object.__setattr__(self, "throughput", throughput)
        object.__setattr__(self, "file_digests", MappingProxyType(dict(self.file_digests)))

    def throughput_at(self, wavelengths_m: ArrayLike) -> np.ndarray:
        wavelengths = np.asarray(wavelengths_m, dtype=float)
        if not np.all(np.isfinite(wavelengths)) or np.any(wavelengths <= 0.0):
            raise ValueError("throughput wavelengths must be positive and finite")
        return np.asarray(np.interp(wavelengths, self.wavelengths_m, self.throughput, left=0.0, right=0.0))

    def nodes(self, count: int) -> np.ndarray:
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, Integral) or count < 1:
            raise ValueError("wavelength node count must be an integer >= 1")
        edges = np.linspace(*self.support_m, int(count) + 1)
        return (edges[1:] + edges[:-1]) / 2.0

    def bin_edges(self, nodes_m: ArrayLike) -> np.ndarray:
        nodes = np.asarray(nodes_m, dtype=float)
        if nodes.ndim != 1 or nodes.size == 0 or not np.all(np.isfinite(nodes)) or np.any(nodes <= 0.0) or np.any(np.diff(nodes) <= 0.0):
            raise ValueError("wavelength nodes must be a nonempty strictly increasing finite vector")
        interior = (nodes[1:] + nodes[:-1]) / 2.0
        return np.concatenate(([self.support_m[0]], np.clip(interior, *self.support_m), [self.support_m[1]]))

    def integration_grid(self, sed: SED | None = None) -> tuple[np.ndarray, np.ndarray]:
        if sed is None:
            return self.wavelengths_m, self.throughput
        knots = sed.knots_m()
        inside = knots[(knots > self.support_m[0]) & (knots < self.support_m[1])]
        if inside.size == 0:
            return self.wavelengths_m, self.throughput
        wavelengths = np.unique(np.concatenate((self.wavelengths_m, inside)))
        throughput = self.throughput_at(wavelengths)
        wavelengths.setflags(write=False)
        throughput.setflags(write=False)
        return wavelengths, throughput

    def digest(self) -> str:
        return mapping_digest({"wavelengths": array_digest(self.wavelengths_m), "throughput": array_digest(self.throughput)})


def build_bandpass(spec: BandpassSpec) -> Bandpass:
    support_nm = (spec.min_nm, spec.max_nm) if isinstance(spec, TopHatBand) else spec.support_nm
    support = tuple(value / 1.0e9 for value in support_nm)
    if not (0.0 < support[0] < support[1]) or not np.all(np.isfinite(support)):
        raise ValueError("bandpass support must be positive, finite and ordered")
    factors = spec.factors if isinstance(spec, ProductBand) else (spec,)
    curves, knots, files = [], [], {}
    for factor in factors:
        if isinstance(factor, (TableBand, TableFactor)):
            if isinstance(factor.power, (bool, np.bool_)) or not isinstance(factor.power, Integral) or factor.power < 1:
                raise ValueError("a table response power must be an integer >= 1")
            table = read_table(factor.table)
            if table.wavelengths_m[0] > support[0] or table.wavelengths_m[-1] < support[1]:
                raise ValueError(f"{factor.table.path}: table does not cover bandpass support {support}")
            if np.any(table.values < 0.0) or np.any(table.values > 1.0):
                raise ValueError(f"{factor.table.path}: throughput table values must be in [0, 1]")
            path = str(factor.table.path)
            if path in files and files[path] != table.digest:
                raise ValueError(f"{path}: spectral input changed between factor reads")
            files[path] = table.digest
            knots.extend(table.wavelengths_m[(table.wavelengths_m > support[0]) & (table.wavelengths_m < support[1])])
            curves.append((factor, table))
        else:
            if isinstance(factor, TopHatBand) and (factor.min_nm > support_nm[0] or factor.max_nm < support_nm[1]):
                raise ValueError("a top-hat factor must contain the product support")
            curves.append((factor, None))
    dense = np.exp(np.linspace(np.log(support[0]), np.log(support[1]), 10001))
    dense[0], dense[-1] = support
    wavelengths = np.unique(np.concatenate((dense, knots)))
    throughput = np.ones_like(wavelengths)
    for factor, table in curves:
        if table is not None:
            throughput *= np.interp(wavelengths, table.wavelengths_m, table.values) ** factor.power
        else:
            throughput *= factor.throughput if isinstance(factor, TopHatBand) else factor.value
    return Bandpass(support, wavelengths, throughput, spec.label, files, spec)

"""Deterministic named-member populations and strict effective configuration binding."""

from __future__ import annotations

import math
import re
from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from scipy.special import ndtr, ndtri

from ..config.checks import ConfigError, Integer, Key, ListOf, MapOf, Nullable, Real, Table, Text, Variants
from ..config.schema import ConfigSource, EngineConfig, parse_config, resolve_config
from ..identity import file_digest, json_ready, mapping_digest
from ..seeding import member_seed, stream_rng
from .catalog import CATALOG_TABLE, Catalog, CatalogSpec, load_catalog
from .derivations import DERIVATION_TABLE, Derivation, parse_derivation
from .distributions import (Constant, DISTRIBUTION_TABLE, Distribution, PopulationError, Reference, finite_value,
                            open_uniforms, parse_distribution, resolve_value)

__all__ = ["Copula", "PopulationSpec", "PopulationMember", "sample_population", "iter_population_members", "POPULATION_TABLE"]


def _integer(value, name):
    try:
        return Integer(min = 0)(value, name)
    except ConfigError as error:
        raise PopulationError(str(error)) from None


def _mapping(value, path):
    if not isinstance(value, Mapping):
        raise PopulationError(f"{path}: must be a mapping")
    return dict(value)


def _variable_name(name, path):
    if not isinstance(name, str) or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None:
        raise PopulationError(f"{path}: expected variable identifier, got {name!r}")


@dataclass(frozen = True)
class Copula:
    name: str
    variables: tuple[str, ...]
    correlation: np.ndarray
    cholesky: np.ndarray = field(init = False, repr = False, compare = False)

    def __post_init__(self):
        _variable_name(self.name, "copula name")
        if not isinstance(self.variables, (list, tuple)):
            raise PopulationError(f"copulas.{self.name}.variables: sequence required")
        variables = tuple(self.variables)
        if len(variables) < 2 or len(set(variables)) != len(variables):
            raise PopulationError(f"copulas.{self.name}.variables: at least2 distinct variables required")
        for variable in variables:
            _variable_name(variable, f"copulas.{self.name}.variables")
        try:
            raw = np.asarray(self.correlation)
        except (ValueError, TypeError):
            raise PopulationError(f"copulas.{self.name}.correlation: rectangular numeric matrix required") from None
        if raw.dtype.kind not in "fiu":
            raise PopulationError(f"copulas.{self.name}.correlation: finite numeric values required without coercion")
        try:
            correlation = np.array(raw, dtype = float, copy = True)
        except (ValueError, TypeError):
            raise PopulationError(f"copulas.{self.name}.correlation: numeric matrix required") from None
        if (correlation.shape != (len(variables),) * 2 or not np.all(np.isfinite(correlation))
                or not np.array_equal(correlation, correlation.T) or not np.array_equal(np.diag(correlation), np.ones(len(variables)))):
            raise PopulationError(f"copulas.{self.name}.correlation: finite exactly symmetric matrix with unit diagonal required")
        try:
            cholesky = np.linalg.cholesky(correlation)
        except np.linalg.LinAlgError:
            raise PopulationError(f"copulas.{self.name}.correlation: positive definite matrix required") from None
        correlation.flags.writeable = False
        cholesky.flags.writeable = False
        object.__setattr__(self, "variables", variables)
        object.__setattr__(self, "correlation", correlation)
        object.__setattr__(self, "cholesky", cholesky)

    def to_mapping(self):
        return {"variables": list(self.variables), "correlation": self.correlation.tolist()}


def _reference_size(variable):
    if isinstance(variable, Derivation):
        return "unknown" if variable.kind == "function" else variable.size
    if isinstance(variable, Constant):
        if isinstance(variable.value, (tuple, list)):
            return len(variable.value)
        if isinstance(variable.value, (Mapping, str, bool)) or variable.value is None:
            return "non_numeric"
    return None


def _check_reference(reference, known, path, *, scalar):
    if reference.variable not in known:
        raise PopulationError(f"{path}: reference {reference.name()} is not an earlier/catalog variable")
    size = known[reference.variable]
    if reference.index is not None:
        if size not in {"unknown"} and (not isinstance(size, int) or reference.index >= size):
            raise PopulationError(f"{path}: reference {reference.name()} outside vector")
    elif scalar and (isinstance(size, int) or size == "non_numeric"):
        raise PopulationError(f"{path}: scalar input needs an indexed numeric reference, got {reference.name()}")


@dataclass(frozen = True)
class PopulationSpec:
    variables: Mapping[str, Distribution | Derivation]
    copulas: Mapping[str, Copula] = field(default_factory = dict)
    catalog: CatalogSpec | None = None
    bind: Mapping[str, Reference] = field(default_factory = dict)
    max_attempts: int = 1

    def __post_init__(self):
        variables = _mapping(self.variables, "population.variables")
        copulas = _mapping(self.copulas, "population.copulas")
        binds = _mapping(self.bind, "population.bind")
        if self.catalog is not None and not isinstance(self.catalog, CatalogSpec):
            raise PopulationError("population.catalog: expected CatalogSpec")
        try:
            attempts = Integer(min = 1)(self.max_attempts, "population.max_attempts")
        except ConfigError as error:
            raise PopulationError(str(error)) from None
        known = {} if self.catalog is None else { ** {name: None for name in self.catalog.columns},
                                               ** {name: "non_numeric" for name in self.catalog.text_columns}}
        for name, variable in variables.items():
            _variable_name(name, "population.variables")
            if name in known:
                raise PopulationError(f"population.variables.{name}: overlaps catalog variable")
            if not isinstance(variable, (Distribution, Derivation)):
                raise PopulationError(f"population.variables.{name}: expected distribution/derivation")
            for reference in variable.references():
                _check_reference(reference, known, f"population.variables.{name}", scalar = True)
            known[name] = _reference_size(variable)
        coupled = set()
        for name, copula in copulas.items():
            if not isinstance(copula, Copula) or copula.name != name:
                raise PopulationError(f"population.copulas.{name}: mismatched Copula")
            for variable in copula.variables:
                if variable not in variables or isinstance(variables[variable], (Derivation, Constant)):
                    raise PopulationError(f"population.copulas.{name}: {variable} must name a nonconstant distribution")
                if variable in coupled:
                    raise PopulationError(f"population.copulas.{name}: {variable} belongs to multiple copulas")
                coupled.add(variable)
        if not binds:
            raise PopulationError("population.bind: required non-empty mapping")
        for path, reference in binds.items():
            if not isinstance(path, str) or not path or any(not part for part in path.split(".")):
                raise PopulationError(f"population.bind: invalid dotted path {path!r}")
            if path.split(".")[0] in {"run_name", "seed"}:
                raise PopulationError(f"population.bind.{path}: reserved identity path")
            if not isinstance(reference, Reference):
                raise PopulationError(f"population.bind.{path}: expected Reference")
            _check_reference(reference, known, f"population.bind.{path}", scalar = False)
        for path in binds:
            for other in binds:
                if path != other and other.startswith(path + "."):
                    raise PopulationError(f"population.bind: overlapping paths {path!r} and {other!r}")
        object.__setattr__(self, "variables", MappingProxyType(variables))
        object.__setattr__(self, "copulas", MappingProxyType(copulas))
        object.__setattr__(self, "bind", MappingProxyType(binds))
        object.__setattr__(self, "max_attempts", attempts)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any], *, base_dir = None):
        try:
            values = POPULATION_TABLE.read(mapping, "population")
            variables = {}
            for name, value in values["variables"].items():
                path = f"population.variables.{name}"
                kind = value.get("kind") if isinstance(value, Mapping) else None
                variables[name] = (parse_derivation(value, path) if isinstance(kind, str) and kind in {"vector", "polar_offset", "ell_comps", "shear_components", "multipole_components", "function"}
                                 else parse_distribution(value, path))
            copulas = {}
            for name, value in values["copulas"].items():
                fields = COPULA_TABLE.read(value, f"population.copulas.{name}")
                copulas[name] = Copula(name, fields["variables"], fields["correlation"])
            catalog = None if values["catalog"] is None else CatalogSpec.from_mapping(values["catalog"], base_dir = base_dir)
            binds = {path: Reference.parse(value, f"population.bind.{path}") for path, value in values["bind"].items()}
            return cls(variables, copulas, catalog, binds, values["max_attempts"])
        except ConfigError as error:
            raise PopulationError(str(error)) from None

    def to_mapping(self):
        return {"variables": {name: value.to_mapping() for name, value in self.variables.items()},
                "copulas": {name: value.to_mapping() for name, value in self.copulas.items()},
                "catalog": None if self.catalog is None else self.catalog.to_mapping(),
                "bind": {path: reference.name() for path, reference in self.bind.items()}, "max_attempts": self.max_attempts}

    def digest(self):
        return mapping_digest({"spec": self.to_mapping(), "catalog_sha256": None if self.catalog is None else file_digest(self.catalog.path)})


_VARIABLE_NAME = Text(pattern=r"[A-Za-z_][A-Za-z0-9_]*")
_REFERENCE_NAME = Text(pattern=r"[A-Za-z_][A-Za-z0-9_]*(?:\[[0-9]+\])?")
VARIABLE_TABLE = Variants("kind", {**DISTRIBUTION_TABLE.tables, **DERIVATION_TABLE.tables})
COPULA_TABLE = Table((Key("variables", ListOf(_VARIABLE_NAME, min_length=2, unique=True), "coupled distributions"),
                      Key("correlation", ListOf(ListOf(Real(), min_length=2), min_length=2), "normal-score correlation matrix")))
POPULATION_TABLE = Table((
    Key("variables", MapOf(key=_VARIABLE_NAME, value=VARIABLE_TABLE), "ordered variables"),
    Key("copulas", MapOf(key=_VARIABLE_NAME, value=COPULA_TABLE), "Gaussian copulas", {}),
    Key("catalog", Nullable(CATALOG_TABLE), "catalog", None),
    Key("bind", MapOf(Text(), _REFERENCE_NAME, min_length=1), "existing effective configuration paths"),
    Key("max_attempts", Integer(min=1), "rejection limit", 1),
))


@dataclass(frozen = True)
class PopulationMember:
    index: int
    run_name: str
    seed: int
    attempt: int
    values: Mapping[str, Any]
    config: EngineConfig

    def __post_init__(self):
        object.__setattr__(self, "values", MappingProxyType(deepcopy(dict(self.values))))

    def to_mapping(self):
        return {"index": self.index, "run_name": self.run_name, "seed": self.seed, "attempt": self.attempt,
                "values": json_ready(self.values), "config_digest": self.config.digest()}


def _spec(spec):
    return spec if isinstance(spec, PopulationSpec) else PopulationSpec.from_mapping(spec)


def _draw(spec: PopulationSpec, seed: int, index: int, attempt: int, catalog: Catalog | None):
    try:
        values = {} if catalog is None else catalog.row(index)
    except PopulationError as error:
        raise PopulationError(f"member {index}: {error}") from error
    uniforms = {}
    for name, copula in spec.copulas.items():
        scores = ndtri(open_uniforms(stream_rng(seed, "population.copula/" + name, index, attempt), len(copula.variables)))
        correlated = copula.cholesky @ scores
        for variable, u in zip(copula.variables, np.clip(ndtr(correlated), 2. ** - 53, 1. - 2. ** - 53), strict = True):
            uniforms[variable] = float(u)
    for name, variable in spec.variables.items():
        try:
            resolve = lambda value: resolve_value(value, values)
            if isinstance(variable, Derivation):
                value = variable.evaluate(resolve)
            else:
                u = 0.5 if isinstance(variable, Constant) else uniforms.get(name)
                if u is None:
                    u = float(open_uniforms(stream_rng(seed, "population/" + name, index, attempt), 1)[0])
                value = variable.quantile(u, resolve)
            values[name] = finite_value(value, f"member {index}, variable {name}")
            if isinstance(value, tuple):
                values[name] = tuple(values[name])
        except PopulationError as error:
            raise PopulationError(f"member {index}, variable {name}: {error}") from error
    return values


def sample_population(spec: PopulationSpec | Mapping[str, Any], count: int, *, seed: int, start: int = 0) -> list[dict[str, Any]]:
    spec = _spec(spec)
    count, seed, start = (_integer(value, name) for value, name in ((count, "count"), (seed, "seed"), (start, "start")))
    catalog = None if spec.catalog is None else load_catalog(spec.catalog)
    return [_draw(spec, seed, index, 0, catalog) for index in range(start, start + count)]


def _key(container, segment, path):
    if isinstance(container, Mapping):
        if segment in container:
            return segment
        if segment.isdecimal() and int(segment) in container:
            return int(segment)
    elif isinstance(container, list) and segment.isdecimal() and int(segment) < len(container):
        return int(segment)
    raise PopulationError(f"population.bind.{path}: must name an existing effective configuration key/index")


def _set_existing(mapping, path, value):
    parts = path.split(".")
    container = mapping
    for segment in parts[: - 1]:
        container = container[_key(container, segment, path)]
    key = _key(container, parts[- 1], path)
    container[key] = deepcopy(value)


def iter_population_members(base: ConfigSource, spec: PopulationSpec | Mapping[str, Any], count: int, *, seed: int, start: int = 0,
                            name_prefix: str = "system", base_dir: Path | str | None = None) -> Iterator[PopulationMember]:
    spec = spec if isinstance(spec, PopulationSpec) else PopulationSpec.from_mapping(spec, base_dir = base_dir)
    count, seed, start = (_integer(value, name) for value, name in ((count, "count"), (seed, "seed"), (start, "start")))
    if not isinstance(name_prefix, str) or re.fullmatch(r"[A-Za-z0-9_]+", name_prefix) is None:
        raise PopulationError("name_prefix: letters/digits/underscores required")
    config = resolve_config(base, base_dir = base_dir)
    mapping = config.to_mapping()
    for path in spec.bind:
        _set_existing(deepcopy(mapping), path, None)
    catalog = None if spec.catalog is None else load_catalog(spec.catalog)
    for index in range(start, start + count):
        for attempt in range(spec.max_attempts):
            try:
                values = _draw(spec, seed, index, attempt, catalog)
                selected = deepcopy(mapping)
                selected["run_name"] = f"{name_prefix}_{index:06d}"
                selected["seed"] = member_seed(seed, index)
                for path, reference in spec.bind.items():
                    _set_existing(selected, path, json_ready(reference.get(values)))
                member_config = parse_config(selected, base_dir = base_dir)
            except (PopulationError, ConfigError) as error:
                last = error
                continue
            yield PopulationMember(index, selected["run_name"], selected["seed"], attempt, values, member_config)
            break
        else:
            raise PopulationError(f"member {index}: no valid draw in {spec.max_attempts} attempts; last error: {last}") from last

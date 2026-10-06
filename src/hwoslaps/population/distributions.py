"""Strict one-dimensional population laws, sampled by an open-uniform quantile."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
from scipy.special import ndtr, ndtri

from ..config.checks import AnyValue, ConfigError, Key, ListOf, Nullable, Real, Table, Text, Union, Variants
from ..identity import json_ready

__all__ = ["PopulationError", "Reference", "Constant", "Choice", "Uniform", "LogUniform", "Normal",
           "TruncatedNormal", "LogNormal", "TruncatedLogNormal", "Distribution", "parse_distribution",
           "open_uniforms", "read_value", "resolve_value", "finite_value", "REFERENCE_TABLE", "VALUE_CHECK", "DISTRIBUTION_TABLE"]


class PopulationError(ValueError):
    """An invalid population specification or draw, with its input/member context."""


@dataclass(frozen = True)
class Reference:
    variable: str
    index: int | None = None

    def __post_init__(self):
        if not isinstance(self.variable, str) or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.variable) is None:
            raise PopulationError(f"invalid variable reference {self.variable!r}")
        if self.index is not None and (isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0):
            raise PopulationError(f"reference {self.variable}: index must be a non-negative integer")

    @classmethod
    def parse(cls, value: Any, path: str = "reference") -> Reference:
        if isinstance(value, Mapping):
            if set(value) != {"var"}:
                raise PopulationError(f"{path}: reference must have exactly the key var")
            value = value["var"]
        if not isinstance(value, str):
            raise PopulationError(f"{path}: reference must be a variable name or name[index]")
        match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)(?:\[([0-9]+)\])?", value)
        if match is None:
            raise PopulationError(f"{path}: invalid reference {value!r}")
        return cls(match[1], None if match[2] is None else int(match[2]))

    def name(self) -> str:
        return self.variable + ("" if self.index is None else f"[{self.index}]")

    def get(self, values: Mapping[str, Any]) -> Any:
        if self.variable not in values:
            raise PopulationError(f"reference {self.name()}: variable is unavailable")
        value = values[self.variable]
        if self.index is None:
            return value
        if not isinstance(value, (list, tuple)) or self.index >= len(value):
            raise PopulationError(f"reference {self.name()}: index outside vector")
        return value[self.index]


Value = float | Reference


def read_value(value: Any, path: str) -> Value:
    if isinstance(value, Mapping):
        return Reference.parse(value, path)
    if isinstance(value, str):
        raise PopulationError(f"{path}: must be a finite number, not text {value!r}; unquote numeric values "
                              "(YAML1.1 may read1e7 without a decimal point as text)")
    try:
        return Real()(value, path)
    except ConfigError as error:
        raise PopulationError(str(error)) from None


def resolve_value(value: Value, values: Mapping[str, Any]) -> float:
    if isinstance(value, Reference):
        return read_value(value.get(values), f"reference {value.name()}")
    return value


def finite_value(value: Any, path: str) -> Any:
    """An independent JSON-shaped value; reject unsupported/nonfinite nested values."""
    try:
        return json_ready(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise PopulationError(f"{path}: {error}") from None


def _number(value: float, name: str, *, positive = False) -> float:
    value = read_value(value, name)
    if isinstance(value, Reference) or (positive and value <= 0.0):
        raise PopulationError(f"{name}: must be {'positive ' if positive else ''}finite number")
    return value


def _bounds(low, high, *, positive = False):
    low, high = _number(low, "low", positive = positive), _number(high, "high", positive = positive)
    if not low < high:
        raise PopulationError(f"high must exceed low, got [{low!r}, {high!r}]")
    return low, high


def _finite_scalar(value):
    return _number(value, "distribution result")


class _Law:
    kind: str

    def references(self) -> tuple[Reference, ...]:
        return tuple(getattr(self, field.name) for field in fields(self)
                     if isinstance(getattr(self, field.name), Reference))

    def to_mapping(self) -> dict[str, Any]:
        return {"kind": self.kind, ** {field.name:({"var": value.name()} if isinstance(value, Reference)
                                                   else deepcopy(value))
                                    for field in fields(self) for value in (getattr(self, field.name),)}}

    def quantile(self, u: float, resolve: Callable[[Value], float]) -> Any:
        if not 0.0 < u < 1.0:
            raise PopulationError(f"{self.kind}: quantile input must be in (0,1)")
        try:
            with np.errstate(over = "ignore", invalid = "ignore", divide = "ignore"):
                return self._quantile(u, resolve)
        except (OverflowError, FloatingPointError) as error:
            raise PopulationError(f"{self.kind}: draw exceeded finite numerical support") from error

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if not isinstance(value, Reference):
                object.__setattr__(self, field.name, read_value(value, f"{self.kind}.{field.name}"))
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name in {"std", "median", "sigma_ln"} and not isinstance(value, Reference):
                _number(value, f"{self.kind}.{field.name}", positive = True)
        if hasattr(self, "low") and not isinstance(self.low, Reference) and not isinstance(self.high, Reference):
            _bounds(self.low, self.high, positive = self.kind in {"log_uniform", "truncated_lognormal"})
        elif hasattr(self, "low") and self.kind in {"log_uniform", "truncated_lognormal"}:
            for name in ("low", "high"):
                value = getattr(self, name)
                if not isinstance(value, Reference):
                    _number(value, name, positive = True)


@dataclass(frozen = True)
class Constant(_Law):
    value: Any
    kind = "constant"
    def __post_init__(self):
        object.__setattr__(self, "value", finite_value(self.value, "constant.value"))
    def references(self):
        return ()
    def quantile(self, u, resolve):
        return deepcopy(self.value)


@dataclass(frozen = True)
class Choice(_Law):
    values: tuple[Any, ...]
    weights: tuple[float, ...] | None = None
    kind = "choice"
    def __post_init__(self):
        if not isinstance(self.values, (tuple, list)) or not self.values:
            raise PopulationError("choice.values: must be a non-empty sequence")
        object.__setattr__(self, "values", tuple(finite_value(value, "choice.values") for value in self.values))
        if self.weights is not None:
            if not isinstance(self.weights, (tuple, list)) or len(self.weights) != len(self.values):
                raise PopulationError("choice.weights: length must equal values")
            weights = tuple(_number(value, "choice.weights") for value in self.weights)
            if any(value < 0 for value in weights) or not math.isfinite(sum(weights)) or sum(weights) <= 0:
                raise PopulationError("choice.weights: finite non-negative weights with positive finite sum required")
            object.__setattr__(self, "weights", weights)
    def references(self):
        return ()
    def _quantile(self, u, resolve):
        weights = np.ones(len(self.values)) if self.weights is None else np.asarray(self.weights)
        cumulative = np.cumsum(weights) / np.sum(weights)
        index = min(int(np.searchsorted(cumulative, u, side = "right")), len(self.values) - 1)
        return deepcopy(self.values[index])


@dataclass(frozen = True)
class Uniform(_Law):
    low: Value
    high: Value
    kind = "uniform"
    def _quantile(self, u, resolve):
        low, high = _bounds(resolve(self.low), resolve(self.high))
        return _finite_scalar(low + u * (high - low))


@dataclass(frozen = True)
class LogUniform(_Law):
    low: Value
    high: Value
    kind = "log_uniform"
    def _quantile(self, u, resolve):
        low, high = _bounds(resolve(self.low), resolve(self.high), positive = True)
        value = _finite_scalar(np.exp(np.log(low) + u * (np.log(high) - np.log(low))))
        return float(np.clip(value, low, high))


@dataclass(frozen = True)
class Normal(_Law):
    mean: Value
    std: Value
    kind = "normal"
    def _quantile(self, u, resolve):
        mean, std = _number(resolve(self.mean), "mean"), _number(resolve(self.std), "std", positive = True)
        return _finite_scalar(mean + std * ndtri(u))


@dataclass(frozen = True)
class TruncatedNormal(_Law):
    mean: Value
    std: Value
    low: Value
    high: Value
    kind = "truncated_normal"
    def _quantile(self, u, resolve):
        mean, std = _number(resolve(self.mean), "mean"), _number(resolve(self.std), "std", positive = True)
        low, high = _bounds(resolve(self.low), resolve(self.high))
        a, b = (low - mean) / std, (high - mean) / std
        if a <= 0:
            probability = ndtr(a) + u * (ndtr(b) - ndtr(a))
            t = ndtri(probability)
        else:
            probability = ndtr(- b) + (1 - u) * (ndtr(- a) - ndtr(- b))
            t = - ndtri(probability)
        value = _finite_scalar(mean + std * t)
        return float(np.clip(value, low, high))


@dataclass(frozen = True)
class LogNormal(_Law):
    median: Value
    sigma_ln: Value
    kind = "lognormal"
    def _quantile(self, u, resolve):
        median = _number(resolve(self.median), "median", positive = True)
        sigma = _number(resolve(self.sigma_ln), "sigma_ln", positive = True)
        return _finite_scalar(median * np.exp(sigma * ndtri(u)))


@dataclass(frozen = True)
class TruncatedLogNormal(_Law):
    median: Value
    sigma_ln: Value
    low: Value
    high: Value
    kind = "truncated_lognormal"
    def _quantile(self, u, resolve):
        median = _number(resolve(self.median), "median", positive = True)
        sigma = _number(resolve(self.sigma_ln), "sigma_ln", positive = True)
        low, high = _bounds(resolve(self.low), resolve(self.high), positive = True)
        logvalue = TruncatedNormal(math.log(median), sigma, math.log(low), math.log(high)).quantile(u, lambda v: v)
        value = _finite_scalar(np.exp(logvalue))
        return float(np.clip(value, low, high))


Distribution = Constant | Choice | Uniform | LogUniform | Normal | TruncatedNormal | LogNormal | TruncatedLogNormal
_LAWS = {law.kind: law for law in (Constant, Choice, Uniform, LogUniform, Normal, TruncatedNormal, LogNormal, TruncatedLogNormal)}


REFERENCE_TABLE = Table((Key("var", Text(pattern=r"[A-Za-z_][A-Za-z0-9_]*(?:\[[0-9]+\])?"),
                                  "earlier variable, optionally indexed"),))
VALUE_CHECK = Union((Real(), REFERENCE_TABLE))

_DISTRIBUTION_TABLES = {}
for _kind, _law in _LAWS.items():
    _keys = []
    for _item in fields(_law):
        if _item.name == "weights":
            _keys.append(Key("weights", Nullable(ListOf(Real(min=0.0))), "choice weights", None))
        elif _law is Constant:
            _keys.append(Key("value", AnyValue(), "constant JSON-shaped value"))
        elif _law is Choice:
            _keys.append(Key("values", ListOf(AnyValue(), min_length=1), "choice values"))
        else:
            _keys.append(Key(_item.name, VALUE_CHECK, "finite numeric parameter or earlier reference"))
    _DISTRIBUTION_TABLES[_kind] = Table(tuple(_keys))
DISTRIBUTION_TABLE = Variants("kind", _DISTRIBUTION_TABLES)


def parse_distribution(mapping: Mapping[str, Any], path: str) -> Distribution:
    try:
        values = DISTRIBUTION_TABLE.read(mapping, path)
        law = _LAWS[values.pop("kind")]
        return law(**values)
    except (ConfigError, PopulationError) as error:
        raise PopulationError(f"{path}: {error}") from None


def open_uniforms(rng: np.random.Generator, size: int) -> np.ndarray:
    return (2 * rng.integers(0, 2 ** 52, size = size, dtype = np.int64) + 1) * 2.0 ** - 53

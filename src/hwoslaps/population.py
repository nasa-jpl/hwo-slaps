"""Declarative, reproducible parameter populations for forecasting experiments.

This module samples independent parameter distributions; it does not impose an
astrophysical population, selection function, telescope, or source morphology.
Each member and parameter has its own deterministic random stream, so adding a
parameter, changing population size, or partitioning work does not move
existing
samples. Correlations and conditional distributions require an explicit caller
recipe rather than an implicit assumption in this sampler.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from copy import deepcopy
import hashlib
import math
from typing import Any

import numpy as np

__all__ = ["sample_population", "iter_population_configs"]


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0):
        qualifier = "positive finite" if positive else "finite"
        raise ValueError(f"{name} must be a {qualifier} number")
    return number


def _rng(seed: int, index: int, parameter: str) -> np.random.Generator:
    digest = hashlib.sha256(parameter.encode("utf-8")).digest()
    stream = tuple(int.from_bytes(digest[i:i + 4], "little") for i in range(0, 32, 4))
    sequence = np.random.SeedSequence(seed, spawn_key=(index, *stream))
    return np.random.Generator(np.random.PCG64(sequence))


def _noise_seed(seed: int, index: int) -> int:
    """Injectively encode population seed and member index as a Python integer.

    Cantor pairing avoids the birthday collisions of drawing a 32-bit seed.
    NumPy SeedSequence accepts arbitrary non-negative integer entropy.
    """
    total = seed + index
    return total * (total + 1) // 2 + index


def _checked_draw(evaluate, rng: np.random.Generator, name: str) -> Any:
    """Reject draws outside finite numerical support."""
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            value = evaluate(rng)
    except (OverflowError, FloatingPointError) as exc:
        raise ValueError(f"{name} draw exceeded finite numerical support") from exc

    def check(item):
        if isinstance(item, (float, np.floating)) and not math.isfinite(item):
            raise ValueError(f"{name} distribution produced a non-finite draw")
        if isinstance(item, Mapping):
            for nested in item.values():
                check(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                check(nested)
    check(value)
    return value


def _distribution(spec: Mapping[str, Any], name: str):
    """Validate a distribution once and return its random-stream evaluator."""
    if not isinstance(spec, Mapping):
        raise ValueError(f"{name} distribution must be a mapping")
    kind = spec.get("distribution")
    fields = {
        "constant": {"distribution", "value"},
        "choice": {"distribution", "values", "weights"},
        "uniform": {"distribution", "low", "high"},
        "log_uniform": {"distribution", "low", "high"},
        "normal": {"distribution", "mean", "std"},
        "truncated_normal": {"distribution", "mean", "std", "low", "high"},
        "lognormal": {"distribution", "median", "sigma_ln"},
    }
    if not isinstance(kind, str) or kind not in fields:
        raise ValueError(f"{name} has unsupported distribution {kind!r}")
    unexpected = set(spec) - fields[kind]
    if unexpected:
        raise ValueError(f"{name} contains unsupported fields: {sorted(unexpected)}")
    required = fields[kind] - {"weights"}
    missing = required - set(spec)
    if missing:
        raise ValueError(f"{name} is missing fields: {sorted(missing)}")
    if kind == "constant":
        value = deepcopy(spec["value"])
        return lambda rng: deepcopy(value)
    if kind == "choice":
        values = spec["values"]
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(f"{name}.values must be a non-empty list")
        values = deepcopy(values)
        weights = spec.get("weights")
        if weights is not None:
            if not isinstance(weights, (list, tuple)) or len(weights) != len(values):
                raise ValueError(f"{name}.weights must match values")
            weights = np.asarray([_number(w, f"{name}.weights") for w in weights])
            if np.any(weights < 0) or not np.isfinite(weights.sum()) or weights.sum() <= 0:
                raise ValueError(f"{name}.weights must be non-negative with positive sum")
            weights = weights / weights.sum()
        return lambda rng: deepcopy(values[int(rng.choice(len(values), p=weights))])
    if kind in {"uniform", "log_uniform", "truncated_normal"}:
        low = _number(spec["low"], f"{name}.low", positive=kind == "log_uniform")
        high = _number(spec["high"], f"{name}.high", positive=kind == "log_uniform")
        if high <= low:
            raise ValueError(f"{name}.high must exceed low")
    if kind == "uniform":
        return lambda rng: float(rng.uniform(low, high))
    if kind == "log_uniform":
        return lambda rng: float(np.exp(rng.uniform(math.log(low), math.log(high))))
    if kind in {"normal", "truncated_normal"}:
        mean = _number(spec["mean"], f"{name}.mean")
        std = _number(spec["std"], f"{name}.std", positive=True)
        if kind == "normal":
            return lambda rng: float(rng.normal(mean, std))
        from scipy.stats import truncnorm
        lower, upper = (low - mean) / std, (high - mean) / std
        return lambda rng: float(truncnorm.rvs(lower, upper, loc=mean, scale=std, random_state=rng))
    median = _number(spec["median"], f"{name}.median", positive=True)
    sigma = _number(spec["sigma_ln"], f"{name}.sigma_ln", positive=True)
    return lambda rng: float(rng.lognormal(math.log(median), sigma))


def _evaluators(parameters: Mapping[str, Mapping[str, Any]]):
    if not isinstance(parameters, Mapping) or not parameters:
        raise ValueError("parameters must be a non-empty mapping")
    evaluators = {}
    for name, spec in parameters.items():
        if not isinstance(name, str) or not name or any(not part for part in name.split(".")):
            raise ValueError("parameter names must be non-empty dotted paths")
        evaluators[name] = _distribution(spec, name)
    return evaluators


def sample_population(
    parameters: Mapping[str, Mapping[str, Any]],
    count: int,
    *,
    seed: int,
    start: int = 0,
) -> list[dict[str, Any]]:
    """Sample independent distributions with stable member identities.

    ``parameters`` maps arbitrary parameter names to distribution
    specifications.
    Supported distributions are constant, choice, uniform, log_uniform, normal,
    truncated_normal, and lognormal. Normal uses mean/std; lognormal uses the
    median and natural-log standard deviation. ``start`` permits deterministic
    chunking. This function changes neither its inputs nor global RNG state.
    """
    count, seed, start = _integer(count, "count"), _integer(seed, "seed"), _integer(start, "start")
    evaluators = _evaluators(parameters)
    return [
        {name: _checked_draw(evaluate, _rng(seed, index, name), name)
         for name, evaluate in evaluators.items()}
        for index in range(start, start + count)
    ]


def _set_existing(config: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    target = config
    for key in parts[:-1]:
        if key not in target or not isinstance(target[key], dict):
            raise ValueError(f"population parameter {path!r} does not name an existing configuration field")
        target = target[key]
    if parts[-1] not in target:
        raise ValueError(f"population parameter {path!r} does not name an existing configuration field")
    target[parts[-1]] = deepcopy(value)


def iter_population_configs(
    base_config: Mapping[str, Any],
    parameters: Mapping[str, Mapping[str, Any]],
    count: int,
    *,
    seed: int,
    start: int = 0,
    name_prefix: str = "system",
    validate: bool = True,
) -> Iterator[dict[str, Any]]:
    """Yield engine configurations with sampled fields and unique member seeds.

    Parameter names are dotted configuration paths. Mappings/lists may be
    replaced as complete values, so ``choice`` can select source templates or
    whole instrument blocks. Overlapping parent/child paths are rejected.
    ``run_name`` and ``global_seed`` are generated from the member identity;
    they cannot be population parameters. Load/resolve file paths before using
    the resulting configurations, or supply a base directory to
    ``run_pipeline``.
    Validation delegates to the engine's existing physical configuration
    checks.
    """
    if not isinstance(base_config, Mapping):
        raise ValueError("base_config must be a mapping")
    if not isinstance(name_prefix, str) or not name_prefix or not all(
        char.isalnum() or char == "_" for char in name_prefix
    ):
        raise ValueError("name_prefix must contain only letters, digits and underscores")
    if not isinstance(validate, bool):
        raise ValueError("validate must be boolean")
    count, seed, start = _integer(count, "count"), _integer(seed, "seed"), _integer(start, "start")
    evaluators = _evaluators(parameters)
    paths = set(evaluators)
    if paths & {"run_name", "global_seed"}:
        raise ValueError("run_name and global_seed are derived from population identity")
    for path in paths:
        for other in paths:
            if path != other and other.startswith(path + "."):
                raise ValueError(f"population paths {path!r} and {other!r} overlap")
        _set_existing(deepcopy(dict(base_config)), path, None)
    if validate:
        from .config.validation import validate_or_raise
    for index in range(start, start + count):
        config = deepcopy(dict(base_config))
        config["run_name"] = f"{name_prefix}_{index:06d}"
        config["global_seed"] = _noise_seed(seed, index)
        for path, evaluate in evaluators.items():
            _set_existing(config, path, _checked_draw(evaluate, _rng(seed, index, path), path))
        if validate:
            validate_or_raise(config)
        yield config

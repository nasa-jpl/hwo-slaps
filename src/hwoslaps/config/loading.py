"""Portable configuration loading and composition.

Configuration files are plain mappings. Later files override earlier files;
mapping values merge recursively, while sequences and scalars replace whole
values. Paths belong to the file that declares them, so reusable instrument
and scene fragments need no assumptions about the caller's working directory.

``read_yaml``, ``parse_yaml_value`` and ``dump_yaml`` read and write YAML with the
YAML 1.2 core schema (``1.0e8`` and ``1e8`` are floats, ``010`` is 10, ``yes`` and
``1:20`` are strings, no timestamps or merge keys) and reject duplicate keys, which
PyYAML would resolve silently by keeping the last one. ``set_path`` and
``parse_assignment`` edit configurations by dotted key path.
"""

from __future__ import annotations

import math
import re
from collections.abc import Hashable, Mapping, Sequence
from copy import deepcopy
from os import PathLike
from pathlib import Path
from typing import Any

import yaml

from .checks import ConfigError
from .validation import validate_or_raise

Config = dict[str, Any]
ConfigPath = str | PathLike[str]

# Only schema-defined filesystem fields are resolved. Arbitrary strings such
# as cosmology names, device labels and model identifiers remain untouched.
_PATH_FIELDS = (
    ("plotting", "output_dir"),
    ("lensing", "source_galaxy", "light", "asset_path"),
    ("modeling", "fisher", "covariance_path"),
    ("psf", "kernel", "path"),
    ("psf", "fit_kernel", "path"),
    ("modeling", "fit_psf", "psf", "kernel", "path"),
    ("modeling", "fit_psf", "psf", "fit_kernel", "path"),
    ("modeling", "fit_psf", "delta", "prior_table"),
)


def merge_configs(*configs: Mapping[str, Any]) -> Config:
    """Return a recursively merged copy without changing any input mapping."""
    merged: Config = {}
    for config in configs:
        if not isinstance(config, Mapping):
            raise ValueError("Each configuration must be a mapping")
        for key, value in config.items():
            if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
                merged[key] = merge_configs(merged[key], value)
            else:
                merged[key] = deepcopy(value)
    return merged


def resolve_config_paths(
    config: Mapping[str, Any], *, base_dir: ConfigPath | None = None
) -> Config:
    """Copy a configuration and make schema-defined filesystem paths absolute.

    ``base_dir`` defaults to the current working directory for configurations
    constructed in Python. File loading supplies each YAML file's directory.
    ``None`` path values remain unchanged for the validator to interpret.
    """
    resolved = merge_configs(config)
    root = Path.cwd() if base_dir is None else Path(base_dir).expanduser().resolve()
    for keys in _PATH_FIELDS:
        section: Any = resolved
        for key in keys[:-1]:
            if not isinstance(section, Mapping):
                break
            section = section.get(key)
        else:
            if not isinstance(section, dict) or keys[-1] not in section:
                continue
            value = section[keys[-1]]
            if value is None:
                continue
            if not isinstance(value, (str, PathLike)):
                raise ValueError(f"{'.'.join(keys)} must be a filesystem path string")
            if not str(value).strip():
                raise ValueError(f"{'.'.join(keys)} must not be empty")
            path = Path(value).expanduser()
            section[keys[-1]] = str((root / path).resolve())
    return resolved


def load_config(
    paths: ConfigPath | Sequence[ConfigPath],
    *,
    overrides: Mapping[str, Any] | None = None,
    base_dir: ConfigPath | None = None,
    validate: bool = True,
) -> Config:
    """Load one YAML configuration or compose an ordered sequence of files.

    Relative paths are resolved against the directory of the file declaring
    them *before* composition. ``base_dir`` explicitly overrides this rule,
    for example when replaying historical repository-relative configurations.
    Python ``overrides`` resolve paths against ``base_dir`` or the caller's
    working directory. The final composed document is validated once.
    """
    if isinstance(paths, (str, PathLike)):
        config_paths = [paths]
    elif isinstance(paths, Sequence):
        config_paths = list(paths)
    else:
        raise ValueError("Configuration paths must be a path or sequence of paths")
    if not config_paths:
        raise ValueError("At least one configuration file is required")

    documents = []
    for path_value in config_paths:
        path = Path(path_value).expanduser().resolve()
        try:
            with path.open("r", encoding="utf-8") as stream:
                document = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML in {path}: {exc}") from exc
        if not isinstance(document, dict):
            raise ValueError(f"Configuration {path} must contain a YAML mapping")
        documents.append(
            resolve_config_paths(document, base_dir=path.parent if base_dir is None else base_dir)
        )
    config = merge_configs(*documents)
    if overrides is not None:
        config = merge_configs(config, resolve_config_paths(overrides, base_dir=base_dir))
    if validate:
        validate_or_raise(config)
    return config


_CORE_SCHEMA = (
    ("tag:yaml.org,2002:null", r"^(?:~|null|Null|NULL|)$", ["~", "n", "N", ""]),
    ("tag:yaml.org,2002:bool", r"^(?:true|True|TRUE|false|False|FALSE)$", list("tTfF")),
    ("tag:yaml.org,2002:int", r"^(?:[-+]?[0-9]+|0o[0-7]+|0x[0-9a-fA-F]+)$", list("-+0123456789")),
    ("tag:yaml.org,2002:float",
     r"^(?:[-+]?(?:\.[0-9]+|[0-9]+(?:\.[0-9]*)?)(?:[eE][-+]?[0-9]+)?|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$",
     list("-+0123456789.")),
)
_SPECIAL_FLOATS = {".inf": math.inf, "+.inf": math.inf, "-.inf": -math.inf, ".nan": math.nan}


def _core_schema(cls):
    """Replace the YAML 1.1 implicit resolvers of ``cls`` by the YAML 1.2 core schema."""
    cls.yaml_implicit_resolvers = {}
    for tag, pattern, first in _CORE_SCHEMA:
        cls.add_implicit_resolver(tag, re.compile(pattern), first)
    return cls


@_core_schema
class _Loader(yaml.SafeLoader):
    def construct_core_int(self, node):
        text = self.construct_scalar(node)
        base = {"0o": 8, "0x": 16}.get(text[:2], 10)
        try:
            return int(text[2:] if base != 10 else text, base)
        except ValueError:
            raise yaml.constructor.ConstructorError(
                None, None, f"invalid integer {text!r}", node.start_mark) from None

    def construct_core_float(self, node):
        text = self.construct_scalar(node)
        if text.lower() in _SPECIAL_FLOATS:
            return _SPECIAL_FLOATS[text.lower()]
        try:
            return float(text)
        except ValueError:
            raise yaml.constructor.ConstructorError(
                None, None, f"invalid float {text!r}", node.start_mark) from None

    def construct_mapping(self, node, deep=False):
        if isinstance(node, yaml.MappingNode):
            seen = set()
            for key_node, _ in node.value:
                key = self.construct_object(key_node, deep=deep)
                if isinstance(key, Hashable):
                    if key in seen:
                        mark = key_node.start_mark
                        raise ConfigError("", f"{mark.name}:{mark.line + 1}: duplicate key {key!r}")
                    seen.add(key)
        return super().construct_mapping(node, deep=deep)


_Loader.add_constructor("tag:yaml.org,2002:int", _Loader.construct_core_int)
_Loader.add_constructor("tag:yaml.org,2002:float", _Loader.construct_core_float)


@_core_schema
class _Dumper(yaml.SafeDumper):
    """Quotes every string the YAML 1.2 loader would read as null, a boolean or a number."""


def _load(stream: Any) -> Any:
    loader = _Loader(stream)
    try:
        return loader.get_single_data()
    finally:
        loader.dispose()


def read_yaml(path: ConfigPath) -> dict[str, Any]:
    """Read one YAML mapping with YAML 1.2 core scalars and duplicate keys rejected."""
    location = Path(path)
    if not location.is_file():
        raise ConfigError("", f"{location}: no such file")
    with location.open("r", encoding="utf-8") as stream:
        try:
            document = _load(stream)
        except yaml.MarkedYAMLError as exc:
            mark = exc.problem_mark or exc.context_mark
            where = f"{location}:{mark.line + 1}" if mark is not None else str(location)
            problem = " ".join(part for part in (exc.context, exc.problem) if part)
            raise ConfigError("", f"{where}: {problem}") from exc
        except yaml.YAMLError as exc:
            raise ConfigError("", f"{location}: {exc}") from exc
    if not isinstance(document, dict):
        raise ConfigError("", f"{location}: the document must be a mapping, got {type(document).__name__}")
    return document


def parse_yaml_value(text: str) -> Any:
    """Parse one YAML value (a scalar, flow list or flow mapping) with the YAML 1.2 core schema."""
    try:
        return _load(text)
    except yaml.YAMLError as exc:
        raise ConfigError("", f"cannot parse {text!r} as YAML: {exc}") from exc


def dump_yaml(mapping: Mapping[str, Any]) -> str:
    """Block-style YAML in key order that ``read_yaml`` reads back to an equal mapping."""
    if not isinstance(mapping, Mapping):
        raise TypeError(f"dump_yaml needs a mapping, got {type(mapping).__name__}")
    return yaml.dump(dict(mapping), Dumper=_Dumper, default_flow_style=False, sort_keys=False,
                     allow_unicode=True)


def set_path(mapping: Mapping[str, Any], dotted: str, value: Any, *, create: bool) -> dict[str, Any]:
    """A deep copy of ``mapping`` with the value at the dotted key path set.

    With ``create=False`` every segment must already exist; with ``create=True``
    missing intermediate mappings are created. Integer-keyed maps are set whole.
    """
    segments = dotted.split(".")
    if not all(segments):
        raise ConfigError(dotted, "empty segment in the key path")
    result = deepcopy(dict(mapping))
    node = result
    for depth, segment in enumerate(segments):
        where = ".".join(segments[:depth + 1])
        if segment not in node and not create:
            raise ConfigError(where, "no such key")
        if depth == len(segments) - 1:
            node[segment] = deepcopy(value)
            break
        child = node.setdefault(segment, {})
        if not isinstance(child, dict):
            raise ConfigError(where, f"is not a mapping, got {child!r}")
        node = child
    return result


def parse_assignment(text: str) -> tuple[str, Any]:
    """Split ``PATH=VALUE`` (a ``--set`` argument) at the first ``=``; the value is YAML."""
    dotted, separator, value = text.partition("=")
    if not separator or not dotted or dotted != dotted.strip():
        raise ConfigError("", f"expected PATH=VALUE, got {text!r}")
    return dotted, parse_yaml_value(value)


__all__ = [
    "Config", "dump_yaml", "load_config", "merge_configs", "parse_assignment", "parse_yaml_value",
    "read_yaml", "resolve_config_paths", "set_path",
]

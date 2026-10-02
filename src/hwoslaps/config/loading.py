"""Portable configuration loading and composition.

Configuration files are plain mappings. Later files override earlier files;
mapping values merge recursively, while sequences and scalars replace whole
values. Paths belong to the file that declares them, so reusable instrument
and scene fragments need no assumptions about the caller's working directory.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from os import PathLike
from pathlib import Path
from typing import Any

import yaml

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




__all__ = ["Config", "load_config", "merge_configs", "resolve_config_paths"]

"""The engine configuration boundary and its scientific identity.

Each section supplies its own key table and parser. Files compose before the root
is read; a path belongs to the file declaring it. Scientific identities include
file bytes and component order, while run labels remain bookkeeping.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass, replace
from os import PathLike
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeAlias

from .checks import ConfigError, FilePath, Identifier, Integer, Key, Nullable, Table
from .loading import read_yaml
from ..fisher.spec import CROSS_RULES as FISHER_CROSS_RULES, FORECAST_TABLE, ForecastSpec
from ..identity import file_digest, mapping_digest
from ..instrument import CROSS_RULES as INSTRUMENT_CROSS_RULES, INSTRUMENT_TABLE, InstrumentSpec
from ..observation.normalization import CROSS_RULES as OBSERVATION_CROSS_RULES
from ..observation.observation import OBSERVATION_TABLE, ObservationSpec
from ..optics.providers import CROSS_RULES as OPTICS_CROSS_RULES, PSF_TABLE, PsfSpec, parse_psf
from ..scene.cosmology import COSMOLOGY_TABLE, CosmologySpec, parse_cosmology
from ..scene.spec import SCENE_TABLE, SceneSpec, parse_scene

__all__ = ["ConfigSource", "EngineConfig", "ROOT_TABLE", "compose_config", "load_config",
           "parse_config", "resolve_config"]

ROOT_TABLE = Table((
    Key("run_name", Identifier(), "run label, excluded from scientific identity", "run"),
    Key("seed", Integer(min=0), "root of named scene-randomness streams; never a noise seed"),
    Key("cosmology", COSMOLOGY_TABLE, "cosmology of the lensing scene"),
    Key("scene", SCENE_TABLE, "smooth scene, subhalo hypothesis and injected halo"),
    Key("psf", PSF_TABLE, "truth and model point spread functions"),
    Key("instrument", INSTRUMENT_TABLE, "instrument and detector noise parameters"),
    Key("observation", OBSERVATION_TABLE, "exposure and sky background"),
    Key("forecast", Nullable(FORECAST_TABLE), "Fisher forecast inputs", None),
), rules=OPTICS_CROSS_RULES + INSTRUMENT_CROSS_RULES + OBSERVATION_CROSS_RULES + FISHER_CROSS_RULES,
   doc="Engine configuration. Section-local rules run before these cross-section rules.")


def _copy_spec(value: Any, memo: dict[int, Any]) -> Any:
    """Copy typed section graphs, including read-only mapping proxies.

    Each field is copied without parsing, validation or file access.
    """
    if isinstance(value, Mapping):
        copied = {deepcopy(key, memo): _copy_spec(item, memo) for key, item in value.items()}
        return MappingProxyType(copied) if isinstance(value, MappingProxyType) else copied
    if is_dataclass(value) and not isinstance(value, type):
        return replace(value, **{item.name: _copy_spec(getattr(value, item.name), memo)
                                 for item in fields(value) if item.init})
    if isinstance(value, tuple):
        return tuple(_copy_spec(item, memo) for item in value)
    if isinstance(value, list):
        return [_copy_spec(item, memo) for item in value]
    return deepcopy(value, memo)


@dataclass(frozen=True, eq=False, init=False)
class EngineConfig:
    """Configuration built by parse_config or load_config and changed through replace.

    Typed sections and their effective mapping are created together. Arbitrary typed
    construction would let executed inputs disagree with replay and scientific identity.
    """

    run_name: str
    seed: int
    cosmology: CosmologySpec
    scene: SceneSpec
    psf: PsfSpec
    instrument: InstrumentSpec
    observation: ObservationSpec
    forecast: ForecastSpec | None
    _values: Mapping[str, Any] = field(repr=False)

    __hash__ = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("construct EngineConfig with parse_config or load_config; "
                        "change inputs with EngineConfig.replace(overrides)")

    def __deepcopy__(self, memo: dict[int, Any]) -> EngineConfig:
        copy = object.__new__(type(self))
        memo[id(self)] = copy
        for item in fields(self):
            object.__setattr__(copy, item.name, _copy_spec(getattr(self, item.name), memo))
        return copy

    def to_mapping(self) -> dict[str, Any]:
        """An independent effective mapping, with table defaults and absolute paths."""
        return deepcopy(dict(self._values))

    def _identity_record(self, *, content: bool, manifest: Mapping[str, str] | None = None) -> dict[str, Any]:
        def record(path: str, check: FilePath) -> Any:
            if not content:
                return path
            return {"file_sha256": file_digest(path) if manifest is None else manifest[path]}
        values = ROOT_TABLE.digest_record(self._values, record)
        del values["run_name"]
        return values

    def digest(self) -> str:
        """Recompute scientific identity from the current bytes of every referenced file."""
        return self.capture_identity()["config_digest"]

    def comparison_digest(self) -> str:
        """Identity of all scientific inputs except the model PSF."""
        return self.capture_identity()["comparison_digest"]

    def capture_identity(self) -> dict[str, Any]:
        """Both scientific digests from one captured manifest of the referenced files."""
        manifest = dict(self.file_digests())
        values = self._identity_record(content=True, manifest=manifest)
        config_digest = mapping_digest(values)
        del values["psf"]["model"]
        return {"config_digest": config_digest, "comparison_digest": mapping_digest(values),
                "file_digests": manifest}

    def file_digests(self) -> Mapping[str, str]:
        """Current SHA-256 of every referenced file, keyed by its absolute path."""
        digests: dict[str, str] = {}
        def record(path: str, check: FilePath) -> str:
            if path not in digests:
                digests[path] = file_digest(path)
            return path
        ROOT_TABLE.transform_paths(self._values, record)
        return digests

    def replace(self, overrides: Mapping[str, Any], *, base_dir: PathLike[str] | str | None = None) -> EngineConfig:
        """Compose an overlay with this configuration and parse the resulting inputs."""
        if not isinstance(overrides, Mapping):
            raise ConfigError("", "overrides must be a mapping")
        resolved = ROOT_TABLE.transform_paths(overrides, _resolver(base_dir))
        return parse_config(ROOT_TABLE.merge(self.to_mapping(), resolved))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EngineConfig):
            return NotImplemented
        return self._identity_record(content=False) == other._identity_record(content=False)


ConfigPath: TypeAlias = str | PathLike[str]
ConfigSource: TypeAlias = EngineConfig | Mapping[str, Any] | ConfigPath | Sequence[ConfigPath]


def _resolver(base_dir: ConfigPath | None):
    root = Path.cwd() if base_dir is None else Path(base_dir).expanduser().resolve()
    def resolve(value: str, check: FilePath) -> str:
        return str((root / Path(value).expanduser()).resolve())
    return resolve


def compose_config(paths: ConfigPath | Sequence[ConfigPath], *, overrides: Mapping[str, Any] | None = None,
                   base_dir: ConfigPath | None = None) -> dict[str, Any]:
    """Compose files in order, resolving each file's paths against its own directory.

    ``base_dir`` applies only to Python overrides. The final mapping has not yet
    been read through the tables, so required keys may be supplied by later files.
    """
    if isinstance(paths, (str, PathLike)):
        locations = [paths]
    elif isinstance(paths, Sequence):
        locations = list(paths)
    else:
        raise ConfigError("", "configuration paths must be a path or a sequence of paths")
    if not locations:
        raise ConfigError("", "at least one configuration file is required")
    result: dict[str, Any] = {}
    for location in locations:
        if not isinstance(location, (str, PathLike)):
            raise ConfigError("", f"configuration path must be a path, got {location!r}")
        path = Path(location).expanduser().resolve()
        document = ROOT_TABLE.transform_paths(read_yaml(path), _resolver(path.parent))
        result = ROOT_TABLE.merge(result, document)
    if overrides is not None:
        if not isinstance(overrides, Mapping):
            raise ConfigError("", "overrides must be a mapping")
        result = ROOT_TABLE.merge(result, ROOT_TABLE.transform_paths(overrides, _resolver(base_dir)))
    return result


def parse_config(mapping: Mapping[str, Any], *, base_dir: ConfigPath | None = None) -> EngineConfig:
    """Read engine inputs strictly and build the typed section specifications."""
    if not isinstance(mapping, Mapping):
        raise ConfigError("", "configuration must be a mapping")
    values = ROOT_TABLE.read(ROOT_TABLE.transform_paths(mapping, _resolver(base_dir)), "")
    sections = dict(
        run_name=values["run_name"], seed=values["seed"], cosmology=parse_cosmology(values["cosmology"]),
        scene=parse_scene(values["scene"]), psf=parse_psf(values["psf"]),
        instrument=InstrumentSpec.from_values(values["instrument"]),
        observation=ObservationSpec.from_values(values["observation"]),
        forecast=None if values["forecast"] is None else ForecastSpec.from_values(values["forecast"]),
        _values=values,
    )
    config = object.__new__(EngineConfig)
    for name, value in sections.items():
        object.__setattr__(config, name, value)
    return config


def load_config(paths: ConfigPath | Sequence[ConfigPath], *, overrides: Mapping[str, Any] | None = None,
                base_dir: ConfigPath | None = None) -> EngineConfig:
    """Compose engine YAML files and read the resulting configuration."""
    return parse_config(compose_config(paths, overrides=overrides, base_dir=base_dir))


def resolve_config(source: ConfigSource, *, base_dir: ConfigPath | None = None) -> EngineConfig:
    """Resolve the configuration input accepted by engine entry points."""
    if isinstance(source, EngineConfig):
        return source
    if isinstance(source, Mapping):
        return parse_config(source, base_dir=base_dir)
    return load_config(source, base_dir=base_dir)

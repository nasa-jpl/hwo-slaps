"""Portable, versioned persistence for explicit subhalo forecast results."""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .modeling.forecast_results import ForecastResult

_SCHEMA_VERSION = 1
_REQUIRED_ARRAYS = (
    "masses_msun", "positions_yx", "q_asimov", "fisher_raw", "fisher_profiled",
    "sigma_amplitude", "degradation",
)
_OPTIONAL_ARRAYS = (
    "amplitude_hat", "q_mismatch", "z_mismatch", "amplitude_spurious",
    "q_spurious", "z_spurious",
)


def save_forecast_result(result: ForecastResult, path: str | Path) -> Path:
    """Atomically publish one NPZ result, refusing to replace an existing file.

    Arrays retain undefined diagnostics such as NaN and infinity. Runtime
    provenance is explicit JSON data; this function does not discover source,
    configuration, study, or environment identity on the caller's behalf.
    """
    destination = Path(path)
    provenance = result.runtime_provenance
    if provenance is not None and not isinstance(provenance, Mapping):
        raise ValueError("runtime_provenance must be a mapping or None")
    metadata = json.dumps(
        None if provenance is None else dict(provenance), sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    )
    payload = {name: np.asarray(getattr(result, name)) for name in _REQUIRED_ARRAYS}
    payload.update({
        name: np.asarray(getattr(result, name)) for name in _OPTIONAL_ARRAYS
        if getattr(result, name) is not None
    })
    payload.update(schema_version=np.asarray(_SCHEMA_VERSION),
                   runtime_provenance_json=np.asarray(metadata))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=destination.parent, prefix=".forecast-", suffix=".npz",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        # A hard link publishes the completed bytes without an overwrite race.
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def load_forecast_result(path: str | Path) -> ForecastResult:
    """Load a current forecast artifact without pickle or rendering imports."""
    from .modeling.forecast_results import ForecastResult

    with np.load(path, allow_pickle=False) as stored:
        required = {*_REQUIRED_ARRAYS, "schema_version", "runtime_provenance_json"}
        missing = required.difference(stored.files)
        unknown = set(stored.files).difference(required | set(_OPTIONAL_ARRAYS))
        if missing or unknown:
            raise ValueError(
                f"Invalid forecast artifact fields: missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}"
            )
        version = stored["schema_version"]
        if version.shape != () or version.dtype.kind not in "iu" or int(version) != _SCHEMA_VERSION:
            raise ValueError("Unsupported forecast artifact schema_version")
        metadata = stored["runtime_provenance_json"]
        if metadata.shape != () or metadata.dtype.kind != "U":
            raise ValueError("runtime_provenance_json must be a scalar JSON string")
        provenance = json.loads(str(metadata))
        if provenance is not None and not isinstance(provenance, dict):
            raise ValueError("runtime_provenance must be a mapping or None")
        payload = {name: stored[name] for name in _REQUIRED_ARRAYS}
        payload.update({name: stored[name] for name in _OPTIONAL_ARRAYS if name in stored.files})
        return ForecastResult(**payload, runtime_provenance=provenance)

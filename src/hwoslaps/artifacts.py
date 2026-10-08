"""Versioned scientific artifacts, published atomically without replacing existing files.

Forecast NPZ schema 2 records the arrays and layout of a ForecastResult. Observation
NPZ schema 1 records detector images, exposure, distinct kernels and fiducial sampling.
Case JSON schema 1 records the typed nonlinear result. Loaders accept only these
versions and never unpickle. Equal NPZ content has equal ZIP timestamps and bytes.
"""
from __future__ import annotations

import io
import json
import os
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

from .config.loading import dump_yaml
from .identity import canonical_json, json_ready, native_ready, read_file_snapshot

if TYPE_CHECKING:
    from .fisher.result import ForecastResult
    from .inference.result import CaseResult
    from .observation.observation import Observation
    from .scene.image_source import ImageAsset

__all__ = ["save_forecast", "load_forecast", "save_observation", "load_observation",
           "save_case", "load_case", "load_case_snapshot", "save_image_asset", "write_json", "write_yaml"]
_STATISTICS = ("fisher_raw", "fisher_profiled", "sigma_amplitude", "q_asimov", "degradation")
_AMPLITUDES = {"amplitude_hat": ("amplitude_hat", "q_mismatch", "z_mismatch"),
               "amplitude_spurious": ("amplitude_spurious", "q_spurious", "z_spurious")}
_FORECAST_REQUIRED = {"schema_version", "artifact_kind", "masses_msun", "positions_yx", "positions_json",
                      "config_json", "provenance_json", "psf_relation", *_STATISTICS}
_GRID_ARRAYS = {"grid_y_coords", "grid_x_coords", "grid_indices"}


def _publish(path, writer) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=".hwoslaps-", dir=destination.parent,
                                         delete=False) as stream:
            temporary = Path(stream.name)
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def _npz(path, members) -> Path:
    def write(stream):
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in sorted(members):
                value = members[name]
                array = np.ascontiguousarray(value) if np.ndim(value) else np.asarray(value)
                if array.dtype.hasobject:
                    raise TypeError(f"artifact member {name!r} has object dtype")
                buffer = io.BytesIO()
                np.lib.format.write_array(buffer, array, allow_pickle=False)
                info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, buffer.getvalue())
    return _publish(path, write)


def _read_npz(path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != len(set(archive.files)):
            raise ValueError("duplicate artifact members")
        return {name: archive[name] for name in archive.files}


def _scalar(members, name, kind):
    if name not in members:
        raise ValueError(f"missing artifact member {name!r}")
    value = members[name]
    if value.shape != () or value.dtype.kind not in kind:
        raise ValueError(f"{name} must be a scalar of kind {kind}")
    return value.item()


def _keys(mapping, expected, what):
    if not isinstance(mapping, Mapping) or set(mapping) != set(expected):
        raise ValueError(f"{what} must have exactly the keys {sorted(expected)}")


def _json_value(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result
    return json_ready(json.loads(text, object_pairs_hook=unique))


def _json_member(members, name):
    return _json_value(_scalar(members, name, "U"))


def _float_array(members, name):
    if name not in members:
        raise ValueError(f"missing artifact member {name!r}")
    array = members[name]
    if array.dtype != np.dtype("float64"):
        raise ValueError(f"{name} must have float64 dtype, got {array.dtype}")
    array.setflags(write=False)
    return array


def write_json(path, payload: Mapping[str, Any]) -> Path:
    """Publish finite JSON values without replacing an existing destination.

    Parameters
    ----------
    path : path-like
        Destination; missing parent directories are created.
    payload : mapping
        Portable values. Integer keys become JSON strings; colliding keys,
        non-finite values and unsupported objects raise an error.

    Returns
    -------
    Path
        Destination after atomic publication. Existing files raise FileExistsError.
    """
    if not isinstance(payload, Mapping):
        raise TypeError("write_json requires a mapping")
    text = json.dumps(json_ready(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    return _publish(path, lambda stream: stream.write(text.encode("utf-8")))


def write_yaml(path, mapping: Mapping[str, Any]) -> Path:
    """Publish portable YAML while preserving native integer scientific keys.

    Parameters
    ----------
    path : path-like
        Destination; missing parent directories are created.
    mapping : mapping
        Finite portable values, including integer-keyed wavefront maps. Path
        values become strings, tuples become lists and signed zero is normalized.

    Returns
    -------
    Path
        Destination after atomic publication. Existing files raise FileExistsError.

    Notes
    -----
    Key collisions and unsupported values raise an error before anything is written.
    Unlike JSON transport, YAML retains string and integer mapping key types.
    """
    text = dump_yaml(native_ready(mapping))
    return _publish(path, lambda stream: stream.write(text.encode("utf-8")))


def save_forecast(result: ForecastResult, path) -> Path:
    """Publish a version-2 forecast NPZ archive without overwriting a file.

    Parameters
    ----------
    result : ForecastResult
        Mass-by-position statistics, layout and captured metadata. Masses use
        solar masses and positions use ``(y, x)`` arcsec coordinates.
    path : path-like
        Destination filename.

    Returns
    -------
    Path
        Published archive, including available mismatch/spurious amplitudes.

    Notes
    -----
    Configuration and provenance are stored in JSON form. This preserves result
    metadata rather than reconstructing a live preparation.
    """
    from .fisher.result import ForecastResult
    if not isinstance(result, ForecastResult):
        raise TypeError("save_forecast requires a ForecastResult")
    positions = result.positions
    layout = {"kind": positions.kind, "centre_yx": positions.centre_yx,
              "domain_radius_arcsec": positions.domain_radius_arcsec,
              "spacing_arcsec": None if positions.grid is None else positions.grid.spacing_arcsec}
    members = {"schema_version": np.asarray(2, dtype=np.int64), "artifact_kind": np.asarray("forecast"),
               "masses_msun": result.masses_msun, "positions_yx": positions.positions_yx,
               "positions_json": np.asarray(canonical_json(layout)),
               "config_json": np.asarray(json.dumps(json_ready(result.config), separators=(",", ":"), allow_nan=False)),
               "provenance_json": np.asarray(canonical_json(result.provenance)),
               "psf_relation": np.asarray(result.psf_relation)}
    members.update({name: getattr(result, name) for name in _STATISTICS})
    for amplitude, group in _AMPLITUDES.items():
        if getattr(result, amplitude) is not None:
            members.update({name: getattr(result, name) for name in group})
    for name in ("cell_areas_arcsec2", "boundary"):
        if getattr(positions, name) is not None:
            members[name] = getattr(positions, name)
    if positions.grid is not None:
        members.update(grid_y_coords=positions.grid.y_coords, grid_x_coords=positions.grid.x_coords,
                       grid_indices=positions.grid.indices)
    return _npz(path, members)


def load_forecast(path) -> ForecastResult:
    """Load and validate the current forecast archive without unpickling.

    Parameters
    ----------
    path : path-like
        Version-2 forecast NPZ filename.

    Returns
    -------
    ForecastResult
        Read-only arrays, spatial layout, PSF relation and JSON-form metadata.

    Notes
    -----
    The loader checks member sets, array types/shapes, grouped optional fields
    and byte equality of stored derived statistics with the reconstructed result.
    It does not reopen the original referenced scene or PSF files.
    """
    from .fisher.positions import GridIndex, PositionSet
    from .fisher.result import ForecastResult
    members = _read_npz(path)
    optional = {"cell_areas_arcsec2", "boundary", *_GRID_ARRAYS,
                *(name for group in _AMPLITUDES.values() for name in group)}
    if not _FORECAST_REQUIRED <= members.keys() or members.keys() - _FORECAST_REQUIRED - optional:
        raise ValueError("invalid forecast artifact member set")
    if _scalar(members, "schema_version", "iu") != 2 or _scalar(members, "artifact_kind", "U") != "forecast":
        raise ValueError("unsupported forecast artifact schema")
    layout = _json_member(members, "positions_json")
    _keys(layout, ("kind", "centre_yx", "domain_radius_arcsec", "spacing_arcsec"), "forecast positions")
    present_grid = _GRID_ARRAYS & members.keys()
    if present_grid and present_grid != _GRID_ARRAYS:
        raise ValueError("grid arrays must be present together")
    if (layout["kind"] == "grid") != bool(present_grid):
        raise ValueError("grid metadata and arrays disagree")
    grid = None if not present_grid else GridIndex(_float_array(members, "grid_y_coords"),
        _float_array(members, "grid_x_coords"), members["grid_indices"], layout["spacing_arcsec"])
    if "boundary" in members and members["boundary"].dtype != bool:
        raise ValueError("boundary must have bool dtype")
    positions = PositionSet(layout["kind"], _float_array(members, "positions_yx"), tuple(layout["centre_yx"]),
                            layout["domain_radius_arcsec"],
                            _float_array(members, "cell_areas_arcsec2") if "cell_areas_arcsec2" in members else None,
                            members.get("boundary"), grid)
    amplitudes = {}
    for amplitude, group in _AMPLITUDES.items():
        present = set(group) & members.keys()
        if present and present != set(group):
            raise ValueError(f"{group} must be present together")
        amplitudes[amplitude] = _float_array(members, amplitude) if present else None
    result = ForecastResult(_float_array(members, "masses_msun"), positions,
                            _float_array(members, "fisher_raw"), _float_array(members, "fisher_profiled"),
                            amplitudes["amplitude_hat"], amplitudes["amplitude_spurious"],
                            _scalar(members, "psf_relation", "U"), _json_member(members, "config_json"),
                            _json_member(members, "provenance_json"))
    for name in (*_STATISTICS, *(key for group in _AMPLITUDES.values() for key in group)):
        if name in members:
            stored = _float_array(members, name)
            derived = np.asarray(getattr(result, name))
            if stored.shape != derived.shape or stored.tobytes() != derived.tobytes():
                raise ValueError(f"forecast statistic {name} is inconsistent with the stored information")
    return result


def save_observation(observation: Observation, path) -> Path:
    """Publish a version-1 detector observation NPZ without overwriting a file.

    Parameters
    ----------
    observation : Observation
        Expected or noisy images and noise map in ADU, light-rate maps in e-/s,
        exposure, distinct detector kernels, sampling and captured identities.
    path : path-like
        Destination filename.

    Returns
    -------
    Path
        Published archive. Expected observations reuse their expected image
        as data; noisy observations additionally store their sampled data.
    """
    from .observation.observation import Observation
    if not isinstance(observation, Observation):
        raise TypeError("save_observation requires an Observation")
    metadata = observation.to_mapping()
    for key in ("data_digest", "expected_digest", "noise_map_digest", "sampling"):
        del metadata[key]
    from ._version import __version__
    metadata.update(schema="hwoslaps.observation", version=1, hwoslaps_version=__version__)
    members = {"metadata_json": np.asarray(canonical_json(metadata)),
               "sampling_json": np.asarray(canonical_json(observation.sampling)),
               "expected_adu": observation.expected_adu, "noise_map_adu": observation.noise_map_adu,
               "light_rate_e_per_s": observation.light_rate_e_per_s}
    if observation.kind == "noisy":
        members["data_adu"] = observation.data_adu
    for plane, rate in observation.light_rate_by_plane_e_per_s.items():
        members[f"light__{plane}_e_per_s"] = rate
    for index, kernel in enumerate(observation.psfs.kernels):
        members[f"kernel_{index}"] = kernel.kernel
    return _npz(path, members)


def load_observation(path) -> Observation:
    """Reconstruct a detector observation from its NPZ archive.

    Parameters
    ----------
    path : path-like
        Version-1 observation NPZ filename.

    Returns
    -------
    Observation
        Validated detector arrays, exposure, kernel identities, light planes,
        sampling and optional injected halo, in their original units.

    Notes
    -----
    Pickled members raise an error. Kernel bytes must match their recorded identity;
    injected halo reconstruction retains its normal physical validation requirements.
    """
    from .instrument import Detector
    from .observation.expected import Exposure
    from .observation.normalization import PhotometryRecord
    from .observation.observation import Observation
    from .optics.kernels import DetectorPSF, KernelBinding
    from .scene.halos import Halo
    from .scene.spec import GridSpec
    members = _read_npz(path)
    metadata = _json_member(members, "metadata_json")
    _keys(metadata, ("schema", "version", "kind", "noise_seed", "grid", "exposure", "kernels",
                     "subhalo", "config_digest", "photometry", "hwoslaps_version"), "observation metadata")
    if metadata["schema"] != "hwoslaps.observation" or type(metadata["version"]) is not int or metadata["version"] != 1:
        raise ValueError("unsupported observation artifact schema")
    binding = metadata["kernels"]
    _keys(binding, ("kernels", "groups"), "kernel binding")
    kernels = []
    for index, record in enumerate(binding["kernels"]):
        _keys(record, ("identity", "source"), "kernel record")
        identity = record["identity"]
        kernel = DetectorPSF.from_array(_float_array(members, f"kernel_{index}"), identity["pixel_scale_arcsec"],
                                        normalize=False, source=record["source"])
        if kernel.kernel_identity().to_mapping() != identity:
            raise ValueError(f"kernel_{index} does not match its recorded identity")
        kernels.append(kernel)
    psfs = KernelBinding(tuple(kernels), binding["groups"])
    required = {"metadata_json", "sampling_json", "expected_adu", "noise_map_adu", "light_rate_e_per_s",
                "light__source_e_per_s", *(f"kernel_{index}" for index in range(len(kernels)))}
    if metadata["kind"] == "noisy":
        required.add("data_adu")
    if members.keys() != required and members.keys() != required | {"light__lens_e_per_s"}:
        raise ValueError("invalid observation artifact member set")
    grid = metadata["grid"]
    _keys(grid, ("shape", "pixel_scale_arcsec", "over_sample_size"), "observation grid")
    exposure = metadata["exposure"]
    _keys(exposure, ("detector", "exposure_time_s", "exposure_count", "sky_rate_e_per_s"), "exposure")
    detector = exposure["detector"]
    _keys(detector, ("gain_e_per_adu", "read_noise_e", "dark_current_e_per_s"), "detector")
    response = Exposure(Detector(**detector), exposure["exposure_time_s"], exposure["sky_rate_e_per_s"],
                        exposure["exposure_count"])
    expected = _float_array(members, "expected_adu")
    total = _float_array(members, "light_rate_e_per_s")
    by_plane = {plane: _float_array(members, f"light__{plane}_e_per_s") for plane in ("lens", "source")
                if f"light__{plane}_e_per_s" in members}
    if "lens" not in by_plane:
        if by_plane["source"].tobytes() != total.tobytes():
            raise ValueError("source-only observation rates disagree")
        by_plane["source"] = total
    photometry = metadata["photometry"]
    return Observation(metadata["kind"], _float_array(members, "data_adu") if metadata["kind"] == "noisy" else expected,
                       expected, _float_array(members, "noise_map_adu"), total, by_plane,
                       GridSpec(tuple(grid["shape"]), grid["pixel_scale_arcsec"], grid["over_sample_size"]),
                       response, psfs, metadata["noise_seed"],
                       None if metadata["subhalo"] is None else Halo.from_mapping(metadata["subhalo"]),
                       metadata["config_digest"], None if photometry is None else PhotometryRecord(**photometry),
                       _json_member(members, "sampling_json"))


def save_case(result: CaseResult, path) -> Path:
    """Publish a version-1 nonlinear case JSON archive without overwriting a file.

    Parameters
    ----------
    result : CaseResult
        Typed physical trial, observation record, role outcomes and provenance.
    path : path-like
        Destination filename.

    Returns
    -------
    Path
        Published case archive with portable finite JSON values.
    """
    from .inference.result import CaseResult
    if not isinstance(result, CaseResult):
        raise TypeError("save_case requires a CaseResult")
    return write_json(path, {"schema": "hwoslaps.case", "version": 1, "result": result.to_mapping()})


def load_case(path) -> CaseResult:
    """Load a case through the full typed physical archive validator.

    Parameters
    ----------
    path : path-like
        Version-1 nonlinear case JSON filename.

    Returns
    -------
    CaseResult
        Reconstructed trial, observation and role records. Physical validation
        retains the normal runtime dependencies of those values.
    """
    return load_case_snapshot(path)[0]


def load_case_snapshot(path) -> tuple[CaseResult, str]:
    """Validate a case and identify the exact bytes consumed by its loader.

    Parameters
    ----------
    path : path-like
        Version-1 nonlinear case JSON filename.

    Returns
    -------
    CaseResult
        Fully validated typed case.
    str
        Full SHA-256 hex digest of the same UTF-8 bytes decoded for the case.
    """
    from .inference.result import CaseResult
    content, digest = read_file_snapshot(path)
    value = _json_value(content.decode("utf-8"))
    _keys(value, ("schema", "version", "result"), "case artifact")
    if value["schema"] != "hwoslaps.case" or type(value["version"]) is not int or value["version"] != 1:
        raise ValueError("unsupported case artifact schema")
    return CaseResult.from_mapping(value["result"]), digest


def save_image_asset(asset: ImageAsset, path) -> Path:
    from .scene.image_source import ImageAsset
    if not isinstance(asset, ImageAsset):
        raise TypeError("save_image_asset requires an ImageAsset")
    return _npz(path, {"sb": asset.sb, "pixel_scale_arcsec": np.asarray(asset.pixel_scale_arcsec),
                       "metadata_json": np.asarray(canonical_json(asset.metadata))})

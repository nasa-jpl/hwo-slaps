"""Shared PSF-mismatch construction and serialization helpers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np


__all__ = (
    "PsfMismatchSpec",
    "build_psf_mismatch_spec",
    "generate_fit_psf",
)

_BANK_VERSION_PACKAGES = ("hwoslaps", "hcipy", "numpy")


def _kernel_sha256(kernel: Any) -> str:
    """Return the canonical SHA-256 for a native detector kernel."""
    array = np.ascontiguousarray(np.asarray(kernel, dtype=np.float64))
    if array.ndim != 2:
        raise ValueError("PSF bank kernels must be two-dimensional")
    prefix = f"{array.shape[0]}x{array.shape[1]}:".encode("utf-8")
    return hashlib.sha256(prefix + array.tobytes()).hexdigest()


def _resolve_prior_table_path(path: Any) -> Path:
    """Resolve a prior table through absolute, CWD, then repository paths.

    Notes
    -----
    The repository-relative fallback intentionally supports the project's
    repo-checkout execution model. Installed-package layouts must provide an
    absolute path or a path resolvable from the current working directory.
    """
    requested = Path(path).expanduser()
    if requested.is_absolute():
        candidates = (requested,)
    else:
        import hwoslaps

        repository_root = Path(hwoslaps.__file__).resolve().parents[2]
        candidates = (Path.cwd() / requested, repository_root / requested)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    attempted = ", ".join(str(candidate.resolve()) for candidate in candidates)
    raise FileNotFoundError(
        f"PSF bank prior table {path!s} was not found; tried: {attempted}"
    )


def _empty_aberrations() -> dict:
    """Return an all-disabled aberrations block."""
    return {
        "enable_segment_pistons": False,
        "enable_segment_tiptilts": False,
        "enable_segment_hexikes": False,
        "enable_global_zernikes": False,
        "segment_pistons": {},
        "segment_tiptilts": {},
        "segment_hexikes": {},
        "global_zernikes": {},
    }


def _canonical_aberrations(aberrations: dict) -> dict:
    """Return aberrations with canonical float coefficients and pair lists."""
    canonical = deepcopy(aberrations)
    if "segment_pistons" in canonical:
        canonical["segment_pistons"] = {
            key: float(value)
            for key, value in canonical["segment_pistons"].items()
        }
    if "segment_tiptilts" in canonical:
        canonical["segment_tiptilts"] = {
            key: [float(value[0]), float(value[1])]
            for key, value in canonical["segment_tiptilts"].items()
        }
    if "segment_hexikes" in canonical:
        canonical["segment_hexikes"] = {
            segment: {
                mode: float(value) for mode, value in modes.items()
            }
            for segment, modes in canonical["segment_hexikes"].items()
        }
    if "global_zernikes" in canonical:
        canonical["global_zernikes"] = {
            key: float(value)
            for key, value in canonical["global_zernikes"].items()
        }
    return canonical


def _flat_int_map_to_wire(mapping: Optional[dict]) -> Optional[list]:
    """Encode one integer-keyed scalar map as sorted entries."""
    if mapping is None:
        return None
    return [[int(key), float(value)] for key, value in sorted(mapping.items())]


def _nested_int_map_to_wire(mapping: Optional[dict]) -> Optional[list]:
    """Encode nested integer-keyed scalar maps as sorted entries."""
    if mapping is None:
        return None
    return [
        [int(segment), _flat_int_map_to_wire(modes)]
        for segment, modes in sorted(mapping.items())
    ]


def _flat_int_map_from_wire(entries: Optional[list]) -> Optional[dict]:
    """Decode sorted scalar entries to integer-keyed maps."""
    if entries is None:
        return None
    return {int(key): float(value) for key, value in entries}


def _nested_int_map_from_wire(entries: Optional[list]) -> Optional[dict]:
    """Decode nested sorted entries to integer-keyed maps."""
    if entries is None:
        return None
    return {
        int(segment): _flat_int_map_from_wire(modes)
        for segment, modes in entries
    }


def _aberrations_to_wire(aberrations: dict) -> dict:
    """Encode integer-keyed aberration maps without JSON key coercion."""
    wire = deepcopy(aberrations)
    if "segment_pistons" in wire:
        wire["segment_pistons"] = _flat_int_map_to_wire(
            aberrations["segment_pistons"]
        )
    if "segment_tiptilts" in wire:
        wire["segment_tiptilts"] = [
            [int(key), [float(value[0]), float(value[1])]]
            for key, value in sorted(aberrations["segment_tiptilts"].items())
        ]
    if "segment_hexikes" in wire:
        wire["segment_hexikes"] = _nested_int_map_to_wire(
            aberrations["segment_hexikes"]
        )
    if "global_zernikes" in wire:
        wire["global_zernikes"] = _flat_int_map_to_wire(
            aberrations["global_zernikes"]
        )
    return wire


def _aberrations_from_wire(wire: dict) -> dict:
    """Restore integer-keyed aberration maps from typed entries."""
    aberrations = deepcopy(wire)
    if "segment_pistons" in aberrations:
        aberrations["segment_pistons"] = _flat_int_map_from_wire(
            wire["segment_pistons"]
        )
    if "segment_tiptilts" in aberrations:
        aberrations["segment_tiptilts"] = {
            int(key): [float(value[0]), float(value[1])]
            for key, value in wire["segment_tiptilts"]
        }
    if "segment_hexikes" in aberrations:
        aberrations["segment_hexikes"] = _nested_int_map_from_wire(
            wire["segment_hexikes"]
        )
    if "global_zernikes" in aberrations:
        aberrations["global_zernikes"] = _flat_int_map_from_wire(
            wire["global_zernikes"]
        )
    return aberrations


def _current_versions() -> dict:
    """Return the software versions recorded for generated PSFs."""
    from ..provenance import _package_version

    return {
        package: _package_version(package)
        for package in _BANK_VERSION_PACKAGES
    }


def _canonical_psf_config(psf_config: dict) -> dict:
    """Return a copied PSF config with canonical aberration coefficients."""
    canonical = deepcopy(psf_config)
    aberrations = _canonical_aberrations(
        canonical["aberrations"]
    )
    if "segment_pistons" in aberrations:
        aberrations["segment_pistons"] = {
            key: _canonical_float(value)
            for key, value in aberrations["segment_pistons"].items()
        }
    if "segment_tiptilts" in aberrations:
        aberrations["segment_tiptilts"] = {
            key: [_canonical_float(value[0]), _canonical_float(value[1])]
            for key, value in aberrations["segment_tiptilts"].items()
        }
    if "segment_hexikes" in aberrations:
        aberrations["segment_hexikes"] = {
            segment: {
                mode: _canonical_float(value) for mode, value in modes.items()
            }
            for segment, modes in aberrations["segment_hexikes"].items()
        }
    if "global_zernikes" in aberrations:
        aberrations["global_zernikes"] = {
            key: _canonical_float(value)
            for key, value in aberrations["global_zernikes"].items()
        }
    canonical["aberrations"] = aberrations
    return canonical


def _canonical_float(value: Any) -> float:
    """Return a float with signed zero normalized."""
    canonical = float(value)
    return 0.0 if canonical == 0.0 else canonical


def _identity_from_payload(payload: dict) -> str:
    """Return a 16-hex identity from canonical JSON inputs."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _sum_flat_maps(base: dict, draw: dict) -> dict:
    """Add two scalar maps over their key union."""
    return {
        key: float(base.get(key, 0.0)) + float(draw.get(key, 0.0))
        for key in sorted(set(base) | set(draw))
    }


def _sum_nested_maps(base: dict, draw: dict) -> dict:
    """Add two nested scalar maps over both key unions."""
    summed = {}
    for segment in sorted(set(base) | set(draw)):
        modes = _sum_flat_maps(
            base.get(segment, {}),
            draw.get(segment, {}),
        )
        if modes:
            summed[segment] = modes
    return summed


def _difference_flat_maps(fit: dict, truth: dict) -> dict:
    """Subtract truth scalar coefficients from fit over their key union."""
    return {
        key: float(fit.get(key, 0.0)) - float(truth.get(key, 0.0))
        for key in sorted(set(fit) | set(truth))
    }


def _difference_nested_maps(fit: dict, truth: dict) -> dict:
    """Subtract truth nested coefficients from fit over both key unions."""
    return {
        segment: _difference_flat_maps(
            fit.get(segment, {}),
            truth.get(segment, {}),
        )
        for segment in sorted(set(fit) | set(truth))
    }


def _add_draw_to_truth(truth_aberrations: dict, draw: dict) -> dict:
    """Add one draw to the truth aberrations as applied by the generator."""
    summed = _empty_aberrations()
    for flag, map_name in (
        ("enable_segment_pistons", "segment_pistons"),
        ("enable_segment_tiptilts", "segment_tiptilts"),
    ):
        values = (
            deepcopy(truth_aberrations[map_name])
            if truth_aberrations[flag]
            else {}
        )
        summed[map_name] = values
        summed[flag] = bool(truth_aberrations[flag] and values)

    truth_segment = (
        truth_aberrations["segment_hexikes"]
        if truth_aberrations["enable_segment_hexikes"]
        else {}
    )
    truth_global = (
        truth_aberrations["global_zernikes"]
        if truth_aberrations["enable_global_zernikes"]
        else {}
    )
    summed["segment_hexikes"] = _sum_nested_maps(
        truth_segment,
        draw["segment_hexikes"],
    )
    summed["global_zernikes"] = _sum_flat_maps(
        truth_global,
        draw["global_zernikes"],
    )
    summed["enable_segment_hexikes"] = bool(summed["segment_hexikes"])
    summed["enable_global_zernikes"] = bool(summed["global_zernikes"])
    return _canonical_aberrations(summed)


@dataclass(frozen=True)
class PsfMismatchSpec:
    """One deterministic explicit or additive fit-PSF specification.

    Parameters
    ----------
    mode : `str`
        Mismatch mode, ``"delta"`` or ``"explicit"``.
    delta_id : `str`
        Stable 16-hex mismatch identity.
    fit_psf_config : `dict`
        Canonical fit-side PSF configuration.
    requested_amplitude_rms_nm : `float`, optional
        Requested additive draw RMS.
    measured_draw_rms_nm : `float`, optional
        Measured additive draw RMS.
    family : `str`, optional
        Delta draw family.
    seed : `int`, optional
        Delta draw seed.
    draw_aberrations : `dict`, optional
        Canonical delta-only raw aberrations.
    orthonormal_segment : `dict`, optional
        Segment-side orthonormal coefficients.
    orthonormal_global : `dict`, optional
        Global orthonormal coefficients.
    prior_table_path : `str`, optional
        Resolved delta prior-table path.
    prior_table_sha256 : `str`, optional
        Delta prior-table content digest.
    truth_psf_config_hash : `str`
        Canonical truth PSF configuration hash.
    fit_psf_config_hash : `str`
        Canonical fit PSF configuration hash.
    lensing_pixel_scale : `float`
        Detector pixel scale used for kernel generation.
    versions : `dict`
        HWO-SLAPS, HCIPy, and NumPy versions.

    Notes
    -----
    Unknown top-level PSF keys participate in the configuration hashes. This
    conservative over-sensitivity can split identities for identical kernels,
    but cannot merge distinct kernels; strict PSF-schema tightening is
    deferred to a dedicated compatibility pass.
    """

    mode: str
    delta_id: str
    fit_psf_config: dict
    requested_amplitude_rms_nm: Optional[float]
    measured_draw_rms_nm: Optional[float]
    family: Optional[str]
    seed: Optional[int]
    draw_aberrations: Optional[dict]
    orthonormal_segment: Optional[dict]
    orthonormal_global: Optional[dict]
    prior_table_path: Optional[str]
    prior_table_sha256: Optional[str]
    truth_psf_config_hash: str
    fit_psf_config_hash: str
    lensing_pixel_scale: float
    versions: dict


def build_psf_mismatch_spec(full_config: dict) -> PsfMismatchSpec:
    """Build one deterministic fit-side PSF mismatch specification.

    Parameters
    ----------
    full_config : `dict`
        Validated full configuration with fit-PSF delta or explicit mode.

    Returns
    -------
    spec : `PsfMismatchSpec`
        Canonical fit PSF, identity, draw, and provenance.

    Raises
    ------
    ValueError
        Raised for an unsupported mode or violated exact-RMS contract.
    """
    from ..provenance import config_hash

    fit_psf = full_config["modeling"]["fit_psf"]
    mode = str(fit_psf.get("mode", "")).lower()
    if mode not in {"delta", "explicit"}:
        raise ValueError(
            "build_psf_mismatch_spec requires modeling.fit_psf.mode to be "
            "'delta' or 'explicit'"
        )
    truth_psf_config = _canonical_psf_config(full_config["psf"])
    truth_hash = config_hash(truth_psf_config)
    lensing_pixel_scale = _canonical_float(
        full_config["lensing"]["grid"]["pixel_scale"]
    )

    if mode == "explicit":
        fit_psf_config = _canonical_psf_config(fit_psf["psf"])
        fit_hash = config_hash(fit_psf_config)
        delta_id = _identity_from_payload({
            "schema": "psf_mismatch_explicit_v1",
            "fit_psf_config_hash": fit_hash,
            "truth_psf_config_hash": truth_hash,
            "lensing_pixel_scale": lensing_pixel_scale,
        })
        return PsfMismatchSpec(
            mode=mode,
            delta_id=delta_id,
            fit_psf_config=fit_psf_config,
            requested_amplitude_rms_nm=None,
            measured_draw_rms_nm=None,
            family=None,
            seed=None,
            draw_aberrations=None,
            orthonormal_segment=None,
            orthonormal_global=None,
            prior_table_path=None,
            prior_table_sha256=None,
            truth_psf_config_hash=truth_hash,
            fit_psf_config_hash=fit_hash,
            lensing_pixel_scale=lensing_pixel_scale,
            versions=_current_versions(),
        )

    delta = fit_psf["delta"]
    prior_path = _resolve_prior_table_path(delta["prior_table"])
    prior_bytes = prior_path.read_bytes()
    prior_sha256 = hashlib.sha256(prior_bytes).hexdigest()
    amplitude = _canonical_float(delta["amplitude_rms_nm"])
    seed = int(delta["seed"])
    family = str(delta.get("family", "combined")).lower()
    draw_aberrations = _empty_aberrations()
    orthonormal_segment = None
    orthonormal_global = None
    measured = 0.0

    if amplitude == 0.0:
        fit_psf_config = deepcopy(truth_psf_config)
    else:
        from .families import (
            draw_weighted_combined_family,
            draw_weighted_global_zernike_family,
            draw_weighted_segment_hexike_family,
            measure_aperture_rms_nm,
            parse_mode_weight_prior,
            realize_weighted_draw,
        )
        from .opd_basis import ApertureBasisTransform
        from .telescope_models import create_hcipy_telescope

        prior = parse_mode_weight_prior(prior_bytes)
        telescope_data = create_hcipy_telescope(truth_psf_config)
        transform = ApertureBasisTransform(
            telescope_data,
            global_mode_nolls=sorted(prior.global_weights),
            segment_mode_nolls=sorted(prior.segment_weights),
        )
        segments = range(len(telescope_data["segments"]))
        rng = np.random.default_rng(np.random.SeedSequence(seed))
        if family == "combined":
            orthonormal_segment, orthonormal_global = (
                draw_weighted_combined_family(
                    rng,
                    segments,
                    prior,
                    amplitude,
                )
            )
        elif family == "global":
            orthonormal_segment = {}
            orthonormal_global = draw_weighted_global_zernike_family(
                rng,
                prior,
                amplitude,
            )
        elif family == "segment":
            orthonormal_segment = draw_weighted_segment_hexike_family(
                rng,
                segments,
                prior,
                amplitude,
            )
            orthonormal_global = {}
        else:
            raise ValueError(
                "modeling.fit_psf.delta.family must be one of: "
                "'combined', 'global', 'segment'"
            )
        segment_raw, global_raw = realize_weighted_draw(
            telescope_data,
            transform,
            amplitude,
            segment_coefficients=orthonormal_segment or None,
            global_coefficients=orthonormal_global or None,
        )
        draw_aberrations.update({
            "enable_segment_hexikes": bool(segment_raw),
            "segment_hexikes": segment_raw,
            "enable_global_zernikes": bool(global_raw),
            "global_zernikes": global_raw,
        })
        draw_aberrations = _canonical_aberrations(draw_aberrations)
        fit_psf_config = deepcopy(truth_psf_config)
        fit_psf_config["aberrations"] = _add_draw_to_truth(
            truth_psf_config["aberrations"],
            draw_aberrations,
        )
        measured = float(measure_aperture_rms_nm(
            telescope_data,
            segment_hexikes=segment_raw or None,
            global_zernikes=global_raw or None,
        ))
        tolerance = 1.0e-9*max(1.0, amplitude)
        if abs(measured - amplitude) > tolerance:
            raise ValueError(
                "PSF mismatch measured draw RMS does not match the requested "
                f"amplitude: {measured:.17g} != {amplitude:.17g} nm"
            )
        truth_aberrations = truth_psf_config["aberrations"]
        fit_aberrations = fit_psf_config["aberrations"]
        truth_segment = (
            truth_aberrations["segment_hexikes"]
            if truth_aberrations["enable_segment_hexikes"]
            else {}
        )
        fit_segment = (
            fit_aberrations["segment_hexikes"]
            if fit_aberrations["enable_segment_hexikes"]
            else {}
        )
        truth_global = (
            truth_aberrations["global_zernikes"]
            if truth_aberrations["enable_global_zernikes"]
            else {}
        )
        fit_global = (
            fit_aberrations["global_zernikes"]
            if fit_aberrations["enable_global_zernikes"]
            else {}
        )
        effective = float(measure_aperture_rms_nm(
            telescope_data,
            segment_hexikes=(
                _difference_nested_maps(fit_segment, truth_segment) or None
            ),
            global_zernikes=(
                _difference_flat_maps(fit_global, truth_global) or None
            ),
        ))
        if effective == 0.0:
            raise ValueError(
                "floating-point addition against the truth coefficients "
                "completely erased the requested delta: effective "
                f"aperture RMS 0 for requested amplitude "
                f"{amplitude:.17g} nm"
            )
        if abs(effective - amplitude) > tolerance:
            raise ValueError(
                "floating-point addition against the truth coefficients "
                "destroyed the requested delta: "
                f"{effective:.17g} != {amplitude:.17g} nm"
            )

    from ..config.validation import validate_psf_config

    validate_psf_config(fit_psf_config)
    fit_hash = config_hash(fit_psf_config)
    delta_id = _identity_from_payload({
        "schema": "psf_mismatch_delta_v1",
        "prior_table_sha256": prior_sha256,
        "amplitude_rms_nm": amplitude,
        "seed": int(seed),
        "family": family,
        "truth_psf_config_hash": truth_hash,
        "lensing_pixel_scale": lensing_pixel_scale,
    })
    return PsfMismatchSpec(
        mode=mode,
        delta_id=delta_id,
        fit_psf_config=fit_psf_config,
        requested_amplitude_rms_nm=amplitude,
        measured_draw_rms_nm=measured,
        family=family,
        seed=seed,
        draw_aberrations=draw_aberrations,
        orthonormal_segment=deepcopy(orthonormal_segment),
        orthonormal_global=deepcopy(orthonormal_global),
        prior_table_path=str(prior_path),
        prior_table_sha256=prior_sha256,
        truth_psf_config_hash=truth_hash,
        fit_psf_config_hash=fit_hash,
        lensing_pixel_scale=lensing_pixel_scale,
        versions=_current_versions(),
    )


def generate_fit_psf(
    psf_config: dict,
    full_config: dict,
) -> tuple[np.ndarray, float, float]:
    """Generate one validated native fit-side PSF kernel.

    Parameters
    ----------
    psf_config : `dict`
        Complete fit-side PSF configuration.
    full_config : `dict`
        Full configuration supplying the detector pixel scale.

    Returns
    -------
    kernel : `numpy.ndarray`
        Native C-contiguous float64 PSF kernel.
    kernel_pixel_scale : `float`
        Kernel pixel scale in arcseconds per pixel.
    measured_total_rms_nm : `float`
        Measured total fit-side wavefront RMS.
    """
    from ..config.validation import validate_psf_config
    from .generator import generate_psf_system
    from .utils import pyauto_kernel_native

    canonical = _canonical_psf_config(psf_config)
    validate_psf_config(canonical)
    psf_data = generate_psf_system(canonical, full_config=full_config)
    kernel = np.ascontiguousarray(
        pyauto_kernel_native(psf_data.kernel),
        dtype=np.float64,
    )
    return (
        kernel,
        float(psf_data.kernel_pixel_scale),
        float(psf_data.total_rms_nm),
    )

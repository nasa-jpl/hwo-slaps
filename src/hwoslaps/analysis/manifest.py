"""Sweep-manifest parsing and expansion into per-run configurations.

A study manifest is a YAML document that names a baseline configuration and
declares sweeps over subhalo mass, single PSF modes, and random PSF-state
ensembles. :func:`expand_manifest` turns a manifest plus its baseline
configuration into an ordered list of fully specified `RunSpec` entries.

Notes
-----
The expansion and seeding semantics are load-bearing for reproducing the
SPIE 2026 study ensembles and must not change silently:

- Sweeps expand in a fixed order: mass sweep, primary PSF sweep, optional
  PSF sweeps, PSF ensemble.
- Each random ensemble draw is seeded as ``base_seed + n_runs_so_far + 1``
  where ``n_runs_so_far`` counts every run expanded before it.
- Random segment-hexike draws normalize the flattened coefficient vector
  to ``target_rms*sqrt(n_segments)`` so the aperture RMS approximates the
  target for equal-area segments; global-Zernike draws normalize the
  coefficient vector to the target RMS directly.
- ``combined`` ensemble draws split the nominal RMS budget with equal
  variance, ``target/sqrt(2)`` per family.
"""

from __future__ import annotations

__all__ = ("RunSpec", "expand_manifest", "load_manifest")

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml


@dataclass(frozen=True)
class RunSpec:
    """One fully expanded study run.

    Parameters
    ----------
    sweep : `str`
        Name of the sweep this run belongs to.
    run_name : `str`
        Unique run identifier used for output directories.
    config : `dict`
        Complete pipeline configuration for this run.
    mass_msun : `float`
        Injected subhalo mass in solar masses.
    psf_case : `str`
        PSF case label, e.g. ``perfect`` or the perturbation family.
    psf_family : `str`
        PSF perturbation family, or ``none`` for the perfect PSF.
    psf_mode : `str`
        Description of the perturbed mode or mode range.
    psf_amplitude : `float`
        Nominal perturbation amplitude.
    psf_units : `str`
        Units of ``psf_amplitude``, e.g. ``nm RMS``.
    """

    sweep: str
    run_name: str
    config: Dict[str, Any] = field(repr=False)
    mass_msun: float
    psf_case: str
    psf_family: str
    psf_mode: str
    psf_amplitude: float
    psf_units: str


def load_manifest(path: str | Path) -> Dict[str, Any]:
    """Load a study manifest from YAML.

    Parameters
    ----------
    path : `str` or `pathlib.Path`
        Manifest file path.

    Returns
    -------
    manifest : `dict`
        Parsed manifest document.
    """
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _amplitude_label(amplitude: float) -> str:
    """Return a filesystem-safe label for an amplitude value."""
    return str(float(amplitude)).replace(".", "p").replace("-", "m")


def _num_segments_from_rings(num_rings: int) -> int:
    """Return the segment count of a hexagonal aperture with ``num_rings``."""
    rings = int(num_rings)
    if rings < 0:
        raise ValueError("num_rings must be non-negative")
    return 1 + 3*rings*(rings + 1)


def _parse_mode_range(raw_modes: Any) -> List[int]:
    """Parse a mode list or an inclusive ``"start-stop"`` range string."""
    if isinstance(raw_modes, str):
        if "-" in raw_modes:
            start, stop = raw_modes.split("-", 1)
            return list(range(int(start), int(stop) + 1))
        return [int(raw_modes)]
    return [int(mode) for mode in raw_modes]


def _normalize_vector(values: np.ndarray, target_norm: float) -> np.ndarray:
    """Rescale a vector to the requested Euclidean norm."""
    norm = float(np.linalg.norm(values))
    if target_norm == 0.0:
        return np.zeros_like(values, dtype=float)
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero random PSF coefficient vector")
    return np.asarray(values, dtype=float)*(float(target_norm)/norm)


def _random_segment_hexikes(
    *,
    rng: np.random.Generator,
    segments: List[int],
    modes: List[int],
    target_aperture_rms_nm: float,
) -> Dict[int, Dict[int, float]]:
    """Draw random segment-hexike coefficients at a target aperture RMS.

    Notes
    -----
    For equal-area segment modes the aperture RMS is approximately
    ``sqrt(sum(coeff**2)/n_segments)``, so the flattened coefficient vector
    is normalized to ``target*sqrt(n_segments)``. The generated PSF records
    the measured pupil RMS.
    """
    if float(target_aperture_rms_nm) == 0.0:
        return {}
    raw = rng.standard_normal((len(segments), len(modes)))
    scaled = _normalize_vector(raw.ravel(), float(target_aperture_rms_nm)*np.sqrt(len(segments)))
    matrix = scaled.reshape((len(segments), len(modes)))
    return {
        int(segment): {
            int(mode): float(matrix[seg_idx, mode_idx]) for mode_idx, mode in enumerate(modes)
        }
        for seg_idx, segment in enumerate(segments)
    }


def _random_global_zernikes(
    *,
    rng: np.random.Generator,
    modes: List[int],
    target_rms_nm: float,
) -> Dict[int, float]:
    """Draw random global-Zernike coefficients at a target RMS."""
    if float(target_rms_nm) == 0.0:
        return {}
    coeffs = _normalize_vector(rng.standard_normal(len(modes)), float(target_rms_nm))
    return {int(mode): float(coeffs[idx]) for idx, mode in enumerate(modes)}


def _set_perfect_psf(config: Dict[str, Any]) -> None:
    """Disable and clear every aberration family in-place."""
    aberr = config["psf"]["aberrations"]
    for family in ("segment_pistons", "segment_tiptilts", "segment_hexikes", "global_zernikes"):
        aberr[f"enable_{family}"] = False
        aberr[family] = {}


def _set_segment_hexike(
    config: Dict[str, Any], *, segment: int, mode_noll: int, amplitude_nm: float
) -> None:
    """Configure a single segment-hexike perturbation in-place."""
    _set_perfect_psf(config)
    if float(amplitude_nm) == 0.0:
        return
    aberr = config["psf"]["aberrations"]
    aberr["enable_segment_hexikes"] = True
    aberr["segment_hexikes"] = {int(segment): {int(mode_noll): float(amplitude_nm)}}


def _set_global_zernike(config: Dict[str, Any], *, mode_noll: int, amplitude_nm: float) -> None:
    """Configure a single global-Zernike perturbation in-place."""
    _set_perfect_psf(config)
    if float(amplitude_nm) == 0.0:
        return
    aberr = config["psf"]["aberrations"]
    aberr["enable_global_zernikes"] = True
    aberr["global_zernikes"] = {int(mode_noll): float(amplitude_nm)}


def _set_coefficient_ensembles(
    config: Dict[str, Any],
    *,
    segment_hexikes: Dict[int, Dict[int, float]],
    global_zernikes: Dict[int, float],
) -> None:
    """Apply drawn ensemble coefficient maps in-place."""
    aberr = config["psf"]["aberrations"]
    if segment_hexikes:
        aberr["enable_segment_hexikes"] = True
        aberr["segment_hexikes"] = segment_hexikes
    if global_zernikes:
        aberr["enable_global_zernikes"] = True
        aberr["global_zernikes"] = global_zernikes


def _set_fisher_common(
    config: Dict[str, Any],
    *,
    mode: str,
    run_map: bool,
    mode_scan: bool,
    manifest: Dict[str, Any],
) -> None:
    """Apply the shared Fisher settings for one expanded run in-place."""
    fisher = config["modeling"]["fisher"]
    fisher["mode"] = "both" if run_map else mode
    fisher["compute_psf_mode_scan"] = bool(mode_scan)
    manifest_map = manifest.get("map", {})
    fisher["map"]["num_angles"] = int(manifest_map.get("num_angles", fisher["map"]["num_angles"]))
    fisher["map"]["offset_pixels"] = float(
        manifest_map.get("offset_pixels", fisher["map"]["offset_pixels"])
    )
    fisher["map"]["explicit_positions_yx"] = None


def _base_run_config(
    manifest: Dict[str, Any], baseline: Dict[str, Any], mass_value: float
) -> Dict[str, Any]:
    """Copy the baseline config and apply per-run manifest basics."""
    config = deepcopy(baseline)
    config["plotting"]["output_dir"] = str(manifest["output_root"])
    config["lensing"]["subhalo"]["mass"] = float(mass_value)
    return config


def _append_mass_sweep_runs(
    *,
    runs: List[RunSpec],
    manifest: Dict[str, Any],
    baseline: Dict[str, Any],
    study_name: str,
) -> None:
    """Expand the perfect-PSF mass sweep, if enabled."""
    mass_sweep = manifest.get("mass_sweep", {})
    if not mass_sweep.get("enabled", False):
        return
    for mass in mass_sweep["masses"]:
        run_name = f"{study_name}_mass_{mass['label']}_perfect"
        config = _base_run_config(manifest, baseline, mass["value"])
        config["run_name"] = run_name
        _set_perfect_psf(config)
        _set_fisher_common(
            config,
            mode="local",
            run_map=bool(mass.get("run_map", False)),
            mode_scan=False,
            manifest=manifest,
        )
        runs.append(
            RunSpec(
                sweep="perfect_mass",
                run_name=run_name,
                config=config,
                mass_msun=float(mass["value"]),
                psf_case="perfect",
                psf_family="none",
                psf_mode="none",
                psf_amplitude=0.0,
                psf_units="",
            )
        )


def _append_psf_sweep_runs(
    *,
    runs: List[RunSpec],
    manifest: Dict[str, Any],
    baseline: Dict[str, Any],
    study_name: str,
    psf_sweep: Dict[str, Any],
) -> None:
    """Expand one single-mode PSF amplitude sweep, if enabled."""
    if not psf_sweep.get("enabled", False):
        return
    pivot_mass = psf_sweep["pivot_mass"]
    family = str(psf_sweep["family"])
    map_amplitudes = {float(val) for val in psf_sweep.get("map_amplitudes", [])}
    scan_amplitudes = {float(val) for val in psf_sweep.get("mode_scan_amplitudes", [])}

    for amplitude in psf_sweep["amplitudes"]:
        amp = float(amplitude)
        amp_label = _amplitude_label(amp)
        config = _base_run_config(manifest, baseline, pivot_mass["value"])
        if family == "segment_hexikes":
            segment = int(psf_sweep["segment"])
            mode_noll = int(psf_sweep["mode_noll"])
            run_name = (
                f"{study_name}_hexike_s{segment}_n{mode_noll}_a{amp_label}nm_{pivot_mass['label']}"
            )
            sweep_name = "segment_hexike_amplitude"
            psf_case = "perfect" if amp == 0.0 else "segment_hexike"
            psf_mode = f"segment_{segment}_noll_{mode_noll}"
            _set_segment_hexike(config, segment=segment, mode_noll=mode_noll, amplitude_nm=amp)
        elif family == "global_zernikes":
            mode_noll = int(psf_sweep["mode_noll"])
            run_name = f"{study_name}_global_zernike_n{mode_noll}_a{amp_label}nm_{pivot_mass['label']}"
            sweep_name = "global_zernike_amplitude"
            psf_case = "perfect" if amp == 0.0 else "global_zernike"
            psf_mode = f"global_zernike_noll_{mode_noll}"
            _set_global_zernike(config, mode_noll=mode_noll, amplitude_nm=amp)
        else:
            raise ValueError(f"Unsupported PSF sweep family: {family}")

        config["run_name"] = run_name
        _set_fisher_common(
            config,
            mode="local",
            run_map=amp in map_amplitudes,
            mode_scan=amp in scan_amplitudes,
            manifest=manifest,
        )
        runs.append(
            RunSpec(
                sweep=sweep_name,
                run_name=run_name,
                config=config,
                mass_msun=float(pivot_mass["value"]),
                psf_case=psf_case,
                psf_family=family,
                psf_mode=psf_mode,
                psf_amplitude=amp,
                psf_units=str(psf_sweep["units"]),
            )
        )


def _ensemble_pivot_masses(ensemble: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the pivot-mass entries of an ensemble block."""
    if "pivot_masses" in ensemble:
        return list(ensemble["pivot_masses"])
    if "masses" in ensemble:
        return list(ensemble["masses"])
    return [ensemble["pivot_mass"]]


def _ensemble_psf_mode(
    family: str, segment_modes: List[int], global_modes: List[int]
) -> str:
    """Return the mode-range description for one ensemble family."""
    segment_part = f"segment_hexike_noll_{segment_modes[0]}-{segment_modes[-1]}"
    global_part = f"global_zernike_noll_{global_modes[0]}-{global_modes[-1]}"
    if family == "combined":
        return f"{segment_part}+{global_part}"
    return segment_part if family == "segment_only" else global_part


def _append_psf_ensemble_runs(
    *,
    runs: List[RunSpec],
    manifest: Dict[str, Any],
    baseline: Dict[str, Any],
    study_name: str,
) -> None:
    """Expand the random PSF-state ensemble, if enabled."""
    ensemble = manifest.get("psf_ensemble_sweep", {})
    if not ensemble.get("enabled", False):
        return

    pivot_masses = _ensemble_pivot_masses(ensemble)
    amplitudes = [float(value) for value in ensemble["amplitudes"]]
    draws = int(ensemble["draws_per_amplitude"])
    draw_index_offset = int(ensemble.get("draw_index_offset", 0))
    base_seed = int(ensemble.get("seed", baseline.get("global_seed", 0)))
    units = str(ensemble.get("units", "nm RMS"))
    split_mode = str(ensemble.get("combined_rms_split", "equal_variance"))

    segment_block = ensemble.get("segment_hexikes", {})
    global_block = ensemble.get("global_zernikes", {})
    num_segments = _num_segments_from_rings(int(baseline["psf"]["telescope"]["num_rings"]))
    segments = segment_block.get("segments", "all")
    segment_ids = (
        list(range(num_segments)) if segments == "all" else [int(segment) for segment in segments]
    )
    segment_modes = _parse_mode_range(segment_block.get("mode_nolls", []))
    global_modes = _parse_mode_range(global_block.get("mode_nolls", []))
    families = [str(family) for family in ensemble.get("families", [])]

    for pivot_mass in pivot_masses:
        if any(amplitude == 0.0 for amplitude in amplitudes):
            config = _base_run_config(manifest, baseline, pivot_mass["value"])
            config["global_seed"] = base_seed
            _set_perfect_psf(config)
            run_name = f"{study_name}_perfect_reference_{pivot_mass['label']}"
            config["run_name"] = run_name
            _set_fisher_common(
                config, mode="local", run_map=False, mode_scan=False, manifest=manifest
            )
            runs.append(
                RunSpec(
                    sweep="psf_ensemble_perfect_reference",
                    run_name=run_name,
                    config=config,
                    mass_msun=float(pivot_mass["value"]),
                    psf_case="perfect",
                    psf_family="none",
                    psf_mode="none",
                    psf_amplitude=0.0,
                    psf_units=units,
                )
            )

        for family in families:
            if family not in {"segment_only", "global_only", "combined"}:
                raise ValueError(f"Unsupported PSF ensemble family: {family}")
            for amplitude in amplitudes:
                if amplitude == 0.0:
                    continue
                amp_label = _amplitude_label(amplitude)
                for draw_idx in range(draws):
                    draw_label = draw_index_offset + draw_idx
                    seed = base_seed + len(runs) + 1
                    rng = np.random.default_rng(seed)
                    config = _base_run_config(manifest, baseline, pivot_mass["value"])
                    config["global_seed"] = seed
                    _set_perfect_psf(config)

                    segment_budget = amplitude
                    global_budget = amplitude
                    if family == "combined" and split_mode == "equal_variance":
                        segment_budget = amplitude/np.sqrt(2.0)
                        global_budget = amplitude/np.sqrt(2.0)

                    segment_coeffs: Dict[int, Dict[int, float]] = {}
                    global_coeffs: Dict[int, float] = {}
                    if family in {"segment_only", "combined"}:
                        segment_coeffs = _random_segment_hexikes(
                            rng=rng,
                            segments=segment_ids,
                            modes=segment_modes,
                            target_aperture_rms_nm=segment_budget,
                        )
                    if family in {"global_only", "combined"}:
                        global_coeffs = _random_global_zernikes(
                            rng=rng, modes=global_modes, target_rms_nm=global_budget
                        )
                    _set_coefficient_ensembles(
                        config, segment_hexikes=segment_coeffs, global_zernikes=global_coeffs
                    )

                    run_name = (
                        f"{study_name}_{family}_a{amp_label}nm_"
                        f"d{draw_label:03d}_{pivot_mass['label']}"
                    )
                    config["run_name"] = run_name
                    _set_fisher_common(
                        config, mode="local", run_map=False, mode_scan=False, manifest=manifest
                    )
                    runs.append(
                        RunSpec(
                            sweep=f"psf_ensemble_{family}",
                            run_name=run_name,
                            config=config,
                            mass_msun=float(pivot_mass["value"]),
                            psf_case=family,
                            psf_family=family,
                            psf_mode=_ensemble_psf_mode(family, segment_modes, global_modes),
                            psf_amplitude=amplitude,
                            psf_units=units,
                        )
                    )


def expand_manifest(manifest: Dict[str, Any], baseline: Dict[str, Any]) -> List[RunSpec]:
    """Expand a study manifest into fully specified run configurations.

    Parameters
    ----------
    manifest : `dict`
        Parsed study manifest, e.g. from `load_manifest`.
    baseline : `dict`
        Baseline pipeline configuration copied into every run.

    Returns
    -------
    runs : `list` of `RunSpec`
        Ordered run specifications. The order is deterministic and part of
        the reproducibility contract, because ensemble draw seeds depend on
        the number of previously expanded runs.
    """
    study_name = str(manifest["study_name"])
    runs: List[RunSpec] = []
    _append_mass_sweep_runs(
        runs=runs, manifest=manifest, baseline=baseline, study_name=study_name
    )
    _append_psf_sweep_runs(
        runs=runs,
        manifest=manifest,
        baseline=baseline,
        study_name=study_name,
        psf_sweep=manifest.get("psf_sweep", {}),
    )
    for psf_sweep in manifest.get("optional_psf_sweeps", []):
        _append_psf_sweep_runs(
            runs=runs,
            manifest=manifest,
            baseline=baseline,
            study_name=study_name,
            psf_sweep=psf_sweep,
        )
    _append_psf_ensemble_runs(
        runs=runs, manifest=manifest, baseline=baseline, study_name=study_name
    )
    return runs

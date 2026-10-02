"""Explicit simulation and prepared subhalo forecasting.

A prepared forecast owns one smooth scene, truth/fit kernels, expected detector
weights, and a profiled likelihood workspace. It can evaluate many masses and
positions without rebuilding the observation or changing scientific conventions.
Saving, plotting, and nonlinear searches are separate explicit operations.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from os import PathLike
from typing import Any

import numpy as np

from .config import load_config, resolve_config_paths
from .config.validation import (
    validate_lensing_config, validate_observation_config,
    validate_modeling_config, validate_psf_config,
)


@dataclass(frozen=True, init=False)
class PreparedForecast:
    """Reusable scientific context; use one context per concurrent worker.

    ``scene`` and ``observation`` are the smooth, no-subhalo expectation.
    Truth and fitted PSFs are distinct objects when a mismatch is requested.
    The detector owns caches for its current mass/position evaluation; a
    context is not a shared mutable resource for concurrent threaded calls.
    """
    _config: dict[str, Any]
    _truth_identity: dict[str, Any]
    _fit_identity: dict[str, Any]
    scene: Any
    truth_psf: Any
    fit_psf: Any
    observation: Any
    detector: Any

    def __init__(self, config, scene, truth_psf, fit_psf, observation, detector):
        object.__setattr__(self, "_config", deepcopy(config))
        for name, value in (
            ("scene", scene), ("truth_psf", truth_psf), ("fit_psf", fit_psf),
            ("observation", observation), ("detector", detector),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_truth_identity", _kernel_identity(truth_psf))
        object.__setattr__(self, "_fit_identity", _kernel_identity(fit_psf))

    @property
    def config(self):
        """Return an independent input snapshot; changes require preparation."""
        return deepcopy(self._config)

    def validate_identity(self):
        """Reject changed kernels rather than relabel a cached likelihood."""
        if _kernel_identity(self.truth_psf) != self._truth_identity:
            raise ValueError("truth PSF changed after preparation; prepare a new forecast")
        if _kernel_identity(self.fit_psf) != self._fit_identity:
            raise ValueError("model PSF changed after preparation; prepare a new forecast")

    def trial(self, mass_msun, position_yx, *, fisher_q=None, case_id=None):
        """Construct a physical trial for this scene's declared subhalo model."""
        from .modeling.nonlinear.trial import trial_from_fisher_map_position
        return trial_from_fisher_map_position(
            self.config, self.scene, mass_msun, position_yx,
            fisher_q=fisher_q, case_id=case_id,
        )


def _configuration(config, *, base_dir=None, require_forecast=False, provided_psf=False):
    if isinstance(config, Mapping):
        resolved = resolve_config_paths(config, base_dir=base_dir)
    else:
        resolved = load_config(config, base_dir=base_dir, validate=False)
    seed = resolved.get("global_seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("global_seed must be an explicit non-negative integer")
    for section in ("lensing", "observation"):
        if not isinstance(resolved.get(section), dict):
            raise ValueError(f"{section} must be a configuration mapping")
    validate_lensing_config(resolved["lensing"])
    validate_observation_config(resolved["observation"])
    if not provided_psf:
        if not isinstance(resolved.get("psf"), dict):
            raise ValueError("psf must define an optical or detector-kernel provider")
        validate_psf_config(resolved["psf"])
    if require_forecast:
        if not isinstance(resolved.get("modeling"), dict) or "fisher" not in resolved["modeling"]:
            raise ValueError("a forecast requires modeling.fisher settings")
    if "modeling" in resolved:
        validate_modeling_config(resolved["modeling"])
    resolved.setdefault("run_name", "forecast" if require_forecast else "simulation")
    return resolved


def _validate_psf(psf, pixel_scale):
    from .psf.utils import pyauto_kernel_pixel_scales, validate_detector_kernel
    if not hasattr(psf, "kernel"):
        raise TypeError("provide a PSF object, for example DetectorPSF.from_array(values, pixel_scale)")
    validate_detector_kernel(psf.kernel)
    scales = np.asarray(pyauto_kernel_pixel_scales(psf.kernel), dtype=float)
    if not np.allclose(scales, float(pixel_scale), rtol=0, atol=1e-12):
        raise ValueError("PSF angular sampling must match the scene grid; no resampling is inferred")
    return psf


def _kernel_from_file(spec, *, pixel_scale):
    from .psf import DetectorPSF
    path = spec["path"]
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            key = spec.get("array_key", "kernel")
            if key not in loaded.files:
                raise ValueError(f"kernel archive must contain the declared array {key!r}")
            values = np.array(loaded[key], copy=True)
        finally:
            loaded.close()
    else:
        values = loaded
    psf = DetectorPSF.from_array(
        values, spec["pixel_scale_arcsec"],
        normalize=spec.get("normalize", True), config={"source_path": str(path)},
    )
    identity = _kernel_identity(psf)
    if "kernel_sha256" in spec and identity["kernel_sha256"] != spec["kernel_sha256"]:
        raise ValueError("kernel file does not match its declared detector-response identity")
    if "shape_native" in spec and identity["shape_native"] != list(spec["shape_native"]):
        raise ValueError("kernel file does not match its declared support")
    return _validate_psf(psf, pixel_scale)


def _truth_psf(config, supplied=None):
    scale = config["lensing"]["grid"]["pixel_scale"]
    if supplied is not None:
        return _validate_psf(supplied, scale)
    spec = config["psf"]
    if spec.get("provider", "optical") == "kernel":
        return _kernel_from_file(spec["kernel"], pixel_scale=scale)
    from .psf import generate_psf_system
    if spec.get("hres_psf", {}).get("save_highres_psf_npy", False):
        raise ValueError("PSF export is an explicit output operation, not a simulation setting")
    return generate_psf_system(spec, full_config=config, target_pixel_scale=scale)


def _kernel_identity(psf):
    from .psf.mismatch import _kernel_sha256
    from .psf.utils import pyauto_kernel_native, pyauto_kernel_pixel_scales
    values = pyauto_kernel_native(psf.kernel)
    return {
        "kernel_sha256": _kernel_sha256(values),
        "pixel_scale_arcsec": float(pyauto_kernel_pixel_scales(psf.kernel)[0]),
        "shape_native": list(values.shape),
    }


def _supplied_psf_config(config, psf):
    from .psf import DetectorPSF
    if isinstance(psf, DetectorPSF):
        identity = _kernel_identity(psf)
        original = config.get("psf") or {}
        declared_kernel = original.get("kernel") or {}
        # A file provider remains replayable; a caller-owned object has no
        # invented path. Its actual normalized bytes are recorded instead.
        if psf.config and psf.config.get("source_path") == declared_kernel.get("path"):
            config["psf"] = {**deepcopy(original), "provider": "kernel",
                             "kernel": {**deepcopy(declared_kernel), **identity}}
        else:
            config["psf"] = {"provider": "kernel", "kernel": identity}


def simulate(
    config: Mapping | str | PathLike | PreparedForecast,
    *, psf=None, trial=None, injected=None, seed=None, sample_noise=True, base_dir=None,
):
    """Render an observation, optionally injecting an explicit subhalo trial.

    A PreparedForecast reuses its truth PSF. A trial supplies mass, position,
    and model; without a trial the configuration's subhalo flag is respected.
    Set ``injected=False`` for a no-subhalo control at the same observing setup;
    an explicit trial can still identify the intended fitting/search hypothesis.
    ``sample_noise=False`` returns the expected image and variance without a
    detector random draw. No artifact or plot is written by this operation.
    """
    if not isinstance(sample_noise, bool):
        raise ValueError("sample_noise must be boolean")
    if injected is not None and not isinstance(injected, bool):
        raise ValueError("injected must be boolean or None")
    if isinstance(config, PreparedForecast):
        config.validate_identity()
        settings = config.config
        truth = config.truth_psf if psf is None else _validate_psf(psf, config.scene.pixel_scale)
    else:
        settings = _configuration(config, base_dir=base_dir, provided_psf=psf is not None)
        truth = _truth_psf(settings, psf)
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        settings["global_seed"] = seed
    if trial is not None:
        hypothesis = settings["lensing"]["subhalo"]
        hypothesis.update(enabled=True, mass=float(trial.mass_msun), model=trial.model)
        hypothesis["position"] = {"type": "direct", "centre": list(trial.position_yx_arcsec)}
        if not np.isclose(trial.lens_redshift, settings["lensing"]["lens_galaxy"]["redshift"]):
            raise ValueError("trial lens redshift must match the prepared scene")
        if not np.isclose(trial.source_redshift, settings["lensing"]["source_galaxy"]["redshift"]):
            raise ValueError("trial source redshift must match the prepared scene")
        validate_lensing_config(settings["lensing"])
    if injected is not None:
        settings["lensing"]["subhalo"]["enabled"] = injected
    from .lensing import generate_lensing_system
    from .observation import generate_observation
    scene = generate_lensing_system(settings["lensing"], full_config=settings)
    observation = generate_observation(
        scene, truth, settings["observation"], full_config=settings,
        sample_noise=sample_noise,
    )
    observation.metadata["subhalo"] = deepcopy(settings["lensing"]["subhalo"])
    observation.metadata["truth_kernel"] = _kernel_identity(truth)
    return observation


def prepare_forecast(config, *, psf=None, fit_psf=None, backend=None, base_dir=None) -> PreparedForecast:
    """Prepare a smooth scene and profiled workspace for a mass/position search.

    Provide detector PSFs explicitly or select their file provider in YAML.
    Optical PSF coefficient nuisance modes require an optical provider. A
    kernel mismatch is bound by actual kernel bytes and angular sampling.
    """
    settings = _configuration(
        config, base_dir=base_dir, require_forecast=True, provided_psf=psf is not None,
    )
    model_psf = (settings["modeling"].get("fit_psf") or {}).get("psf") or {}
    if model_psf.get("hres_psf", {}).get("save_highres_psf_npy", False):
        raise ValueError("Model PSF export is an explicit output operation, not a forecast setting")
    fit_spec = deepcopy((settings.get("psf") or {}).get("fit_kernel"))
    truth = _truth_psf(settings, psf)
    _supplied_psf_config(settings, truth)
    if fit_psf is None and fit_spec is not None:
        fit_psf = _kernel_from_file(fit_spec, pixel_scale=settings["lensing"]["grid"]["pixel_scale"])
        # Replay must bind the model calibration too, not only truth bytes.
        settings["psf"]["fit_kernel"] = {**fit_spec, **_kernel_identity(fit_psf)}
    if fit_psf is not None:
        _validate_psf(fit_psf, settings["lensing"]["grid"]["pixel_scale"])
        if (settings["modeling"].get("fit_psf") or {}).get("mode") == "delta":
            raise ValueError("a delta optical mismatch binds its own model PSF")
        settings["modeling"]["fit_psf"] = {"mode": "kernel", **_kernel_identity(fit_psf)}
    fisher = settings["modeling"]["fisher"]
    from .psf import DetectorPSF
    if (
        isinstance(truth, DetectorPSF) or isinstance(fit_psf, DetectorPSF)
    ) and (fisher.get("include_psf_nuisance", False) or fisher.get("compute_psf_mode_scan", False)):
        raise ValueError(
            "Optical coefficient nuisance modes require an optical PSF provider; "
            "external detector kernels have no declared wavefront basis"
        )
    if backend is not None:
        if backend not in {"reference", "jax"}:
            raise ValueError("backend must be reference or jax")
        fisher["map"]["engine"] = backend
    from .lensing import generate_lensing_system
    from .modeling.fisher_detector import FisherDetector
    from .observation import generate_observation
    baseline = deepcopy(settings)
    baseline["lensing"]["subhalo"]["enabled"] = False
    scene = generate_lensing_system(baseline["lensing"], full_config=baseline)
    observation = generate_observation(
        scene, truth, baseline["observation"], full_config=baseline, sample_noise=False,
    )
    detector = FisherDetector(
        observation_baseline=observation, lensing_baseline=scene, psf_data=truth,
        full_config=settings, fisher_config=fisher, fit_psf_data=fit_psf,
    )
    actual_fit = detector.model_psf_data
    return PreparedForecast(deepcopy(settings), scene, truth, actual_fit, observation, detector)


def forecast(prepared: PreparedForecast, *, masses=None, positions=None, domain_positions=None):
    """Evaluate explicit masses and positions on a reusable prepared context.

    When omitted, masses use the declared hypothesis mass and positions use the
    configured candidate layout. Sparse positions may declare their full
    numerical domain separately, preserving interpolation support and clipping.
    The result reports q; it does not claim a calibrated freed-search p-value.
    """
    if not isinstance(prepared, PreparedForecast):
        raise TypeError("prepare a forecast with prepare_forecast first")
    prepared.validate_identity()
    config = prepared.config
    if masses is None:
        masses = [config["lensing"]["subhalo"]["mass"]]
    masses = np.atleast_1d(np.asarray(masses, dtype=float))
    if positions is None:
        positions = prepared.detector.candidate_positions()
    if domain_positions is None:
        domain_positions = prepared.detector.domain_positions()
    result = prepared.detector.evaluate_masses(
        masses, positions, domain_positions_yx=domain_positions,
    )
    # Numerical backends return a fresh result with explicit execution metadata.
    # Record scientific identity without copying a potentially large mass bank.
    from .provenance import config_hash
    provenance = dict(result.runtime_provenance or {})
    provenance.update(
        config_hash=config_hash(config),
        truth_kernel=_kernel_identity(prepared.truth_psf),
        fit_kernel=_kernel_identity(prepared.fit_psf),
        statistic="profiled_linear_gaussian_q",
        mass_definition=("point_mass" if config["lensing"]["subhalo"]["model"] == "PointMass"
                         else "M200"),
        coordinate_order="y,x", position_unit="arcsec", mass_unit="solar_mass",
    )
    object.__setattr__(result, "runtime_provenance", provenance)
    return result

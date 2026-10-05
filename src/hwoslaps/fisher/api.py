"""Prepare reusable smooth-scene forecasts and evaluate supplied halo masses."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral
from os import PathLike
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike

from .._version import __version__
from ..config.checks import ConfigError
from ..config.schema import ConfigSource, EngineConfig, resolve_config
from ..identity import array_digest, json_ready, validate_file_manifest, validate_loaded_file
from ..observation.normalization import resolve_observing
from ..observation.observation import Observation, observe
from ..scene.builder import Scene, build_scene, native_sampling_variation
from ..scene.cosmology import Cosmology
from ..scene.critical_curve import effective_einstein_radius
from ..scene.halos import Halo, make_halo
from ..scene.image_source import frozen_value, load_image_asset
from ..scene.perturbers import realize_perturbers
from ..scene.spec import SceneSpec, pixel_centres_yx
from .data_space import (DataSpace, all_pixels_mask, annulus_mask, build_data_space,
                         grid_centre_yx, load_noise_covariance, psf_border_mask, source_snr_mask)
from .engines.base import EngineContext, TemplateEngine, make_engine
from .nuisances import NuisanceDesign, build_nuisance_design, resolve_nuisances
from .positions import PositionSet, explicit_positions, grid_positions, ring_positions
from .psf_pair import PsfPair, bind_psfs, truth_provider, validate_loaded_psf_files
from .renderer import SceneRenderer
from .result import ForecastResult
from .statistics import ProfileLikelihoodWorkspace

__all__ = ["Execution", "PreparedForecast", "forecast", "prepare_forecast"]


@dataclass(frozen=True)
class Execution:
    engine: Literal["reference", "jax"] = "reference"
    reference_workers: int = 1
    batch_size: int = 16
    progress: bool = False

    def __post_init__(self) -> None:
        if self.engine not in ("reference", "jax"):
            raise ValueError(f"engine must be reference or jax, got {self.engine!r}")
        for name in ("reference_workers", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be an integer >= 1, got {value!r}")
        if not isinstance(self.progress, bool):
            raise ValueError("progress must be a bool")


@dataclass(frozen=True, eq=False)
class PreparedForecast:
    _config: EngineConfig
    scene: Scene
    psfs: PsfPair
    observation: Observation
    positions: PositionSet
    renderer: SceneRenderer
    data_space: DataSpace
    nuisances: NuisanceDesign
    workspace: ProfileLikelihoodWorkspace
    mean_model_adu: np.ndarray
    engine: TemplateEngine
    execution: Execution
    record: Mapping[str, Any]

    @property
    def config(self) -> EngineConfig:
        return deepcopy(self._config)

    @property
    def mean_truth_adu(self) -> np.ndarray:
        return self.observation.expected_adu

    @property
    def sigma_adu(self) -> np.ndarray:
        return self.observation.noise_map_adu

    @property
    def mask(self) -> np.ndarray:
        return self.data_space.mask

    def hypothesis(self, mass_msun: float, position_yx: tuple[float, float]) -> Halo:
        spec = self.scene.spec
        redshift = spec.lens.redshift if spec.subhalo_redshift is None else spec.subhalo_redshift
        return make_halo(spec.subhalo, mass_msun, position_yx, redshift=redshift,
                         source_redshift=spec.source.redshift, cosmology=self.scene.cosmology)

    def validate_identity(self) -> None:
        for side, binding in (("truth", self.psfs.truth_kernels), ("model", self.psfs.model_kernels)):
            recorded = self.record[side + "_kernels"]
            if binding.to_mapping() != json_ready(recorded):
                raise ValueError(f"the {side} kernel changed; prepare the forecast again")
            for index, kernel in enumerate(binding.kernels):
                digest = array_digest(np.asarray(kernel.convolver().kernel.native))
                if digest != self.record[side + "_convolver_digests"][index]:
                    raise ValueError(f"the {side} convolver kernel {index} changed; prepare the forecast again")
        validate_file_manifest(self.record["file_digests"])

    def close(self) -> None:
        self.engine.close()

    def __enter__(self) -> PreparedForecast:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _positions(spec: Any, scene: Scene) -> PositionSet:
    centre = scene.spec.lens_centre
    if spec.kind == "grid":
        return grid_positions(centre, spacing_arcsec=spec.spacing_arcsec, half_width_arcsec=spec.half_width_arcsec,
                              annulus=spec.annulus)
    if spec.kind == "ring":
        if spec.radius == "einstein_radius":
            radius = scene.einstein_radius()
        elif spec.radius == "critical_curve":
            radius = effective_einstein_radius(scene.spec, scene.cosmology)
        else:
            radius = spec.radius
        return ring_positions(centre, count=spec.count, radius_arcsec=radius, offset_arcsec=spec.offset_arcsec)
    return explicit_positions(spec.positions_yx, centre)


def _mask(spec: Any, scene: Scene, observation: Observation, psfs: PsfPair) -> np.ndarray:
    shape = scene.spec.grid.shape
    if spec.kind == "all_pixels":
        return all_pixels_mask(shape)
    if spec.kind == "source_snr":
        source_adu = observation.exposure.signal_adu(observation.light_rate_by_plane_e_per_s["source"])
        return source_snr_mask(source_adu, observation.noise_map_adu, spec.snr_min)
    if spec.kind == "annulus":
        y, x = pixel_centres_yx(shape, scene.pixel_scale_arcsec)
        centre = scene.spec.lens_centre if spec.about == "lens" else grid_centre_yx(y, x)
        return annulus_mask(y, x, centre_yx=centre, inner_arcsec=spec.inner_arcsec, outer_arcsec=spec.outer_arcsec)
    kernel_shape = tuple(max(kernel.shape[axis] for kernel in psfs.model_kernels.kernels) for axis in (0, 1))
    return psf_border_mask(shape, kernel_shape)


def _assets(spec: SceneSpec) -> dict[str, Any]:
    paths = dict.fromkeys(str(component.values["asset_path"]) for galaxy in (spec.lens, spec.source)
                          for component in galaxy.light if component.type == "Image")
    return {path: load_image_asset(path) for path in paths}


def _constant(renderer: SceneRenderer, scene: Scene, binding: Any) -> float | np.ndarray:
    offset = renderer.exposure.background_adu
    if not scene.spec.lens.light:
        return offset
    return offset + renderer.exposure.signal_adu(renderer.light_rate(scene, binding, planes=("lens",)))


def _validate_loaded_files(manifest: Mapping[str, str], assets: Mapping[str, Any], psfs: PsfPair) -> None:
    for path, asset in assets.items():
        validate_loaded_file(path, asset.digest, manifest)
    validate_loaded_psf_files(psfs.truth, psfs.truth_kernels, manifest)
    validate_loaded_psf_files(psfs.model.provider, psfs.model_kernels, manifest)
    if psfs.model.knowledge_error is not None:
        draw = psfs.model.knowledge_error.draw
        if draw.spec.prior.kind == "path":
            validate_loaded_file(draw.spec.prior.path, draw.prior_digest, manifest)


def prepare_forecast(config: ConfigSource, *, execution: Execution = Execution(),
                     base_dir: PathLike[str] | None = None) -> PreparedForecast:
    resolved = deepcopy(resolve_config(config, base_dir=base_dir))
    if resolved.forecast is None:
        raise ConfigError("forecast", "a forecast section is required to prepare a forecast")
    if not isinstance(execution, Execution):
        raise TypeError("execution must be an Execution")
    identity = resolved.capture_identity()
    config_digest = identity["config_digest"]
    comparison_digest = identity["comparison_digest"]
    file_digests = identity["file_digests"]
    cosmology = Cosmology(resolved.cosmology)
    truth = truth_provider(resolved)
    setup = resolve_observing(resolved.scene, resolved.instrument, resolved.observation, truth=truth)
    perturbers = realize_perturbers(setup.scene, cosmology, seed=resolved.seed)
    assets = _assets(setup.scene)
    smooth = build_scene(setup.scene, cosmology, subhalo=None, perturbers=perturbers, assets=assets)
    sampling = native_sampling_variation(smooth)
    psfs = bind_psfs(resolved, setup.scene, setup.instrument, truth=truth)
    observation = observe(smooth, psfs.truth_kernels, setup.exposure, config_digest=config_digest,
                          photometry=setup.photometry, sampling=sampling)
    renderer = SceneRenderer(setup.scene, cosmology, perturbers, setup.exposure, assets=assets)
    mean_model = renderer.mean_adu(smooth, psfs.model_kernels) if psfs.mismatched else observation.expected_adu
    mean_model.flags.writeable = False
    positions = _positions(resolved.forecast.positions, smooth)
    mask = _mask(resolved.forecast.mask, smooth, observation, psfs)
    covariance_path = resolved.forecast.noise_covariance
    covariance = None if covariance_path is None else load_noise_covariance(
        covariance_path, mask.size, file_sha256=file_digests[str(covariance_path)])
    data_space = build_data_space(mask, observation.noise_map_adu, covariance)
    parameters = resolve_nuisances(setup.scene, resolved.forecast.nuisances, psfs)
    design = build_nuisance_design(parameters, renderer=renderer, smooth_scene=smooth, psfs=psfs)
    workspace = ProfileLikelihoodWorkspace(data_space.whiten(data_space.design(design.images)),
                                           design.prior_precision, design.names)
    bias = data_space.whiten(data_space.flatten(observation.expected_adu - mean_model)) if psfs.mismatched else None
    hypothesis_redshift = setup.scene.lens.redshift if setup.scene.subhalo_redshift is None else setup.scene.subhalo_redshift
    model_constant = _constant(renderer, smooth, psfs.model_kernels)
    truth_constant = _constant(renderer, smooth, psfs.truth_kernels) if psfs.mismatched else model_constant
    context = EngineContext(renderer=renderer, smooth_scene=smooth, hypothesis_model=setup.scene.subhalo,
                            hypothesis_redshift=hypothesis_redshift, source_redshift=setup.scene.source.redshift,
                            cosmology=cosmology, model_kernels=psfs.model_kernels,
                            truth_kernels=psfs.truth_kernels if psfs.mismatched else None,
                            exposure=setup.exposure, mean_model_adu=mean_model, data_space=data_space,
                            workspace=workspace, bias_whitened=bias, lens_centre_yx=setup.scene.lens_centre,
                            domain_radius_arcsec=positions.domain_radius_arcsec, model_constant_adu=model_constant,
                            truth_constant_adu=truth_constant)
    record = {"config_digest": config_digest, "comparison_digest": comparison_digest,
              "file_digests": file_digests, "psf_relation": psfs.relation,
              "truth_kernels": psfs.truth_kernels.to_mapping(), "model_kernels": psfs.model_kernels.to_mapping(),
              "truth_convolver_digests": [array_digest(np.asarray(kernel.convolver().kernel.native))
                                           for kernel in psfs.truth_kernels.kernels],
              "model_convolver_digests": [array_digest(np.asarray(kernel.convolver().kernel.native))
                                           for kernel in psfs.model_kernels.kernels],
              "nuisance_names": list(design.names), "mask_digest": data_space.digest(),
              "pixel_count": data_space.pixel_count, "sampling": dict(observation.sampling)}
    _validate_loaded_files(file_digests, assets, psfs)
    engine = make_engine(execution.engine, context, execution)
    prepared = PreparedForecast(resolved, smooth, psfs, observation, positions, renderer, data_space,
                                design, workspace, mean_model, engine, execution, frozen_value(record))
    try:
        prepared.validate_identity()
    except BaseException:
        prepared.close()
        raise
    return prepared


def _provenance(prepared: PreparedForecast, positions: PositionSet) -> dict[str, Any]:
    spec = prepared.scene.spec
    covariance = prepared._config.forecast.noise_covariance
    draw = prepared.psfs.model.knowledge_error
    condition = prepared.workspace.condition_number
    return {key: json_ready(prepared.record[key]) for key in ("config_digest", "comparison_digest", "file_digests", "sampling")} | {
        "truth_kernels": prepared.psfs.truth_kernels.to_mapping(), "model_kernels": prepared.psfs.model_kernels.to_mapping(),
        "psf_relation": prepared.psfs.relation, "knowledge_error": None if draw is None else draw.to_mapping(),
        "spectral": prepared.psfs.spectral, "photometry": None if prepared.observation.photometry is None else prepared.observation.photometry.to_mapping(),
        "statistic": "profiled_linear_gaussian_q", "halo_model": spec.subhalo.type,
        "subhalo_redshift": spec.lens.redshift if spec.subhalo_redshift is None else spec.subhalo_redshift,
        "mass_definition": spec.subhalo.mass_definition, "perturbers": [halo.to_mapping() for halo in prepared.scene.perturbers],
        "cosmology": prepared.scene.cosmology.to_mapping(),
        "source_model": {"components": [component.type for component in spec.source.light],
                         "profiled_parameters": sum(name.startswith("source.light.") for name in prepared.nuisances.names)},
        "coordinate_order": "y,x", "units": {"mass": "solar_mass", "position": "arcsec"},
        "engine": dict(prepared.engine.describe()) | {"reference_workers": prepared.execution.reference_workers,
                                                     "batch_size": prepared.execution.batch_size}, "nuisance_names": list(prepared.nuisances.names),
        "nuisance_rank": prepared.workspace.nuisance_rank, "gram_condition_number": condition if np.isfinite(condition) else None,
        "mask": {"kind": prepared._config.forecast.mask.kind, "pixel_count": prepared.record["pixel_count"],
                 "digest": prepared.record["mask_digest"]},
        "noise_covariance": None if covariance is None else prepared.record["file_digests"][str(covariance)],
        "positions": {"kind": positions.kind, "count": len(positions), "domain_radius_arcsec": positions.domain_radius_arcsec},
        "hwoslaps_version": __version__,
    }


def forecast(prepared: PreparedForecast, *, masses_msun: ArrayLike,
             positions: PositionSet | ArrayLike | None = None) -> ForecastResult:
    if not isinstance(prepared, PreparedForecast):
        raise TypeError("prepare a forecast with prepare_forecast first")
    prepared.validate_identity()
    masses = np.asarray(masses_msun, dtype=float)
    if masses.ndim != 1 or masses.size == 0 or not np.all(np.isfinite(masses)) or np.any(masses <= 0.0):
        raise ValueError("masses_msun must be a non-empty vector of positive finite masses")
    selected = prepared.positions if positions is None else positions
    if not isinstance(selected, PositionSet):
        selected = explicit_positions(selected, prepared.scene.spec.lens_centre)
    radii = np.hypot(*(selected.positions_yx - prepared.scene.spec.lens_centre).T)
    if np.max(radii) > prepared.positions.domain_radius_arcsec + 1.0e-12:
        raise ValueError("positions lie outside the prepared domain")
    banks = prepared.engine.evaluate(selected.positions_yx, masses)
    def stack(name: str) -> np.ndarray | None:
        if getattr(banks[0], name) is None:
            return None
        return np.stack([getattr(bank, name) for bank in banks])
    return ForecastResult(masses, selected, stack("fisher_raw"), stack("fisher_profiled"),
                          stack("amplitude_hat"), stack("amplitude_spurious"), prepared.psfs.relation,
                          prepared._config.to_mapping(), _provenance(prepared, selected))

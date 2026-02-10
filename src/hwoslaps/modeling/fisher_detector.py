"""Fisher v1 detector for expected subhalo detectability.

This implementation is intentionally independent from legacy modeling methods.
It computes Asimov detectability and projects out a hard-coded v1 nuisance set.
In v2, nuisance selection should become fully config-driven.
"""

from dataclasses import dataclass
from copy import deepcopy
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import autolens as al

from ..lensing import generate_lensing_system
from ..lensing.utils import LensingData, get_einstein_ring_position
from ..observation.utils import ObservationData
from ..psf.utils import PSFData
from .utils_fisher import FisherLocalData, FisherMapData


@dataclass
class _NuisanceSpec:
    """Descriptor for a single hard-coded v1 nuisance direction."""

    name: str
    path: Optional[Tuple[Any, ...]]
    step_mode: str  # additive or multiplicative
    step_key: Optional[str] = None


class FisherDetector:
    """Compute Fisher v1 detectability metrics for one pipeline scenario.

    Notes
    -----
    v1 nuisance directions are hard-coded to match the current Isothermal lens
    and Exponential source parameterization. This must move to config-driven
    nuisance selection in v2.
    """

    _GRAM_CONDITION_MAX = 1.0e12

    def __init__(
        self,
        observation_baseline: ObservationData,
        lensing_baseline: LensingData,
        psf_data: PSFData,
        full_config: Dict[str, Any],
        fisher_config: Dict[str, Any],
    ):
        self.observation_baseline = observation_baseline
        self.lensing_baseline = lensing_baseline
        self.psf_data = psf_data
        self.full_config = deepcopy(full_config)
        self.fisher_config = deepcopy(fisher_config)

        self.include_background_offset = bool(self.fisher_config["include_background_offset"])
        self.snr_threshold = float(self.fisher_config["snr_threshold"])
        self.finite_diff = deepcopy(self.fisher_config["finite_diff"])
        self.map_config = deepcopy(self.fisher_config["map"])

        # Template configs for deterministic mean-image generation.
        self.baseline_config_template = deepcopy(self.full_config)
        self.baseline_config_template["lensing"]["subhalo"]["enabled"] = False

        self.map_config_template = deepcopy(self.full_config)
        self.map_config_template["lensing"]["subhalo"]["enabled"] = True
        self.map_config_template["lensing"]["subhalo"]["position"] = {
            "type": "direct",
            "centre": [0.0, 0.0],
        }

        # Cache baseline means/mask/weights and nuisance Jacobian.
        self.mu0_adu_2d = self._mean_adu_from_observation(self.observation_baseline)
        self.source_adu_2d = self._source_adu_from_observation(self.observation_baseline)
        self.sigma_adu_2d = self.observation_baseline.noise_map.native

        self.mask_2d = self._build_mask()
        self.mask_flat = self.mask_2d.flatten()
        self.pixels_unmasked = int(np.sum(self.mask_flat))

        self.nuisance_specs = self._build_nuisance_specs()
        self.n_nuisance = len(self.nuisance_specs)

        if self.pixels_unmasked <= self.n_nuisance + 2:
            raise ValueError(
                "Degenerate Fisher mask: number of unmasked pixels must exceed "
                "nuisance directions + 2."
            )

        sigma_masked = self.sigma_adu_2d.flatten()[self.mask_flat]
        eps = 1e-12
        self.weight_1d = 1.0 / np.maximum(sigma_masked**2, eps)

        self.jacobian = self._build_nuisance_jacobian()
        if self.jacobian.shape[0] != self.pixels_unmasked:
            raise ValueError("Nuisance Jacobian row count does not match masked pixel count.")
        if self.jacobian.shape[1] != self.n_nuisance:
            raise ValueError("Nuisance Jacobian column count does not match nuisance directions.")

        self.gram_matrix = (self.jacobian * self.weight_1d[:, None]).T @ self.jacobian
        self.gram_condition_number = float(np.linalg.cond(self.gram_matrix))
        if not np.isfinite(self.gram_condition_number) or self.gram_condition_number > self._GRAM_CONDITION_MAX:
            raise ValueError(
                f"Ill-conditioned nuisance Gram matrix (cond={self.gram_condition_number:.3e})."
            )

    def compute_local(self, observation_test: ObservationData, lensing_test: LensingData) -> FisherLocalData:
        """Compute Fisher local detectability at the injected test position."""
        mu1_adu_2d = self._mean_adu_from_observation(observation_test)
        signal_1d = (mu1_adu_2d - self.mu0_adu_2d).flatten()[self.mask_flat]
        metrics = self._compute_profiled_metrics(signal_1d)

        return FisherLocalData(
            snr_asimov=metrics["snr_asimov"],
            delta_chi2_raw=metrics["delta_chi2_raw"],
            delta_chi2_profiled=metrics["delta_chi2_profiled"],
            degradation=metrics["degradation"],
            pixels_unmasked=self.pixels_unmasked,
            n_nuisance=self.n_nuisance,
            gram_condition_number=self.gram_condition_number,
            true_subhalo_position=lensing_test.subhalo_position,
            true_subhalo_mass=lensing_test.subhalo_mass,
            true_subhalo_model=lensing_test.subhalo_model,
        )

    def compute_map(self) -> FisherMapData:
        """Compute Fisher detectability map over candidate subhalo positions."""
        positions_yx = self._candidate_positions()
        snr_values = []
        profiled_values = []
        raw_values = []

        for position_yx in positions_yx:
            mu1_adu_2d = self._mean_adu_for_position(position_yx)
            signal_1d = (mu1_adu_2d - self.mu0_adu_2d).flatten()[self.mask_flat]
            metrics = self._compute_profiled_metrics(signal_1d)
            snr_values.append(metrics["snr_asimov"])
            profiled_values.append(metrics["delta_chi2_profiled"])
            raw_values.append(metrics["delta_chi2_raw"])

        snr_array = np.array(snr_values, dtype=float)
        profiled_array = np.array(profiled_values, dtype=float)
        raw_array = np.array(raw_values, dtype=float)

        return FisherMapData(
            positions_yx=np.array(positions_yx, dtype=float),
            snr_asimov_by_position=snr_array,
            delta_chi2_profiled_by_position=profiled_array,
            delta_chi2_raw_by_position=raw_array,
            num_positions=len(positions_yx),
            median_snr_asimov=float(np.median(snr_array)),
            p25_snr_asimov=float(np.percentile(snr_array, 25)),
            p75_snr_asimov=float(np.percentile(snr_array, 75)),
            min_snr_asimov=float(np.min(snr_array)),
            max_snr_asimov=float(np.max(snr_array)),
        )

    def _build_nuisance_specs(self) -> List[_NuisanceSpec]:
        """Hard-coded v1 nuisance set; migrate to config selection in v2."""
        specs = [
            _NuisanceSpec(
                name="lens.centre_y",
                path=("lensing", "lens_galaxy", "mass", "centre", 0),
                step_mode="additive",
                step_key="centre_arcsec",
            ),
            _NuisanceSpec(
                name="lens.centre_x",
                path=("lensing", "lens_galaxy", "mass", "centre", 1),
                step_mode="additive",
                step_key="centre_arcsec",
            ),
            _NuisanceSpec(
                name="lens.einstein_radius",
                path=("lensing", "lens_galaxy", "mass", "einstein_radius"),
                step_mode="additive",
                step_key="einstein_radius_arcsec",
            ),
            _NuisanceSpec(
                name="lens.ell_comp_1",
                path=("lensing", "lens_galaxy", "mass", "ell_comps", 0),
                step_mode="additive",
                step_key="ell_comp",
            ),
            _NuisanceSpec(
                name="lens.ell_comp_2",
                path=("lensing", "lens_galaxy", "mass", "ell_comps", 1),
                step_mode="additive",
                step_key="ell_comp",
            ),
            _NuisanceSpec(
                name="source.centre_y",
                path=("lensing", "source_galaxy", "light", "centre", 0),
                step_mode="additive",
                step_key="centre_arcsec",
            ),
            _NuisanceSpec(
                name="source.centre_x",
                path=("lensing", "source_galaxy", "light", "centre", 1),
                step_mode="additive",
                step_key="centre_arcsec",
            ),
            _NuisanceSpec(
                name="source.ell_comp_1",
                path=("lensing", "source_galaxy", "light", "ell_comps", 0),
                step_mode="additive",
                step_key="ell_comp",
            ),
            _NuisanceSpec(
                name="source.ell_comp_2",
                path=("lensing", "source_galaxy", "light", "ell_comps", 1),
                step_mode="additive",
                step_key="ell_comp",
            ),
            _NuisanceSpec(
                name="source.intensity",
                path=("lensing", "source_galaxy", "light", "intensity"),
                step_mode="multiplicative",
                step_key="source_intensity_frac",
            ),
            _NuisanceSpec(
                name="source.effective_radius",
                path=("lensing", "source_galaxy", "light", "effective_radius"),
                step_mode="multiplicative",
                step_key="source_reff_frac",
            ),
        ]
        if self.include_background_offset:
            specs.append(
                _NuisanceSpec(
                    name="observation.background_offset_adu",
                    path=None,
                    step_mode="additive",
                    step_key=None,
                )
            )
        return specs

    def _build_mask(self) -> np.ndarray:
        """Build Fisher-specific source SNR mask in ADU domain."""
        eps = 1e-12
        snr_source = self.source_adu_2d / np.maximum(self.sigma_adu_2d, eps)
        mask = snr_source > self.snr_threshold
        if not np.any(mask):
            raise ValueError("Degenerate Fisher mask: no pixels above fisher.snr_threshold.")
        return mask

    def _build_nuisance_jacobian(self) -> np.ndarray:
        """Central-difference derivatives for all nuisance directions."""
        jac_columns: List[np.ndarray] = []
        for spec in self.nuisance_specs:
            if spec.name == "observation.background_offset_adu":
                column = np.ones(self.pixels_unmasked, dtype=float)
                jac_columns.append(column)
                continue

            plus_config = deepcopy(self.baseline_config_template)
            minus_config = deepcopy(self.baseline_config_template)
            step = self._apply_perturbation(plus_config, minus_config, spec)

            mu_plus = self._mean_adu_from_config(plus_config)
            mu_minus = self._mean_adu_from_config(minus_config)
            derivative_2d = (mu_plus - mu_minus) / (2.0 * step)
            jac_columns.append(derivative_2d.flatten()[self.mask_flat])

        return np.column_stack(jac_columns)

    def _apply_perturbation(
        self,
        plus_config: Dict[str, Any],
        minus_config: Dict[str, Any],
        spec: _NuisanceSpec,
    ) -> float:
        """Apply +/- perturbations for one nuisance parameter."""
        if spec.path is None:
            raise ValueError("Unexpected nuisance path None for finite-difference perturbation.")
        if spec.step_key is None:
            raise ValueError("Missing finite-difference step key.")

        step_cfg = float(self.finite_diff[spec.step_key])
        base_value = float(self._get_path_value(plus_config, spec.path))

        if spec.step_mode == "additive":
            step = step_cfg
        elif spec.step_mode == "multiplicative":
            step = abs(base_value) * step_cfg
        else:
            raise ValueError(f"Unknown nuisance step mode: {spec.step_mode}")

        if step <= 0:
            raise ValueError(f"Non-positive finite-difference step for nuisance {spec.name}.")

        self._set_path_value(plus_config, spec.path, base_value + step)
        self._set_path_value(minus_config, spec.path, base_value - step)
        return step

    def _candidate_positions(self) -> List[Tuple[float, float]]:
        """Build map candidate positions using explicit list or ring sampling."""
        explicit = self.map_config.get("explicit_positions_yx")
        if explicit:
            return [tuple(float(v) for v in pair) for pair in explicit]

        num_angles = int(self.map_config["num_angles"])
        offset_pixels = float(self.map_config["offset_pixels"])
        einstein_radius = float(self.lensing_baseline.lens_einstein_radius)
        pixel_scale = float(self.lensing_baseline.pixel_scale)

        positions = []
        for angle in np.linspace(0.0, 360.0, num_angles, endpoint=False):
            positions.append(
                get_einstein_ring_position(
                    angle_deg=float(angle),
                    einstein_radius=einstein_radius,
                    offset_pixels=offset_pixels,
                    pixel_scale=pixel_scale,
                )
            )
        return positions

    def _mean_adu_for_position(self, position_yx: Tuple[float, float]) -> np.ndarray:
        """Generate mean ADU image for a specific direct subhalo position."""
        config = deepcopy(self.map_config_template)
        config["lensing"]["subhalo"]["position"] = {
            "type": "direct",
            "centre": [float(position_yx[0]), float(position_yx[1])],
        }
        return self._mean_adu_from_config(config)

    def _mean_adu_from_config(self, config: Dict[str, Any]) -> np.ndarray:
        """Generate deterministic mean ADU image from a full top-level config."""
        lensing_data = generate_lensing_system(config["lensing"], full_config=config)
        return self._mean_adu_from_lensing(lensing_data=lensing_data, observation_config=config["observation"])

    def _mean_adu_from_lensing(self, lensing_data: LensingData, observation_config: Dict[str, Any]) -> np.ndarray:
        """Compute noiseless PSF-convolved mean image in ADU from a lensing scene."""
        psf_kernel = self._ensure_odd_kernel(self.psf_data.kernel)
        exposure_time = float(observation_config["exposure_time"])
        detector = observation_config["detector"]
        gain = float(detector["gain"])
        sky_background = float(detector["sky_background"])
        dark_current = float(detector["dark_current"])

        mask = al.Mask2D.all_false(
            shape_native=lensing_data.image.shape,
            pixel_scales=lensing_data.pixel_scale,
        )
        lensed_image = al.Array2D(values=lensing_data.image, mask=mask)

        simulator_noiseless = al.SimulatorImaging(
            exposure_time=exposure_time,
            psf=psf_kernel,
            background_sky_level=0.0,
            normalize_psf=False,
            add_poisson_noise_to_data=False,
            noise_seed=0,
        )
        noiseless_dataset = simulator_noiseless.via_image_from(image=lensed_image)
        source_only_eps = noiseless_dataset.data.native

        source_e = source_only_eps * exposure_time
        sky_e = sky_background * exposure_time
        dark_e = dark_current * exposure_time
        return (source_e + sky_e + dark_e) / gain

    def _mean_adu_from_observation(self, observation_data: ObservationData) -> np.ndarray:
        """Compute mean ADU image from ObservationData fields."""
        exposure_time = observation_data.exposure_time
        gain = observation_data.gain
        sky_background = observation_data.sky_background
        dark_current = observation_data.dark_current

        source_e = observation_data.noiseless_source_eps * exposure_time
        sky_e = sky_background * exposure_time
        dark_e = dark_current * exposure_time
        return (source_e + sky_e + dark_e) / gain

    def _source_adu_from_observation(self, observation_data: ObservationData) -> np.ndarray:
        """Source-only mean ADU from ObservationData."""
        source_e = observation_data.noiseless_source_eps * observation_data.exposure_time
        return source_e / observation_data.gain

    def _compute_profiled_metrics(self, signal_1d: np.ndarray) -> Dict[str, float]:
        """Compute raw/profiled DeltaChi2 and SNR from a masked signal vector."""
        raw = float(np.sum(self.weight_1d * signal_1d * signal_1d))

        rhs = (self.jacobian * self.weight_1d[:, None]).T @ signal_1d
        coeffs = np.linalg.solve(self.gram_matrix, rhs)
        profiled_signal = signal_1d - self.jacobian @ coeffs
        profiled = float(np.sum(self.weight_1d * profiled_signal * profiled_signal))
        if profiled < 0:
            if profiled > -1e-10:
                profiled = 0.0
            else:
                raise ValueError("Profiled DeltaChi2 is significantly negative, indicating numerical instability.")

        snr_asimov = float(np.sqrt(profiled))
        degradation = float(profiled / raw) if raw > 0 else 0.0
        return {
            "delta_chi2_raw": raw,
            "delta_chi2_profiled": profiled,
            "snr_asimov": snr_asimov,
            "degradation": degradation,
        }

    @staticmethod
    def _get_path_value(config: Dict[str, Any], path: Tuple[Any, ...]) -> Any:
        """Get a nested config value by path components."""
        current: Any = config
        for key in path:
            current = current[key]
        return current

    @staticmethod
    def _set_path_value(config: Dict[str, Any], path: Tuple[Any, ...], value: Any) -> None:
        """Set a nested config value by path components."""
        current: Any = config
        for key in path[:-1]:
            current = current[key]
        current[path[-1]] = value

    @staticmethod
    def _ensure_odd_kernel(kernel: al.Kernel2D) -> al.Kernel2D:
        """Ensure PSF kernel dimensions are odd for SimulatorImaging."""
        kernel_array = kernel.native
        if kernel_array.shape[0] % 2 == 1 and kernel_array.shape[1] % 2 == 1:
            return kernel

        if kernel_array.shape[0] % 2 == 0:
            kernel_array = kernel_array[:-1, :]
        if kernel_array.shape[1] % 2 == 0:
            kernel_array = kernel_array[:, :-1]

        return al.Kernel2D.no_mask(
            values=kernel_array,
            pixel_scales=kernel.pixel_scales,
            normalize=True,
        )

"""Fisher / Asimov detector integrated with AutoLens + HCIPy.

This module is the package-aware bridge between the statistical core in
:mod:`hwoslaps.modeling.fisher_core` and the HWO-SLAPS forward
model built on PyAutoLens and HCIPy.

The detector computes profiled linear-Gaussian amplitude tests for
Asimov / Fisher forecasts.

Key features
------------
- Uses deterministic mean images, not noisy realizations.
- Supports nuisance profiling with optional Gaussian priors.
- Builds PSF nuisance / systematic modes directly from the HCIPy aberration
  configuration used by the codebase.
- Computes a spurious-subhalo susceptibility scan for PSF modes.
- Reuses the forward-model conventions already present in the pipeline
  (`generate_lensing_system`, `generate_psf_system`, the PyAutoLens
  convolver).

Caveats
-------
This is still a local detector. Its statistical object is rigorous, explicit,
and sweepable, but it remains a surrogate for the full nonlinear profile
likelihood and therefore should be calibrated against sparse full fits before
making requirement-level claims.
"""

from __future__ import annotations

import contextlib
import io
import multiprocessing
import os
import sys
from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import autolens as al
import numpy as np

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:  # pragma: no cover - optional dependency
    _tqdm = None

from ..lensing import generate_lensing_system
from ..lensing.utils import LensingData, get_einstein_ring_position
from ..observation.utils import ObservationData
from ..psf.generator import generate_psf_system
from ..psf.utils import (
    PSFData,
    make_pyauto_convolver,
    make_pyauto_kernel,
    pyauto_kernel_native,
    pyauto_kernel_pixel_scales,
)
from .fisher_adapter import (
    compute_asimov_from_images,
    evaluate_signal_bank_from_images,
    extract_masked_covariance,
    flatten_masked_image,
    scan_systematic_modes_from_images,
    stack_masked_images,
)
from .fisher_core import ProfileLikelihoodWorkspace, Whitener, detectable_area
from .utils_fisher import (
    FisherGridMapData,
    FisherLocalData,
    FisherMapData,
    FisherModeCouplingData,
    FisherModeScanData,
)


@dataclass(frozen=True)
class _ScalarNuisanceSpec:
    """Descriptor for one scalar nuisance parameter."""

    name: str
    path: Optional[Tuple[Any, ...]]
    step_mode: str
    step_key: Optional[str] = None
    prior_sigma: Optional[float] = None


@dataclass(frozen=True)
class _PsfModeSpec:
    """Descriptor for one PSF/systematic mode coefficient."""

    name: str
    family: str
    path: Tuple[Any, ...]
    enable_flag_path: Tuple[Any, ...]
    step: float
    prior_sigma: Optional[float] = None


@dataclass(frozen=True)
class _GridLayout:
    """Node layout for a 2D sensitivity grid map.

    ``positions_yx`` lists only the evaluated nodes in row-major order and
    ``node_indices`` gives each one's ``(i, j)`` into the 2D arrays.
    """

    y_coords: np.ndarray
    x_coords: np.ndarray
    spacing_arcsec: float
    centre_yx: Tuple[float, float]
    evaluated_mask: np.ndarray
    positions_yx: Tuple[Tuple[float, float], ...]
    node_indices: Tuple[Tuple[int, int], ...]


def _mean_adu_from_lensing_arrays(
    lensing_data: LensingData,
    observation_config: Dict[str, Any],
    psf_kernel,
) -> np.ndarray:
    """Noiseless PSF-convolved mean image in ADU for one lensing scene.

    Shared by the in-process detector paths and the grid-map worker
    processes; ``psf_kernel`` must already have odd dimensions.
    """
    return _mean_adu_images_from_lensing_arrays(
        lensing_data=lensing_data,
        observation_config=observation_config,
        psf_kernels=(psf_kernel,),
    )[0]


def _mean_adu_images_from_lensing_arrays(
    lensing_data: LensingData,
    observation_config: Dict[str, Any],
    psf_kernels: Sequence[Any],
) -> Tuple[np.ndarray, ...]:
    """Apply one or more PSF kernels to one prepared lensing scene.

    Ray tracing and construction of the AutoLens image are shared across all
    kernels.  The returned noiseless ADU images follow the kernel order.
    """
    if not psf_kernels:
        raise ValueError("psf_kernels must contain at least one kernel.")
    exposure_time = float(observation_config["exposure_time"])
    throughput = float(observation_config["throughput"])
    detector = observation_config["detector"]
    gain = float(detector["gain"])
    sky_background = float(detector["sky_background"])
    dark_current = float(detector["dark_current"])

    mask = al.Mask2D.all_false(
        shape_native=lensing_data.image.shape,
        pixel_scales=lensing_data.pixel_scale,
    )
    lensed_image = al.Array2D(values=lensing_data.image, mask=mask)

    sky_e = sky_background * exposure_time
    dark_e = dark_current * exposure_time
    images = []
    for psf_kernel in psf_kernels:
        # Direct noiseless convolution; SimulatorImaging is not used because
        # it evaluates Poisson noise internally even with noise disabled,
        # which rejects the epsilon-scale FFT-roundoff negatives produced
        # where a compactly supported source is exactly zero. These mean
        # images are deterministic and enter the Fisher algebra linearly,
        # so the roundoff negatives are kept as-is (clamping them would
        # break bit-level agreement with the truth-template convolutions).
        convolved_eps = make_pyauto_convolver(psf_kernel).convolved_image_from(
            image=lensed_image,
            blurring_image=None,
        )
        source_only_eps = np.asarray(convolved_eps.native) * throughput
        source_e = source_only_eps * exposure_time
        images.append((source_e + sky_e + dark_e) / gain)
    return tuple(images)


_GRID_WORKER_STATE: Dict[str, Any] = {}

# Grid-map worker processes are CPU-bound template factories; cap library
# threading so a large pool does not oversubscribe cores, and keep JAX off
# the GPUs so many workers never contend for accelerator contexts.
_GRID_WORKER_ENV = {
    "JAX_PLATFORMS": "cpu",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def _grid_worker_init(
    config_template: Dict[str, Any],
    kernel_native: np.ndarray,
    kernel_pixel_scales,
    mu0_adu_2d: np.ndarray,
    mask_2d: np.ndarray,
    truth_kernel_native: Optional[np.ndarray] = None,
    truth_kernel_pixel_scales=None,
    truth_config_template: Optional[Dict[str, Any]] = None,
) -> None:
    _GRID_WORKER_STATE["config_template"] = config_template
    _GRID_WORKER_STATE["kernel"] = make_pyauto_kernel(
        values=kernel_native,
        pixel_scales=kernel_pixel_scales,
        normalize=False,
    )
    _GRID_WORKER_STATE["mu0"] = mu0_adu_2d
    _GRID_WORKER_STATE["mask"] = mask_2d
    _GRID_WORKER_STATE["truth_config_template"] = truth_config_template
    if truth_kernel_native is None:
        _GRID_WORKER_STATE["truth_kernel"] = None
    else:
        _GRID_WORKER_STATE["truth_kernel"] = make_pyauto_kernel(
            values=truth_kernel_native,
            pixel_scales=truth_kernel_pixel_scales,
            normalize=False,
        )


def _grid_worker_signal(position_yx: Tuple[float, float]) -> np.ndarray:
    config = deepcopy(_GRID_WORKER_STATE["config_template"])
    config["lensing"]["subhalo"]["position"] = {
        "type": "direct",
        "centre": [float(position_yx[0]), float(position_yx[1])],
    }
    lensing_data = generate_lensing_system(config["lensing"], full_config=config)
    model_kernel = _GRID_WORKER_STATE["kernel"]
    truth_kernel = _GRID_WORKER_STATE["truth_kernel"]
    if truth_kernel is None:
        model_image = _mean_adu_images_from_lensing_arrays(
            lensing_data=lensing_data,
            observation_config=config["observation"],
            psf_kernels=(model_kernel,),
        )[0]
        return flatten_masked_image(
            model_image - _GRID_WORKER_STATE["mu0"],
            mask=_GRID_WORKER_STATE["mask"],
        )
    truth_config_template = _GRID_WORKER_STATE["truth_config_template"]
    if truth_config_template is None:
        model_image, truth_image = _mean_adu_images_from_lensing_arrays(
            lensing_data=lensing_data,
            observation_config=config["observation"],
            psf_kernels=(model_kernel, truth_kernel),
        )
    else:
        model_image = _mean_adu_images_from_lensing_arrays(
            lensing_data=lensing_data,
            observation_config=config["observation"],
            psf_kernels=(model_kernel,),
        )[0]
        truth_config = deepcopy(truth_config_template)
        truth_config["lensing"]["subhalo"]["position"] = {
            "type": "direct",
            "centre": [float(position_yx[0]), float(position_yx[1])],
        }
        truth_lensing_data = generate_lensing_system(
            truth_config["lensing"], full_config=truth_config
        )
        truth_image = _mean_adu_images_from_lensing_arrays(
            lensing_data=truth_lensing_data,
            observation_config=truth_config["observation"],
            psf_kernels=(truth_kernel,),
        )[0]
    model_signal = flatten_masked_image(
        model_image - _GRID_WORKER_STATE["mu0"],
        mask=_GRID_WORKER_STATE["mask"],
    )
    data_residual = flatten_masked_image(
        truth_image - _GRID_WORKER_STATE["mu0"],
        mask=_GRID_WORKER_STATE["mask"],
    )
    return np.stack((model_signal, data_residual), axis=0)


class FisherDetector:
    """Fisher / Asimov detector for HWO-SLAPS.

    Parameters
    ----------
    observation_baseline
        Baseline observation generated with the smooth lens model.
    lensing_baseline
        Baseline lensing scene (pre-PSF image and tracer).
    psf_data
        PSF used for the science observation under test.
    full_config
        Full pipeline configuration.
    fisher_config
        ``modeling.fisher`` configuration block.
    """

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

        fit_psf_config = self.full_config["modeling"].get("fit_psf")
        fit_lens_config = self.full_config["modeling"].get("fit_lens")
        self.fit_psf_mode = str(
            (fit_psf_config or {}).get("mode", "matched")
        ).lower()
        self.fit_psf_delta = None
        if self.fit_psf_mode == "bank":
            raise ValueError(
                "modeling.fit_psf mode 'bank' is a nonlinear-only "
                "treatment; use 'matched' or 'explicit' for Fisher runs"
            )
        self.psf_mismatch_enabled = self.fit_psf_mode in {
            "explicit",
            "delta",
        }
        self.lens_mismatch_enabled = bool(
            fit_lens_config is not None
            and str(fit_lens_config.get("mode", "")).lower() == "explicit"
        )
        self.mismatch_enabled = (
            self.psf_mismatch_enabled or self.lens_mismatch_enabled
        )
        self.fit_full_config = deepcopy(self.full_config)
        if self.psf_mismatch_enabled:
            assert fit_psf_config is not None
            if self.fit_psf_mode == "delta":
                from ..psf.mismatch import (
                    _aberrations_to_wire,
                    _flat_int_map_to_wire,
                    _kernel_sha256,
                    _nested_int_map_to_wire,
                    build_psf_mismatch_spec,
                )

                mismatch_spec = build_psf_mismatch_spec(self.full_config)
                regenerated_truth = self._quiet_generate_psf_system(
                    self.full_config
                )
                truth_kernel_sha256 = _kernel_sha256(
                    pyauto_kernel_native(regenerated_truth.kernel)
                )
                supplied_truth_kernel_sha256 = _kernel_sha256(
                    pyauto_kernel_native(self.psf_data.kernel)
                )
                if truth_kernel_sha256 != supplied_truth_kernel_sha256:
                    raise ValueError(
                        "psf_data was not generated by full_config['psf']: "
                        "regenerated truth kernel sha256 "
                        f"{truth_kernel_sha256}, supplied psf_data kernel "
                        f"sha256 {supplied_truth_kernel_sha256}"
                    )
                self.fit_full_config["psf"] = deepcopy(
                    mismatch_spec.fit_psf_config
                )
                self.fit_psf_delta = {
                    "delta_id": mismatch_spec.delta_id,
                    "requested_amplitude_rms_nm": (
                        mismatch_spec.requested_amplitude_rms_nm
                    ),
                    "measured_draw_rms_nm": (
                        mismatch_spec.measured_draw_rms_nm
                    ),
                    "family": mismatch_spec.family,
                    "seed": mismatch_spec.seed,
                    "prior_table_path": mismatch_spec.prior_table_path,
                    "prior_table_sha256": (
                        mismatch_spec.prior_table_sha256
                    ),
                    "truth_psf_config_hash": (
                        mismatch_spec.truth_psf_config_hash
                    ),
                    "fit_psf_config_hash": (
                        mismatch_spec.fit_psf_config_hash
                    ),
                    "draw_aberrations": _aberrations_to_wire(
                        mismatch_spec.draw_aberrations
                    ),
                    "orthonormal_segment": _nested_int_map_to_wire(
                        mismatch_spec.orthonormal_segment
                    ),
                    "orthonormal_global": _flat_int_map_to_wire(
                        mismatch_spec.orthonormal_global
                    ),
                    "versions": deepcopy(mismatch_spec.versions),
                    "truth_kernel_sha256": truth_kernel_sha256,
                }
            else:
                self.fit_full_config["psf"] = deepcopy(
                    fit_psf_config["psf"]
                )
            self.fit_psf_data = self._quiet_generate_psf_system(
                self.fit_full_config
            )
            if self.fit_psf_mode == "delta":
                assert self.fit_psf_delta is not None
                self.fit_psf_delta["fit_kernel_sha256"] = _kernel_sha256(
                    pyauto_kernel_native(self.fit_psf_data.kernel)
                )
            self.model_psf_data = self.fit_psf_data
        else:
            self.fit_psf_data = None
            self.model_psf_data = self.psf_data
        if self.lens_mismatch_enabled:
            assert fit_lens_config is not None
            fit_lens_galaxy = fit_lens_config["lens_galaxy"]
            target = self.fit_full_config["lensing"]["lens_galaxy"]
            target["mass"] = deepcopy(fit_lens_galaxy["mass"])
            if "shear" in fit_lens_galaxy:
                target["shear"] = deepcopy(fit_lens_galaxy["shear"])
            else:
                target.pop("shear", None)

        self.include_background_offset = bool(self.fisher_config["include_background_offset"])
        self.snr_threshold = float(self.fisher_config["snr_threshold"])
        self.finite_diff = deepcopy(self.fisher_config["finite_diff"])
        self.map_config = deepcopy(self.fisher_config["map"])
        self.map_type = str(self.map_config.get("type", "")).lower()
        if self.map_type not in {"ring", "grid", "explicit"}:
            raise ValueError(
                "modeling.fisher.map.type must be one of: 'ring', 'grid', 'explicit'"
            )
        self.show_progress = self._progress_enabled()
        self.show_timing = self._timing_enabled()

        self.mask_mode = str(self.fisher_config.get("mask_mode", "source_snr")).lower()
        self.include_psf_nuisance = bool(self.fisher_config.get("include_psf_nuisance", False))
        self.compute_psf_mode_scan = bool(
            self.fisher_config.get("compute_psf_mode_scan", False)
        )
        z_tolerance = self.fisher_config.get("mode_scan_z_tolerance", 1.0)
        self.mode_scan_z_tolerance = None if z_tolerance is None else float(z_tolerance)

        self.prior_sigmas = deepcopy(self.fisher_config.get("prior_sigmas", {}))
        self.psf_mode_steps = deepcopy(self.fisher_config.get("psf_mode_steps", {}))
        self.psf_mode_prior_sigmas = deepcopy(
            self.fisher_config.get("psf_mode_prior_sigmas", {})
        )
        if "psf_mode_selection" in self.fisher_config:
            raise ValueError(
                "modeling.fisher.psf_mode_selection is not supported; use modeling.fisher.psf_basis"
            )
        self.psf_basis_config = deepcopy(self.fisher_config.get("psf_basis"))
        self.fit_psf_mode_selection = deepcopy(
            self.fisher_config.get("fit_psf_mode_selection")
        )
        self.scan_psf_mode_selection = deepcopy(
            self.fisher_config.get("scan_psf_mode_selection")
        )

        if (self.include_psf_nuisance or self.compute_psf_mode_scan) and not self.psf_basis_config:
            raise ValueError(
                "modeling.fisher.psf_basis is required when PSF nuisance "
                "fitting or PSF mode scanning is enabled"
            )
        if self.compute_psf_mode_scan and not self.scan_psf_mode_selection:
            raise ValueError(
                "modeling.fisher.scan_psf_mode_selection is required when compute_psf_mode_scan is true"
            )

        covariance_path = self.fisher_config.get("covariance_path")
        if covariance_path is not None:
            covariance_path = os.path.expanduser(str(covariance_path))
            if not os.path.exists(covariance_path):
                raise ValueError(
                    f"modeling.fisher.covariance_path does not exist: {covariance_path}"
                )
            self.full_covariance = np.load(covariance_path)
        else:
            self.full_covariance = None

        # Template configs for deterministic mean-image generation.
        self.baseline_config_template = deepcopy(self.fit_full_config)
        self.baseline_config_template["lensing"]["subhalo"]["enabled"] = False

        self.map_config_template = deepcopy(self.fit_full_config)
        self.map_config_template["lensing"]["subhalo"]["enabled"] = True
        self.map_config_template["lensing"]["subhalo"]["position"] = {
            "type": "direct",
            "centre": [0.0, 0.0],
        }
        self.map_config_template_truth = deepcopy(self.full_config)
        self.map_config_template_truth["lensing"]["subhalo"]["enabled"] = True
        self.map_config_template_truth["lensing"]["subhalo"]["position"] = {
            "type": "direct",
            "centre": [0.0, 0.0],
        }
        self._candidate_positions_cache: Optional[List[Tuple[float, float]]] = None
        self._grid_layout_cache: Optional[_GridLayout] = None
        self._jax_grid_engine = None

        self.fit_psf_config_template = self._build_science_psf_config_template(
            self.fit_full_config
        )
        self.scan_psf_config_template = self._build_science_psf_config_template(
            self.full_config
        )

        self.mu0_adu_2d = self._mean_adu_from_observation(self.observation_baseline)
        self.lensing_baseline_fit = None
        if self.lens_mismatch_enabled:
            self.lensing_baseline_fit = generate_lensing_system(
                self.baseline_config_template["lensing"],
                full_config=self.baseline_config_template,
            )
            self.mu0_model_adu_2d = self._mean_adu_from_lensing(
                lensing_data=self.lensing_baseline_fit,
                observation_config=self.baseline_config_template["observation"],
            )
        elif self.psf_mismatch_enabled:
            self.mu0_model_adu_2d = self._mean_adu_from_config(
                self.baseline_config_template
            )
        else:
            self.mu0_model_adu_2d = self.mu0_adu_2d
        self.source_adu_2d = self._source_adu_from_observation(self.observation_baseline)
        self.sigma_adu_2d = self.observation_baseline.noise_map.native

        self.mask_2d = self._build_mask()
        self.pixels_unmasked = int(np.count_nonzero(self.mask_2d))
        if self.pixels_unmasked <= 0:
            raise ValueError("Degenerate Fisher mask: no pixels selected for analysis.")

        self.scalar_nuisance_specs = self._build_scalar_nuisance_specs()
        self.n_scalar_nuisances = len(self.scalar_nuisance_specs)

        self.instrument_psf_mode_specs = self._build_psf_mode_specs_from_selection(
            self.psf_basis_config,
            context="modeling.fisher.psf_basis",
        )
        self.instrument_psf_mode_names = [spec.name for spec in self.instrument_psf_mode_specs]
        self._instrument_psf_mode_name_set = set(self.instrument_psf_mode_names)

        self.fit_psf_mode_specs = self._resolve_fit_psf_mode_specs()
        self.scan_psf_mode_specs = self._resolve_scan_psf_mode_specs()
        self._validate_psf_mode_spec_sets()

        self.n_psf_fit_modes = len(self.fit_psf_mode_specs)
        self.n_psf_scan_modes = len(self.scan_psf_mode_specs)
        if self.map_type == "grid":
            self.n_map_positions = len(self._grid_layout().positions_yx)
        else:
            self.n_map_positions = len(self._candidate_positions())
        self._log_modeling_summary()

        self.scalar_nuisance_images = self._timed_call(
            "scalar nuisance derivatives",
            self._build_scalar_nuisance_images,
            count=self.n_scalar_nuisances,
            unit="direction",
        )
        self.fit_psf_mode_images = self._timed_call(
            "PSF fit derivatives",
            self._build_psf_mode_images,
            self.fit_psf_mode_specs,
            desc="Fisher PSF fit derivatives",
            role="fit",
            count=self.n_psf_fit_modes,
            unit="mode",
        )
        self.scan_psf_mode_images = self._timed_call(
            "PSF scan derivatives",
            self._build_psf_mode_images,
            self.scan_psf_mode_specs,
            desc="Fisher PSF scan derivatives",
            role="scan",
            count=self.n_psf_scan_modes,
            unit="mode",
        )

        self.nuisance_names: List[str] = [spec.name for spec in self.scalar_nuisance_specs]
        self.nuisance_images: List[np.ndarray] = list(self.scalar_nuisance_images)
        self.prior_precision_diagonal: List[float] = [
            self._precision_from_sigma(spec.prior_sigma) for spec in self.scalar_nuisance_specs
        ]

        if self.include_psf_nuisance:
            self.nuisance_names.extend(spec.name for spec in self.fit_psf_mode_specs)
            self.nuisance_images.extend(self.fit_psf_mode_images)
            self.prior_precision_diagonal.extend(
                self._precision_from_sigma(spec.prior_sigma) for spec in self.fit_psf_mode_specs
            )

        self.n_nuisance = len(self.nuisance_names)
        self.n_psf_modes = len(self.fit_psf_mode_specs)
        self.psf_mode_names = [spec.name for spec in self.fit_psf_mode_specs]
        self.psf_mode_sigmas = [spec.prior_sigma for spec in self.fit_psf_mode_specs]
        self.psf_fit_mode_names = [spec.name for spec in self.fit_psf_mode_specs]
        self.psf_scan_mode_names = [spec.name for spec in self.scan_psf_mode_specs]
        self.psf_scan_mode_sigmas = [spec.prior_sigma for spec in self.scan_psf_mode_specs]

        self.prior_precision = self._build_prior_precision_matrix(self.prior_precision_diagonal)
        self.sigma_masked = flatten_masked_image(self.sigma_adu_2d, mask=self.mask_2d)

        if self.full_covariance is not None:
            self.masked_covariance = extract_masked_covariance(
                self.full_covariance,
                mask=self.mask_2d,
                image_shape=self.mu0_adu_2d.shape,
            )
            whitener = Whitener.from_covariance(self.masked_covariance)
        else:
            self.masked_covariance = None
            whitener = Whitener.from_sigma(self.sigma_masked)
        self.whitener = whitener
        if self.mismatch_enabled:
            bias_masked = flatten_masked_image(
                self.mu0_adu_2d - self.mu0_model_adu_2d,
                mask=self.mask_2d,
            )
            self._bias_whitened = whitener.apply(bias_masked)
        else:
            self._bias_whitened = None

        if self.n_nuisance > 0:
            nuisance_design = stack_masked_images(self.nuisance_images, mask=self.mask_2d)
            nuisance_whitened = whitener.apply(nuisance_design)
        else:
            nuisance_design = None
            nuisance_whitened = None
        self.nuisance_design = nuisance_design

        self.workspace = ProfileLikelihoodWorkspace(
            nuisance_whitened=nuisance_whitened,
            prior_precision=self.prior_precision,
            nuisance_names=self.nuisance_names,
        )
        self.gram_condition_number = float(self.workspace.nuisance_condition_number)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_local(
        self,
        observation_test: ObservationData,
        lensing_test: LensingData,
    ) -> FisherLocalData:
        """Compute local profiled Asimov detectability.

        The calculation uses the injected position. Existing Fisher fields
        describe the fit-side template alone. With an explicit fit PSF or fit
        lens, optional mismatch fields project truth-side data and the smooth
        truth-minus-fit residual onto that profiled template. Lens mismatch
        requires separate truth- and fit-side ray traces.
        """
        mu1_adu_2d = self._mean_adu_from_observation(observation_test)
        if self.mismatch_enabled:
            if self.lens_mismatch_enabled:
                if lensing_test.subhalo_position is None:
                    raise ValueError(
                        "Lens-mismatch local evaluation requires a subhalo position."
                    )
                mu1_model_adu_2d = self._mean_adu_for_position(
                    lensing_test.subhalo_position
                )
            else:
                mu1_model_adu_2d = self._mean_adu_from_lensing(
                    lensing_data=lensing_test,
                    observation_config=self.full_config["observation"],
                )
            smooth_mean_image = self.mu0_model_adu_2d
            subhalo_mean_image = mu1_model_adu_2d
        else:
            mu1_model_adu_2d = mu1_adu_2d
            smooth_mean_image = self.mu0_adu_2d
            subhalo_mean_image = mu1_adu_2d
        result = self._timed_call(
            "local Asimov evaluation",
            compute_asimov_from_images,
            smooth_mean_image=smooth_mean_image,
            subhalo_mean_image=subhalo_mean_image,
            sigma_image=None if self.masked_covariance is not None else self.sigma_adu_2d,
            nuisance_images=self.nuisance_images if self.nuisance_images else None,
            prior_precision=self.prior_precision,
            mask=self.mask_2d,
            amplitude_true=1.0,
            nuisance_names=self.nuisance_names,
            covariance=self.full_covariance,
        )

        mode_scan = None
        if self.compute_psf_mode_scan and self.scan_psf_mode_images:
            mode_scan = self._compute_local_mode_scan(mu1_model_adu_2d)

        amplitude_hat_mismatch = None
        q_mismatch = None
        z_mismatch = None
        amplitude_spurious = None
        q_spurious = None
        z_spurious = None
        if self.mismatch_enabled:
            template_masked = flatten_masked_image(
                mu1_model_adu_2d - self.mu0_model_adu_2d,
                mask=self.mask_2d,
            )
            data_residual_masked = flatten_masked_image(
                mu1_adu_2d - self.mu0_model_adu_2d,
                mask=self.mask_2d,
            )
            template_whitened = self.whitener.apply(template_masked)
            data_residual_whitened = self.whitener.apply(data_residual_masked)
            mismatch = self.workspace.spurious_from_bias(
                template_whitened,
                data_residual_whitened,
            )
            assert self._bias_whitened is not None
            spurious = self.workspace.spurious_from_bias(
                template_whitened,
                self._bias_whitened,
            )
            amplitude_hat_mismatch = float(mismatch.amplitude_spurious)
            z_mismatch = float(
                amplitude_hat_mismatch / result.sigma_amplitude_profiled
            )
            q_mismatch = float(z_mismatch**2)
            amplitude_spurious = float(spurious.amplitude_spurious)
            z_spurious = float(spurious.z_spurious)
            q_spurious = float(z_spurious**2)

        return FisherLocalData(
            snr_asimov=float(result.z_asimov_local),
            delta_chi2_raw=float(result.fisher_raw),
            delta_chi2_profiled=float(result.fisher_profiled),
            degradation=float(result.degradation),
            pixels_unmasked=self.pixels_unmasked,
            n_nuisance=self.n_nuisance,
            gram_condition_number=self.gram_condition_number,
            true_subhalo_position=lensing_test.subhalo_position,
            true_subhalo_mass=lensing_test.subhalo_mass,
            true_subhalo_model=lensing_test.subhalo_model,
            fisher_raw=float(result.fisher_raw),
            fisher_profiled=float(result.fisher_profiled),
            sigma_amplitude_raw=float(result.sigma_amplitude_raw),
            sigma_amplitude_profiled=float(result.sigma_amplitude_profiled),
            q_asimov_local=float(result.q_asimov_local),
            z_asimov_local=float(result.z_asimov_local),
            local_p_one_sided=float(result.local_p_one_sided),
            absorbed_fraction=float(result.absorbed_fraction),
            residual_norm_whitened=float(result.residual_norm_whitened),
            nuisance_prior_penalty=float(result.nuisance_prior_penalty),
            nuisance_rank=int(result.nuisance_rank),
            whitened_size=int(result.whitened_size),
            psf_mode_scan=mode_scan,
            mismatch_enabled=self.mismatch_enabled,
            amplitude_hat_mismatch=amplitude_hat_mismatch,
            q_mismatch=q_mismatch,
            z_mismatch=z_mismatch,
            amplitude_spurious=amplitude_spurious,
            q_spurious=q_spurious,
            z_spurious=z_spurious,
        )

    def compute_map(self) -> FisherMapData:
        """Compute a signal-bank detectability map over candidate positions."""
        if self.mismatch_enabled:
            raise ValueError(
                "fit_psf and fit_lens mismatch are unsupported for legacy "
                "bank maps; use map.type: grid for mismatch runs."
            )
        positions_yx = self._candidate_positions()
        build_start = perf_counter()
        subhalo_mean_images = [
            self._mean_adu_for_position(pos)
            for pos in self._progress_iter(
                positions_yx,
                desc="Fisher map templates",
                total=len(positions_yx),
            )
        ]
        self._log_timing(
            "map template generation",
            perf_counter() - build_start,
            count=len(positions_yx),
            unit="position",
        )
        result = self._timed_call(
            "map bank evaluation",
            evaluate_signal_bank_from_images,
            smooth_mean_image=self.mu0_adu_2d,
            subhalo_mean_images=subhalo_mean_images,
            sigma_image=None if self.masked_covariance is not None else self.sigma_adu_2d,
            nuisance_images=self.nuisance_images if self.nuisance_images else None,
            prior_precision=self.prior_precision,
            mask=self.mask_2d,
            amplitude_true=1.0,
            nuisance_names=self.nuisance_names,
            covariance=self.full_covariance,
        )

        snr = np.asarray(result.z_asimov_local, dtype=float)
        raw = np.asarray(result.fisher_raw, dtype=float)
        profiled = np.asarray(result.fisher_profiled, dtype=float)

        return FisherMapData(
            positions_yx=np.asarray(positions_yx, dtype=float),
            snr_asimov_by_position=snr,
            delta_chi2_profiled_by_position=profiled,
            delta_chi2_raw_by_position=raw,
            num_positions=len(positions_yx),
            median_snr_asimov=float(np.median(snr)),
            p25_snr_asimov=float(np.percentile(snr, 25)),
            p75_snr_asimov=float(np.percentile(snr, 75)),
            min_snr_asimov=float(np.min(snr)),
            max_snr_asimov=float(np.max(snr)),
            fisher_raw_by_position=raw,
            fisher_profiled_by_position=profiled,
            q_asimov_local_by_position=np.asarray(result.q_asimov_local, dtype=float),
            z_asimov_local_by_position=snr,
            sigma_amplitude_profiled_by_position=np.asarray(
                result.sigma_amplitude_profiled, dtype=float
            ),
            degradation_by_position=np.asarray(result.degradation, dtype=float),
            absorbed_fraction_by_position=np.asarray(result.absorbed_fraction, dtype=float),
        )

    _GRID_EVAL_BATCH = 256

    def compute_grid_map(self) -> FisherGridMapData:
        """Compute a 2D sensitivity grid map with streaming bank evaluation.

        Templates are generated, projected, and discarded in batches so
        memory stays flat regardless of grid size.  With
        ``map.num_workers > 1`` template generation fans out over spawned
        worker processes; the statistical evaluation always runs in the
        parent against the workspace built at construction time.

        Existing Fisher arrays describe fit-side templates. With an explicit
        fit PSF or fit lens, paired truth-side data residuals are evaluated in
        the same streaming batches and populate the optional mismatch arrays.
        Lens mismatch costs about twice the per-node ray-tracing work.
        """
        if self.map_type != "grid":
            raise ValueError("compute_grid_map requires modeling.fisher.map.type: 'grid'.")

        layout = self._grid_layout()
        positions = list(layout.positions_yx)
        n_positions = len(positions)
        num_workers = int(self.map_config.get("num_workers", 1))
        threshold = float(self.map_config.get("detection_q_threshold", 10.0))

        start = perf_counter()
        signal_iter = self._grid_signal_iterator(positions, num_workers=num_workers)
        progressed = self._progress_iter(
            signal_iter,
            desc="Fisher grid map templates",
            total=n_positions,
        )

        batch: List[np.ndarray] = []
        results = []

        def _flush() -> None:
            if not batch:
                return
            if self.mismatch_enabled:
                signal_matrix = np.column_stack([pair[0] for pair in batch])
                data_matrix = np.column_stack([pair[1] for pair in batch])
                signal_whitened = self.whitener.apply(signal_matrix)
                data_whitened = self.whitener.apply(data_matrix)
                assert self._bias_whitened is not None
                results.append(
                    self.workspace.evaluate_signal_bank(
                        signal_whitened.T,
                        data_bank_whitened=data_whitened.T,
                        bias_whitened=self._bias_whitened,
                    )
                )
            else:
                signal_matrix = np.column_stack(batch)
                whitened = self.whitener.apply(signal_matrix)
                results.append(self.workspace.evaluate_signal_bank(whitened.T))
            batch.clear()

        for signal in progressed:
            batch.append(np.asarray(signal, dtype=float))
            if len(batch) >= self._GRID_EVAL_BATCH:
                _flush()
        _flush()
        self._log_timing(
            "grid map streaming evaluation",
            perf_counter() - start,
            count=n_positions,
            unit="position",
        )

        fisher_raw = np.concatenate([np.atleast_1d(r.fisher_raw) for r in results])
        fisher_profiled = np.concatenate([np.atleast_1d(r.fisher_profiled) for r in results])
        q_asimov = np.concatenate([np.atleast_1d(r.q_asimov_local) for r in results])
        z_asimov = np.concatenate([np.atleast_1d(r.z_asimov_local) for r in results])
        sigma_amplitude = np.concatenate(
            [np.atleast_1d(r.sigma_amplitude_profiled) for r in results]
        )
        degradation = np.concatenate([np.atleast_1d(r.degradation) for r in results])
        absorbed_fraction = np.concatenate(
            [np.atleast_1d(r.absorbed_fraction) for r in results]
        )
        if self.mismatch_enabled:
            amplitude_hat = np.concatenate(
                [np.atleast_1d(r.amplitude_hat) for r in results]
            )
            q_mismatch = np.concatenate(
                [np.atleast_1d(r.q_mismatch) for r in results]
            )
            z_mismatch = np.concatenate(
                [np.atleast_1d(r.z_mismatch) for r in results]
            )
            amplitude_spurious = np.concatenate(
                [np.atleast_1d(r.amplitude_spurious) for r in results]
            )
            q_spurious = np.concatenate(
                [np.atleast_1d(r.q_spurious) for r in results]
            )
            z_spurious = np.concatenate(
                [np.atleast_1d(r.z_spurious) for r in results]
            )
        else:
            amplitude_hat = None
            q_mismatch = None
            z_mismatch = None
            amplitude_spurious = None
            q_spurious = None
            z_spurious = None

        node_idx = np.asarray(layout.node_indices, dtype=int)
        shape = layout.evaluated_mask.shape

        def _to_2d(values: np.ndarray) -> np.ndarray:
            grid = np.full(shape, np.nan, dtype=float)
            grid[node_idx[:, 0], node_idx[:, 1]] = values
            return grid

        detectable_flat = q_asimov >= threshold
        detectable_mask_2d = np.zeros(shape, dtype=bool)
        detectable_mask_2d[node_idx[:, 0], node_idx[:, 1]] = detectable_flat

        cell_area = layout.spacing_arcsec * layout.spacing_arcsec
        area_arcsec2 = detectable_area(q_asimov, cell_area, threshold)

        mismatch_detectable_mask_2d = None
        mismatch_area_arcsec2 = None
        num_mismatch_detectable = None
        false_positive_mask_2d = None
        false_positive_area_arcsec2 = None
        num_false_positive = None
        max_z_spurious = None
        if self.mismatch_enabled:
            assert amplitude_hat is not None
            assert q_mismatch is not None
            assert amplitude_spurious is not None
            assert q_spurious is not None
            assert z_spurious is not None
            mismatch_detectable_flat = (
                (amplitude_hat > 0.0) & (q_mismatch >= threshold)
            )
            false_positive_flat = (
                (amplitude_spurious > 0.0) & (q_spurious >= threshold)
            )
            mismatch_detectable_mask_2d = np.zeros(shape, dtype=bool)
            mismatch_detectable_mask_2d[
                node_idx[:, 0], node_idx[:, 1]
            ] = mismatch_detectable_flat
            false_positive_mask_2d = np.zeros(shape, dtype=bool)
            false_positive_mask_2d[
                node_idx[:, 0], node_idx[:, 1]
            ] = false_positive_flat
            num_mismatch_detectable = int(
                np.count_nonzero(mismatch_detectable_flat)
            )
            num_false_positive = int(np.count_nonzero(false_positive_flat))
            mismatch_area_arcsec2 = float(num_mismatch_detectable * cell_area)
            false_positive_area_arcsec2 = float(
                num_false_positive * cell_area
            )
            max_z_spurious = float(np.nanmax(z_spurious))

        subhalo_config = self.full_config["lensing"]["subhalo"]
        source_image_asset = self.lensing_baseline.source_image_asset or {}

        return FisherGridMapData(
            y_coords=layout.y_coords.copy(),
            x_coords=layout.x_coords.copy(),
            spacing_arcsec=layout.spacing_arcsec,
            centre_yx=layout.centre_yx,
            detection_q_threshold=threshold,
            evaluated_mask_2d=layout.evaluated_mask.copy(),
            detectable_mask_2d=detectable_mask_2d,
            q_asimov_2d=_to_2d(q_asimov),
            z_asimov_2d=_to_2d(z_asimov),
            fisher_raw_2d=_to_2d(fisher_raw),
            fisher_profiled_2d=_to_2d(fisher_profiled),
            sigma_amplitude_profiled_2d=_to_2d(sigma_amplitude),
            degradation_2d=_to_2d(degradation),
            absorbed_fraction_2d=_to_2d(absorbed_fraction),
            num_positions_evaluated=n_positions,
            num_detectable=int(np.count_nonzero(detectable_flat)),
            detectable_area_arcsec2=float(area_arcsec2),
            max_z_asimov=float(np.max(z_asimov)),
            median_z_asimov=float(np.median(z_asimov)),
            subhalo_mass=float(subhalo_config["mass"]) if "mass" in subhalo_config else None,
            subhalo_model=str(subhalo_config["model"]) if "model" in subhalo_config else None,
            lens_einstein_radius=float(self.lensing_baseline.lens_einstein_radius),
            mismatch_enabled=self.mismatch_enabled,
            amplitude_hat_2d=None if amplitude_hat is None else _to_2d(amplitude_hat),
            q_mismatch_2d=None if q_mismatch is None else _to_2d(q_mismatch),
            z_mismatch_2d=None if z_mismatch is None else _to_2d(z_mismatch),
            mismatch_detectable_mask_2d=mismatch_detectable_mask_2d,
            mismatch_detectable_area_arcsec2=mismatch_area_arcsec2,
            num_mismatch_detectable=num_mismatch_detectable,
            amplitude_spurious_2d=(
                None if amplitude_spurious is None else _to_2d(amplitude_spurious)
            ),
            q_spurious_2d=None if q_spurious is None else _to_2d(q_spurious),
            z_spurious_2d=None if z_spurious is None else _to_2d(z_spurious),
            false_positive_mask_2d=false_positive_mask_2d,
            false_positive_area_arcsec2=false_positive_area_arcsec2,
            num_false_positive=num_false_positive,
            max_z_spurious=max_z_spurious,
            source_image_asset_path=source_image_asset.get("asset_path"),
            source_image_asset_sha256_16=source_image_asset.get("sha256_16"),
        )

    def _grid_signal_iterator(
        self,
        positions: Sequence[Tuple[float, float]],
        num_workers: int,
    ) -> Iterator[np.ndarray]:
        """Yield masked signal vectors for grid nodes, serial or pooled."""
        engine = str(self.map_config.get("engine", "reference")).lower()
        if engine == "jax":
            yield from self._grid_signal_iterator_jax(positions)
            return
        if num_workers <= 1:
            for position in positions:
                if self.mismatch_enabled:
                    mu1_model_adu_2d, mu1_truth_adu_2d = (
                        self._mean_adu_pair_for_position(position)
                    )
                    yield np.stack(
                        (
                            flatten_masked_image(
                                mu1_model_adu_2d - self.mu0_model_adu_2d,
                                mask=self.mask_2d,
                            ),
                            flatten_masked_image(
                                mu1_truth_adu_2d - self.mu0_model_adu_2d,
                                mask=self.mask_2d,
                            ),
                        ),
                        axis=0,
                    )
                else:
                    mu1_adu_2d = self._mean_adu_for_position(position)
                    yield flatten_masked_image(
                        mu1_adu_2d - self.mu0_adu_2d,
                        mask=self.mask_2d,
                    )
            return
        yield from self._grid_signal_iterator_parallel(positions, num_workers)

    def _grid_signal_iterator_jax(
        self,
        positions: Sequence[Tuple[float, float]],
    ) -> Iterator[np.ndarray]:
        if self._jax_grid_engine is None:
            from .fisher_grid_jax import JaxGridTemplateEngine

            kernel = self._ensure_odd_kernel(self.model_psf_data.kernel)
            truth_kernel_native = None
            if self.mismatch_enabled:
                truth_kernel = self._ensure_odd_kernel(self.psf_data.kernel)
                truth_kernel_native = np.asarray(
                    pyauto_kernel_native(truth_kernel),
                    dtype=float,
                )
            self._jax_grid_engine = JaxGridTemplateEngine(
                lensing_baseline=self.lensing_baseline,
                lensing_baseline_fit=self.lensing_baseline_fit,
                map_config_template=deepcopy(self.map_config_template),
                psf_kernel_native=np.asarray(pyauto_kernel_native(kernel), dtype=float),
                mu0_adu_2d=self.mu0_model_adu_2d,
                mask_2d=self.mask_2d,
                truth_psf_kernel_native=truth_kernel_native,
                candidate_positions=positions,
                truth_lens_centre_yx=self._grid_layout().centre_yx,
            )
        yield from self._jax_grid_engine.signal_iterator(positions)

    def _grid_signal_iterator_parallel(
        self,
        positions: Sequence[Tuple[float, float]],
        num_workers: int,
    ) -> Iterator[np.ndarray]:
        kernel = self._ensure_odd_kernel(self.model_psf_data.kernel)
        truth_kernel_native = None
        truth_kernel_pixel_scales = None
        if self.mismatch_enabled:
            truth_kernel = self._ensure_odd_kernel(self.psf_data.kernel)
            truth_kernel_native = np.asarray(
                pyauto_kernel_native(truth_kernel),
                dtype=float,
            )
            truth_kernel_pixel_scales = pyauto_kernel_pixel_scales(truth_kernel)
        initargs = (
            deepcopy(self.map_config_template),
            np.asarray(pyauto_kernel_native(kernel), dtype=float),
            pyauto_kernel_pixel_scales(kernel),
            np.asarray(self.mu0_model_adu_2d, dtype=float),
            np.asarray(self.mask_2d, dtype=bool),
            truth_kernel_native,
            truth_kernel_pixel_scales,
            (
                deepcopy(self.map_config_template_truth)
                if self.lens_mismatch_enabled
                else None
            ),
        )
        saved_env = {key: os.environ.get(key) for key in _GRID_WORKER_ENV}
        os.environ.update(_GRID_WORKER_ENV)
        try:
            context = multiprocessing.get_context("spawn")
            with context.Pool(
                processes=num_workers,
                initializer=_grid_worker_init,
                initargs=initargs,
            ) as pool:
                chunksize = max(1, len(positions) // (num_workers * 8))
                yield from pool.imap(_grid_worker_signal, positions, chunksize=chunksize)
        finally:
            for key, value in saved_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    # ------------------------------------------------------------------
    # Forward-model runtime helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _progress_enabled() -> bool:
        if _tqdm is None:
            return False
        disable_env = os.environ.get("HWOSLAPS_DISABLE_TQDM", "").strip().lower()
        if disable_env in {"1", "true", "yes", "on"}:
            return False
        stream = sys.stderr
        isatty = getattr(stream, "isatty", None)
        if callable(isatty):
            try:
                return bool(isatty())
            except Exception:
                return False
        wrapped_streams = getattr(stream, "_streams", ())
        for wrapped in wrapped_streams:
            wrapped_isatty = getattr(wrapped, "isatty", None)
            if callable(wrapped_isatty):
                try:
                    if wrapped_isatty():
                        return True
                except Exception:
                    continue
        return False

    @staticmethod
    def _timing_enabled() -> bool:
        disable_env = os.environ.get("HWOSLAPS_DISABLE_FISHER_TIMING", "").strip().lower()
        return disable_env not in {"1", "true", "yes", "on"}

    def _timed_call(
        self,
        label: str,
        func,
        *args,
        count: Optional[int] = None,
        unit: str = "item",
        **kwargs,
    ):
        if not self.show_timing:
            return func(*args, **kwargs)
        start = perf_counter()
        result = func(*args, **kwargs)
        self._log_timing(label, perf_counter() - start, count=count, unit=unit)
        return result

    def _log_modeling_summary(self) -> None:
        if not self.show_timing:
            return
        print(
            "[Fisher] modeling summary: "
            f"pixels={self.pixels_unmasked}, "
            f"scalar_nuisances={self.n_scalar_nuisances}, "
            f"psf_fit_modes={self.n_psf_fit_modes}, "
            f"psf_scan_modes={self.n_psf_scan_modes}, "
            f"map_positions={self.n_map_positions}"
        )

    def _log_timing(
        self,
        label: str,
        elapsed_s: float,
        *,
        count: Optional[int] = None,
        unit: str = "item",
    ) -> None:
        if not self.show_timing:
            return
        message = f"[Fisher] timing: {label} finished in {elapsed_s:.2f} s"
        if count is not None:
            if count <= 0:
                message += f" ({count} {unit}s)"
            else:
                rate = count / elapsed_s if elapsed_s > 0.0 else float('inf')
                sec_per = elapsed_s / count
                message += (
                    f" ({count} {unit}s, {rate:.3f} {unit}/s, {sec_per:.3f} s/{unit})"
                )
        print(message)

    def _progress_iter(
        self,
        iterable: Iterable[Any],
        *,
        desc: str,
        total: Optional[int] = None,
    ) -> Iterable[Any]:
        if not self.show_progress:
            return iterable
        return _tqdm(
            iterable,
            desc=desc,
            total=total,
            dynamic_ncols=True,
            leave=False,
            mininterval=0.2,
        )

    def _progress_wrapper(
        self,
        *,
        desc: str,
        total: Optional[int] = None,
    ):
        if not self.show_progress:
            return None

        def _wrap(iterable: Iterable[int]) -> Iterable[int]:
            return self._progress_iter(iterable, desc=desc, total=total)

        return _wrap

    def _candidate_positions(self) -> List[Tuple[float, float]]:
        """Build candidate positions for the ring and explicit map types."""
        if self._candidate_positions_cache is not None:
            return list(self._candidate_positions_cache)

        if self.map_type == "grid":
            raise ValueError(
                "Grid maps do not use _candidate_positions; use _grid_layout instead."
            )

        if self.map_type == "explicit":
            explicit = self.map_config.get("explicit_positions_yx")
            if explicit is None or len(explicit) == 0:
                raise ValueError(
                    "modeling.fisher.map.explicit_positions_yx must be non-empty when map.type is 'explicit'."
                )
            positions = []
            numeric_types = (int, float, np.integer, np.floating)
            for i, pair in enumerate(explicit):
                if not isinstance(pair, Sequence) or isinstance(pair, (str, bytes)):
                    raise ValueError(
                        f"modeling.fisher.map.explicit_positions_yx[{i}] must be a length-2 coordinate pair."
                    )
                if len(pair) != 2:
                    raise ValueError(
                        f"modeling.fisher.map.explicit_positions_yx[{i}] must be a length-2 coordinate pair."
                    )
                y, x = pair
                if (
                    isinstance(y, bool)
                    or isinstance(x, bool)
                    or not isinstance(y, numeric_types)
                    or not isinstance(x, numeric_types)
                ):
                    raise ValueError(
                        f"modeling.fisher.map.explicit_positions_yx[{i}] must contain numeric coordinates."
                    )
                y = float(y)
                x = float(x)
                if not np.isfinite(y) or not np.isfinite(x):
                    raise ValueError(
                        f"modeling.fisher.map.explicit_positions_yx[{i}] must contain finite coordinates."
                    )
                positions.append((y, x))
            self._candidate_positions_cache = positions
            return list(self._candidate_positions_cache)

        ring_config = self.map_config["ring"]
        num_angles = int(ring_config["num_angles"])
        offset_pixels = float(ring_config["offset_pixels"])
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
        self._candidate_positions_cache = positions
        return list(self._candidate_positions_cache)

    def _grid_layout(self) -> _GridLayout:
        """Build the node layout for a 2D sensitivity grid map.

        The grid is a square lattice centred on the configured lens mass
        centre, optionally restricted to an annulus around that centre.
        """
        if self._grid_layout_cache is not None:
            return self._grid_layout_cache
        if self.map_type != "grid":
            raise ValueError("_grid_layout is only available when map.type is 'grid'.")

        grid_config = self.map_config["grid"]
        spacing = float(grid_config["spacing_arcsec"])
        half_width = float(grid_config["half_width_arcsec"])
        if spacing <= 0.0 or not np.isfinite(spacing):
            raise ValueError("modeling.fisher.map.grid.spacing_arcsec must be positive and finite.")
        if half_width < spacing or not np.isfinite(half_width):
            raise ValueError(
                "modeling.fisher.map.grid.half_width_arcsec must be finite and >= spacing_arcsec."
            )

        # Candidate geometry is defined by the truth/data lens centre.
        lens_centre = self.full_config["lensing"]["lens_galaxy"]["mass"]["centre"]
        centre_y = float(lens_centre[0])
        centre_x = float(lens_centre[1])

        n_half = int(np.floor(half_width / spacing + 1.0e-9))
        offsets = spacing * np.arange(-n_half, n_half + 1, dtype=float)
        y_coords = centre_y + offsets
        x_coords = centre_x + offsets

        yy = y_coords[:, None]
        xx = x_coords[None, :]
        radius = np.hypot(yy - centre_y, xx - centre_x)

        annulus = grid_config.get("annulus")
        if annulus is None:
            evaluated_mask = np.ones((y_coords.size, x_coords.size), dtype=bool)
        else:
            r_min = float(annulus["r_min_arcsec"])
            r_max = float(annulus["r_max_arcsec"])
            evaluated_mask = (radius >= r_min) & (radius <= r_max)
            if not np.any(evaluated_mask):
                raise ValueError(
                    "modeling.fisher.map.grid.annulus selects no grid nodes; "
                    "widen the annulus or refine the spacing."
                )

        node_indices = []
        positions = []
        for i in range(y_coords.size):
            for j in range(x_coords.size):
                if evaluated_mask[i, j]:
                    node_indices.append((i, j))
                    positions.append((float(y_coords[i]), float(x_coords[j])))

        self._grid_layout_cache = _GridLayout(
            y_coords=y_coords,
            x_coords=x_coords,
            spacing_arcsec=spacing,
            centre_yx=(centre_y, centre_x),
            evaluated_mask=evaluated_mask,
            positions_yx=tuple(positions),
            node_indices=tuple(node_indices),
        )
        return self._grid_layout_cache

    def _mean_adu_for_position(self, position_yx: Tuple[float, float]) -> np.ndarray:
        """Generate mean ADU image for a specific direct subhalo position."""
        config = deepcopy(self.map_config_template)
        config["lensing"]["subhalo"]["position"] = {
            "type": "direct",
            "centre": [float(position_yx[0]), float(position_yx[1])],
        }
        return self._mean_adu_from_config(config)

    def _mean_adu_pair_for_position(
        self,
        position_yx: Tuple[float, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate fit- and truth-side node images.

        PSF-only mismatch shares one ray trace. Lens mismatch regenerates both
        macro models and therefore costs about twice the per-node work.
        """
        config = deepcopy(self.map_config_template)
        config["lensing"]["subhalo"]["position"] = {
            "type": "direct",
            "centre": [float(position_yx[0]), float(position_yx[1])],
        }
        lensing_data = generate_lensing_system(
            config["lensing"],
            full_config=config,
        )
        model_kernel = self._ensure_odd_kernel(self.model_psf_data.kernel)
        truth_kernel = self._ensure_odd_kernel(self.psf_data.kernel)
        if self.lens_mismatch_enabled:
            model_image = _mean_adu_images_from_lensing_arrays(
                lensing_data=lensing_data,
                observation_config=config["observation"],
                psf_kernels=(model_kernel,),
            )[0]
            truth_config = deepcopy(self.map_config_template_truth)
            truth_config["lensing"]["subhalo"]["position"] = {
                "type": "direct",
                "centre": [float(position_yx[0]), float(position_yx[1])],
            }
            truth_lensing_data = generate_lensing_system(
                truth_config["lensing"],
                full_config=truth_config,
            )
            truth_image = _mean_adu_images_from_lensing_arrays(
                lensing_data=truth_lensing_data,
                observation_config=truth_config["observation"],
                psf_kernels=(truth_kernel,),
            )[0]
        else:
            model_image, truth_image = _mean_adu_images_from_lensing_arrays(
                lensing_data=lensing_data,
                observation_config=config["observation"],
                psf_kernels=(model_kernel, truth_kernel),
            )
        return model_image, truth_image

    def _mean_adu_from_config(self, config: Dict[str, Any]) -> np.ndarray:
        """Generate a deterministic mean ADU image.

        The image is generated from a full top-level configuration.
        """
        lensing_data = generate_lensing_system(config["lensing"], full_config=config)
        return self._mean_adu_from_lensing(
            lensing_data=lensing_data,
            observation_config=config["observation"],
        )

    def _mean_adu_from_lensing(
        self,
        lensing_data: LensingData,
        observation_config: Dict[str, Any],
    ) -> np.ndarray:
        """Compute a noiseless PSF-convolved mean image.

        The returned image is in ADU and generated from a lensing scene.
        """
        return _mean_adu_from_lensing_arrays(
            lensing_data=lensing_data,
            observation_config=observation_config,
            psf_kernel=self._ensure_odd_kernel(self.model_psf_data.kernel),
        )

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

    def _source_adu_from_kernel(
        self,
        kernel,
        lensing_baseline: Optional[LensingData] = None,
    ) -> np.ndarray:
        """Apply a PSF kernel to the baseline source.

        The kernel may be a derivative kernel.
        """
        psf_kernel = self._ensure_odd_kernel(kernel)
        psf_convolver = make_pyauto_convolver(psf_kernel)
        if lensing_baseline is None:
            lensing_baseline = self.lensing_baseline

        mask = al.Mask2D.all_false(
            shape_native=lensing_baseline.image.shape,
            pixel_scales=lensing_baseline.pixel_scale,
        )
        lensed_image = al.Array2D(values=lensing_baseline.image, mask=mask)
        # PSF derivative kernels are signed and therefore cannot pass through
        # the simulator's Poisson-count path. Use the raw linear convolution.
        source_only_eps = (
            psf_convolver.convolved_image_via_real_space_from(
                image=lensed_image,
                blurring_image=None,
            ).native
            * self.observation_baseline.throughput
        )
        source_e = source_only_eps * self.observation_baseline.exposure_time
        return source_e / self.observation_baseline.gain

    @staticmethod
    def _quiet_generate_psf_system(config: Dict[str, Any]) -> PSFData:
        """Generate a PSF system while silencing verbose HCIPy diagnostics."""
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            return generate_psf_system(config["psf"], full_config=config)

    # ------------------------------------------------------------------
    # Scalar nuisances
    # ------------------------------------------------------------------

    def _build_scalar_nuisance_specs(self) -> List[_ScalarNuisanceSpec]:
        """Build scalar nuisance specifications for the source-light schema.

        Image sources omit the two source ellipticity directions because the
        profile has no ellipticity.  With the optional background direction,
        Image scenes therefore have 10 scalar nuisances instead of 12.

        Prior-sigma caveat: for Clumpy and Image the `source.intensity` and
        `source.effective_radius` directions perturb the dimensionless
        `flux_scale` / `size_scale`, so any configured prior sigmas for
        these names are fractional there, while for Exponential and Sersic
        they are in the parameter's own units.
        """
        analysis_config = getattr(self, "fit_full_config", self.full_config)
        light_type = analysis_config["lensing"]["source_galaxy"]["light"]["type"]
        analysis_lens = analysis_config["lensing"]["lens_galaxy"]
        analysis_mass = analysis_lens["mass"]
        light_root = ("lensing", "source_galaxy", "light")
        if light_type == "Clumpy":
            centre_root = light_root + ("host", "centre")
            ell_comps_root = light_root + ("host", "ell_comps")
        else:
            centre_root = light_root + ("centre",)
            ell_comps_root = light_root + ("ell_comps",)
        if light_type in {"Clumpy", "Image"}:
            intensity_path = light_root + ("flux_scale",)
            effective_radius_path = light_root + ("size_scale",)
        else:
            intensity_path = light_root + ("intensity",)
            effective_radius_path = light_root + ("effective_radius",)

        specs = [
            _ScalarNuisanceSpec(
                name="lens.centre_y",
                path=("lensing", "lens_galaxy", "mass", "centre", 0),
                step_mode="additive",
                step_key="centre_arcsec",
                prior_sigma=self._lookup_prior_sigma("lens.centre_y"),
            ),
            _ScalarNuisanceSpec(
                name="lens.centre_x",
                path=("lensing", "lens_galaxy", "mass", "centre", 1),
                step_mode="additive",
                step_key="centre_arcsec",
                prior_sigma=self._lookup_prior_sigma("lens.centre_x"),
            ),
            _ScalarNuisanceSpec(
                name="lens.einstein_radius",
                path=("lensing", "lens_galaxy", "mass", "einstein_radius"),
                step_mode="additive",
                step_key="einstein_radius_arcsec",
                prior_sigma=self._lookup_prior_sigma("lens.einstein_radius"),
            ),
            _ScalarNuisanceSpec(
                name="lens.ell_comp_1",
                path=("lensing", "lens_galaxy", "mass", "ell_comps", 0),
                step_mode="additive",
                step_key="ell_comp",
                prior_sigma=self._lookup_prior_sigma("lens.ell_comp_1"),
            ),
            _ScalarNuisanceSpec(
                name="lens.ell_comp_2",
                path=("lensing", "lens_galaxy", "mass", "ell_comps", 1),
                step_mode="additive",
                step_key="ell_comp",
                prior_sigma=self._lookup_prior_sigma("lens.ell_comp_2"),
            ),
        ]
        if analysis_mass["type"] == "PowerLaw":
            specs.append(
                _ScalarNuisanceSpec(
                    name="lens.slope",
                    path=("lensing", "lens_galaxy", "mass", "slope"),
                    step_mode="additive",
                    step_key="slope",
                    prior_sigma=self._lookup_prior_sigma("lens.slope"),
                )
            )
            for order in ("m3", "m4"):
                if order not in analysis_mass.get("multipoles", {}):
                    continue
                for component_index in (0, 1):
                    name = f"lens.multipole_{order}_{component_index + 1}"
                    specs.append(
                        _ScalarNuisanceSpec(
                            name=name,
                            path=(
                                "lensing",
                                "lens_galaxy",
                                "mass",
                                "multipoles",
                                order,
                                component_index,
                            ),
                            step_mode="additive",
                            step_key="multipole_comp",
                            prior_sigma=self._lookup_prior_sigma(name),
                        )
                    )
        if "shear" in analysis_lens:
            for component_index in (0, 1):
                name = f"lens.shear_{component_index + 1}"
                specs.append(
                    _ScalarNuisanceSpec(
                        name=name,
                        path=(
                            "lensing",
                            "lens_galaxy",
                            "shear",
                            component_index,
                        ),
                        step_mode="additive",
                        step_key="shear_comp",
                        prior_sigma=self._lookup_prior_sigma(name),
                    )
                )
        specs.extend(
            [
                _ScalarNuisanceSpec(
                    name="source.centre_y",
                    path=centre_root + (0,),
                    step_mode="additive",
                    step_key="centre_arcsec",
                    prior_sigma=self._lookup_prior_sigma("source.centre_y"),
                ),
                _ScalarNuisanceSpec(
                    name="source.centre_x",
                    path=centre_root + (1,),
                    step_mode="additive",
                    step_key="centre_arcsec",
                    prior_sigma=self._lookup_prior_sigma("source.centre_x"),
                ),
            ]
        )
        if light_type != "Image":
            specs.extend(
                [
                    _ScalarNuisanceSpec(
                        name="source.ell_comp_1",
                        path=ell_comps_root + (0,),
                        step_mode="additive",
                        step_key="ell_comp",
                        prior_sigma=self._lookup_prior_sigma(
                            "source.ell_comp_1"
                        ),
                    ),
                    _ScalarNuisanceSpec(
                        name="source.ell_comp_2",
                        path=ell_comps_root + (1,),
                        step_mode="additive",
                        step_key="ell_comp",
                        prior_sigma=self._lookup_prior_sigma(
                            "source.ell_comp_2"
                        ),
                    ),
                ]
            )
        specs.extend(
            [
                _ScalarNuisanceSpec(
                    name="source.intensity",
                    path=intensity_path,
                    step_mode="multiplicative",
                    step_key="source_intensity_frac",
                    prior_sigma=self._lookup_prior_sigma("source.intensity"),
                ),
                _ScalarNuisanceSpec(
                    name="source.effective_radius",
                    path=effective_radius_path,
                    step_mode="multiplicative",
                    step_key="source_reff_frac",
                    prior_sigma=self._lookup_prior_sigma(
                        "source.effective_radius"
                    ),
                ),
            ]
        )
        if self.include_background_offset:
            specs.append(
                _ScalarNuisanceSpec(
                    name="observation.background_offset_adu",
                    path=None,
                    step_mode="additive",
                    step_key=None,
                    prior_sigma=self._lookup_prior_sigma("observation.background_offset_adu"),
                )
            )
        return specs

    def _build_scalar_nuisance_images(self) -> List[np.ndarray]:
        images: List[np.ndarray] = []
        for spec in self._progress_iter(
            self.scalar_nuisance_specs,
            desc="Fisher scalar nuisances",
            total=len(self.scalar_nuisance_specs),
        ):
            if spec.path is None:
                images.append(np.ones_like(self.mu0_adu_2d, dtype=float))
                continue
            plus_config = deepcopy(self.baseline_config_template)
            minus_config = deepcopy(self.baseline_config_template)
            step = self._apply_scalar_perturbation(plus_config, minus_config, spec)
            mu_plus = self._mean_adu_from_config(plus_config)
            mu_minus = self._mean_adu_from_config(minus_config)
            images.append((mu_plus - mu_minus) / (2.0 * step))
        return images

    def _apply_scalar_perturbation(
        self,
        plus_config: Dict[str, Any],
        minus_config: Dict[str, Any],
        spec: _ScalarNuisanceSpec,
    ) -> float:
        if spec.path is None:
            raise ValueError("Unexpected nuisance path None for scalar perturbation.")
        if spec.step_key is None:
            raise ValueError(f"Missing finite-difference step key for nuisance {spec.name}.")

        step_cfg = float(self.finite_diff[spec.step_key])
        base_value = float(self._get_path_value_or_default(plus_config, spec.path, 0.0))

        if spec.step_mode == "additive":
            step = step_cfg
        elif spec.step_mode == "multiplicative":
            step = abs(base_value) * step_cfg
            if step == 0.0:
                step = step_cfg
        else:
            raise ValueError(f"Unknown nuisance step mode: {spec.step_mode}")

        if step <= 0.0 or not np.isfinite(step):
            raise ValueError(f"Invalid finite-difference step for nuisance {spec.name}: {step}")

        if spec.name == "lens.slope":
            lower_slope = base_value - step
            upper_slope = base_value + step
            if lower_slope <= 1.0 or upper_slope >= 3.0:
                raise ValueError(
                    "PowerLaw slope finite difference violates the supported "
                    f"bounds (1, 3): slope={base_value}, step={step}, "
                    f"perturbed interval=[{lower_slope}, {upper_slope}]."
                )

        self._set_path_value_create(plus_config, spec.path, base_value + step)
        self._set_path_value_create(minus_config, spec.path, base_value - step)
        light = plus_config["lensing"]["source_galaxy"]["light"]
        if light["type"] == "Clumpy" and spec.name in {
            "source.centre_y",
            "source.centre_x",
        }:
            coordinate_index = int(spec.path[-1])
            for clump in light["clumps"]:
                clump["centre"][coordinate_index] += step
            for clump in minus_config["lensing"]["source_galaxy"]["light"][
                "clumps"
            ]:
                clump["centre"][coordinate_index] -= step
        return float(step)

    # ------------------------------------------------------------------
    # PSF/systematic modes
    # ------------------------------------------------------------------

    def _build_psf_mode_specs_from_selection(
        self,
        selection_config: Optional[Dict[str, Any]],
        *,
        context: str,
    ) -> List[_PsfModeSpec]:
        if not selection_config:
            return []

        supported = {
            "segment_pistons",
            "segment_tiptilts",
            "segment_hexikes",
            "global_zernikes",
        }
        unsupported = sorted(set(selection_config.keys()) - supported)
        if unsupported:
            raise ValueError(
                f"{context} contains unsupported PSF families: {unsupported}. "
                f"Supported families are: {sorted(supported)}"
            )

        specs: List[_PsfModeSpec] = []

        for seg_id in self._parse_segment_selection(
            selection_config.get("segment_pistons"),
            family="segment_pistons",
            context=f"{context}.segment_pistons",
        ):
            name = f"psf.segment_pistons[{seg_id}]"
            specs.append(
                _PsfModeSpec(
                    name=name,
                    family="segment_pistons",
                    path=("psf", "aberrations", "segment_pistons", int(seg_id)),
                    enable_flag_path=("psf", "aberrations", "enable_segment_pistons"),
                    step=self._lookup_psf_step("segment_pistons", name),
                    prior_sigma=self._lookup_psf_prior_sigma("segment_pistons", name),
                )
            )

        for seg_id in self._parse_segment_selection(
            selection_config.get("segment_tiptilts"),
            family="segment_tiptilts",
            context=f"{context}.segment_tiptilts",
        ):
            for comp_idx, comp_name in enumerate(("tip", "tilt")):
                name = f"psf.segment_tiptilts[{seg_id}].{comp_name}"
                specs.append(
                    _PsfModeSpec(
                        name=name,
                        family="segment_tiptilts",
                        path=("psf", "aberrations", "segment_tiptilts", int(seg_id), comp_idx),
                        enable_flag_path=("psf", "aberrations", "enable_segment_tiptilts"),
                        step=self._lookup_psf_step("segment_tiptilts", name),
                        prior_sigma=self._lookup_psf_prior_sigma("segment_tiptilts", name),
                    )
                )

        for seg_id, mode_noll in self._parse_segment_hexike_selection(
            selection_config.get("segment_hexikes"),
            context=f"{context}.segment_hexikes",
        ):
            name = f"psf.segment_hexikes[{seg_id}][{mode_noll}]"
            specs.append(
                _PsfModeSpec(
                    name=name,
                    family="segment_hexikes",
                    path=("psf", "aberrations", "segment_hexikes", int(seg_id), int(mode_noll)),
                    enable_flag_path=("psf", "aberrations", "enable_segment_hexikes"),
                    step=self._lookup_psf_step("segment_hexikes", name),
                    prior_sigma=self._lookup_psf_prior_sigma("segment_hexikes", name),
                )
            )

        for mode in self._parse_global_zernike_selection(
            selection_config.get("global_zernikes"),
            context=f"{context}.global_zernikes",
        ):
            name = f"psf.global_zernikes[{mode}]"
            specs.append(
                _PsfModeSpec(
                    name=name,
                    family="global_zernikes",
                    path=("psf", "aberrations", "global_zernikes", int(mode)),
                    enable_flag_path=("psf", "aberrations", "enable_global_zernikes"),
                    step=self._lookup_psf_step("global_zernikes", name),
                    prior_sigma=self._lookup_psf_prior_sigma("global_zernikes", name),
                )
            )

        return specs

    def _resolve_fit_psf_mode_specs(self) -> List[_PsfModeSpec]:
        if not self.include_psf_nuisance:
            return []
        if self.fit_psf_mode_selection is None:
            specs = list(self.instrument_psf_mode_specs)
        else:
            specs = self._build_psf_mode_specs_from_selection(
                self.fit_psf_mode_selection,
                context="modeling.fisher.fit_psf_mode_selection",
            )
        self._validate_psf_subset(
            specs,
            context="modeling.fisher.fit_psf_mode_selection",
        )
        return specs

    def _resolve_scan_psf_mode_specs(self) -> List[_PsfModeSpec]:
        if not self.compute_psf_mode_scan:
            return []
        specs = self._build_psf_mode_specs_from_selection(
            self.scan_psf_mode_selection,
            context="modeling.fisher.scan_psf_mode_selection",
        )
        self._validate_psf_subset(
            specs,
            context="modeling.fisher.scan_psf_mode_selection",
        )
        return specs

    def _validate_psf_subset(
        self,
        specs: Sequence[_PsfModeSpec],
        *,
        context: str,
    ) -> None:
        missing = sorted(spec.name for spec in specs if spec.name not in self._instrument_psf_mode_name_set)
        if missing:
            raise ValueError(
                f"{context} contains modes outside modeling.fisher.psf_basis: {missing}"
            )

    def _validate_psf_mode_spec_sets(self) -> None:
        fit_names = {spec.name for spec in self.fit_psf_mode_specs}
        scan_names = {spec.name for spec in self.scan_psf_mode_specs}
        overlap = sorted(fit_names & scan_names)
        if overlap:
            raise ValueError(
                "PSF fit and scan bases must be disjoint. Overlapping modes: "
                f"{overlap}"
            )

    def _build_psf_mode_images(
        self,
        specs: Sequence[_PsfModeSpec],
        *,
        desc: str,
        role: str,
    ) -> List[np.ndarray]:
        return [
            self._psf_derivative_image(spec, role=role)
            for spec in self._progress_iter(
                specs,
                desc=desc,
                total=len(specs),
            )
        ]

    def _build_science_psf_config_template(
        self,
        full_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        config = deepcopy(self.full_config if full_config is None else full_config)
        aberr = config.setdefault("psf", {}).setdefault("aberrations", {})
        family_flags = {
            "segment_pistons": "enable_segment_pistons",
            "segment_tiptilts": "enable_segment_tiptilts",
            "segment_hexikes": "enable_segment_hexikes",
            "global_zernikes": "enable_global_zernikes",
        }
        for family, flag in family_flags.items():
            if not bool(aberr.get(flag, False)):
                aberr[family] = {}
            elif aberr.get(family) is None:
                aberr[family] = {}
        return config

    def _science_psf_base_value(
        self,
        spec: _PsfModeSpec,
        *,
        role: str = "fit",
    ) -> float:
        if role == "fit":
            template = self.fit_psf_config_template
        elif role == "scan":
            template = self.scan_psf_config_template
        else:
            raise ValueError("PSF derivative role must be 'fit' or 'scan'.")
        enabled = bool(self._get_path_value_or_default(
            template, spec.enable_flag_path, False))
        if not enabled:
            return 0.0
        return float(self._get_path_value_or_default(template, spec.path, 0.0))

    def _psf_derivative_image(
        self,
        spec: _PsfModeSpec,
        *,
        role: str,
    ) -> np.ndarray:
        if role == "fit":
            template = self.fit_psf_config_template
        elif role == "scan":
            template = self.scan_psf_config_template
        else:
            raise ValueError("PSF derivative role must be 'fit' or 'scan'.")
        plus_config = deepcopy(template)
        minus_config = deepcopy(template)

        base_value = self._science_psf_base_value(spec, role=role)
        self._set_path_value_create(plus_config, spec.enable_flag_path, True)
        self._set_path_value_create(minus_config, spec.enable_flag_path, True)
        self._ensure_psf_derivative_container(plus_config, spec)
        self._ensure_psf_derivative_container(minus_config, spec)
        self._set_path_value_create(plus_config, spec.path, base_value + spec.step)
        self._set_path_value_create(minus_config, spec.path, base_value - spec.step)

        psf_plus = self._quiet_generate_psf_system(plus_config)
        psf_minus = self._quiet_generate_psf_system(minus_config)
        kernel_plus = self._ensure_odd_kernel(psf_plus.kernel)
        kernel_minus = self._ensure_odd_kernel(psf_minus.kernel)

        derivative_kernel = (
            pyauto_kernel_native(kernel_plus) - pyauto_kernel_native(kernel_minus)
        ) / (2.0 * spec.step)
        derivative_kernel_obj = make_pyauto_kernel(
            values=derivative_kernel,
            pixel_scales=pyauto_kernel_pixel_scales(kernel_plus),
            normalize=False,
        )
        lensing_baseline = self.lensing_baseline
        if role == "fit" and self.lens_mismatch_enabled:
            assert self.lensing_baseline_fit is not None
            lensing_baseline = self.lensing_baseline_fit
        return self._source_adu_from_kernel(
            derivative_kernel_obj,
            lensing_baseline=lensing_baseline,
        )

    @staticmethod
    def _ensure_psf_derivative_container(config: Dict[str, Any], spec: _PsfModeSpec) -> None:
        """Prepare nested PSF coefficient containers before path assignment.

        Segment hexikes use integer dictionary keys at both the segment and
        mode levels. The generic path setter cannot infer that the segment
        level should be a dictionary rather than a list when the science PSF is
        perfect and the family is absent, so handle that structure explicitly.
        """
        if spec.family != "segment_hexikes":
            return
        segment_id = int(spec.path[-2])
        aberr = config.setdefault("psf", {}).setdefault("aberrations", {})
        family = aberr.setdefault("segment_hexikes", {})
        if family is None:
            family = {}
            aberr["segment_hexikes"] = family
        if not isinstance(family, dict):
            raise ValueError("psf.aberrations.segment_hexikes must be a dictionary.")
        if family.get(segment_id) is None:
            family[segment_id] = {}
        if not isinstance(family[segment_id], dict):
            raise ValueError(
                f"psf.aberrations.segment_hexikes[{segment_id}] must be a dictionary."
            )

    def _compute_local_mode_scan(self, mu1_adu_2d: np.ndarray) -> FisherModeScanData:
        all_sigmas_known = all(sigma is not None for sigma in self.psf_scan_mode_sigmas)
        syst_cov = None
        if all_sigmas_known and self.psf_scan_mode_sigmas:
            sigmas_arr = np.asarray(self.psf_scan_mode_sigmas, dtype=float)
            syst_cov = np.diag(sigmas_arr * sigmas_arr)

        scan = self._timed_call(
            "local mode scan",
            scan_systematic_modes_from_images,
            smooth_mean_image=self.mu0_model_adu_2d,
            subhalo_mean_image=mu1_adu_2d,
            systematic_mode_images=self.scan_psf_mode_images,
            sigma_image=None if self.masked_covariance is not None else self.sigma_adu_2d,
            nuisance_images=self.nuisance_images if self.nuisance_images else None,
            prior_precision=self.prior_precision,
            mask=self.mask_2d,
            nuisance_names=self.nuisance_names,
            mode_names=self.psf_scan_mode_names,
            z_tolerance=self.mode_scan_z_tolerance,
            systematic_covariance=syst_cov,
            covariance=self.full_covariance,
            progress=self._progress_wrapper(
                desc="Fisher mode scan",
                total=len(self.scan_psf_mode_images),
            ),
            count=len(self.scan_psf_mode_images),
            unit="mode",
        )

        couplings: List[FisherModeCouplingData] = []
        for coupling, sigma in zip(scan.couplings, self.psf_scan_mode_sigmas):
            one_sigma_z = None if sigma is None else float(coupling.z_per_unit * sigma)
            couplings.append(
                FisherModeCouplingData(
                    mode_name=str(coupling.mode_name),
                    amplitude_per_unit=float(coupling.amplitude_per_unit),
                    z_per_unit=float(coupling.z_per_unit),
                    one_sigma_z=one_sigma_z,
                    tolerance_for_zmax=(
                        None
                        if self.mode_scan_z_tolerance is None
                        else float(coupling.tolerance_for_zmax)
                        if coupling.tolerance_for_zmax is not None
                        else None
                    ),
                )
            )

        return FisherModeScanData(
            couplings=couplings,
            sigma_amplitude_profiled=float(scan.sigma_amplitude_profiled),
            fisher_profiled=float(scan.fisher_profiled),
            rms_spurious_amplitude=(
                None
                if scan.rms_spurious_amplitude is None
                else float(scan.rms_spurious_amplitude)
            ),
            rms_spurious_z=None if scan.rms_spurious_z is None else float(scan.rms_spurious_z),
            z_tolerance=self.mode_scan_z_tolerance,
        )

    # ------------------------------------------------------------------
    # Mask / priors / config parsing helpers
    # ------------------------------------------------------------------

    def _build_mask(self) -> np.ndarray:
        if self.mask_mode == "all_pixels":
            return np.ones_like(self.mu0_adu_2d, dtype=bool)
        if self.mask_mode != "source_snr":
            raise ValueError(
                "modeling.fisher.mask_mode must be 'source_snr' or 'all_pixels'."
            )

        eps = 1.0e-12
        snr_source = self.source_adu_2d / np.maximum(self.sigma_adu_2d, eps)
        mask = snr_source > self.snr_threshold
        if not np.any(mask):
            raise ValueError("Degenerate Fisher mask: no pixels above fisher.snr_threshold.")
        return mask

    def _lookup_prior_sigma(self, name: str) -> Optional[float]:
        value = self.prior_sigmas.get(name)
        if value is None:
            return None
        sigma = float(value)
        if sigma <= 0.0 or not np.isfinite(sigma):
            raise ValueError(f"Invalid prior sigma for {name}: {sigma}")
        return sigma

    def _lookup_psf_step(self, family: str, name: str) -> float:
        value = self.psf_mode_steps.get(name, self.psf_mode_steps.get(family))
        if value is None:
            defaults = {
                "segment_pistons": 1.0,
                "segment_tiptilts": 0.1,
                "segment_hexikes": 1.0,
                "global_zernikes": 1.0,
            }
            value = defaults[family]
        step = float(value)
        if step <= 0.0 or not np.isfinite(step):
            raise ValueError(f"Invalid PSF finite-difference step for {name}: {step}")
        return step

    def _lookup_psf_prior_sigma(self, family: str, name: str) -> Optional[float]:
        value = self.psf_mode_prior_sigmas.get(name, self.psf_mode_prior_sigmas.get(family))
        if value is None:
            return None
        sigma = float(value)
        if sigma <= 0.0 or not np.isfinite(sigma):
            raise ValueError(f"Invalid PSF prior sigma for {name}: {sigma}")
        return sigma

    @staticmethod
    def _precision_from_sigma(sigma: Optional[float]) -> float:
        if sigma is None:
            return 0.0
        return float(1.0 / (sigma * sigma))

    @staticmethod
    def _build_prior_precision_matrix(diagonal: Sequence[float]) -> np.ndarray:
        diag = np.asarray(diagonal, dtype=float)
        if diag.size == 0:
            return np.zeros((0, 0), dtype=float)
        return np.diag(diag)

    def _all_segment_ids(self) -> List[int]:
        return list(range(int(self.psf_data.num_segments)))

    def _coerce_segment_id(self, value: Any, *, context: str) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{context} must be a non-negative integer segment id.")
        seg_id = int(value)
        if seg_id < 0 or seg_id >= int(self.psf_data.num_segments):
            raise ValueError(
                f"{context}={seg_id} is outside the available segment range "
                f"[0, {int(self.psf_data.num_segments) - 1}]"
            )
        return seg_id

    def _coerce_mode_noll(self, value: Any, *, context: str) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{context} must be a 1-based integer Noll index.")
        mode = int(value)
        if mode < 1:
            raise ValueError(f"{context} must be a 1-based integer Noll index.")
        return mode

    def _parse_segment_selection(
        self,
        selection: Any,
        *,
        family: str,
        context: str,
    ) -> List[int]:
        if selection is None:
            return []
        value = selection
        if isinstance(selection, dict):
            if "segments" not in selection:
                raise ValueError(f"{context} must contain a 'segments' field.")
            value = selection["segments"]
        if isinstance(value, str):
            if value.lower() != "all":
                raise ValueError(
                    f"{context} for {family} must be 'all', a list of segment "
                    "ids, or a dict with a 'segments' field."
                )
            segment_ids = self._all_segment_ids()
        elif isinstance(value, (list, tuple)):
            segment_ids = [
                self._coerce_segment_id(seg_id, context=f"{context}[{idx}]")
                for idx, seg_id in enumerate(value)
            ]
        else:
            raise ValueError(
                f"{context} for {family} must be 'all', a list of segment ids, "
                "or a dict with a 'segments' field."
            )
        return sorted(set(segment_ids))

    def _parse_global_zernike_selection(self, selection: Any, *, context: str) -> List[int]:
        if selection is None:
            return []
        value = selection.get("mode_nolls") if isinstance(selection, dict) else selection
        if not isinstance(value, (list, tuple)):
            raise ValueError(
                f"{context} must be a list of 1-based Noll indices or a dict with a 'mode_nolls' field."
            )
        modes = [self._coerce_mode_noll(mode, context=f"{context}[{idx}]") for idx, mode in enumerate(value)]
        return sorted(set(modes))

    def _parse_segment_hexike_selection(
        self,
        selection: Any,
        *,
        context: str,
    ) -> List[Tuple[int, int]]:
        if selection is None:
            return []
        pairs: List[Tuple[int, int]] = []

        if isinstance(selection, dict) and ("segments" in selection or "mode_nolls" in selection):
            if "segments" not in selection or "mode_nolls" not in selection:
                raise ValueError(
                    f"{context} cross-product form must contain both 'segments' and 'mode_nolls'."
                )
            segment_ids = self._parse_segment_selection(
                selection["segments"],
                family="segment_hexikes",
                context=f"{context}.segments",
            )
            modes = self._parse_global_zernike_selection(
                selection["mode_nolls"],
                context=f"{context}.mode_nolls",
            )
            pairs = [(seg_id, mode) for seg_id in segment_ids for mode in modes]
            return sorted(set(pairs))

        if isinstance(selection, dict):
            for seg_key, modes_cfg in selection.items():
                seg_id = self._coerce_segment_id(seg_key, context=f"{context}[segment]")
                modes = self._parse_global_zernike_selection(
                    modes_cfg,
                    context=f"{context}[{seg_id}]",
                )
                pairs.extend((seg_id, mode) for mode in modes)
            return sorted(set(pairs))

        if isinstance(selection, (list, tuple)):
            for idx, pair in enumerate(selection):
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    raise ValueError(
                        f"{context}[{idx}] must be a (segment, mode_noll) pair."
                    )
                seg_id = self._coerce_segment_id(pair[0], context=f"{context}[{idx}][0]")
                mode = self._coerce_mode_noll(pair[1], context=f"{context}[{idx}][1]")
                pairs.append((seg_id, mode))
            return sorted(set(pairs))

        raise ValueError(
            f"{context} must be either {{segments, mode_nolls}}, a mapping "
            "seg->modes, or a list of (segment, mode_noll) pairs."
        )

    # ------------------------------------------------------------------
    # Nested-config helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_path_value_or_default(config: Dict[str, Any], path: Tuple[Any, ...], default: Any) -> Any:
        current: Any = config
        for key in path:
            if isinstance(current, list):
                if not isinstance(key, int) or key < 0 or key >= len(current):
                    return default
                current = current[key]
                continue
            if not isinstance(current, dict) or key not in current:
                return default
            current = current[key]
        return current

    @staticmethod
    def _set_path_value_create(config: Dict[str, Any], path: Tuple[Any, ...], value: Any) -> None:
        current: Any = config
        for idx, key in enumerate(path[:-1]):
            next_key = path[idx + 1]
            if isinstance(current, list):
                if not isinstance(key, int) or key < 0:
                    raise ValueError(f"Invalid list index in path: {key}")
                while len(current) <= key:
                    current.append([] if isinstance(next_key, int) else {})
                if current[key] is None:
                    current[key] = [] if isinstance(next_key, int) else {}
                current = current[key]
                continue

            if not isinstance(current, dict):
                raise ValueError(f"Cannot descend into non-container object for path element {key}")
            if key not in current or current[key] is None:
                current[key] = [] if isinstance(next_key, int) else {}
            current = current[key]

        last = path[-1]
        if isinstance(current, list):
            if not isinstance(last, int) or last < 0:
                raise ValueError(f"Invalid list index in path: {last}")
            while len(current) <= last:
                current.append(0.0)
            current[last] = value
            return

        if not isinstance(current, dict):
            raise ValueError(f"Cannot assign into non-container object for path element {last}")
        current[last] = value

    @staticmethod
    def _ensure_odd_kernel(kernel):
        kernel_array = pyauto_kernel_native(kernel)
        if kernel_array.shape[0] % 2 == 1 and kernel_array.shape[1] % 2 == 1:
            return kernel

        if kernel_array.shape[0] % 2 == 0:
            kernel_array = kernel_array[:-1, :]
        if kernel_array.shape[1] % 2 == 0:
            kernel_array = kernel_array[:, :-1]

        return make_pyauto_kernel(
            values=kernel_array,
            pixel_scales=pyauto_kernel_pixel_scales(kernel),
            normalize=False,
        )

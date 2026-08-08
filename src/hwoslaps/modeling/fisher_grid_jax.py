"""JAX batch engine for 2D Fisher sensitivity grid-map templates.

This module accelerates ``FisherDetector.compute_grid_map`` template
generation by evaluating batches of subhalo positions with JAX instead of
constructing one PyAutoLens scene per node.  It contains no physics of its
own: every physical ingredient is extracted once, exactly, from the same
PyAutoLens objects the reference path uses, and the per-node computation is
pure geometry.

Build-time extraction (all exact, evaluated once):

- the over-sampled image-plane coordinates of the baseline grid, whose
  consecutive-block mean binning reproduces ``tracer.image_2d_from``
  bitwise for uniform over-sampling;
- the macro-model deflection field at those coordinates, from the baseline
  (no-subhalo) tracer;
- the subhalo's radial deflection profile, sampled from the PyAutoLens
  mass profile on a dense logarithmic radius table (profile parameters are
  position-independent; position enters only as the profile centre);
- all source light profiles in deterministic galaxy/profile order, using
  analytic Sersic parameters or zero-padded image samples.  Both evaluators
  are verified against the PyAutoLens profiles at build time on random
  points, so unsupported profile types fail loudly instead of silently
  diverging.

Per node the kernel computes: subhalo deflection by 1D interpolation of
the radius table, traced coordinates, analytic source brightness,
block-mean binning, FFT PSF convolution (equivalent to the simulator's
zero-padded same-mode convolution), the ADU transform, and the masked
signal vector.  Accuracy relative to the reference path is set by the
radial table and FFT round-off and is gated by the equivalence tests.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from ..lensing.generator import _create_subhalo
from ..lensing.image_source import ImageSource
from ..lensing.utils import LensingData

_RADIAL_SAMPLES = 8192
_MISMATCH_RADIAL_SAMPLES = 32768
_RADIAL_R_MIN_ARCSEC = 1.0e-6
_DEFAULT_BATCH_SIZE = 16
_SOURCE_VERIFY_POINTS = 128
_SOURCE_VERIFY_RTOL = 1.0e-10
_IMAGE_SOURCE_VERIFY_RTOL = 1.0e-9
_IMAGE_SUPPORT_GRID_SIZE = 16
_RADIAL_TABLE_MARGIN_FRACTION = 1.0e-6
_EXTENDED_RADIAL_SAMPLE_FACTOR = 4


def _sersic_constant(sersic_index: float) -> float:
    n = float(sersic_index)
    return (
        2.0 * n
        - 1.0 / 3.0
        + 4.0 / (405.0 * n)
        + 46.0 / (25515.0 * n**2)
        + 131.0 / (1148175.0 * n**3)
        - 2194697.0 / (30690717750.0 * n**4)
    )


class JaxGridTemplateEngine:
    """Batched grid-map template generator on JAX (CPU or GPU).

    The engine produces masked signal vectors identical (to interpolation
    accuracy) to ``mu1(position) - mu0`` flattened over ``mask_2d``.

    Parameters
    ----------
    lensing_baseline : `~hwoslaps.lensing.utils.LensingData`
        Baseline (no-subhalo) lensing scene supplying the over-sampled
        grid, tracer, redshifts, and cosmology.
    map_config_template : `dict`
        Config template the reference per-node path consumes; supplies
        the subhalo model and the observation/detector parameters.
    psf_kernel_native : `numpy.ndarray`
        Native-resolution PSF kernel; both dimensions must be odd.
    truth_psf_kernel_native : `numpy.ndarray`, optional
        Truth-side PSF kernel. When supplied, the engine yields fit-template /
        truth-residual pairs, using separate node scenes when a fit baseline
        is also supplied for lens mismatch.
    lensing_baseline_fit : `~hwoslaps.lensing.utils.LensingData`, optional
        Retained fit-side no-subhalo scene for macro-lens mismatch. Its grid
        geometry and source must match ``lensing_baseline`` exactly.
    mu0_adu_2d : `numpy.ndarray`
        Baseline (no-subhalo) mean image in ADU on the native grid.
    mask_2d : `numpy.ndarray`
        Boolean mask selecting the pixels of the signal vector.
    candidate_positions : sequence of `(float, float)`, optional
        Candidate subhalo positions that the engine will evaluate.  The
        positions determine the required radial-deflection table extent.
    truth_lens_centre_yx : `(float, float)`, optional
        Truth/data lens centre defining candidate-position geometry.
    batch_size : `int`, optional
        Number of grid positions evaluated per vmapped batch.
    """

    def __init__(
        self,
        *,
        lensing_baseline: LensingData,
        lensing_baseline_fit: Optional[LensingData] = None,
        map_config_template: Dict[str, Any],
        psf_kernel_native: np.ndarray,
        mu0_adu_2d: np.ndarray,
        mask_2d: np.ndarray,
        truth_psf_kernel_native: Optional[np.ndarray] = None,
        candidate_positions: Optional[Sequence[Tuple[float, float]]] = None,
        truth_lens_centre_yx: Optional[Tuple[float, float]] = None,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ):
        import jax

        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp

        self._jax = jax
        self._jnp = jnp
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")

        grid = lensing_baseline.grid
        tracer = lensing_baseline.tracer
        shape_native = tuple(int(s) for s in lensing_baseline.image.shape)
        n_pix = shape_native[0] * shape_native[1]

        over_sampled = np.asarray(grid.over_sampled, dtype=float)
        sub_per_pix = over_sampled.shape[0] // n_pix
        if over_sampled.shape[0] != n_pix * sub_per_pix or not bool(
            np.all(np.asarray(grid.over_sampler.sub_is_uniform))
        ):
            raise ValueError(
                "JAX grid engine requires uniform over-sampling with "
                "consecutive-block binning."
            )

        deflections_macro = np.asarray(
            tracer.deflections_yx_2d_from(grid=grid.over_sampled), dtype=float
        )
        traced_macro = over_sampled - deflections_macro

        deflections_macro_fit = deflections_macro
        traced_macro_fit = traced_macro
        if lensing_baseline_fit is not None:
            fit_grid = lensing_baseline_fit.grid
            fit_shape_native = tuple(
                int(s) for s in lensing_baseline_fit.image.shape
            )
            if fit_shape_native != shape_native:
                raise ValueError(
                    "Truth and fit baselines must share native grid shape."
                )
            if float(lensing_baseline_fit.pixel_scale) != float(
                lensing_baseline.pixel_scale
            ):
                raise ValueError(
                    "Truth and fit baselines must share grid pixel scale."
                )
            fit_over_sampled = np.asarray(fit_grid.over_sampled, dtype=float)
            fit_sub_per_pix = fit_over_sampled.shape[0] // n_pix
            fit_uniform = bool(
                np.all(np.asarray(fit_grid.over_sampler.sub_is_uniform))
            )
            if (
                fit_over_sampled.shape[0] != n_pix * fit_sub_per_pix
                or not fit_uniform
                or fit_sub_per_pix != sub_per_pix
            ):
                raise ValueError(
                    "Truth and fit baselines must share uniform over-sampling."
                )
            if not np.array_equal(fit_over_sampled, over_sampled):
                raise ValueError(
                    "Truth and fit baselines must share over-sampled grid geometry."
                )
            fit_tracer = lensing_baseline_fit.tracer
            deflections_macro_fit = np.asarray(
                fit_tracer.deflections_yx_2d_from(grid=fit_grid.over_sampled),
                dtype=float,
            )
            traced_macro_fit = fit_over_sampled - deflections_macro_fit

        if truth_lens_centre_yx is None:
            self._query_centre_yx = self._reference_centre_yx(
                map_config_template
            )
        else:
            self._query_centre_yx = self._coerce_centre_yx(
                truth_lens_centre_yx
            )
        candidate_positions_array = self._coerce_candidate_positions(
            candidate_positions,
            map_config_template,
        )
        self._image_radius_max = self._max_radius_from_centre(
            over_sampled,
            self._query_centre_yx,
        )

        subhalo_profile = self._build_subhalo_profile(
            lensing_baseline=lensing_baseline,
            map_config_template=map_config_template,
        )
        radii, alpha_radial = self._sample_radial_deflection(
            subhalo_profile,
            over_sampled=over_sampled,
            num_samples=(
                _MISMATCH_RADIAL_SAMPLES
                if truth_psf_kernel_native is not None
                else _RADIAL_SAMPLES
            ),
            candidate_positions=candidate_positions_array,
            centre_yx=self._query_centre_yx,
        )
        self._radial_r_max = float(radii[-1])

        source_profiles, image_profiles = self._extract_source_profiles(
            tracer=tracer,
            traced_macro=traced_macro,
        )
        if lensing_baseline_fit is not None:
            truth_plane_redshifts = self._plane_redshift_sequence(tracer)
            fit_plane_redshifts = self._plane_redshift_sequence(
                lensing_baseline_fit.tracer
            )
            if truth_plane_redshifts != fit_plane_redshifts:
                raise ValueError(
                    "Truth and fit baseline plane redshifts must be identical; "
                    f"truth={truth_plane_redshifts}, "
                    f"fit={fit_plane_redshifts}."
                )
            fit_source_profiles, fit_image_profiles = (
                self._extract_source_profiles(
                    tracer=lensing_baseline_fit.tracer,
                    traced_macro=traced_macro_fit,
                )
            )
            if not self._source_profile_sets_identical(
                source_profiles,
                image_profiles,
                fit_source_profiles,
                fit_image_profiles,
            ):
                raise ValueError(
                    "Truth and fit baselines must contain identical source profiles."
                )

        observation_config = map_config_template["observation"]
        detector_config = observation_config["detector"]
        exposure_time = float(observation_config["exposure_time"])
        throughput = float(observation_config["throughput"])
        gain = float(detector_config["gain"])
        sky_e = float(detector_config["sky_background"]) * exposure_time
        dark_e = float(detector_config["dark_current"]) * exposure_time

        kernel = np.asarray(psf_kernel_native, dtype=float)
        if kernel.shape[0] % 2 != 1 or kernel.shape[1] % 2 != 1:
            raise ValueError("psf_kernel_native must have odd dimensions.")
        fft_shape = (
            shape_native[0] + kernel.shape[0] - 1,
            shape_native[1] + kernel.shape[1] - 1,
        )
        kernel_fft = np.fft.rfft2(kernel, s=fft_shape)

        mask_flat_idx = np.flatnonzero(np.asarray(mask_2d, dtype=bool).reshape(-1))
        if mask_flat_idx.size == 0:
            raise ValueError("mask_2d selects zero pixels.")

        self._shape_native = shape_native
        self._sub_per_pix = sub_per_pix
        self._crop_y = kernel.shape[0] // 2
        self._crop_x = kernel.shape[1] // 2
        self._fft_shape = fft_shape

        self._coords = jnp.asarray(over_sampled)
        self._alpha_macro_fit = jnp.asarray(deflections_macro_fit)
        self._alpha_macro_truth = jnp.asarray(deflections_macro)
        self._lens_mismatch_enabled = lensing_baseline_fit is not None
        self._log_radii = jnp.asarray(np.log(radii))
        self._alpha_radial = jnp.asarray(alpha_radial)
        self._source_profiles = tuple(source_profiles)
        self._image_profiles = tuple(
            {
                **params,
                "sb_padded": jnp.asarray(params["sb_padded"]),
            }
            for params in image_profiles
        )
        self._kernel_fft = jnp.asarray(kernel_fft)
        self._truth_kernel_fft = None
        self._truth_fft_shape = None
        self._truth_crop_y = None
        self._truth_crop_x = None
        if truth_psf_kernel_native is not None:
            truth_kernel = np.asarray(truth_psf_kernel_native, dtype=float)
            if (
                truth_kernel.shape[0] % 2 != 1
                or truth_kernel.shape[1] % 2 != 1
            ):
                raise ValueError(
                    "truth_psf_kernel_native must have odd dimensions."
                )
            self._truth_fft_shape = (
                shape_native[0] + truth_kernel.shape[0] - 1,
                shape_native[1] + truth_kernel.shape[1] - 1,
            )
            self._truth_kernel_fft = jnp.asarray(
                np.fft.rfft2(truth_kernel, s=self._truth_fft_shape)
            )
            self._truth_crop_y = truth_kernel.shape[0] // 2
            self._truth_crop_x = truth_kernel.shape[1] // 2
        self._mu0_flat = jnp.asarray(np.asarray(mu0_adu_2d, dtype=float).reshape(-1))
        self._mask_flat_idx = jnp.asarray(mask_flat_idx)
        self._scale_source = throughput * exposure_time
        self._offset_adu = (sky_e + dark_e) / gain
        self._gain = gain

        self._batch_signals = jax.jit(jax.vmap(self._signal_for_position))

    # ------------------------------------------------------------------
    # Build-time extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_subhalo_profile(
        lensing_baseline: LensingData,
        map_config_template: Dict[str, Any],
    ):
        """Build the subhalo mass profile once, centred at the origin.

        Profile parameters (Einstein radius or kappa_s / scale radius)
        depend only on mass, concentration model, and cosmology, so a
        single origin-centred profile serves every node via translation.
        """
        subhalo_config = dict(map_config_template["lensing"]["subhalo"])
        subhalo_config["position"] = {"type": "direct", "centre": [0.0, 0.0]}
        lens_galaxy = lensing_baseline.tracer.galaxies[0]
        subhalo, _ = _create_subhalo(
            subhalo_config,
            lensing_baseline.lens_redshift,
            lensing_baseline.source_redshift,
            lens_galaxy,
            pixel_scale=lensing_baseline.pixel_scale,
            cosmology=lensing_baseline.tracer.cosmology,
        )
        return subhalo

    @staticmethod
    def _reference_centre_yx(map_config_template: Dict[str, Any]) -> np.ndarray:
        """Return the lens/grid centre used for radial extent bounds."""
        try:
            centre = map_config_template["lensing"]["lens_galaxy"]["mass"]["centre"]
        except (KeyError, TypeError):
            return np.zeros(2, dtype=float)
        return JaxGridTemplateEngine._coerce_centre_yx(centre)

    @staticmethod
    def _coerce_centre_yx(centre) -> np.ndarray:
        """Return one finite length-two lens centre."""
        centre_array = np.asarray(centre, dtype=float)
        if centre_array.shape != (2,) or not np.all(np.isfinite(centre_array)):
            raise ValueError("Lens mass centre must be a finite length-2 coordinate.")
        return centre_array

    @staticmethod
    def _max_radius_from_centre(points: np.ndarray, centre_yx: np.ndarray) -> float:
        """Return the largest Euclidean radius of points around a centre."""
        offsets = np.asarray(points, dtype=float) - centre_yx[None, :]
        return float(np.max(np.hypot(offsets[:, 0], offsets[:, 1])))

    @classmethod
    def _coerce_candidate_positions(
        cls,
        candidate_positions: Optional[Sequence[Tuple[float, float]]],
        map_config_template: Dict[str, Any],
    ) -> np.ndarray:
        """Coerce candidate positions, retaining a safe direct-use fallback."""
        if candidate_positions is None:
            candidate_positions = ()
            try:
                fisher_map = map_config_template["modeling"]["fisher"]["map"]
                if str(fisher_map.get("type", "")).lower() == "grid":
                    grid_config = fisher_map["grid"]
                    spacing = float(grid_config["spacing_arcsec"])
                    half_width = float(grid_config["half_width_arcsec"])
                    n_half = int(np.floor(half_width / spacing + 1.0e-9))
                    offsets = spacing * np.arange(-n_half, n_half + 1, dtype=float)
                    lens_centre = cls._reference_centre_yx(map_config_template)
                    yy, xx = np.meshgrid(
                        lens_centre[0] + offsets,
                        lens_centre[1] + offsets,
                        indexing="ij",
                    )
                    candidate_positions = np.column_stack((yy.ravel(), xx.ravel()))
            except (KeyError, TypeError, ValueError):
                candidate_positions = ()

        positions = np.asarray(candidate_positions, dtype=float)
        if positions.size == 0:
            return np.empty((0, 2), dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("candidate_positions must have shape (n_positions, 2).")
        if not np.all(np.isfinite(positions)):
            raise ValueError("candidate_positions contains non-finite values.")
        return positions

    @classmethod
    def _sample_radial_deflection(
        cls,
        subhalo_profile,
        over_sampled: np.ndarray,
        num_samples: int,
        candidate_positions: Optional[np.ndarray] = None,
        centre_yx: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        import autolens as al

        if centre_yx is None:
            centre_yx = np.zeros(2, dtype=float)
        centre_yx = np.asarray(centre_yx, dtype=float)
        image_radius = cls._max_radius_from_centre(over_sampled, centre_yx)
        candidate_radius = 0.0
        if candidate_positions is not None and len(candidate_positions) > 0:
            candidate_radius = cls._max_radius_from_centre(
                candidate_positions,
                centre_yx,
            )

        span_y = float(np.ptp(over_sampled[:, 0]))
        span_x = float(np.ptp(over_sampled[:, 1]))
        base_r_max = 4.0 * float(np.hypot(span_y, span_x))
        minimum_r_max = _RADIAL_R_MIN_ARCSEC * (1.0 + _RADIAL_TABLE_MARGIN_FRACTION)
        base_r_max = max(base_r_max, minimum_r_max)
        required_r_max = image_radius + candidate_radius
        required_r_max += _RADIAL_TABLE_MARGIN_FRACTION * max(required_r_max, 1.0)
        r_max = max(base_r_max, required_r_max, minimum_r_max)

        sample_count = int(num_samples)
        if r_max > base_r_max:
            base_log_span = np.log(base_r_max / _RADIAL_R_MIN_ARCSEC)
            required_log_span = np.log(r_max / _RADIAL_R_MIN_ARCSEC)
            sample_count = int(
                np.ceil(sample_count * required_log_span / base_log_span)
            )
            # Endpoint queries are the most sensitive to interpolation error;
            # retain extra log-space density when the table is extended.
            sample_count *= _EXTENDED_RADIAL_SAMPLE_FACTOR
        radii = np.logspace(
            np.log10(_RADIAL_R_MIN_ARCSEC),
            np.log10(r_max),
            sample_count,
        )
        sample_grid = al.Grid2DIrregular(
            values=np.column_stack([np.zeros_like(radii), radii])
        )
        deflections = np.asarray(
            subhalo_profile.deflections_yx_2d_from(grid=sample_grid), dtype=float
        )
        alpha_radial = deflections[:, 1]
        if not np.all(np.isfinite(alpha_radial)):
            raise ValueError("Subhalo radial deflection table contains non-finite values.")
        return radii, alpha_radial

    @staticmethod
    def _sersic_params_from_profile(light_profile) -> Dict[str, float]:
        centre = tuple(float(v) for v in light_profile.centre)
        ell = tuple(float(v) for v in light_profile.ell_comps)
        sersic_index = float(getattr(light_profile, "sersic_index", 1.0))
        return {
            "centre_y": centre[0],
            "centre_x": centre[1],
            "e1": ell[0],
            "e2": ell[1],
            "intensity": float(light_profile.intensity),
            "effective_radius": float(light_profile.effective_radius),
            "sersic_index": sersic_index,
            "sersic_b": float(
                getattr(light_profile, "sersic_constant", _sersic_constant(sersic_index))
            ),
        }

    @staticmethod
    def _sersic_brightness_np(params: Dict[str, float], points: np.ndarray) -> np.ndarray:
        y = points[:, 0] - params["centre_y"]
        x = points[:, 1] - params["centre_x"]
        fac = np.hypot(params["e1"], params["e2"])
        axis_ratio = (1.0 - fac) / (1.0 + fac)
        angle = 0.5 * np.arctan2(params["e1"], params["e2"])
        x_rot = x * np.cos(angle) + y * np.sin(angle)
        y_rot = -x * np.sin(angle) + y * np.cos(angle)
        radius = np.sqrt(axis_ratio) * np.sqrt(x_rot**2 + (y_rot / axis_ratio) ** 2)
        return params["intensity"] * np.exp(
            -params["sersic_b"]
            * (
                (radius / params["effective_radius"]) ** (1.0 / params["sersic_index"])
                - 1.0
            )
        )

    @staticmethod
    def _image_params_from_profile(light_profile) -> Dict[str, Any]:
        """Extract one ImageSource into immutable JAX-kernel parameters."""
        sb = np.asarray(light_profile.sb, dtype=float)
        theta = np.deg2rad(float(light_profile.rotation_deg))
        return {
            "sb_padded": np.pad(sb, 1, mode="constant"),
            "pixel_scale_arcsec": float(light_profile.pixel_scale_arcsec),
            "size_scale": float(light_profile.size_scale),
            "amplitude": float(light_profile.total_flux * light_profile.flux_scale),
            "centre_y": float(light_profile.centre[0]),
            "centre_x": float(light_profile.centre[1]),
            "rotation_cos": float(np.cos(theta)),
            "rotation_sin": float(np.sin(theta)),
            "row_c": (sb.shape[0] - 1) / 2.0,
            "col_c": (sb.shape[1] - 1) / 2.0,
        }

    @staticmethod
    def _bilinear_gather_np(
        array: np.ndarray,
        rows: np.ndarray,
        cols: np.ndarray,
    ) -> np.ndarray:
        """Evaluate a padded array bilinearly with exact zero outside."""
        in_bounds = (
            (rows >= 0.0)
            & (rows <= array.shape[0] - 1)
            & (cols >= 0.0)
            & (cols <= array.shape[1] - 1)
        )
        row0 = np.floor(rows).astype(int)
        col0 = np.floor(cols).astype(int)
        row1 = row0 + 1
        col1 = col0 + 1
        row0_clip = np.clip(row0, 0, array.shape[0] - 1)
        row1_clip = np.clip(row1, 0, array.shape[0] - 1)
        col0_clip = np.clip(col0, 0, array.shape[1] - 1)
        col1_clip = np.clip(col1, 0, array.shape[1] - 1)
        row_weight = rows - row0
        col_weight = cols - col0
        values = (
            (1.0 - row_weight)
            * (1.0 - col_weight)
            * array[row0_clip, col0_clip]
            + (1.0 - row_weight)
            * col_weight
            * array[row0_clip, col1_clip]
            + row_weight
            * (1.0 - col_weight)
            * array[row1_clip, col0_clip]
            + row_weight
            * col_weight
            * array[row1_clip, col1_clip]
        )
        return np.where(in_bounds, values, 0.0)

    @classmethod
    def _image_brightness_np(
        cls,
        params: Dict[str, Any],
        points: np.ndarray,
    ) -> np.ndarray:
        """Evaluate the image-source convention in pure NumPy."""
        dy = points[:, 0] - params["centre_y"]
        dx = points[:, 1] - params["centre_x"]
        u = dx * params["rotation_cos"] + dy * params["rotation_sin"]
        v = -dx * params["rotation_sin"] + dy * params["rotation_cos"]
        scale = params["pixel_scale_arcsec"] * params["size_scale"]
        cols = u / scale + params["col_c"] + 1.0
        rows = v / scale + params["row_c"] + 1.0
        return params["amplitude"] * cls._bilinear_gather_np(
            params["sb_padded"], rows, cols
        )

    @staticmethod
    def _image_support_points(params: Dict[str, Any]) -> np.ndarray:
        """Build deterministic sky-plane probes for an image-source asset."""
        sb = np.asarray(params["sb_padded"])[1:-1, 1:-1]
        row_grid = np.linspace(
            0,
            sb.shape[0] - 1,
            min(sb.shape[0], _IMAGE_SUPPORT_GRID_SIZE),
            dtype=int,
        )
        col_grid = np.linspace(
            0,
            sb.shape[1] - 1,
            min(sb.shape[1], _IMAGE_SUPPORT_GRID_SIZE),
            dtype=int,
        )
        grid_rows, grid_cols = np.meshgrid(row_grid, col_grid, indexing="ij")
        pixel_indices = np.column_stack((grid_rows.ravel(), grid_cols.ravel()))
        grid_values = sb[pixel_indices[:, 0], pixel_indices[:, 1]]
        supported_grid = pixel_indices[grid_values != 0.0]

        first_nonzero = None
        for row, values in enumerate(sb):
            nonzero = np.flatnonzero(values != 0.0)
            if nonzero.size:
                first_nonzero = (row, int(nonzero[0]))
                break
        if first_nonzero is not None:
            pixel_indices = np.vstack((supported_grid, first_nonzero))
        elif supported_grid.size:
            pixel_indices = supported_grid
        pixel_indices = np.unique(pixel_indices, axis=0)

        offsets = np.array(
            [[0.0, 0.0], [-0.5, 0.0], [0.5, 0.0], [0.0, -0.5], [0.0, 0.5]],
            dtype=float,
        )
        row_col = pixel_indices[:, None, :] + offsets[None, :, :]
        row_col = row_col.reshape(-1, 2)
        scale = params["pixel_scale_arcsec"] * params["size_scale"]
        u = (row_col[:, 1] - params["col_c"]) * scale
        v = (row_col[:, 0] - params["row_c"]) * scale
        dx = u * params["rotation_cos"] - v * params["rotation_sin"]
        dy = u * params["rotation_sin"] + v * params["rotation_cos"]
        return np.column_stack(
            (params["centre_y"] + dy, params["centre_x"] + dx)
        )

    @classmethod
    def _extract_source_profiles(
        cls,
        tracer,
        traced_macro: np.ndarray,
    ) -> Tuple[List[Dict[str, float]], List[Dict[str, Any]]]:
        """Extract every analytic and image source light profile.

        The analytic parametrization is verified numerically against the
        PyAutoLens profile on random points spanning the macro-traced
        footprint, at deterministic macro-ray coordinates, and for image
        profiles at deterministic asset-support probes.  Any convention drift
        or unsupported profile type raises at build time.
        """
        import autogalaxy as ag
        import autolens as al

        profiles = []
        image_profiles = []
        light_objects = []
        for galaxy in tracer.galaxies:
            if hasattr(galaxy, "cls_list_from"):
                galaxy_profiles = galaxy.cls_list_from(cls=ag.LightProfile)
            else:
                galaxy_profiles = [
                    value
                    for value in galaxy.__dict__.values()
                    if isinstance(value, ag.LightProfile)
                ]
            light_objects.extend(galaxy_profiles)
        if not light_objects:
            raise ValueError("Baseline tracer has no galaxy with a light profile.")

        rng = np.random.default_rng(0)
        low = traced_macro.min(axis=0)
        high = traced_macro.max(axis=0)
        bbox_points = rng.uniform(low, high, size=(_SOURCE_VERIFY_POINTS, 2))
        macro_step = max(1, int(np.ceil(traced_macro.shape[0] / 256)))
        macro_points = traced_macro[::macro_step]
        base_points = np.vstack((bbox_points, macro_points))

        for light in light_objects:
            if isinstance(light, ImageSource):
                params = cls._image_params_from_profile(light)
                points = np.vstack((base_points, cls._image_support_points(params)))
                sample_grid = al.Grid2DIrregular(values=points)
                reference = np.asarray(
                    light.image_2d_from(grid=sample_grid), dtype=float
                )
                analytic = cls._image_brightness_np(params, points)
                if not np.any(reference != 0.0):
                    raise ValueError(
                        "JAX grid engine image profile verification sampled no "
                        "nonzero reference brightness; source support was not hit."
                    )
                if not np.allclose(
                    analytic,
                    reference,
                    rtol=_IMAGE_SOURCE_VERIFY_RTOL,
                    atol=0.0,
                ):
                    raise ValueError(
                        "JAX grid engine could not reproduce image profile "
                        f"{type(light).__name__} bilinearly; unsupported "
                        "profile type or convention drift."
                    )
                image_profiles.append(params)
                continue
            params = cls._sersic_params_from_profile(light)
            points = base_points
            sample_grid = al.Grid2DIrregular(values=points)
            reference = np.asarray(light.image_2d_from(grid=sample_grid), dtype=float)
            analytic = cls._sersic_brightness_np(params, points)
            if not np.allclose(
                analytic, reference, rtol=_SOURCE_VERIFY_RTOL, atol=0.0
            ):
                raise ValueError(
                    "JAX grid engine could not reproduce light profile "
                    f"{type(light).__name__} analytically; unsupported profile "
                    "type or convention drift."
                )
            profiles.append(params)
        return profiles, image_profiles

    @staticmethod
    def _plane_redshift_sequence(tracer) -> Tuple[float, ...]:
        """Return the ordered plane redshifts carried by a tracer."""
        return tuple(float(plane[0].redshift) for plane in tracer.planes)

    @staticmethod
    def _source_profile_sets_identical(
        source_profiles: Sequence[Dict[str, float]],
        image_profiles: Sequence[Dict[str, Any]],
        fit_source_profiles: Sequence[Dict[str, float]],
        fit_image_profiles: Sequence[Dict[str, Any]],
    ) -> bool:
        """Return whether truth and fit extracted sources are identical."""
        if len(source_profiles) != len(fit_source_profiles):
            return False
        if len(image_profiles) != len(fit_image_profiles):
            return False
        for truth, fit in zip(source_profiles, fit_source_profiles):
            if truth != fit:
                return False
        for truth, fit in zip(image_profiles, fit_image_profiles):
            if set(truth) != set(fit):
                return False
            for key in truth:
                truth_value = truth[key]
                fit_value = fit[key]
                if isinstance(truth_value, np.ndarray):
                    if not np.array_equal(truth_value, fit_value):
                        return False
                elif truth_value != fit_value:
                    return False
        return True

    # ------------------------------------------------------------------
    # Per-node kernel
    # ------------------------------------------------------------------

    def _render_for_macro(self, alpha_macro, alpha_sub):
        """Render the common source through one macro deflection field."""
        jnp = self._jnp
        traced = self._coords - alpha_macro - alpha_sub
        brightness = jnp.zeros(traced.shape[0])
        for params in self._source_profiles:
            y = traced[:, 0] - params["centre_y"]
            x = traced[:, 1] - params["centre_x"]
            fac = np.hypot(params["e1"], params["e2"])
            axis_ratio = (1.0 - fac) / (1.0 + fac)
            angle = 0.5 * np.arctan2(params["e1"], params["e2"])
            x_rot = x * np.cos(angle) + y * np.sin(angle)
            y_rot = -x * np.sin(angle) + y * np.cos(angle)
            radius_ell = jnp.sqrt(axis_ratio) * jnp.sqrt(
                x_rot**2 + (y_rot / axis_ratio) ** 2
            )
            brightness = brightness + params["intensity"] * jnp.exp(
                -params["sersic_b"]
                * (
                    (radius_ell / params["effective_radius"])
                    ** (1.0 / params["sersic_index"])
                    - 1.0
                )
            )

        for params in self._image_profiles:
            dy = traced[:, 0] - params["centre_y"]
            dx = traced[:, 1] - params["centre_x"]
            u = dx * params["rotation_cos"] + dy * params["rotation_sin"]
            v = -dx * params["rotation_sin"] + dy * params["rotation_cos"]
            scale = params["pixel_scale_arcsec"] * params["size_scale"]
            cols = u / scale + params["col_c"] + 1.0
            rows = v / scale + params["row_c"] + 1.0
            array = params["sb_padded"]
            in_bounds = (
                (rows >= 0.0)
                & (rows <= array.shape[0] - 1)
                & (cols >= 0.0)
                & (cols <= array.shape[1] - 1)
            )
            row0 = jnp.floor(rows).astype(jnp.int32)
            col0 = jnp.floor(cols).astype(jnp.int32)
            row1 = row0 + 1
            col1 = col0 + 1
            row0 = jnp.clip(row0, 0, array.shape[0] - 1)
            row1 = jnp.clip(row1, 0, array.shape[0] - 1)
            col0 = jnp.clip(col0, 0, array.shape[1] - 1)
            col1 = jnp.clip(col1, 0, array.shape[1] - 1)
            row_weight = rows - jnp.floor(rows)
            col_weight = cols - jnp.floor(cols)
            interpolated = (
                (1.0 - row_weight) * (1.0 - col_weight) * array[row0, col0]
                + (1.0 - row_weight) * col_weight * array[row0, col1]
                + row_weight * (1.0 - col_weight) * array[row1, col0]
                + row_weight * col_weight * array[row1, col1]
            )
            brightness = brightness + params["amplitude"] * jnp.where(
                in_bounds, interpolated, 0.0
            )

        n_pix = self._shape_native[0] * self._shape_native[1]
        image = brightness.reshape(n_pix, self._sub_per_pix).mean(axis=1)
        return image.reshape(self._shape_native)

    def _signal_for_position(self, position_yx):
        jnp = self._jnp

        delta = self._coords - position_yx[None, :]
        radius = jnp.sqrt(delta[:, 0] ** 2 + delta[:, 1] ** 2)
        radius_safe = jnp.clip(radius, jnp.exp(self._log_radii[0]), None)
        alpha_r = jnp.interp(
            jnp.log(radius_safe), self._log_radii, self._alpha_radial
        )
        alpha_sub = alpha_r[:, None] * delta / radius_safe[:, None]
        image = self._render_for_macro(self._alpha_macro_fit, alpha_sub)

        image_fft = jnp.fft.rfft2(image, s=self._fft_shape)
        convolved_model = jnp.fft.irfft2(
            image_fft * self._kernel_fft,
            s=self._fft_shape,
        )
        convolved_model = convolved_model[
            self._crop_y:self._crop_y + self._shape_native[0],
            self._crop_x:self._crop_x + self._shape_native[1],
        ]

        mu1_model_flat = (
            convolved_model.reshape(-1) * self._scale_source / self._gain
            + self._offset_adu
        )
        model_signal = (mu1_model_flat - self._mu0_flat)[self._mask_flat_idx]
        if self._truth_kernel_fft is None:
            return model_signal

        truth_fft_shape = self._truth_fft_shape
        truth_crop_y = self._truth_crop_y
        truth_crop_x = self._truth_crop_x
        assert truth_fft_shape is not None
        assert truth_crop_y is not None
        assert truth_crop_x is not None
        truth_image = image
        if self._lens_mismatch_enabled:
            truth_image = self._render_for_macro(
                self._alpha_macro_truth,
                alpha_sub,
            )
        truth_image_fft = jnp.fft.rfft2(truth_image, s=truth_fft_shape)
        convolved_truth = jnp.fft.irfft2(
            truth_image_fft * self._truth_kernel_fft,
            s=truth_fft_shape,
        )
        convolved_truth = convolved_truth[
            truth_crop_y:truth_crop_y + self._shape_native[0],
            truth_crop_x:truth_crop_x + self._shape_native[1],
        ]
        mu1_truth_flat = (
            convolved_truth.reshape(-1) * self._scale_source / self._gain
            + self._offset_adu
        )
        data_residual = (mu1_truth_flat - self._mu0_flat)[self._mask_flat_idx]
        return jnp.stack((model_signal, data_residual), axis=0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def signal_iterator(
        self, positions: Sequence[Tuple[float, float]]
    ) -> Iterator[np.ndarray]:
        """Yield masked signal vectors or mismatch pairs for each position.

        Parameters
        ----------
        positions : sequence of `tuple` of `float`
            Subhalo (y, x) centres in arcsec, one per grid node.

        Yields
        ------
        signal : `numpy.ndarray`
            Masked ``mu1 - mu0`` signal vector in ADU for one position, or a
            ``(2, n_masked)`` fit-template/truth-residual pair when the truth
            kernel was supplied.
        """
        jnp = self._jnp
        positions_arr = np.asarray(positions, dtype=float)
        for start in range(0, positions_arr.shape[0], self.batch_size):
            batch_positions = positions_arr[start:start + self.batch_size]
            if batch_positions.size:
                position_radius = self._max_radius_from_centre(
                    batch_positions,
                    self._query_centre_yx,
                )
                possible_query_radius = position_radius + self._image_radius_max
                table_r_max = min(
                    self._radial_r_max,
                    float(np.exp(np.asarray(self._log_radii[-1]))),
                )
                if possible_query_radius > table_r_max:
                    raise ValueError(
                        "JAX radial deflection table is too small for the requested "
                        f"positions: possible query radius {possible_query_radius:.6g} "
                        f"exceeds table maximum {table_r_max:.6g}; rebuild the "
                        "engine with the complete candidate position set."
                    )
            batch = jnp.asarray(batch_positions)
            signals = np.asarray(self._batch_signals(batch))
            for row in signals:
                yield row


__all__ = ["JaxGridTemplateEngine"]

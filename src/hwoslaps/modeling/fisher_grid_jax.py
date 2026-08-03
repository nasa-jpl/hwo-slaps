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
- the source light profiles' Sersic parameters, whose analytic JAX
  evaluation is verified against the PyAutoLens profiles at build time on
  random points, so unsupported profile types fail loudly instead of
  silently diverging.

Per node the kernel computes: subhalo deflection by 1D interpolation of
the radius table, traced coordinates, analytic source brightness,
block-mean binning, FFT PSF convolution (equivalent to the simulator's
zero-padded same-mode convolution), the ADU transform, and the masked
signal vector.  Accuracy relative to the reference path is set by the
radial table and FFT round-off and is gated by the equivalence tests.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Sequence, Tuple

import numpy as np

from ..lensing.generator import _create_subhalo
from ..lensing.utils import LensingData

_RADIAL_SAMPLES = 8192
_RADIAL_R_MIN_ARCSEC = 1.0e-6
_DEFAULT_BATCH_SIZE = 16
_SOURCE_VERIFY_POINTS = 128
_SOURCE_VERIFY_RTOL = 1.0e-10


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
    mu0_adu_2d : `numpy.ndarray`
        Baseline (no-subhalo) mean image in ADU on the native grid.
    mask_2d : `numpy.ndarray`
        Boolean mask selecting the pixels of the signal vector.
    batch_size : `int`, optional
        Number of grid positions evaluated per vmapped batch.
    """

    def __init__(
        self,
        *,
        lensing_baseline: LensingData,
        map_config_template: Dict[str, Any],
        psf_kernel_native: np.ndarray,
        mu0_adu_2d: np.ndarray,
        mask_2d: np.ndarray,
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

        subhalo_profile = self._build_subhalo_profile(
            lensing_baseline=lensing_baseline,
            map_config_template=map_config_template,
        )
        radii, alpha_radial = self._sample_radial_deflection(
            subhalo_profile,
            over_sampled=over_sampled,
        )

        source_profiles = self._extract_source_profiles(
            tracer=tracer,
            traced_macro=traced_macro,
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
        self._alpha_macro = jnp.asarray(deflections_macro)
        self._log_radii = jnp.asarray(np.log(radii))
        self._alpha_radial = jnp.asarray(alpha_radial)
        self._source_profiles = tuple(source_profiles)
        self._kernel_fft = jnp.asarray(kernel_fft)
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
    def _sample_radial_deflection(
        subhalo_profile,
        over_sampled: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        import autolens as al

        span_y = float(np.ptp(over_sampled[:, 0]))
        span_x = float(np.ptp(over_sampled[:, 1]))
        r_max = 4.0 * float(np.hypot(span_y, span_x))
        radii = np.logspace(
            np.log10(_RADIAL_R_MIN_ARCSEC),
            np.log10(r_max),
            _RADIAL_SAMPLES,
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

    @classmethod
    def _extract_source_profiles(
        cls,
        tracer,
        traced_macro: np.ndarray,
    ) -> List[Dict[str, float]]:
        """Extract Sersic parameters for every source light profile.

        The analytic parametrization is verified numerically against the
        PyAutoLens profile on random points spanning the macro-traced
        footprint, so any convention drift or unsupported profile type
        raises at build time.
        """
        import autolens as al

        profiles = []
        light_objects = [
            galaxy.light
            for galaxy in tracer.galaxies
            if getattr(galaxy, "light", None) is not None
        ]
        if not light_objects:
            raise ValueError("Baseline tracer has no galaxy with a light profile.")

        rng = np.random.default_rng(0)
        low = traced_macro.min(axis=0)
        high = traced_macro.max(axis=0)
        points = rng.uniform(low, high, size=(_SOURCE_VERIFY_POINTS, 2))
        sample_grid = al.Grid2DIrregular(values=points)

        for light in light_objects:
            params = cls._sersic_params_from_profile(light)
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
        return profiles

    # ------------------------------------------------------------------
    # Per-node kernel
    # ------------------------------------------------------------------

    def _signal_for_position(self, position_yx):
        jnp = self._jnp

        delta = self._coords - position_yx[None, :]
        radius = jnp.sqrt(delta[:, 0] ** 2 + delta[:, 1] ** 2)
        radius_safe = jnp.clip(radius, jnp.exp(self._log_radii[0]), None)
        alpha_r = jnp.interp(
            jnp.log(radius_safe), self._log_radii, self._alpha_radial
        )
        alpha_sub = alpha_r[:, None] * delta / radius_safe[:, None]

        traced = self._coords - self._alpha_macro - alpha_sub
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

        n_pix = self._shape_native[0] * self._shape_native[1]
        image = brightness.reshape(n_pix, self._sub_per_pix).mean(axis=1)
        image = image.reshape(self._shape_native)

        image_fft = jnp.fft.rfft2(image, s=self._fft_shape)
        convolved = jnp.fft.irfft2(image_fft * self._kernel_fft, s=self._fft_shape)
        convolved = convolved[
            self._crop_y:self._crop_y + self._shape_native[0],
            self._crop_x:self._crop_x + self._shape_native[1],
        ]

        mu1_flat = (
            convolved.reshape(-1) * self._scale_source / self._gain + self._offset_adu
        )
        signal = mu1_flat - self._mu0_flat
        return signal[self._mask_flat_idx]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def signal_iterator(
        self, positions: Sequence[Tuple[float, float]]
    ) -> Iterator[np.ndarray]:
        """Yield masked signal vectors for each position, in order.

        Parameters
        ----------
        positions : sequence of `tuple` of `float`
            Subhalo (y, x) centres in arcsec, one per grid node.

        Yields
        ------
        signal : `numpy.ndarray`
            Masked ``mu1 - mu0`` signal vector in ADU for one position.
        """
        jnp = self._jnp
        positions_arr = np.asarray(positions, dtype=float)
        for start in range(0, positions_arr.shape[0], self.batch_size):
            batch = jnp.asarray(positions_arr[start:start + self.batch_size])
            signals = np.asarray(self._batch_signals(batch))
            for row in signals:
                yield row


__all__ = ["JaxGridTemplateEngine"]

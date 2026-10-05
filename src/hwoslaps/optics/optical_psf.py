"""Detector kernels of an optical system by Fraunhofer propagation.

A kernel is the box integral of the propagated intensity over each detector pixel (an
``N x N`` midpoint rule with ``N = detector_oversampling``), normalized to unit sum on its
support; the power beyond the support is discarded and its complement is recorded in the
kernel's source as ``captured_power_fraction``. The detector grid is built in focal-plane
metres, so every wavelength lands on the same pixels.

``check_sampling`` refuses aliased kernels (the sampled pupil makes the focal plane periodic)
and kernels whose pixels are too coarse for the oversampling to resolve the intensity before
binning.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any

import numpy as np

from ..constants import ARCSEC_PER_RAD
from ..identity import mapping_digest
from .kernels import DetectorPSF, deterministic_blas
from .pupils import Pupil, PupilSpec
from .wavefront import WavefrontBasis, WavefrontCoefficients, validate_coefficients

if TYPE_CHECKING:
    from .knowledge_error import WavefrontDraw, WavefrontDrawSpec

__all__ = ["FocalField", "OpticalPSF", "OpticalSpec", "check_sampling"]


@dataclass(frozen=True)
class OpticalSpec:
    """An optical PSF truth: pupil, focal length, wavelength, detector sampling and wavefront.

    ``kernel_shape`` is ``(ny, nx)``, both odd. The wavefront is the listed coefficients or,
    when ``draw`` is set (with no listed coefficients), a draw from a mode-weight prior at an
    exact aperture RMS.
    """

    pupil: PupilSpec
    focal_length_m: float
    wavelength_m: float
    detector_oversampling: int
    kernel_shape: tuple[int, int]
    wavefront: WavefrontCoefficients
    draw: WavefrontDrawSpec | None


def check_sampling(spec: OpticalSpec, *, pixel_scale_arcsec: float, wavelength_m: float) -> None:
    """Refuse an aliased or under-resolved kernel at ``wavelength_m``.

    Aliasing: a pupil of ``pixels`` samples over ``diameter_m`` makes the focal plane periodic
    with period ``pixels * wavelength / diameter_m``, which must exceed the kernel extent.
    Sub-pixel Nyquist: the oversampled pixel ``pixel_scale_arcsec / detector_oversampling``
    must not exceed ``wavelength / (2 diameter_m)``, the Nyquist step of the intensity.
    """
    pupil = spec.pupil
    wavelength_nm = wavelength_m * 1e9
    period = pupil.pixels * wavelength_m / pupil.diameter_m * ARCSEC_PER_RAD
    extent = max(spec.kernel_shape) * pixel_scale_arcsec
    if not period > extent:
        raise ValueError(
            f"aliased kernel at {wavelength_nm:.6g} nm: the focal-plane period {period:.6g} arcsec of a "
            f"{pupil.pixels}-pixel pupil does not exceed the kernel extent {extent:.6g} arcsec; raise "
            "psf.truth.pupil.pixels or reduce psf.truth.kernel_shape")
    step = pixel_scale_arcsec / spec.detector_oversampling
    nyquist = wavelength_m / (2 * pupil.diameter_m) * ARCSEC_PER_RAD
    if not step <= nyquist:
        needed = math.ceil(pixel_scale_arcsec / nyquist)
        raise ValueError(
            f"under-resolved kernel at {wavelength_nm:.6g} nm: the oversampled pixel {step:.6g} arcsec exceeds "
            f"the Nyquist step {nyquist:.6g} arcsec; set psf.truth.detector_oversampling to at least {needed}")


@dataclass(frozen=True)
class FocalField:
    """Focal-plane intensity as a fraction of the pupil power per sample, about the optical axis."""

    intensity: np.ndarray
    x_arcsec: np.ndarray
    y_arcsec: np.ndarray
    sample_arcsec: float
    wavelength_m: float


def _is_odd_size(value: Any) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool) and value >= 1 and value % 2 == 1


def _positive(value: Any, what: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not (
            math.isfinite(float(value)) and float(value) > 0.0):
        raise ValueError(f"{what} must be a finite positive number, got {value!r}")
    return float(value)


class OpticalPSF:
    """Detector kernels of one optical system at one pixel scale (a PSF provider).

    Built by ``optics.providers.build_psf_provider``; ``with_coefficients`` derives a
    provider of the same optics with another wavefront, sharing the pupil and the basis.
    Construction evaluates no kernel. ``draw`` records the prior draw that set the
    coefficients of a drawn truth.
    """

    def __init__(self, spec: OpticalSpec, *, pupil: Pupil, basis: WavefrontBasis, pixel_scale_arcsec: float,
                 coefficients: WavefrontCoefficients, draw: WavefrontDraw | None = None) -> None:
        shape = tuple(spec.kernel_shape)
        if len(shape) != 2 or not all(_is_odd_size(n) for n in shape):
            raise ValueError(f"kernel_shape must be two odd positive integers (ny, nx), got {spec.kernel_shape!r}")
        if not (isinstance(spec.detector_oversampling, Integral) and not isinstance(spec.detector_oversampling, bool)
                and spec.detector_oversampling >= 1):
            raise ValueError(f"detector_oversampling must be an integer >= 1, got {spec.detector_oversampling!r}")
        if pupil.spec != spec.pupil:
            raise ValueError("the pupil was built from another pupil specification")
        if basis.pupil is not pupil:
            raise ValueError("the wavefront basis belongs to another pupil")
        self._spec = spec
        self._pupil = pupil
        self._basis = basis
        self._pixel_scale_arcsec = _positive(pixel_scale_arcsec, "pixel_scale_arcsec")
        self._focal_length_m = _positive(spec.focal_length_m, "focal_length_m")
        self._wavelengths_m = (_positive(spec.wavelength_m, "wavelength_m"),)
        if basis.reference_wavelength_m != min(self._wavelengths_m):
            raise ValueError(f"the basis reference wavelength {basis.reference_wavelength_m!r} m is not the "
                             f"shortest provider wavelength {min(self._wavelengths_m)!r} m")
        for wavelength in self._wavelengths_m:
            check_sampling(spec, pixel_scale_arcsec=self._pixel_scale_arcsec, wavelength_m=wavelength)
        validate_coefficients(coefficients, pupil.spec, "coefficients")
        self._coefficients = coefficients
        self._draw = draw

    @property
    def spec(self) -> OpticalSpec:
        return self._spec

    @property
    def pupil(self) -> Pupil:
        return self._pupil

    @property
    def basis(self) -> WavefrontBasis:
        return self._basis

    @property
    def coefficients(self) -> WavefrontCoefficients:
        return self._coefficients

    @property
    def draw(self) -> WavefrontDraw | None:
        return self._draw

    @property
    def pixel_scale_arcsec(self) -> float:
        return self._pixel_scale_arcsec

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self._spec.kernel_shape[0]), int(self._spec.kernel_shape[1]))

    @property
    def wavelengths_m(self) -> tuple[float, ...]:
        return self._wavelengths_m

    @property
    def reference_wavelength_m(self) -> float:
        return min(self._wavelengths_m)

    @property
    def collecting_area_m2(self) -> float:
        return self._pupil.collecting_area_m2

    def _wavelength(self, wavelength_m: float | None) -> float:
        if wavelength_m is None:
            if len(self._wavelengths_m) != 1:
                raise ValueError(f"the provider propagates {len(self._wavelengths_m)} wavelengths; name one")
            return self._wavelengths_m[0]
        wavelength = _positive(wavelength_m, "wavelength_m")
        check_sampling(self._spec, pixel_scale_arcsec=self._pixel_scale_arcsec, wavelength_m=wavelength)
        return wavelength

    def _coefficients_or_own(self, coefficients: WavefrontCoefficients | None) -> WavefrontCoefficients:
        if coefficients is None:
            return self._coefficients
        validate_coefficients(coefficients, self._pupil.spec, "coefficients")
        return coefficients

    @functools.cached_property
    def _detector_grid(self) -> Any:
        import hcipy

        ny, nx = self.shape
        pixel_scale_rad = self._pixel_scale_arcsec * np.pi / (180 * 3600)
        pixel_m = self._focal_length_m * pixel_scale_rad
        return hcipy.make_uniform_grid(dims=[nx, ny], extent=np.array([nx, ny]) * pixel_m)

    @functools.cached_property
    def _propagator(self) -> Any:
        import hcipy

        supersampled = hcipy.make_supersampled_grid(self._detector_grid, self._spec.detector_oversampling)
        return hcipy.FraunhoferPropagator(self._pupil.grid, supersampled, self._focal_length_m)

    def _binned_power(self, wavelength_m: float, coefficients: WavefrontCoefficients) -> Any:
        """Detector-pixel power (flat HCIPy field, pupil-power units): the one propagation of a kernel."""
        import hcipy

        with deterministic_blas():
            wavefront = self._basis.apply(hcipy.Wavefront(self._pupil.transmission, wavelength_m), coefficients)
            return hcipy.subsample_field(self._propagator(wavefront).power,
                                         subsampling=self._spec.detector_oversampling,
                                         new_grid=self._detector_grid, statistic="sum")

    @functools.cached_property
    def _provider_digest(self) -> str:
        return mapping_digest(self.to_mapping())

    def kernel(self, wavelength_m: float | None = None, *,
               coefficients: WavefrontCoefficients | None = None) -> DetectorPSF:
        """The unit-sum detector kernel at ``wavelength_m`` (default: the only provider wavelength).

        ``coefficients`` replaces the provider's wavefront for this evaluation. The source
        records the wavelength, the provider and coefficients digests and the captured power
        fraction of this propagation.
        """
        wavelength = self._wavelength(wavelength_m)
        wavefront = self._coefficients_or_own(coefficients)
        power = self._binned_power(wavelength, wavefront)
        total = float(np.sum(power))
        if not (np.isfinite(total) and total > 0.0):
            raise ValueError(f"the propagated detector power is not positive and finite: {total!r}")
        normalized = power / total
        values = np.asarray(normalized.shaped, dtype=float)
        if values.shape != self.shape:
            raise RuntimeError(f"the detector grid gave shape {values.shape}, not kernel_shape {self.shape}")
        return DetectorPSF.from_array(values, self._pixel_scale_arcsec, normalize=True, source={
            "kind": "optical", "wavelength_m": wavelength, "provider_digest": self._provider_digest,
            "coefficients_digest": wavefront.digest(), "captured_power_fraction": total / self._pupil.total_power})

    def detector_power(self, wavelength_m: float | None = None, *,
                       coefficients: WavefrontCoefficients | None = None) -> np.ndarray:
        """Unnormalized power per detector pixel, in units of the pupil power, shape ``(ny, nx)``."""
        power = self._binned_power(self._wavelength(wavelength_m), self._coefficients_or_own(coefficients))
        return np.array(power.shaped, dtype=float)

    def pupil_wavefront(self, wavelength_m: float | None = None, *,
                        coefficients: WavefrontCoefficients | None = None) -> Any:
        """The aberrated pupil-plane HCIPy wavefront."""
        import hcipy

        wavelength = self._wavelength(wavelength_m)
        with deterministic_blas():
            return self._basis.apply(hcipy.Wavefront(self._pupil.transmission, wavelength),
                                     self._coefficients_or_own(coefficients))

    def focal_field(self, wavelength_m: float | None = None, *, coefficients: WavefrontCoefficients | None = None,
                    samples_per_lambda_over_d: int, radius_lambda_over_d: float) -> FocalField:
        """Point-sampled focal-plane intensity on a grid in units of ``wavelength / diameter_m``."""
        import hcipy

        wavelength = self._wavelength(wavelength_m)
        wavefront = self.pupil_wavefront(wavelength, coefficients=coefficients)
        grid = hcipy.make_focal_grid(q=samples_per_lambda_over_d, num_airy=radius_lambda_over_d,
                                     pupil_diameter=self._pupil.spec.diameter_m,
                                     focal_length=self._focal_length_m, reference_wavelength=wavelength)
        with deterministic_blas():
            power = hcipy.FraunhoferPropagator(self._pupil.grid, grid, self._focal_length_m)(wavefront).power
        to_arcsec = ARCSEC_PER_RAD / self._focal_length_m
        intensity = np.array(power.shaped, dtype=float) / self._pupil.total_power
        x = np.array(grid.x.reshape(grid.shape), dtype=float) * to_arcsec
        y = np.array(grid.y.reshape(grid.shape), dtype=float) * to_arcsec
        for array in (intensity, x, y):
            array.flags.writeable = False
        return FocalField(intensity, x, y, float(grid.delta[0]) * to_arcsec, wavelength)

    def with_coefficients(self, coefficients: WavefrontCoefficients) -> OpticalPSF:
        """The same optics with another wavefront; shares the pupil and the basis."""
        return OpticalPSF(self._spec, pupil=self._pupil, basis=self._basis,
                          pixel_scale_arcsec=self._pixel_scale_arcsec, coefficients=coefficients)

    def to_mapping(self) -> dict[str, Any]:
        """The provider record: optics, sampling, wavefront and the draw that set it.

        The pupil enters through its specification and the Zernike disc; the sampled pupil's
        own record (``Pupil.to_mapping``: area, power, active and dark segments) follows from
        the specification and stays out of the provider identity.
        """
        pupil = self._pupil.spec
        return {
            "provider": "optical",
            "pupil": {"kind": pupil.kind, **dataclasses.asdict(pupil)},
            "zernike_diameter_m": self._pupil.zernike_diameter_m,
            "focal_length_m": self._focal_length_m,
            "wavelengths_m": list(self._wavelengths_m),
            "reference_wavelength_m": self.reference_wavelength_m,
            "detector_oversampling": self._spec.detector_oversampling,
            "kernel_shape": list(self.shape),
            "pixel_scale_arcsec": self._pixel_scale_arcsec,
            "coefficients": self._coefficients.to_mapping(),
            "coefficients_digest": self._coefficients.digest(),
            "draw": None if self._draw is None else self._draw.to_mapping(),
        }


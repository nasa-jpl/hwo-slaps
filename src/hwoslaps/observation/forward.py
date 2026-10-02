"""Shared expected-image rendering for simulations and sensitivity forecasts."""

from dataclasses import dataclass

import numpy as np

from .noise_models import DetectorMoments, detector_mean_adu, detector_moments


@dataclass(frozen=True)
class ObservationPrediction:
    """Deterministic detector expectation, with no random draws.

    ``source_eps`` retains convolution roundoff; ``moments`` and the noise map
    use nonnegative Poisson rates. Images and uncertainty are in ADU.
    """

    source_eps: np.ndarray
    mean_adu: np.ndarray
    noise_map_adu: np.ndarray
    moments: DetectorMoments


def _lensing_image(lensing_data):
    import autolens as al

    mask = al.Mask2D.all_false(
        shape_native=lensing_data.image.shape,
        pixel_scales=lensing_data.pixel_scale,
    )
    return al.Array2D(values=lensing_data.image, mask=mask)


def _convolved_rate(image, kernel, throughput, pixel_scale):
    from ..psf.utils import (
        make_pyauto_convolver,
        pyauto_kernel_pixel_scales,
        validate_detector_kernel,
    )

    validate_detector_kernel(kernel)
    if not np.allclose(pyauto_kernel_pixel_scales(kernel), pixel_scale, rtol=0.0, atol=1e-12):
        raise ValueError("Pixel scale mismatch between PSF kernel and lensing image")
    if isinstance(throughput, bool) or not isinstance(throughput, (int, float, np.number)) or not np.isfinite(throughput) or throughput <= 0:
        raise ValueError("throughput must be positive and finite")
    convolved = make_pyauto_convolver(kernel).convolved_image_from(
        image=image, blurring_image=None,
    )
    rate = np.asarray(convolved.native) * float(throughput)
    if not np.all(np.isfinite(rate)):
        raise ValueError("PSF-convolved source image must be finite")
    tolerance = 1.0e-10 * float(np.max(np.abs(rate), initial=0.0))
    if float(np.min(rate, initial=0.0)) < -tolerance:
        raise ValueError("PSF-convolved source image has negative values beyond FFT roundoff scale")
    return rate


def convolve_source_rate(lensing_data, kernel, *, throughput=1.0):
    """Convolve one scene at matching angular sampling, returning e-/s.

    The detector-integrated kernel must have odd support and unit flux. The
    returned rate retains epsilon-scale FFT negatives for linear forecasting.
    """
    return _convolved_rate(_lensing_image(lensing_data), kernel, throughput, lensing_data.pixel_scale)


def mean_adu_images_from_lensing_arrays(lensing_data, observation_config, psf_kernels):
    """Render paired truth/model kernels from one scene without noise draws."""
    if not psf_kernels:
        raise ValueError("psf_kernels must contain at least one kernel")
    image = _lensing_image(lensing_data)
    return tuple(
        detector_mean_adu(
            _convolved_rate(image, kernel, observation_config['throughput'], lensing_data.pixel_scale),
            observation_config['exposure_time'], observation_config['detector'],
        )
        for kernel in psf_kernels
    )


def predict_observation(lensing_data, psf_data, observation_config):
    """Predict a scene's source rate, detector mean and uncertainty.

    ``psf_data`` may be optical ``PSFData`` or an external ``DetectorPSF``.
    No seed, random generator, sampled image or output directory is required.
    """
    source_eps = convolve_source_rate(
        lensing_data, psf_data.kernel, throughput=observation_config['throughput'],
    )
    exposure = observation_config['exposure_time']
    detector = observation_config['detector']
    moments = detector_moments(np.maximum(source_eps, 0.0), exposure, detector)
    return ObservationPrediction(
        source_eps=source_eps,
        mean_adu=detector_mean_adu(source_eps, exposure, detector),
        noise_map_adu=np.sqrt(moments.variance_e2) / moments.gain,
        moments=moments,
    )

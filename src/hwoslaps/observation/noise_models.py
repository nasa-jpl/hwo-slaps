"""Detector noise modeling functions.

This module implements realistic detector noise models including
Poisson noise, read noise, dark current, and sky background.
"""

from numbers import Real
from typing import Dict, Optional, Tuple

import numpy as np


def _validate_noise_inputs(
    source_eps: np.ndarray,
    exposure_time: float,
    detector_config: Dict[str, float],
) -> Tuple[np.ndarray, float, Dict[str, float]]:
    """Validate common detector-noise inputs.

    Parameters
    ----------
    source_eps : array-like
        Source flux in electrons per second.
    exposure_time : `float`
        Exposure time in seconds.
    detector_config : `dict`
        Detector configuration with ``gain``, ``read_noise``,
        ``dark_current``, and ``sky_background`` entries.

    Returns
    -------
    source_array : `numpy.ndarray`
        Validated source flux array.
    exposure : `float`
        Validated exposure time.
    detector : `dict`
        Validated detector values converted to floats.

    Raises
    ------
    ValueError
        Raised when any input is nonfinite or outside its physical domain.
    """
    source_array = np.asarray(source_eps, dtype=float)
    if not np.all(np.isfinite(source_array)):
        raise ValueError("source_eps must be finite")
    if np.any(source_array < 0.0):
        raise ValueError("source_eps must be non-negative")

    exposure = _validate_scalar(
        exposure_time,
        "exposure_time",
        positive=True,
    )

    if not isinstance(detector_config, dict):
        raise ValueError("detector_config must be a dictionary")

    detector = {
        "gain": _validate_scalar(
            detector_config.get("gain"),
            "detector_config.gain",
            positive=True,
        ),
        "read_noise": _validate_scalar(
            detector_config.get("read_noise"),
            "detector_config.read_noise",
        ),
        "dark_current": _validate_scalar(
            detector_config.get("dark_current"),
            "detector_config.dark_current",
        ),
        "sky_background": _validate_scalar(
            detector_config.get("sky_background"),
            "detector_config.sky_background",
        ),
    }
    return source_array, exposure, detector


def _validate_scalar(value: object, key_path: str, positive: bool = False) -> float:
    """Validate a finite scalar detector parameter."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{key_path} must be numeric")
    value_float = float(value)
    if not np.isfinite(value_float):
        raise ValueError(f"{key_path} must be finite")
    if positive:
        if value_float <= 0.0:
            raise ValueError(f"{key_path} must be positive")
    elif value_float < 0.0:
        raise ValueError(f"{key_path} must be non-negative")
    return value_float


def apply_detector_noise(
    source_eps: np.ndarray,
    exposure_time: float,
    detector_config: Dict[str, float],
    seed: Optional[int] = None
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Apply realistic detector noise to a source image.

    This function implements a complete detector noise model including:
    - Poisson noise (photon shot noise) on source + sky + dark
    - Read noise (Gaussian)
    - Dark current
    - Sky background

    Parameters
    ----------
    source_eps : `np.ndarray`
        Source flux in e-/s (PSF-convolved, noiseless).
    exposure_time : `float`
        Exposure time in seconds.
    detector_config : `dict`
        Detector configuration with keys: gain, read_noise,
        dark_current, sky_background.
    seed : `int`, optional
        Random seed for reproducibility.

    Returns
    -------
    final_image_adu : `np.ndarray`
        Final image in ADU including all noise.
    components : `dict`
        Dictionary containing individual components:
        - 'source_e': source electrons
        - 'sky_e': sky electrons per pixel
        - 'dark_e': dark electrons per pixel
        - 'detected_e': after Poisson noise
        - 'final_e': after read noise

    Notes
    -----
    The noise model follows standard CCD detector physics:
    1. Convert all components to electrons
    2. Apply Poisson statistics to total expected counts
    3. Add Gaussian read noise
    4. Convert to ADU using gain
    """
    source_eps, exposure_time, detector_config = _validate_noise_inputs(
        source_eps,
        exposure_time,
        detector_config,
    )

    # Use a local random number generator to avoid global RNG side effects
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    # Extract detector parameters
    gain = detector_config['gain']
    read_noise = detector_config['read_noise']
    dark_current = detector_config['dark_current']
    sky_background = detector_config['sky_background']

    # Convert to total electrons for each component
    source_e = source_eps * exposure_time
    dark_e = dark_current * exposure_time  # Per pixel
    sky_e = sky_background * exposure_time  # Per pixel

    # Total expected electrons per pixel
    expected_e = source_e + dark_e + sky_e

    # Apply Poisson noise to the total expected counts
    detected_e = rng.poisson(expected_e).astype(float)

    # Add read noise (Gaussian)
    final_e = detected_e + rng.normal(0.0, read_noise, size=detected_e.shape)

    # Convert to ADU
    final_image_adu = final_e / gain

    # Store components for analysis
    components = {
        'source_e': source_e,
        'sky_e': sky_e,
        'dark_e': dark_e,
        'detected_e': detected_e,
        'final_e': final_e,
        'expected_e': expected_e
    }

    return final_image_adu, components


def create_noise_map(
    source_eps: np.ndarray,
    exposure_time: float,
    detector_config: Dict[str, float]
) -> np.ndarray:
    """Create a proper noise map for the observation.

    The noise map represents the total uncertainty in each pixel,
    including contributions from Poisson noise and read noise.

    Parameters
    ----------
    source_eps : `np.ndarray`
        Source flux in e-/s (PSF-convolved, noiseless).
    exposure_time : `float`
        Exposure time in seconds.
    detector_config : `dict`
        Detector configuration parameters.

    Returns
    -------
    noise_map_adu : `np.ndarray`
        Noise map in ADU.

    Notes
    -----
    The total variance in electrons² is:
    variance = expected_counts + read_noise²

    Where expected_counts includes source, sky, and dark current.
    This follows from Poisson statistics where variance equals mean.
    """
    source_eps, exposure_time, detector_config = _validate_noise_inputs(
        source_eps,
        exposure_time,
        detector_config,
    )

    # Extract detector parameters
    gain = detector_config['gain']
    read_noise = detector_config['read_noise']
    dark_current = detector_config['dark_current']
    sky_background = detector_config['sky_background']

    # Convert to electrons
    source_e = source_eps * exposure_time
    dark_e = dark_current * exposure_time
    sky_e = sky_background * exposure_time

    # Total expected counts
    expected_e = source_e + dark_e + sky_e

    # Variance components (in electrons²):
    # - Poisson variance = expected counts
    # - Read noise variance = read_noise²
    total_variance_e2 = expected_e + read_noise**2

    # Convert to noise in ADU
    noise_map_adu = np.sqrt(total_variance_e2) / gain

    return noise_map_adu

"""Detector noise modeling functions.

This module implements realistic detector noise models including
Poisson noise, read noise, dark current, and sky background.
"""

import numpy as np
from typing import Dict, Tuple, Optional


def get_default_detector_config() -> Dict[str, float]:
    """Deprecated. Configuration must explicitly specify detector parameters."""
    raise RuntimeError("get_default_detector_config is deprecated. Provide detector config explicitly.")


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


def calculate_snr(
    signal_adu: np.ndarray,
    noise_map_adu: np.ndarray
) -> np.ndarray:
    """Calculate signal-to-noise ratio.
    
    Parameters
    ----------
    signal_adu : `np.ndarray`
        Signal array in ADU.
    noise_map_adu : `np.ndarray`
        Noise map in ADU.
        
    Returns
    -------
    snr : `np.ndarray`
        Signal-to-noise ratio array.
    """
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        snr = np.where(noise_map_adu > 0, signal_adu / noise_map_adu, 0)
    
    return snr


def estimate_limiting_magnitude(
    detector_config: Dict[str, float],
    exposure_time: float,
    snr_threshold: float = 5.0,
    pixel_scale: float = 0.05,
    aperture_radius_arcsec: float = 0.5
) -> Dict[str, float]:
    """Estimate limiting magnitude for point source detection.
    
    Parameters
    ----------
    detector_config : `dict`
        Detector configuration parameters.
    exposure_time : `float`
        Exposure time in seconds.
    snr_threshold : `float`, optional
        Required SNR for detection (default: 5.0).
    pixel_scale : `float`, optional
        Pixel scale in arcsec/pixel (default: 0.05).
    aperture_radius_arcsec : `float`, optional
        Aperture radius in arcsec (default: 0.5).
        
    Returns
    -------
    results : `dict`
        Dictionary with limiting magnitude estimates.
    """
    # Calculate aperture area in pixels
    aperture_radius_pix = aperture_radius_arcsec / pixel_scale
    aperture_area_pix = np.pi * aperture_radius_pix**2
    
    # Background noise per pixel
    sky_e = detector_config['sky_background'] * exposure_time
    dark_e = detector_config['dark_current'] * exposure_time
    read_e = detector_config['read_noise']
    
    # Total background noise in aperture
    background_variance_per_pixel = sky_e + dark_e + read_e**2
    total_background_noise = np.sqrt(aperture_area_pix * background_variance_per_pixel)
    
    # Required source counts for detection
    # Using approximation: SNR ≈ S / sqrt(S + B)
    # Where S = source counts, B = background noise²
    required_source_e = snr_threshold * total_background_noise
    
    return {
        'aperture_radius_arcsec': aperture_radius_arcsec,
        'aperture_area_pix': aperture_area_pix,
        'background_noise_e': total_background_noise,
        'required_source_e': required_source_e,
        'required_source_eps': required_source_e / exposure_time
    }
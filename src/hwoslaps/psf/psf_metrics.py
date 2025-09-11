"""PSF quality metrics and measurements.

This module contains functions for measuring PSF quality including FWHM
and Strehl ratio calculations.
"""

import numpy as np
from scipy.optimize import curve_fit
from ..constants import ARCSEC_PER_RAD


def measure_fwhm(psf_field, pixel_scale=None):
    """Measure the Full Width at Half Maximum (FWHM) of a PSF via its radial profile.

    This method is robust for diffraction-limited PSFs with complex features
    (e.g., Airy rings, diffraction spikes) where a simple Gaussian fit to the
    entire image would fail. Uses radial binning and interpolation to find
    the half-maximum crossing of the PSF core.

    Parameters
    ----------
    psf_field : `hcipy.Field`
        The PSF field from HCIPy.
    pixel_scale : `float`, optional
        Pixel scale in arcseconds/pixel. If None, returns FWHM in pixels.

    Returns
    -------
    fwhm : `float`
        FWHM (in arcsec if pixel_scale provided, else pixels).

    Raises
    ------
    ValueError
        Raised if no half-maximum crossing is found within the search radius,
        if the crossing occurs at the first radial bin, or if interpolation
        fails due to numerical issues.

    Notes
    -----
    The algorithm works by:
    1. Finding the precise sub-pixel center via 2D Gaussian fit
    2. Computing radial distances from center
    3. Creating radial profile using fine binning
    4. Interpolating to find exact half-maximum crossing

    Examples
    --------
    >>> fwhm_pixels = measure_fwhm(psf)
    >>> fwhm_arcsec = measure_fwhm(psf, pixel_scale=0.2)
    """
    data = psf_field.intensity.shaped
    
    # --- 1. Find the precise sub-pixel center of the PSF core ---
    # We fit a 2D Gaussian to a small 7x7 cutout around the peak for a robust center.
    peak_y, peak_x = np.unravel_index(np.argmax(data), data.shape)
    
    # Ensure the cutout doesn't go out of bounds
    box_radius = 3
    x_min = max(0, peak_x - box_radius)
    x_max = min(data.shape[1], peak_x + box_radius + 1)
    y_min = max(0, peak_y - box_radius)
    y_max = min(data.shape[0], peak_y + box_radius + 1)
    
    cutout = data[y_min:y_max, x_min:x_max]
    y_cut, x_cut = np.mgrid[:cutout.shape[0], :cutout.shape[1]]

    def gaussian_2d(coords, amplitude, x0, y0, sigma_x, sigma_y, background):
        x, y = coords
        return background + amplitude * np.exp(
            -((x - x0)**2 / (2 * sigma_x**2) + (y - y0)**2 / (2 * sigma_y**2))
        )
    
    try:
        popt, _ = curve_fit(
            gaussian_2d,
            (x_cut.ravel(), y_cut.ravel()),
            cutout.ravel(),
            p0=[cutout.max(), cutout.shape[1]/2, cutout.shape[0]/2, 1, 1, 0]
        )
        center_x = popt[1] + x_min
        center_y = popt[2] + y_min
    except (RuntimeError, ValueError):
        # If fit fails, fall back to the peak pixel
        center_x, center_y = float(peak_x), float(peak_y)

    # --- 2. Calculate radial distances from center ---
    y, x = np.indices(data.shape)
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # --- 3. Create radial profile using binning ---
    # Use finer binning for better precision, especially for sharp PSFs
    max_radius = min(min(data.shape) / 2.0, 50.0)  # Limit search to 50 pixels max
    bin_size = 0.1  # Finer bins for better precision
    n_bins = int(max_radius / bin_size)
    r_bins = np.linspace(0, max_radius, n_bins + 1)
    r_bin_centers = (r_bins[:-1] + r_bins[1:]) / 2
    
    # Bin the intensity data by radius
    intensity_binned = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (r >= r_bins[i]) & (r < r_bins[i + 1])
        if np.sum(mask) > 0:
            intensity_binned[i] = np.mean(data[mask])
        else:
            intensity_binned[i] = 0
    
    # Use a more lenient threshold for valid bins to avoid throwing away low-intensity regions
    peak_intensity = np.max(data)  # Use actual peak from original data
    intensity_threshold = peak_intensity * 1e-6  # Very small threshold
    valid_bins = intensity_binned > intensity_threshold
    
    if not np.any(valid_bins):
        raise ValueError("No valid intensity data found in radial bins.")
    
    r_valid = r_bin_centers[valid_bins]
    intensity_valid = intensity_binned[valid_bins]
    
    # --- 4. Find half-maximum crossing using interpolation ---
    half_max_intensity = peak_intensity / 2.0
    
    # Find the first crossing where intensity drops below half-maximum
    crossing_indices = np.where(intensity_valid < half_max_intensity)[0]
    
    if len(crossing_indices) == 0:
        raise ValueError(
            f"No half-maximum crossing found within search radius of {max_radius:.2f} pixels. "
            f"Peak intensity: {peak_intensity:.6e}, Half-max: {half_max_intensity:.6e}. "
            f"This suggests either the PSF is broader than the search region or there's a "
            f"fundamental issue with the PSF data or analysis parameters."
        )
    
    crossing_idx = crossing_indices[0]
    
    if crossing_idx == 0:
        raise ValueError(
            f"Half-maximum crossing occurs at the first radial bin (r={r_valid[0]:.3f} pixels). "
            f"This suggests the PSF is extremely sharp or there's an issue with centering or "
            f"binning parameters. Consider using finer binning or checking PSF centering."
        )
    
    # Linear interpolation between the two bins surrounding the crossing
    r1, I1 = r_valid[crossing_idx - 1], intensity_valid[crossing_idx - 1]
    r2, I2 = r_valid[crossing_idx], intensity_valid[crossing_idx]
    
    # Validate interpolation inputs
    if I1 == I2:
        raise ValueError(
            f"Cannot interpolate half-maximum crossing: intensities at r={r1:.3f} and "
            f"r={r2:.3f} are identical ({I1:.6e}). This suggests numerical precision "
            f"issues or insufficient resolution in the radial profile."
        )
    
    # Interpolate to find exact crossing point
    r_half = r1 + (r2 - r1) * (half_max_intensity - I1) / (I2 - I1)
    
    fwhm_pixels = 2.0 * r_half

    # --- 5. Convert to physical units if pixel_scale is provided ---
    if pixel_scale is not None:
        fwhm = fwhm_pixels * pixel_scale
    else:
        fwhm = fwhm_pixels

    return fwhm


def calculate_strehl_ratio(aberrated_psf, perfect_psf):
    """Calculate the Strehl ratio of an aberrated PSF.

    The Strehl ratio is defined as the ratio of peak intensities between
    an aberrated PSF and its corresponding perfect (diffraction-limited) PSF.
    It is a standard metric for optical system performance.

    Parameters
    ----------
    aberrated_psf : `hcipy.Field`
        The aberrated PSF to analyze.
    perfect_psf : `hcipy.Field`
        The perfect (diffraction-limited) PSF for comparison.

    Returns
    -------
    strehl_ratio : `float`
        The Strehl ratio, ranging from 0 to 1, where 1 indicates a perfect PSF.

    Notes
    -----
    The calculation assumes both PSFs are normalized to the same total flux
    and are properly centered. The returned value is capped at 1.0 as
    physically the Strehl ratio cannot exceed unity.
    """
    # Get peak intensities
    aberrated_peak = np.max(aberrated_psf.intensity)
    perfect_peak = np.max(perfect_psf.intensity)

    # Strehl ratio is the ratio of peaks
    strehl_ratio = aberrated_peak / perfect_peak

    return min(strehl_ratio, 1.0)


def calculate_psf_pixel_scale(wavelength, pupil_diameter, sampling):
    """Calculate the pixel scale of a PSF in arcseconds per pixel.

    Computes the angular pixel scale for a PSF based on the optical system
    parameters and sampling rate. Uses the standard diffraction relation
    λ/D for the angular scale.

    Parameters
    ----------
    wavelength : `float`
        Wavelength in meters.
    pupil_diameter : `float`
        Pupil diameter in meters.
    sampling : `float`
        Sampling factor (pixels per λ/D).

    Returns
    -------
    pixel_scale : `float`
        Pixel scale in arcseconds per pixel.

    Notes
    -----
    The calculation uses the small angle approximation and converts
    to arcseconds using the factor 206264.8062471 (number of arcseconds in one radian).
    """
    return wavelength / pupil_diameter * ARCSEC_PER_RAD / sampling


def analyze_psf_quality(psf_field, perfect_psf=None, wavelength=None, pupil_diameter=None, sampling=None):
    """Perform comprehensive PSF quality analysis.

    Computes multiple quality metrics for a PSF including FWHM, Strehl ratio,
    peak intensity, and total flux. Can optionally convert measurements to
    physical units if optical system parameters are provided.

    Parameters
    ----------
    psf_field : `hcipy.Field`
        The PSF to analyze.
    perfect_psf : `hcipy.Field`, optional
        Perfect PSF for Strehl ratio calculation.
    wavelength : `float`, optional
        Wavelength in meters for pixel scale calculation.
    pupil_diameter : `float`, optional
        Pupil diameter in meters for pixel scale calculation.
    sampling : `float`, optional
        Sampling factor for pixel scale calculation.

    Returns
    -------
    results : `dict`
        Dictionary containing PSF quality metrics with keys:
        
        ``pixel_scale_arcsec``
            Pixel scale in arcseconds (if parameters provided).
        ``fwhm_arcsec``
            FWHM in arcseconds (if pixel scale available).
        ``fwhm_mas``
            FWHM in milliarcseconds (if pixel scale available).
        ``strehl_ratio``
            Strehl ratio (if perfect PSF provided).
        ``peak_intensity``
            Maximum intensity value.
        ``total_flux``
            Integrated intensity.
        ``peak_to_total_ratio``
            Ratio of peak to total intensity.

    Notes
    -----
    Error conditions in individual measurements (e.g., FWHM calculation)
    are captured in the results dictionary with '_error' suffix keys
    containing error messages.
    """
    results = {}
    
    # Calculate pixel scale if parameters provided
    pixel_scale = None
    if wavelength is not None and pupil_diameter is not None and sampling is not None:
        pixel_scale = calculate_psf_pixel_scale(wavelength, pupil_diameter, sampling)
        results['pixel_scale_arcsec'] = pixel_scale
    
    # Measure FWHM
    try:
        fwhm = measure_fwhm(psf_field, pixel_scale)
        results['fwhm_arcsec'] = fwhm
        if pixel_scale is not None:
            results['fwhm_mas'] = fwhm * 1000
    except Exception as e:
        results['fwhm_error'] = str(e)
    
    # Calculate Strehl ratio if perfect PSF provided
    if perfect_psf is not None:
        try:
            strehl = calculate_strehl_ratio(psf_field, perfect_psf)
            results['strehl_ratio'] = strehl
        except Exception as e:
            results['strehl_error'] = str(e)
    
    # Basic PSF statistics
    intensity = psf_field.intensity
    results['peak_intensity'] = np.max(intensity)
    results['total_flux'] = np.sum(intensity)
    results['peak_to_total_ratio'] = results['peak_intensity'] / results['total_flux']
    
    return results


def print_psf_quality_summary(psf_data, perfect_psf=None):
    """Print a formatted summary of PSF quality metrics.

    Analyzes the PSF and prints a comprehensive report including configuration
    parameters and all computed quality metrics. Optionally includes Strehl
    ratio if a perfect PSF is provided.

    Parameters
    ----------
    psf_data : `PSFData`
        PSF data object containing PSF and telescope parameters.
    perfect_psf : `hcipy.Field`, optional
        Perfect PSF for comparison.

    Notes
    -----
    The summary includes:
    - Wavelength, pupil diameter, and sampling parameters
    - Pixel scale in arcseconds
    - FWHM in both arcseconds and milliarcseconds
    - Strehl ratio (if perfect PSF provided)
    - Peak intensity and total flux
    - Applied aberrations (if any)
    """
    config = psf_data.config
    telescope_config = config['psf']['telescope']
    sim_config = config['psf']['hres_psf']
    
    # Analyze PSF quality
    quality = analyze_psf_quality(
        psf_data.psf,
        perfect_psf=perfect_psf,
        wavelength=sim_config['wavelength'],
        pupil_diameter=telescope_config['pupil_diameter'],
        sampling=sim_config['sampling']
    )
    
    print("=== PSF Quality Summary ===")
    print(f"Wavelength: {sim_config['wavelength']*1e9:.0f} nm")
    print(f"Pupil diameter: {telescope_config['pupil_diameter']:.1f} m")
    print(f"Sampling: {sim_config['sampling']} pixels/lambda/D")
    
    if 'pixel_scale_arcsec' in quality:
        print(f"Pixel scale: {quality['pixel_scale_arcsec']:.6f} arcsec/pixel")
    
    if 'fwhm_arcsec' in quality:
        print(f"FWHM: {quality['fwhm_arcsec']:.6f} arcsec")
        if 'fwhm_mas' in quality:
            print(f"FWHM: {quality['fwhm_mas']:.1f} mas")
    
    if 'strehl_ratio' in quality:
        print(f"Strehl ratio: {quality['strehl_ratio']:.3f}")
    
    print(f"Peak intensity: {quality['peak_intensity']:.6e}")
    print(f"Total flux: {quality['total_flux']:.6e}")
    print(f"Peak/Total ratio: {quality['peak_to_total_ratio']:.6f}")
    
    # Print aberration summary if present
    if psf_data.aberrations:
        print("\n=== Applied Aberrations ===")
        for aberration_type, values in psf_data.aberrations.items():
            if values:
                print(f"{aberration_type}: {values}")
    
    return quality
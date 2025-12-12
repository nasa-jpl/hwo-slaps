"""PSF generation functions for HWO-SLAPS.

This module contains the main PSF generation logic implementing a clean API
for generating aberrated PSFs.

The module implements a diverging-path architecture where high-resolution PSFs
are computed for optical quality metrics while detector-sampled kernels are
generated for science applications.
"""

import numpy as np
import os
import hcipy
from hcipy.optics import Wavefront
from hcipy.field import (
    make_focal_grid,
    make_uniform_grid,
    make_supersampled_grid,
    subsample_field,
)
from hcipy.propagation import FraunhoferPropagator
import autolens as al
from .telescope_models import create_hcipy_telescope
from ..constants import ARCSEC_PER_RAD
from .aberration_models import (
    apply_segment_pistons,
    apply_segment_tiptilts,
    apply_segment_zernikes,
    apply_global_zernikes
)
from .utils import PSFData


def generate_psf_system(config, full_config=None):
    """Generate a PSF system with specified aberrations.

    This is the main API function that creates a complete PSF system
    including telescope setup, aberration application, and PSF generation
    with comprehensive quality analysis and unified data structure.

    Parameters
    ----------
    config : `dict`
        Configuration dictionary containing telescope, simulation, and
        aberration parameters.
    full_config : `dict`, optional
        Full configuration dictionary containing run_name and other top-level
        parameters. If provided, this will be stored in PSFData.

    Returns
    -------
    psf_data : `PSFData`
        Complete PSF system data with unified structure providing direct
        access to all system parameters and quality metrics. Includes
        pre-converted PyAutoLens Kernel2D for immediate use in lensing
        simulations.

    Notes
    -----
    The returned PSFData object contains all information in a flat structure
    with direct property access, including pre-computed quality metrics like
    FWHM and aberration statistics.

    Examples
    --------
    Generate a PSF system and access key properties:
    
    >>> psf_data = generate_psf_system(config)
    >>> print(f"Wavelength: {psf_data.wavelength_nm} nm")
    >>> print(f"FWHM: {psf_data.fwhm_arcsec:.3f} arcsec")
    >>> print(f"Quality: {psf_data.quality_grade}")
    >>> if psf_data.has_aberrations:
    ...     print(f"Total RMS: {psf_data.total_rms_nm:.1f} nm")
    """
    # Automatic sampling calculation and validation.
    
    # Extract parameters from the config structure.
    # Config is the PSF config section, full_config contains everything.
    if full_config is not None and 'lensing' in full_config:
        lensing_config = full_config['lensing']
    else:
        raise ValueError('full_config must be provided and contain a "lensing" key.')
    psf_config = config
    # Use the hres_psf block from the new config structure
    if 'hres_psf' not in psf_config or 'telescope' not in psf_config:
        raise ValueError('psf_config must contain "hres_psf" and "telescope" keys.')
    sim_config = psf_config['hres_psf']
    telescope_config = psf_config['telescope']

    target_pixel_scale = lensing_config['grid']['pixel_scale']
    wavelength = sim_config['wavelength']
    pupil_diameter = telescope_config['pupil_diameter']
    focal_length = telescope_config['focal_length']
    requested_sampling = sim_config['sampling']

    # Calculate the nearest integer subsampling factor.
    res_element_arcsec = (wavelength / pupil_diameter) * ARCSEC_PER_RAD

    # Prevent division by zero if res_element_arcsec is somehow zero.
    if res_element_arcsec <= 0:
        raise ValueError("Calculated resolution element size is not positive. "
                        "Check wavelength and pupil diameter.")

    hres_pixel_scale_initial = res_element_arcsec / requested_sampling
    non_integer_factor = target_pixel_scale / hres_pixel_scale_initial

    N = int(round(non_integer_factor))

    # Validate the subsampling factor.
    if N < 1:
        raise ValueError(
            f"Calculated integer subsampling factor is {N}, which is not valid. "
            f"This implies the high-resolution PSF grid (sampling={requested_sampling}) "
            f"is coarser than the target lensing grid (pixel_scale={target_pixel_scale}). "
            f"Increase the 'hres_psf.sampling' value in the config."
        )

    # Calculate the new, ideal sampling value using the corrected formula.
    used_sampling = (N * res_element_arcsec) / target_pixel_scale

    # Update the config in-memory for this run.
    print(f"Auto-adjusting PSF sampling: requested={requested_sampling}, used={used_sampling:.4f} "
          f"to achieve integer subsampling of N={N}.")
    sim_config['sampling'] = used_sampling

    # Create telescope model.
    telescope_data = create_hcipy_telescope(config)
    
    # Extract aberration configurations (strict: all flags must be explicit)
    aberrations = psf_config['aberrations']
    use_segment_api = aberrations['use_api']
    
    # Apply toggle flags to aberrations (explicit True/False required by validation)
    segment_pistons = aberrations['segment_pistons'] if aberrations['enable_segment_pistons'] else None
    segment_tiptilts = aberrations['segment_tiptilts'] if aberrations['enable_segment_tiptilts'] else None
    segment_hexikes = aberrations['segment_hexikes'] if aberrations['enable_segment_hexikes'] else None
    global_zernikes = aberrations['global_zernikes'] if aberrations['enable_global_zernikes'] else None
    
    # Implement the common trunk approach.
    # Create the pupil_grid and the aberrated pupil-plane Wavefront as the
    # single source of truth.
    
    # Extract telescope components.
    aper = telescope_data['aper']
    hsm = telescope_data['hsm']
    segments = telescope_data['segments']
    wavelength = telescope_data['wavelength']
    num_segments = len(segments)
    
    # Dictionary to store phase screens.
    phase_screens = {}
    
    # Apply segment pistons and tip/tilts via segmented mirror.
    hsm.flatten()
    
    if segment_pistons is not None:
        from .aberration_models import apply_segment_pistons
        apply_segment_pistons(hsm, segment_pistons, wavelength, num_segments)
        
    if segment_tiptilts is not None:
        from .aberration_models import apply_segment_tiptilts
        apply_segment_tiptilts(hsm, segment_tiptilts, num_segments)
    
    # Create initial wavefront from aperture.
    wf_pupil = Wavefront(aper, wavelength)
    
    # Apply segmented mirror.
    wf_pupil = hsm(wf_pupil)
    
    # Apply segment-level Zernikes (hexikes) as phase screen.
    if segment_hexikes is not None:
        if use_segment_api:
            from .aberration_models import apply_segment_zernikes
            phase_screen, hexike_surface = apply_segment_zernikes(
                segment_hexikes, segments, telescope_data, wavelength, use_api=True
            )
            phase_screens['segment_hexikes_api'] = phase_screen
            # Apply hexike phase via the segmented hexike surface to avoid double-application hazards.
            wf_pupil = hexike_surface(wf_pupil)
        else:
            from .aberration_models import apply_segment_zernikes
            phase_screen = apply_segment_zernikes(
                segment_hexikes, segments, telescope_data, wavelength, use_api=False
            )
            phase_screens['segment_hexikes'] = phase_screen
            wf_pupil.electric_field *= np.exp(1j * np.array(phase_screen))
    
    # Apply global Zernikes as phase screen.
    if global_zernikes is not None:
        from .aberration_models import apply_global_zernikes
        phase_screen = apply_global_zernikes(global_zernikes, telescope_data, wavelength)
        phase_screens['global_zernikes'] = phase_screen
        wf_pupil.electric_field *= np.exp(1j * np.array(phase_screen))
    
    # Implement the single high-resolution propagation.
    # Define the high-resolution focal grid using parameters from
    # config['psf']['hres_psf'].
    
    # Create high-resolution focal grid in physical units (meters at focal plane).
    focal_grid_hres = make_focal_grid(
        q=sim_config['sampling'],
        num_airy=sim_config['num_airy'],
        pupil_diameter=pupil_diameter,
        focal_length=focal_length,
        reference_wavelength=wavelength,
    )
    
    # Create FraunhoferPropagator for high-resolution path with correct focal length.
    prop_hres = FraunhoferPropagator(telescope_data['pupil_grid'], focal_grid_hres, focal_length)
    
    # Propagate the pupil wavefront to get the single high-resolution PSF Wavefront.
    wf_psf_hres = prop_hres(wf_pupil)

    # Optionally save the high-resolution PSF intensity before any downsampling.
    saved_highres_psf_path = None
    if sim_config.get('save_highres_psf_npy', False):
        try:
            plotting_cfg = full_config.get('plotting', {}) if full_config else {}
            base_output_dir = plotting_cfg.get('output_dir', os.getcwd())
            run_name = (full_config.get('run_name') if full_config else None) or 'run'
            psf_out_dir = os.path.join(base_output_dir, run_name, 'psf')
            os.makedirs(psf_out_dir, exist_ok=True)
            saved_highres_psf_path = os.path.join(psf_out_dir, 'highres_psf.npy')
            np.save(saved_highres_psf_path, wf_psf_hres.power.shaped)
            print(f"Saved high-resolution PSF intensity to {saved_highres_psf_path}")
        except Exception as e:
            print(f"Warning: Failed to save high-resolution PSF .npy: {e}")
    
    # Implement branch A for metrics calculation.
    # Use wf_psf_hres to calculate all optical metrics (Strehl, FWHM, etc.).
    # This part operates on the explicitly high-resolution PSF.
    
    # Calculate high-resolution pixel scale for metrics.
    pixel_scale_arcsec = (
        telescope_data['wavelength'] / telescope_data['pupil_diameter'] * ARCSEC_PER_RAD / sim_config['sampling']
    )
    
    # Calculate PSF metrics using high-resolution PSF.
    from .psf_metrics import analyze_psf_quality
    quality_metrics = analyze_psf_quality(
        wf_psf_hres,
        wavelength=telescope_data['wavelength'],
        pupil_diameter=telescope_data['pupil_diameter'],
        sampling=sim_config['sampling']
    )
    
    # Implement branch B for kernel generation via detector.
    
    # Get target parameters and enforce odd kernel shape.
    autolens_pixel_scale = target_pixel_scale
    kernel_shape_native = psf_config['kernel']['shape_native'].copy()
    
    # Enforce odd dimensions.
    original_shape = kernel_shape_native.copy()
    for i in range(len(kernel_shape_native)):
        if kernel_shape_native[i] % 2 == 0:
            kernel_shape_native[i] += 1
    
    if kernel_shape_native != original_shape:
        print(f"Warning: Kernel shape changed from {original_shape} to {kernel_shape_native} "
              f"to ensure odd dimensions required for PyAutoLens convolution.")
    
    # Get subsampling factor.
    # Use the integer subsampling factor N calculated above.
    subsampling_factor = N
    
    # Define detector grid in focal-plane meters using small-angle approximation (x ≈ f * theta).
    autolens_pixel_scale_rad = autolens_pixel_scale * np.pi / (180 * 3600)
    pixel_scale_m = focal_length * autolens_pixel_scale_rad
    detector_grid_m = make_uniform_grid(
        dims=kernel_shape_native,
        extent=np.array(kernel_shape_native) * pixel_scale_m,
    )

    # Create a supersampled detector input grid for physical downsampling by an
    # integer factor N. Propagate directly to this supersampled detector grid.
    detector_input_grid = make_supersampled_grid(detector_grid_m, subsampling_factor)
    prop_det = FraunhoferPropagator(telescope_data['pupil_grid'], detector_input_grid, focal_length)
    wf_psf_supersampled = prop_det(wf_pupil)

    # Downsample the supersampled PSF power to the detector grid via summation to conserve flux.
    psf_downsampled = subsample_field(
        wf_psf_supersampled.power, subsampling=subsampling_factor, new_grid=detector_grid_m, statistic='sum'
    )
    
    # Normalize psf_downsampled to sum to 1.
    psf_downsampled_normalized = psf_downsampled / np.sum(psf_downsampled)
    
    # Create the final al.Kernel2D object.
    kernel = al.Kernel2D.no_mask(
        values=psf_downsampled_normalized.shaped,  # Use .shaped to get 2D array.
        pixel_scales=autolens_pixel_scale
    )
    
    # Verify pixel scale matching.
    if not np.allclose(kernel.pixel_scales, autolens_pixel_scale, rtol=1e-10):
        raise ValueError(
            f"Pixel scale mismatch: kernel pixel_scales={kernel.pixel_scales}, "
            f"expected autolens_pixel_scale={autolens_pixel_scale}. "
            f"This indicates a fundamental problem in the downsampling logic."
        )
    
    # Calculate proper total RMS wavefront error including all aberrations.
    # This combines the DM surface OPD (factor 2 on reflection) with OPD
    # implied by any applied phase screens (segment hexikes, global Zernikes).
    try:
        # Start with DM OPD.
        opd_total = telescope_data['hsm'].surface * 2

        # Convert phase screens (stored in radians) to OPD and add.
        # Relation: phase [rad] = 2π * OPD / λ  =>  OPD = phase * λ / (2π)
        if phase_screens:
            lambda_m = telescope_data['wavelength']
            rad_to_opd = lambda_m / (2 * np.pi)
            for screen in phase_screens.values():
                try:
                    opd_total += screen * rad_to_opd
                except Exception:
                    # Fallback if stored as ndarray instead of Field
                    opd_total += np.array(screen) * rad_to_opd

        # Compute RMS over illuminated pupil after removing piston.
        aper_field = telescope_data['aper']
        valid_pixels = aper_field > 0.5
        opd_valid = opd_total[valid_pixels]
        opd_valid -= np.mean(opd_valid)
        total_rms_nm = float(np.sqrt(np.mean(opd_valid**2)) * 1e9)
    except Exception as e:
        print(f"Warning: Could not calculate total RMS including phase screens: {e}")
        total_rms_nm = 0.0
    
    # Calculate individual aberration RMS values for statistics.
    segment_piston_rms_nm = 0.0
    segment_tiptilt_rms_urad = 0.0
    global_zernike_rms_nm = 0.0

    # Calculate segment flat-to-flat from point-to-point.
    segment_point_to_point = telescope_config['segment_point_to_point']
    segment_flat_to_flat = segment_point_to_point * np.sqrt(3) / 2
    
    if segment_pistons:
        segment_piston_rms_nm = np.std(list(segment_pistons.values()))
        
    if segment_tiptilts:
        # RMS magnitude of tip/tilt vector across segments (μrad).
        tiptilts_array = np.array(list(segment_tiptilts.values()))  # shape (N, 2)
        magsq = np.sum(tiptilts_array**2, axis=1)  # tip^2 + tilt^2 per segment
        segment_tiptilt_rms_urad = float(np.sqrt(np.mean(magsq)))
        
    if global_zernikes:
        # Calculate RMS of global Zernikes.
        global_zernike_rms_nm = np.sqrt(np.sum([coeff**2 for coeff in global_zernikes.values()]))
    
    # Update PSFData instantiation.
    # Populate the PSFData object with the results from both branches.
    
    # Create unified PSFData object.
    psf_data = PSFData(
        # Primary data from both branches.
        psf=wf_psf_hres,  # High-resolution PSF from Branch A.
        wavefront=wf_psf_hres,  # High-resolution wavefront from Branch A.
        telescope_data=telescope_data,
        kernel=kernel,  # Physically downsampled kernel from Branch B.
        
        # System parameters.
        wavelength_nm=sim_config['wavelength'] * 1e9,
        pupil_diameter_m=telescope_config['pupil_diameter'],
        focal_length_m=telescope_config['focal_length'],
        pixel_scale_arcsec=pixel_scale_arcsec,
        sampling_factor=sim_config['sampling'],
        requested_sampling_factor=requested_sampling,
        used_sampling_factor=used_sampling,
        integer_subsampling_factor=N,
        num_segments=len(telescope_data['segments']),
        
        # Telescope geometry.
        segment_flat_to_flat_m=segment_flat_to_flat,
        segment_point_to_point_m=telescope_config['segment_point_to_point'],
        gap_size_m=telescope_config['gap_size'],
        num_rings=telescope_config['num_rings'],
        
        # Quality metrics (no default backups; optional keys may be absent)
        fwhm_arcsec=quality_metrics['fwhm_arcsec'] if 'fwhm_arcsec' in quality_metrics else None,
        fwhm_mas=(quality_metrics['fwhm_arcsec'] * 1000) if 'fwhm_arcsec' in quality_metrics else None,
        strehl_ratio=quality_metrics['strehl_ratio'] if 'strehl_ratio' in quality_metrics else None,
        peak_intensity=quality_metrics['peak_intensity'],
        total_flux=quality_metrics['total_flux'],
        
        # Aberration summary.
        total_rms_nm=total_rms_nm,
        segment_piston_rms_nm=float(segment_piston_rms_nm),
        segment_tiptilt_rms_urad=segment_tiptilt_rms_urad,
        global_zernike_rms_nm=global_zernike_rms_nm,
        
        # Aberration flags.
        has_segment_pistons=segment_pistons is not None,
        has_segment_tiptilts=segment_tiptilts is not None,
        has_segment_hexikes=segment_hexikes is not None,
        has_global_zernikes=global_zernikes is not None,
        
        # Kernel metadata.
        kernel_pixel_scale=autolens_pixel_scale,  # Pixel scale of detector-generated kernel.
        highres_psf_npy_path=saved_highres_psf_path,
        
        # Complex data.
        phase_screens=phase_screens,
        aberrations=aberrations,
        config=full_config
    )
    
    return psf_data


def generate_aberrated_psf(
    telescope_data,
    segment_pistons=None,
    segment_tiptilts=None,
    segment_hexikes=None,
    zernike_coeffs=None,
    use_segment_api=False,
    return_all=False
):
    """Generate an aberrated PSF with specified aberrations.

    This function applies various aberrations to the telescope and generates
    the resulting PSF. Aberrations are applied in a specific order to ensure
    proper physical modeling.

    Parameters
    ----------
    telescope_data : `dict`
        Dictionary containing telescope components from create_hcipy_telescope.
        Required keys: 'aper', 'hsm', 'prop', 'segments', 'wavelength', 
        'segment_flat_to_flat'.
    segment_pistons : `dict`, optional
        Dictionary mapping segment indices to piston amplitudes in nanometers
        of wavefront OPD. These are converted to mirror-surface height by
        dividing by two before being sent to the segmented mirror actuators.
    segment_tiptilts : `dict`, optional
        Dictionary mapping segment indices to (tip, tilt) tuples in microradians.
    segment_hexikes : `dict`, optional
        Dictionary mapping segment indices to hexike coefficient dictionaries.
    zernike_coeffs : `dict`, optional
        Dictionary mapping Zernike indices to coefficient values in nm RMS.
    use_segment_api : `bool`, optional
        Whether to use HCIPy's new API for segment-level aberrations. Default False.
    return_all : `bool`, optional
        Whether to return wavefront and phase screens in addition to PSF. Default False.
        
    Returns
    -------
    psf : `hcipy.Field`
        The PSF field.
    wavefront : `hcipy.Wavefront`, optional
        The wavefront at the pupil plane (if return_all=True).
    phase_screens : `dict`, optional
        Dictionary of phase screens by type (if return_all=True).

    Notes
    -----
    Aberrations are applied in the following order:

    1. Segment pistons and tip/tilts (via segmented deformable mirror)
    2. Segment-level Zernikes (as phase screen)
    3. Global Zernikes (as phase screen)

    Examples
    --------
    Generate PSF with segment pistons:
    
    >>> segment_pistons = {i: np.random.randn() * 10 for i in range(37)}
    >>> psf = generate_aberrated_psf(telescope_data, segment_pistons=segment_pistons)
    
    Generate PSF with all aberration types:
    
    >>> psf, wf, screens = generate_aberrated_psf(
    ...     telescope_data,
    ...     segment_pistons=pistons,
    ...     segment_tiptilts=tiptilts,
    ...     segment_hexikes=hexikes,
    ...     zernike_coeffs=zernikes,
    ...     return_all=True
    ... )
    """
    # Extract telescope components.
    aper = telescope_data['aper']
    hsm = telescope_data['hsm']
    prop = telescope_data['prop']
    segments = telescope_data['segments']
    wavelength = telescope_data['wavelength']
    num_segments = len(segments)
    
    # Dictionary to store phase screens.
    phase_screens = {}
    
    # Apply segment pistons and tip/tilts via segmented mirror.
    hsm.flatten()
    
    if segment_pistons is not None:
        # Convert nm to radians of phase: phi = 2 * 2pi * OPD / lambda.
        # Factor of 2 for reflection.
        apply_segment_pistons(hsm, segment_pistons, wavelength, num_segments)
        
    if segment_tiptilts is not None:
        apply_segment_tiptilts(hsm, segment_tiptilts, num_segments)
    
    # Create initial wavefront.
    wf = Wavefront(aper, wavelength)
    
    # Apply segmented mirror.
    wf = hsm(wf)
    
    # Apply segment-level Zernikes (hexikes) as phase screen.
    if segment_hexikes is not None:
        if use_segment_api:
            from .aberration_models import apply_segment_zernikes
            result = apply_segment_zernikes(segment_hexikes, segments, telescope_data, wavelength, use_api=True)
            phase_screen, hexike_surface = result  # API version returns tuple.
            phase_screens['segment_hexikes_api'] = phase_screen
            wf = hexike_surface(wf)
        else:
            from .aberration_models import apply_segment_zernikes
            phase_screen = apply_segment_zernikes(segment_hexikes, segments, telescope_data, wavelength, use_api=False)
            phase_screens['segment_hexikes'] = phase_screen
            wf.electric_field *= np.exp(1j * np.array(phase_screen))
    
    # Apply global Zernikes as phase screen.
    if zernike_coeffs is not None:
        phase_screen = apply_global_zernikes(zernike_coeffs, telescope_data, wavelength)
        phase_screens['global_zernikes'] = phase_screen
        wf.electric_field *= np.exp(1j * np.array(phase_screen))
    
    # Propagate to focal plane.
    psf = prop(wf)
    
    if return_all:
        return psf, wf, phase_screens
    else:
        return psf


def generate_calibrated_segment_aberrations(target_rms_nm, telescope_data, tolerance=0.1, seed=None):
    """Generate segment aberrations calibrated to produce a specific RMS wavefront error.
    
    This function generates random segment pistons and tip/tilts, measures the
    actual RMS wavefront error they produce, and scales them to match the target.
    
    Parameters
    ----------
    target_rms_nm : `float`
        Target RMS wavefront error in nanometers.
    telescope_data : `dict`
        Dictionary containing telescope components.
    tolerance : `float`, optional
        Tolerance for RMS matching in nanometers.
    seed : `int`, optional
        Random seed for reproducibility. If None, uses current random state.
        
    Returns
    -------
    segment_pistons : `dict`
        Dictionary mapping segment indices to piston values that produce target RMS.
    segment_tiptilts : `dict`
        Dictionary mapping segment indices to tip/tilt tuples that produce target RMS.
    """
    from .aberration_models import generate_random_segment_aberrations, calculate_wavefront_rms
    
    hsm = telescope_data['hsm']
    aper = telescope_data['aper']
    wavelength = telescope_data['wavelength']
    segments = telescope_data['segments']
    segment_flat_to_flat = telescope_data['segment_flat_to_flat']
    num_segments = len(segments)
    
    # Generate initial aberrations and measure actual RMS.
    segment_pistons, segment_tiptilts = generate_random_segment_aberrations(
        target_rms_nm, num_segments, segment_flat_to_flat=segment_flat_to_flat, seed=seed)

    hsm.flatten()
    apply_segment_pistons(hsm, segment_pistons, wavelength, num_segments)
    apply_segment_tiptilts(hsm, segment_tiptilts, num_segments)
    actual_rms = calculate_wavefront_rms(hsm, aper, wavelength)

    # Scale to match target and verify.
    scale_factor = target_rms_nm / actual_rms
    segment_pistons = {k: v * scale_factor for k, v in segment_pistons.items()}
    segment_tiptilts = {k: (v[0] * scale_factor, v[1] * scale_factor) 
                       for k, v in segment_tiptilts.items()}

    hsm.flatten()
    apply_segment_pistons(hsm, segment_pistons, wavelength, num_segments)
    apply_segment_tiptilts(hsm, segment_tiptilts, num_segments)
    final_rms = calculate_wavefront_rms(hsm, aper, wavelength)

    print(f"Initial RMS: {actual_rms:.4f} nm, Scale factor: {scale_factor:.3f}, Final RMS: {final_rms:.4f} nm")
    return segment_pistons, segment_tiptilts

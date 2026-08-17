"""Generate high-resolution PSFs and detector-sampled kernels.

The module implements a diverging-path architecture where high-resolution PSFs
are computed for optical quality metrics while detector-sampled kernels are
generated for science images. Both branches share the same aberrated
pupil-plane wavefront so that metrics and detector products describe the same
incoming optical state.
"""

import copy
import os

import numpy as np
from hcipy.field import (
    make_focal_grid,
    make_supersampled_grid,
    make_uniform_grid,
    subsample_field,
)
from hcipy.optics import Wavefront
from hcipy.propagation import FraunhoferPropagator

from ..constants import ARCSEC_PER_RAD
from .aberration_models import (
    apply_global_zernikes,
    apply_segment_pistons,
    apply_segment_tiptilts,
    apply_segment_zernikes,
)
from .telescope_models import create_hcipy_telescope
from .utils import PSFData, make_pyauto_kernel, pyauto_kernel_pixel_scales


def _total_aperture_rms_nm(telescope_data, phase_screens):
    """Compute the total piston-removed aperture OPD RMS in nanometers.

    The OPD combines the segmented-mirror surface (factor 2 on
    reflection) with the OPD implied by any applied phase screens
    (segment hexikes, global Zernikes). The RMS is taken over the
    illuminated pupil only, after removing the mean (piston).

    Parameters
    ----------
    telescope_data : `dict`
        Pupil-side telescope dictionary containing ``hsm`` (segmented
        mirror with a ``surface`` array), ``wavelength`` (meters), and
        ``aper`` (aperture transmission array).
    phase_screens : `dict`
        Mapping from screen name to pupil-plane phase screen in radians.
        May be empty.

    Returns
    -------
    total_rms_nm : `float`
        Piston-removed OPD RMS over the illuminated aperture in
        nanometers.

    Raises
    ------
    RuntimeError
        Raised, chained from the original exception, if any step of the
        calculation fails (for example a phase screen whose shape does
        not match the pupil grid). No fallback value is ever returned.
    """
    try:
        # Start with DM OPD.
        opd_total = telescope_data['hsm'].surface * 2

        # Convert phase screens (stored in radians) to OPD and add.
        # Relation: phase [rad] = 2π * OPD / λ  =>  OPD = phase * λ / (2π)
        if phase_screens:
            lambda_m = telescope_data['wavelength']
            rad_to_opd = lambda_m / (2 * np.pi)
            for screen in phase_screens.values():
                opd_total += np.asarray(screen) * rad_to_opd

        # Compute RMS over illuminated pupil after removing piston.
        aper_field = telescope_data['aper']
        valid_pixels = aper_field > 0.5
        opd_valid = opd_total[valid_pixels]
        opd_valid -= np.mean(opd_valid)
        return float(np.sqrt(np.mean(opd_valid**2)) * 1e9)
    except Exception as e:
        raise RuntimeError(
            f"Total aperture OPD RMS calculation failed: {e}"
        ) from e


def generate_psf_system(config, full_config=None):
    """Generate a PSF system with specified aberrations.

    This function is the canonical PSF runtime path for the project. It builds
    a pupil-side HCIPy telescope model, applies configured segment and global
    aberrations, propagates the shared pupil wavefront to a high-resolution
    focal grid for metrics, and separately propagates it to a supersampled
    detector grid for flux-conserving binning into a PyAutoLens kernel.

    Parameters
    ----------
    config : `dict`
        PSF configuration dictionary containing ``telescope``, ``hres_psf``,
        ``kernel``, and ``aberrations`` blocks.
    full_config : `dict`, optional
        Full configuration dictionary. It must contain the lensing grid pixel
        scale used to set the detector kernel scale. The complete dictionary is
        stored on the returned `PSFData` object for provenance.

    Returns
    -------
    psf_data : `PSFData`
        Complete PSF products and metadata. The object contains the
        high-resolution focal-plane PSF, the final pupil-plane wavefront,
        pupil-side telescope data, detector-sampled PyAutoLens kernel,
        sampling metadata, quality metrics, and aberration summaries.

    Raises
    ------
    ValueError
        Raised if ``full_config`` does not contain a lensing block, required
        PSF blocks are missing, or the requested sampling cannot produce a
        positive integer detector subsampling factor.

    Notes
    -----
    ``PSFData.psf`` is the high-resolution focal-plane PSF used for optical
    metrics. ``PSFData.kernel`` is the detector-sampled science kernel. The
    detector kernel is propagated on its own supersampled focal grid rather
    than being treated as a cosmetic resize of the metric PSF.

    Examples
    --------
    Generate a PSF system and inspect its detector kernel::

        psf_data = generate_psf_system(config["psf"], full_config=config)
        kernel = psf_data.kernel
    """
    # Automatic sampling calculation and validation.

    # Extract parameters from the config structure.
    # Config is the PSF config section, full_config contains everything.
    if full_config is not None and 'lensing' in full_config:
        lensing_config = full_config['lensing']
    else:
        raise ValueError('full_config must be provided and contain a "lensing" key.')
    psf_config = copy.deepcopy(config)
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
    # Apply toggle flags to aberrations.
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
        apply_segment_pistons(hsm, segment_pistons, wavelength, num_segments)

    if segment_tiptilts is not None:
        apply_segment_tiptilts(hsm, segment_tiptilts, num_segments)

    # Create initial wavefront from aperture.
    wf_pupil = Wavefront(aper, wavelength)

    # Apply segmented mirror.
    wf_pupil = hsm(wf_pupil)

    # Apply segment-level Zernikes (hexikes) as phase screen.
    if segment_hexikes is not None:
        phase_screen, hexike_surface = apply_segment_zernikes(
            segment_hexikes, telescope_data, wavelength
        )
        phase_screens['segment_hexikes'] = phase_screen
        wf_pupil = hexike_surface(wf_pupil)

    # Apply global Zernikes as phase screen.
    if global_zernikes is not None:
        phase_screen = apply_global_zernikes(global_zernikes, telescope_data, wavelength)
        phase_screens['global_zernikes'] = phase_screen
        wf_pupil.electric_field *= np.exp(1j * np.array(phase_screen))

    # Implement the single high-resolution propagation.
    # Define the high-resolution focal grid using parameters from
    # config['psf']['hres_psf'].

    # Create the high-resolution focal grid in focal-plane meters.
    focal_grid_hres = make_focal_grid(
        q=sim_config['sampling'],
        num_airy=sim_config['num_airy'],
        pupil_diameter=pupil_diameter,
        focal_length=focal_length,
        reference_wavelength=wavelength,
    )

    # Create the high-resolution propagator with the correct focal length.
    prop_hres = FraunhoferPropagator(telescope_data['pupil_grid'], focal_grid_hres, focal_length)

    # Propagate the pupil wavefront to get one high-resolution PSF wavefront.
    wf_psf_hres = prop_hres(wf_pupil)
    wf_pupil_perfect = Wavefront(aper, wavelength)
    wf_psf_perfect_hres = prop_hres(wf_pupil_perfect)

    # Optionally save high-resolution PSF intensity before downsampling.
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
        telescope_data['wavelength'] / telescope_data['pupil_diameter']
        * ARCSEC_PER_RAD / sim_config['sampling']
    )

    # Calculate PSF metrics using high-resolution PSF.
    from .psf_metrics import analyze_psf_quality
    quality_metrics = analyze_psf_quality(
        wf_psf_hres,
        perfect_psf=wf_psf_perfect_hres,
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

    # Define the detector grid in focal-plane meters.
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
    wf_psf_perfect_supersampled = prop_det(wf_pupil_perfect)

    # Downsample supersampled PSF power by summation to conserve flux.
    psf_downsampled = subsample_field(
        wf_psf_supersampled.power, subsampling=subsampling_factor, new_grid=detector_grid_m, statistic='sum'
    )
    psf_perfect_downsampled = subsample_field(
        wf_psf_perfect_supersampled.power,
        subsampling=subsampling_factor,
        new_grid=detector_grid_m,
        statistic='sum',
    )

    # Normalize detector kernels to unit flux.
    psf_sum = float(np.sum(psf_downsampled))
    perfect_psf_sum = float(np.sum(psf_perfect_downsampled))
    if not np.isfinite(psf_sum) or psf_sum <= 0.0:
        raise ValueError("Detector-sampled PSF has non-positive or non-finite flux.")
    if not np.isfinite(perfect_psf_sum) or perfect_psf_sum <= 0.0:
        raise ValueError(
            "Perfect detector-sampled PSF has non-positive or non-finite flux."
        )
    psf_downsampled_normalized = psf_downsampled / psf_sum
    psf_perfect_downsampled_normalized = psf_perfect_downsampled / perfect_psf_sum
    perfect_kernel = np.asarray(psf_perfect_downsampled_normalized.shaped, dtype=float)
    kernel_diff = (
        np.asarray(psf_downsampled_normalized.shaped, dtype=float)
        - perfect_kernel
    )
    kernel_diff_l2_norm = float(np.linalg.norm(kernel_diff))
    perfect_kernel_l2_norm = float(np.linalg.norm(perfect_kernel))
    kernel_diff_l2_rel = (
        kernel_diff_l2_norm / perfect_kernel_l2_norm
        if perfect_kernel_l2_norm > 0.0
        else None
    )

    # Create the detector-sampled PyAuto kernel array.
    kernel = make_pyauto_kernel(
        # Use .shaped to get a 2D array.
        values=psf_downsampled_normalized.shaped,
        pixel_scales=autolens_pixel_scale
    )

    # Verify pixel scale matching.
    if not np.allclose(pyauto_kernel_pixel_scales(kernel), autolens_pixel_scale, rtol=1e-10):
        raise ValueError(
            f"Pixel scale mismatch: kernel pixel_scales={pyauto_kernel_pixel_scales(kernel)}, "
            f"expected autolens_pixel_scale={autolens_pixel_scale}. "
            f"This indicates a fundamental problem in the downsampling logic."
        )

    # Calculate proper total RMS wavefront error including all aberrations.
    # The helper combines the DM surface OPD (factor 2 on reflection) with
    # OPD implied by any applied phase screens (segment hexikes, global
    # Zernikes) and raises RuntimeError if the calculation fails.
    total_rms_nm = _total_aperture_rms_nm(telescope_data, phase_screens)

    # Calculate individual aberration coefficient summaries for metadata.
    # These are not independent aperture-weighted RMS budget terms.
    # above is the physical OPD RMS over the illuminated pupil.
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
        # Shape is (N, 2).
        tiptilts_array = np.array(list(segment_tiptilts.values()))
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
        wavefront=wf_pupil.copy(),  # Final aberrated pupil-plane wavefront.
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
        raw_peak_ratio_before_clipping=quality_metrics.get('raw_peak_ratio_before_clipping'),
        peak_intensity=quality_metrics['peak_intensity'],
        total_flux=quality_metrics['total_flux'],
        kernel_diff_l2_norm=kernel_diff_l2_norm,
        kernel_diff_l2_rel=kernel_diff_l2_rel,

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
        # Pixel scale of detector-generated kernel.
        kernel_pixel_scale=autolens_pixel_scale,
        highres_psf_npy_path=saved_highres_psf_path,

        # Complex data.
        phase_screens=phase_screens,
        aberrations=aberrations,
        config=full_config
    )

    return psf_data

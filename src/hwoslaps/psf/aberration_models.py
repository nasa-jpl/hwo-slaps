"""Aberration models for PSF generation.

This module contains all aberration functions for applying various types of
optical aberrations to segmented mirrors.

The module supports both manual implementation of segment-level aberrations
and the HCIPy API approach, providing flexibility for different use cases.
"""

import numpy as np
import hcipy


def make_hexike_basis(num_modes, circum_diameter, grid, hexagon_angle=0):
    """Make a hexike basis.

    This is based on [Mahajan2006]_. This function creates a Zernike mode basis and
    numerically orthogonalizes it using Gramm-Schmidt orthogonalization.

    .. [Mahajan2006] Virendra N. Mahajan and Guang-ming Dai,
        "Orthonormal polynomials for hexagonal pupils," Opt.
        Lett. 31, 2462-2464 (2006)

    Parameters
    ----------
    num_modes : `int`
        The number of hexike modes to compute.
    circum_diameter : `float`
        The circumdiameter of the hexagonal aperture.
    grid : `hcipy.Grid`
        The grid on which to compute the mode basis.
    hexagon_angle : `float`, optional
        The rotation angle of the hexagon in radians. Default is 0.

    Returns
    -------
    hexike_basis : `hcipy.ModeBasis`
        The hexike mode basis.
    """
    zernike_basis = hcipy.make_zernike_basis(int(num_modes), circum_diameter, grid)
    # Create hexagonal aperture for this basis.
    hexagonal_aperture = hcipy.make_hexagonal_aperture(circum_diameter, hexagon_angle)(grid)
    sqrt_weights = np.sqrt(grid.weights)

    if np.isscalar(sqrt_weights):
        sqrt_weights = np.array([sqrt_weights])

    # Un-normalize the zernike modes using the weights of the grid.
    # Reshape to properly broadcast: (n_pixels,) -> (n_pixels, 1).
    weights_factor = (hexagonal_aperture * sqrt_weights)[:, np.newaxis]
    zernike_basis.transformation_matrix *= weights_factor

    # Perform Gramm-Schmidt orthogonalization using a QR decomposition.
    Q, R = np.linalg.qr(zernike_basis.transformation_matrix)

    # Correct for negative sign of components of the Q matrix.
    hexike_basis = hcipy.ModeBasis(Q / np.sign(np.diag(R)), grid)

    # Renormalize the resulting functions using the area of a hexagon and the grid weights.
    area_hexagon = 3 * np.sqrt(3) / 8 * circum_diameter**2
    # Reshape sqrt_weights for proper broadcasting.
    normalization_factor = (np.sqrt(area_hexagon) / sqrt_weights)[:, np.newaxis]
    hexike_basis.transformation_matrix *= normalization_factor

    return hexike_basis


def make_segment_modes(segments, segment_centers, segment_diameter, pupil_grid, 
                       num_modes_per_segment, pointy_top=False):
    """Make a list of modes containing segment-level modes for each segment.

    Parameters
    ----------
    segments : `list` of `hcipy.Field`
        List of segment apertures.
    segment_centers : `hcipy.Grid`
        Grid containing the center positions of each segment.
    segment_diameter : `float`
        The circumscribed diameter of each hexagonal segment.
    pupil_grid : `hcipy.Grid`
        The pupil grid.
    num_modes_per_segment : `int`
        The number of modes per segment.
    pointy_top : `bool`, optional
        Whether the hexagons have a pointy top. Default is False (flat top).

    Returns
    -------
    modes : `list` of `numpy.ndarray`
        A list of computed modes, normalized to RMS=1 over each segment.
    """
    modes = []
    angle = 0 if pointy_top else np.pi / 2

    for i, center in enumerate(segment_centers.points):
        # Create hexike basis shifted to segment center.
        basis = make_hexike_basis(num_modes_per_segment, segment_diameter, 
                                  pupil_grid.shifted(-center), angle)

        # Extract the modes and normalize them over the segment.
        for mode in basis:
            mode_array = np.asarray(mode.shaped)
            segment_mask = segments[i].shaped

            # Normalize to RMS=1 over the segment.
            mode_over_segment = mode_array * segment_mask
            rms = np.sqrt(np.mean(mode_over_segment[segment_mask > 0.5]**2))
            if rms > 0:
                mode_array /= rms

            modes.append(mode_array)

    return modes


def nm_to_opd(nm_rms):
    """Convert nanometers of wavefront OPD to meters.

    Parameters
    ----------
    nm_rms : `float`
        Amplitude in nanometers of wavefront optical path difference (OPD).

    Returns
    -------
    opd : `float`
        Optical path difference in meters.

    Notes
    -----
    For a reflective mirror, OPD = 2 × surface_height. Throughout this module,
    piston amplitudes supplied in nanometers are interpreted as wavefront OPD
    and are converted to mirror-surface height by dividing by two when writing
    actuator values.
    """
    return nm_rms * 1e-9


def urad_to_rad(urad):
    """Convert microradians to radians.
    
    Parameters
    ----------
    urad : `float`
        Angle in microradians.
        
    Returns
    -------
    radians : `float`
        Angle in radians.
    """
    return urad * 1e-6


def apply_segment_pistons(hsm, piston_dict, wavelength, num_segments):
    """Apply piston errors to individual segments.

    Parameters
    ----------
    hsm : `hcipy.SegmentedDeformableMirror`
        The segmented mirror object.
    piston_dict : `dict`
        Mapping from segment index to piston amplitudes in nanometers of
        wavefront optical path difference (OPD). These are converted to
        mirror-surface height by dividing by two (reflection doubles OPD)
        before being written to the actuators.
    wavelength : `float`
        Wavelength in meters.
    num_segments : `int`
        Total number of segments.
    """
    hsm.flatten()
    for seg_id, piston_nm in piston_dict.items():
        if seg_id < num_segments:
            # Convert OPD (nm) → surface height (m). A reflective surface
            # doubles the OPD, so surface = OPD / 2.
            piston_m = nm_to_opd(piston_nm) / 2
            hsm.set_segment_actuators(seg_id, piston_m, 0, 0)


def apply_segment_tiptilts(hsm, tiptilt_dict, num_segments):
    """Apply tip and tilt errors to individual segments while preserving piston.

    This function updates the tip and tilt actuator values for each segment in a
    segmented deformable mirror, keeping the existing piston value intact. The
    actuator layout for :class:`hcipy.SegmentedDeformableMirror` consists of three
    contiguous blocks of length ``N`` (number of segments): pistons ``[0..N-1]``,
    tips ``[N..2N-1]``, and tilts ``[2N..3N-1]``. We therefore either fetch the
    current piston using the mirror accessor, or directly read from the piston
    block, and write tip/tilt into their respective blocks.

    Notes
    -----
    - Tip/tilt inputs here are specified as outgoing beam angles. A mirror
      doubles the beam deflection, so the corresponding mirror surface slope is
      half of the outgoing angle; we therefore divide by 2 before writing the
      actuator values.
    - Piston amplitudes are handled by ``apply_segment_pistons`` and are
      specified in nanometers of wavefront OPD. They are converted to surface
      height by dividing by two before being set on the piston actuator.

    Parameters
    ----------
    hsm : `hcipy.SegmentedDeformableMirror`
        The segmented mirror object.
    tiptilt_dict : `dict`
        Mapping from segment index to a 2‑tuple ``(tip_urad, tilt_urad)`` giving
        desired outgoing beam angles in microradians for tip and tilt.
    num_segments : `int`
        Total number of segments in the mirror.
    """
    for seg_id, (tip_urad, tilt_urad) in tiptilt_dict.items():
        if seg_id < num_segments:
            # Preserve existing piston for this segment.
            current_piston, _, _ = hsm.get_segment_actuators(seg_id)

            # Convert outgoing beam angles (µrad) to surface slope (rad).
            tip_rad = urad_to_rad(tip_urad) / 2.0
            tilt_rad = urad_to_rad(tilt_urad) / 2.0

            # Update all three actuators atomically for this segment.
            hsm.set_segment_actuators(seg_id, current_piston, tip_rad, tilt_rad)


def apply_segment_zernikes_manual(segment_zernike_dict, segments, telescope_data, wavelength):
    """Apply Zernike aberrations to individual segments (manual implementation).

    This implementation manually creates hexike basis functions for each segment.

    Parameters
    ----------
    segment_zernike_dict : `dict`
        Dictionary mapping segment ID to another dict of {mode: amplitude_nm}.
        Example: {0: {4: 20, 5: 10}, 5: {6: 15}} applies Z4=20nm and Z5=10nm
        to segment 0, and Z6=15nm to segment 5.
    segments : `list`
        List of segment aperture functions.
    telescope_data : `dict`
        Dictionary containing telescope parameters.
    wavelength : `float`
        Wavelength in meters.

    Returns
    -------
    phase_screen : `hcipy.Field`
        Phase screen containing segment-level Zernike aberrations.
    """
    pupil_grid = telescope_data['pupil_grid']
    segment_flat_to_flat = telescope_data['segment_flat_to_flat']
    gap_size = telescope_data['gap_size']
    num_rings = telescope_data['num_rings']
    segment_point_to_point = telescope_data['segment_point_to_point']
    
    # Calculate segment centers.
    segment_pitch = segment_flat_to_flat + gap_size
    segment_centers = hcipy.make_hexagonal_grid(segment_pitch, num_rings, False)
    mask = segment_centers.ones(dtype='bool')
    segment_centers_grid = segment_centers.subset(mask)

    # The segment diameter is the circumscribed diameter (vertex-to-vertex).
    segment_diameter = segment_point_to_point

    # Initialize phase screen.
    phase_screen = pupil_grid.zeros()

    # Process each segment that has aberrations.
    for seg_id, mode_dict in segment_zernike_dict.items():
        if seg_id < len(segments):
            # Get the center of this segment.
            center = segment_centers_grid.points[seg_id]

            # Find the maximum mode needed for this segment.
            max_mode_for_segment = max(mode_dict.keys())

            # Create hexike basis for this segment only.
            angle = np.pi / 2  # For flat-top hexagons.
            basis = make_hexike_basis(int(max_mode_for_segment + 1), segment_diameter,
                                      pupil_grid.shifted(-center), angle)

            # Apply each requested mode.
            for mode, coeff_nm in mode_dict.items():
                if mode < len(basis):
                    phase_rad = 2 * np.pi * nm_to_opd(coeff_nm) / wavelength
                    # Get the mode as a Field (already 1D).
                    mode_field = basis[mode]
                    # Apply segment mask to ensure mode only affects this segment.
                    segment_mask = segments[seg_id]
                    phase_screen += phase_rad * mode_field * segment_mask

    return phase_screen


def apply_segment_zernikes_api(segment_hexike_dict, telescope_data, wavelength):
    """Apply segment-level hexike aberrations using the HCIPy API.

    This uses the HCIPy API for applying segment-level hexike aberrations.

    Parameters
    ----------
    segment_hexike_dict : `dict`
        Dictionary mapping segment ID to another dict of {mode: amplitude_nm}.
        Example: {0: {0: 100}, 1: {1: 100}} applies hexike mode 0 with 100nm RMS
        to segment 0, and hexike mode 1 with 100nm RMS to segment 1.
    telescope_data : `dict`
        Dictionary containing telescope parameters.
    wavelength : `float`
        Wavelength in meters.

    Returns
    -------
    phase_screen : `hcipy.Field`
        Phase screen containing segment-level aberrations.
    hsm_api : `hcipy.SegmentedDeformableMirror`
        The API-enabled segmented mirror.
    """
    segments = telescope_data['segments']
    pupil_grid = telescope_data['pupil_grid']
    segment_flat_to_flat = telescope_data['segment_flat_to_flat']
    gap_size = telescope_data['gap_size']
    num_rings = telescope_data['num_rings']
    segment_point_to_point = telescope_data['segment_point_to_point']
    
    # Calculate segment centers for the API.
    segment_pitch = segment_flat_to_flat + gap_size
    segment_centers = hcipy.make_hexagonal_grid(segment_pitch, num_rings, False)
    mask = segment_centers.ones(dtype='bool')
    segment_centers_grid = segment_centers.subset(mask)

    # Create segmented mirror with hexike support using new API.
    hsm_api = hcipy.SegmentedDeformableMirror(
        segments,
        segment_diameter=segment_point_to_point,
        hexagon_angle=np.pi/2,  # Flat-top orientation.
        segment_centers=segment_centers_grid,
        pupil_grid=pupil_grid
    )

    # Apply hexike aberrations using the new API method.
    phase_screen_api = hsm_api.apply_segment_hexike_aberrations(segment_hexike_dict, wavelength)

    return phase_screen_api, hsm_api


def apply_segment_zernikes(segment_zernike_dict, segments, telescope_data, wavelength, use_api=False):
    """Apply Zernike/hexike aberrations to individual segments.

    This function provides both the manual implementation and the HCIPy API approach
    for applying segment-level aberrations.

    Parameters
    ----------
    segment_zernike_dict : `dict`
        Dictionary mapping segment ID to another dict of {mode: amplitude_nm}.
    segments : `list`
        List of segment aperture functions.
    telescope_data : `dict`
        Dictionary containing telescope parameters.
    wavelength : `float`
        Wavelength in meters.
    use_api : `bool`, optional
        If True, use the HCIPy API method. If False, use manual implementation.
        Default is False for backward compatibility.

    Returns
    -------
    phase_screen : `hcipy.Field`
        Phase screen containing segment-level aberrations (if use_api=False).
    phase_screen, hsm_api : `tuple`
        Phase screen and API-enabled mirror (if use_api=True).
    """
    if use_api:
        return apply_segment_zernikes_api(segment_zernike_dict, telescope_data, wavelength)
    else:
        return apply_segment_zernikes_manual(segment_zernike_dict, segments, telescope_data, wavelength)


def apply_global_zernikes(zernike_coeffs_nm, telescope_data, wavelength):
    """Apply global Zernike aberrations across the entire pupil.

    Parameters
    ----------
    zernike_coeffs_nm : `dict` or `array_like`
        Global Zernike coefficients in nm RMS.
    telescope_data : `dict`
        Dictionary containing telescope parameters.
    wavelength : `float`
        Wavelength in meters.

    Returns
    -------
    phase_screen : `hcipy.Field`
        Phase screen containing global Zernike aberrations.
    """
    pupil_grid = telescope_data['pupil_grid']
    pupil_diameter = telescope_data['pupil_diameter']
    
    # Create Zernike basis for the full pupil.
    num_zernike_modes = 50
    pupil_diameter_for_zernike = pupil_grid.x.max() - pupil_grid.x.min()
    zernike_basis = hcipy.make_zernike_basis(num_zernike_modes, D=pupil_diameter_for_zernike, grid=pupil_grid)
    
    phase_screen = pupil_grid.zeros()

    if isinstance(zernike_coeffs_nm, dict):
        for mode, coeff_nm in zernike_coeffs_nm.items():
            if mode < len(zernike_basis):
                phase_rad = 2 * np.pi * nm_to_opd(coeff_nm) / wavelength
                phase_screen += phase_rad * zernike_basis[mode]
    else:
        for mode, coeff_nm in enumerate(zernike_coeffs_nm):
            if coeff_nm != 0 and mode < len(zernike_basis):
                phase_rad = 2 * np.pi * nm_to_opd(coeff_nm) / wavelength
                phase_screen += phase_rad * zernike_basis[mode]

    return phase_screen


def generate_random_segment_aberrations(
    target_rms_nm,
    num_segments,
    piston_weight=0.5,
    tiptilt_weight=0.5,
    segment_flat_to_flat=None,
    seed=None,
):
    """Generate random segment pistons and tip/tilts for a target RMS (heuristic).

    This function produces zero-mean random pistons (in nm OPD) and tip/tilts (in µrad)
    as an initial guess to achieve a total RMS wavefront error of ``target_rms_nm``.
    The tip/tilt scaling uses a geometric relation for a hexagon of flat-to-flat
    size ``F`` with radius ``R = F/2``:

    RMS_height [m] ≈ slope [rad] × R / √3  ⇒  slope [µrad] ≈ RMS_nm × √3 / R × 1e-3

    Notes
    -----
    - This mapping is an approximation and depends on aperture discretization and
      basis details. For scientific use, follow with
      ``generate_calibrated_segment_aberrations`` which numerically rescales to the
      exact target RMS on the configured system.
    - If ``segment_flat_to_flat`` is not provided, we generate unit-variance
      tip/tilts in µrad without attempting a physically inconsistent nm→µrad
      conversion. The calibration step should then be used to match the target.

    Parameters
    ----------
    target_rms_nm : `float`
        Target RMS wavefront error in nanometers.
    num_segments : `int`
        Number of segments.
    piston_weight : `float`, optional
        Relative weight of piston errors (0-1). Default is 0.5.
    tiptilt_weight : `float`, optional
        Relative weight of tip/tilt errors (0-1). Default is 0.5.
    segment_flat_to_flat : `float`, optional
        Segment flat-to-flat distance in meters. Required for physically
        meaningful nm→µrad tip/tilt scaling.
    seed : `int`, optional
        Random seed for reproducibility. If None, uses current random state.

    Returns
    -------
    segment_pistons : `dict`
        Dictionary mapping segment indices to piston values (nm OPD).
    segment_tiptilts : `dict`
        Dictionary mapping segment indices to (tip_µrad, tilt_µrad).
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate zero-mean random pistons and tip/tilts for all segments.
    pistons_raw = np.random.randn(num_segments)
    pistons_raw -= np.mean(pistons_raw)
    tips_raw = np.random.randn(num_segments)
    tilts_raw = np.random.randn(num_segments)

    # Scale to desired RMS contributions.
    piston_rms_target = target_rms_nm * np.sqrt(piston_weight)
    tiptilt_rms_target = target_rms_nm * np.sqrt(tiptilt_weight)

    pistons_nm = pistons_raw * (piston_rms_target / np.std(pistons_raw))

    # Convert tip/tilts to microradians (small-angle relation RMS_height ≈ slope*R/√3).
    if segment_flat_to_flat is not None:
        segment_radius = segment_flat_to_flat / 2
        # Match the random vector RMS to the requested nm component before geometry.
        tiptilt_scale = tiptilt_rms_target / np.sqrt(np.var(tips_raw) + np.var(tilts_raw))
        # slope[µrad] ≈ RMS_nm * √3 / R * 1e-3
        geom = np.sqrt(3.0) / segment_radius * 1e-3
        tips_urad = tips_raw * tiptilt_scale * geom
        tilts_urad = tilts_raw * tiptilt_scale * geom
    else:
        # No segment size: generate dimensionally correct angles (µrad) with unit RMS.
        # Rely on downstream numerical calibration to match the requested nm RMS.
        tips_urad = tips_raw
        tilts_urad = tilts_raw

    return ({i: pistons_nm[i] for i in range(num_segments)}, 
            {i: (tips_urad[i], tilts_urad[i]) for i in range(num_segments)})


def calculate_wavefront_rms(hsm, aper, wavelength):
    """Calculate the RMS wavefront error in nm.
    
    Parameters
    ----------
    hsm : `hcipy.SegmentedDeformableMirror`
        The segmented mirror object.
    aper : `hcipy.Field`
        The aperture function.
    wavelength : `float`
        Wavelength in meters.
        
    Returns
    -------
    rms_error : `float`
        RMS wavefront error in nanometers.
    """
    opd_map = hsm.surface * 2  # Factor of 2 for reflection.
    valid_pixels = aper > 0.5
    opd_valid = opd_map[valid_pixels]
    opd_valid -= np.mean(opd_valid)  # Remove piston.
    return np.sqrt(np.mean(opd_valid**2)) * 1e9  # Convert to nm.
"""Apply optical aberrations to segmented HWO pupil models.

This module contains small conversion helpers and HCIPy adapters for segment
pistons, segment tip/tilts, segment hexikes, and global Zernike phase screens.
Unless otherwise noted, nanometer amplitudes are wavefront OPD amplitudes.
"""

import hcipy
import numpy as np


def _validate_segment_id(raw_seg_id, num_segments):
    """Validate a zero-based segment identifier.

    Parameters
    ----------
    raw_seg_id : `int`
        Candidate segment identifier.
    num_segments : `int`
        Number of available segments.

    Returns
    -------
    seg_id : `int`
        Validated segment identifier.

    Raises
    ------
    ValueError
        Raised if ``raw_seg_id`` is not an integer segment index in range.
    """
    if isinstance(raw_seg_id, bool) or not isinstance(raw_seg_id, int):
        raise ValueError('segment indices must be integers.')
    if raw_seg_id < 0 or raw_seg_id >= num_segments:
        raise ValueError(f'segment index {raw_seg_id} is outside the valid 0..{num_segments - 1} range.')
    return raw_seg_id


def _normalize_segment_hexike_dict(segment_hexike_dict, num_segments):
    """Normalize segment and hexike mode indices.

    Parameters
    ----------
    segment_hexike_dict : `dict`
        Mapping from zero-based segment identifiers to mode dictionaries.
        Mode keys are 1-based Noll indices.
    num_segments : `int`
        Number of available segments.

    Returns
    -------
    normalized : `dict`
        Validated mapping with integer segment and mode keys.
    """
    normalized = {}

    for raw_seg_id, raw_mode_dict in segment_hexike_dict.items():
        seg_id = _validate_segment_id(raw_seg_id, num_segments)

        if not raw_mode_dict:
            normalized[seg_id] = {}
            continue

        parsed_mode_dict = {int(raw_mode): coeff for raw_mode, coeff in raw_mode_dict.items()}
        if any(mode_idx < 1 for mode_idx in parsed_mode_dict):
            raise ValueError('Hexike mode indices must be 1-based Noll integers (>= 1).')
        normalized[seg_id] = parsed_mode_dict

    return normalized


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

    Raises
    ------
    ValueError
        Raised if any segment index is invalid.
    """
    hsm.flatten()
    for seg_id, piston_nm in piston_dict.items():
        seg_id = _validate_segment_id(seg_id, num_segments)
        # Convert OPD (nm) → surface height (m). A reflective surface
        # doubles the OPD, so surface = OPD / 2.
        piston_m = nm_to_opd(piston_nm) / 2
        hsm.set_segment_actuators(seg_id, piston_m, 0, 0)


def apply_segment_tiptilts(hsm, tiptilt_dict, num_segments):
    """Apply per-segment tip and tilt errors while preserving piston.

    This function updates the tip and tilt actuator values for each segment in
    a segmented deformable mirror while keeping the existing piston value
    intact.

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
        Mapping from segment index to a 2-tuple
        ``(tip_urad, tilt_urad)`` giving desired outgoing beam angles in
        microradians for tip and tilt.
    num_segments : `int`
        Total number of segments in the mirror.

    Raises
    ------
    ValueError
        Raised if any segment index is invalid.
    """
    for seg_id, (tip_urad, tilt_urad) in tiptilt_dict.items():
        seg_id = _validate_segment_id(seg_id, num_segments)
        # Preserve existing piston for this segment.
        current_piston, _, _ = hsm.get_segment_actuators(seg_id)

        # Convert outgoing beam angles (µrad) to surface slope (rad).
        tip_rad = urad_to_rad(tip_urad) / 2.0
        tilt_rad = urad_to_rad(tilt_urad) / 2.0

        # Update all three actuators atomically for this segment.
        hsm.set_segment_actuators(seg_id, current_piston, tip_rad, tilt_rad)


def apply_segment_zernikes(segment_hexike_dict, telescope_data, wavelength):
    """Apply segment hexike aberrations via HCIPy's segmented surface optic.

    Parameters
    ----------
    segment_hexike_dict : `dict`
        Mapping from zero-based segment ID to ``{mode_noll: amplitude_nm}``.
        Mode keys follow 1-based Noll indexing.
    telescope_data : `dict`
        Pupil-side telescope dictionary returned by
        :func:`hwoslaps.psf.telescope_models.create_hcipy_telescope`.
    wavelength : `float`
        Wavelength in meters.

    Returns
    -------
    phase_screen : `hcipy.Field`
        Phase screen containing segment-level aberrations.
    hexike_surface : `hcipy.SegmentedHexikeSurface`
        The segmented hexike surface optic.

    Raises
    ------
    ValueError
        Raised if any segment or mode index is invalid.
    """
    pupil_grid = telescope_data['pupil_grid']
    segments = telescope_data['segments']
    segment_centers = telescope_data['segment_centers']
    segment_circum_diameter = telescope_data['segment_point_to_point']
    normalized_segment_dict = _normalize_segment_hexike_dict(segment_hexike_dict, len(segments))

    # Determine how many modes per segment are required.
    max_mode = 0
    for mode_dict in normalized_segment_dict.values():
        if mode_dict:
            max_mode = max(max_mode, max(mode_dict.keys()))
    num_modes = max_mode

    # Build or reuse a segmented hexike surface for this telescope geometry.
    hexike_surface = telescope_data.get('segment_hexike_surface')
    expected_shape = (len(segments), num_modes)
    if (hexike_surface is None or hexike_surface.input_grid is not pupil_grid
            or hexike_surface.coefficients.shape != expected_shape):
        hexike_surface = hcipy.SegmentedHexikeSurface(
            segments=segments,
            segment_centers=segment_centers,
            segment_circum_diameter=segment_circum_diameter,
            pupil_grid=pupil_grid,
            num_modes=num_modes,
            hexagon_angle=np.pi / 2  # Flat-top orientation.
        )

        if hexike_surface.coefficients.shape[0] != len(segments):
            raise ValueError(
                'HCIPy segment hexike surface segment count does not match telescope segment count.'
            )

        telescope_data['segment_hexike_surface'] = hexike_surface
    else:
        hexike_surface.flatten()

    # Set per-segment coefficients using HCIPy Noll indexing and
    # surface-height units.
    for seg_id, mode_dict in normalized_segment_dict.items():
        if mode_dict:
            coeffs_m = {
                mode_idx: nm_to_opd(coeff_nm) / 2
                for mode_idx, coeff_nm in mode_dict.items()
            }
            hexike_surface.set_segment_coefficients(seg_id, coeffs_m, indexing='noll')

    phase_screen_api = hexike_surface.phase_for(wavelength)

    return phase_screen_api, hexike_surface


def apply_global_zernikes(zernike_coeffs_nm, telescope_data, wavelength):
    """Apply global Zernike aberrations across the entire pupil.

    Parameters
    ----------
    zernike_coeffs_nm : `dict` or `array_like`
        Global Zernike coefficients in nanometers RMS. Dictionary keys are
        1-based Noll indices.
    telescope_data : `dict`
        Pupil-side telescope dictionary returned by
        :func:`hwoslaps.psf.telescope_models.create_hcipy_telescope`.
    wavelength : `float`
        Wavelength in meters.

    Returns
    -------
    phase_screen : `hcipy.Field`
        Phase screen containing global Zernike aberrations.

    Raises
    ------
    ValueError
        Raised if a dictionary key is not a supported 1-based Noll index.
    """
    pupil_grid = telescope_data['pupil_grid']

    # Create Zernike basis for the full pupil, sized to the highest
    # requested Noll mode.
    num_zernike_modes = 50
    if isinstance(zernike_coeffs_nm, dict):
        integer_modes = [
            mode for mode in zernike_coeffs_nm
            if isinstance(mode, int) and not isinstance(mode, bool)
        ]
        if integer_modes:
            num_zernike_modes = max(num_zernike_modes, max(integer_modes))
    pupil_diameter_for_zernike = pupil_grid.x.max() - pupil_grid.x.min()
    zernike_basis = hcipy.make_zernike_basis(num_zernike_modes, D=pupil_diameter_for_zernike, grid=pupil_grid)

    phase_screen = pupil_grid.zeros()

    if isinstance(zernike_coeffs_nm, dict):
        for mode, coeff_nm in zernike_coeffs_nm.items():
            if isinstance(mode, bool) or not isinstance(mode, int):
                raise ValueError('Global Zernike mode indices must be 1-based Noll integers.')
            if mode < 1 or mode > len(zernike_basis):
                raise ValueError(
                    f'Global Zernike mode {mode} is outside the supported 1..{len(zernike_basis)} Noll range.'
                )
            phase_rad = 2 * np.pi * nm_to_opd(coeff_nm) / wavelength
            phase_screen += phase_rad * zernike_basis[mode - 1]
    else:
        for mode_idx, coeff_nm in enumerate(zernike_coeffs_nm):
            if coeff_nm != 0 and mode_idx < len(zernike_basis):
                phase_rad = 2 * np.pi * nm_to_opd(coeff_nm) / wavelength
                phase_screen += phase_rad * zernike_basis[mode_idx]

    return phase_screen


def generate_random_segment_aberrations(
    target_rms_nm,
    num_segments,
    piston_weight=0.5,
    tiptilt_weight=0.5,
    segment_flat_to_flat=None,
    seed=None,
):
    """Generate random segment pistons and tip/tilts for a target RMS.

    This heuristic produces zero-mean random pistons in nanometers OPD and
    tip/tilts in microradians as an initial guess for a total wavefront error.
    If a segment size is supplied, the tip/tilt scaling uses a geometric
    relation for a hexagon of flat-to-flat size ``F`` with radius ``R = F/2``::

        RMS_height [m] ~= slope [rad] * R / sqrt(3)
        slope [urad] ~= RMS_nm * sqrt(3) / R * 1e-3

    Notes
    -----
    - This mapping is an approximation and depends on aperture
      discretization and basis details. For scientific use, follow with a
      numerical calibration pass that rescales to the exact target RMS on the
      configured system.
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

    Raises
    ------
    ValueError
        Raised if the segment count, RMS target, weights, or supplied segment
        size are outside their physical domains.
    """
    if isinstance(num_segments, bool) or not isinstance(num_segments, int) or num_segments < 2:
        raise ValueError('num_segments must be an integer >= 2.')
    if not np.isfinite(target_rms_nm) or target_rms_nm < 0:
        raise ValueError('target_rms_nm must be finite and non-negative.')
    if not np.isfinite(piston_weight) or piston_weight < 0:
        raise ValueError('piston_weight must be finite and non-negative.')
    if not np.isfinite(tiptilt_weight) or tiptilt_weight < 0:
        raise ValueError('tiptilt_weight must be finite and non-negative.')
    total_weight = piston_weight + tiptilt_weight
    if total_weight <= 0:
        raise ValueError('At least one aberration weight must be positive.')
    if segment_flat_to_flat is not None and (
        not np.isfinite(segment_flat_to_flat) or segment_flat_to_flat <= 0
    ):
        raise ValueError('segment_flat_to_flat must be finite and positive when provided.')

    rng = np.random.default_rng(seed)

    # Generate zero-mean random pistons and tip/tilts for all segments.
    pistons_raw = rng.standard_normal(num_segments)
    pistons_raw -= np.mean(pistons_raw)
    tips_raw = rng.standard_normal(num_segments)
    tilts_raw = rng.standard_normal(num_segments)

    # Scale to desired RMS contributions.
    piston_rms_target = target_rms_nm * np.sqrt(piston_weight / total_weight)
    tiptilt_rms_target = target_rms_nm * np.sqrt(tiptilt_weight / total_weight)

    piston_std = np.std(pistons_raw)
    if piston_rms_target == 0:
        pistons_nm = np.zeros(num_segments)
    elif piston_std == 0:
        raise ValueError('Could not generate non-degenerate piston perturbations.')
    else:
        pistons_nm = pistons_raw * (piston_rms_target / piston_std)

    # Convert tip/tilts to microradians (small-angle relation
    # RMS_height ≈ slope*R/√3).
    if tiptilt_rms_target == 0:
        tips_urad = np.zeros(num_segments)
        tilts_urad = np.zeros(num_segments)
    elif segment_flat_to_flat is not None:
        segment_radius = segment_flat_to_flat / 2
        # Match the random vector RMS to the requested nm component before
        # applying geometry.
        tiptilt_variance = np.var(tips_raw) + np.var(tilts_raw)
        if tiptilt_variance == 0:
            raise ValueError('Could not generate non-degenerate tip/tilt perturbations.')
        tiptilt_scale = tiptilt_rms_target / np.sqrt(tiptilt_variance)
        # slope[µrad] ≈ RMS_nm * √3 / R * 1e-3
        geom = np.sqrt(3.0) / segment_radius * 1e-3
        tips_urad = tips_raw * tiptilt_scale * geom
        tilts_urad = tilts_raw * tiptilt_scale * geom
    else:
        # No segment size: generate dimensionally correct angles (µrad) with
        # unit RMS.  Rely on downstream numerical calibration to match the
        # requested nm RMS.
        tips_urad = tips_raw
        tilts_urad = tilts_raw

    return ({i: pistons_nm[i] for i in range(num_segments)},
            {i: (tips_urad[i], tilts_urad[i]) for i in range(num_segments)})


def calculate_wavefront_rms(hsm, aper, wavelength):
    """Calculate RMS wavefront OPD from a segmented mirror surface.

    This helper measures only the OPD represented by the segmented deformable
    mirror surface. It does not include additional phase-screen optics such as
    segment hexikes or global Zernikes.

    Parameters
    ----------
    hsm : `hcipy.SegmentedDeformableMirror`
        The segmented mirror object.
    aper : `hcipy.Field`
        The aperture function.
    wavelength : `float`
        Wavelength in meters. This parameter is accepted for API compatibility
        and is not used in the surface-height calculation.

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

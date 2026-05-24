"""Build HCIPy pupil-plane telescope models for PSF generation.

This module creates the segmented aperture, pupil grid, segment masks, and
deformable mirror used by the PSF generator. Focal-plane grids and propagators
are built by :func:`hwoslaps.psf.generator.generate_psf_system`, where the
high-resolution and detector-sampled branches choose their own focal grids.
"""

import numpy as np
import hcipy


def create_hcipy_telescope(config):
    """Create the HCIPy pupil-plane telescope model.

    The returned dictionary contains only pupil-side optical components and
    static telescope geometry. It deliberately does not contain focal grids or
    propagators, since those depend on whether the caller is generating the
    high-resolution metric PSF or the detector-sampled kernel.

    Parameters
    ----------
    config : `dict`
        PSF configuration dictionary containing ``telescope`` and
        ``hres_psf`` blocks.

    Returns
    -------
    telescope_data : `dict`
        Dictionary containing the HCIPy pupil grid, supersampled segmented
        aperture, per-segment aperture masks, segmented deformable mirror,
        wavelength, and telescope geometry metadata.
    """
    # Extract parameters from config
    telescope_config = config['telescope']
    sim_config = config['hres_psf']
    
    # Parameters for the pupil function
    gap_size = telescope_config['gap_size']
    segment_point_to_point = telescope_config['segment_point_to_point']
    pupil_diameter = telescope_config['pupil_diameter']
    num_rings = telescope_config['num_rings']
    segment_flat_to_flat = segment_point_to_point * np.sqrt(3) / 2
    
    # Parameters for the simulation
    num_pix = sim_config['num_pix']
    wavelength = sim_config['wavelength']
    
    # HCIPy pupil grid
    pupil_grid = hcipy.make_pupil_grid(dims=num_pix, diameter=pupil_diameter)
    
    # Create segmented aperture
    aper, segments = hcipy.make_hexagonal_segmented_aperture(num_rings,
                                                             segment_flat_to_flat,
                                                             gap_size,
                                                             starting_ring=0,
                                                             return_segments=True)
    segment_pitch = segment_flat_to_flat + gap_size
    segment_centers = hcipy.make_hexagonal_grid(segment_pitch, num_rings, pointy_top=False)
    
    # Apply supersampling (required explicitly by config validation)
    supersampling_factor = telescope_config['supersampling_factor']
    aper = hcipy.evaluate_supersampled(aper, pupil_grid, supersampling_factor)
    segments = hcipy.evaluate_supersampled(segments, pupil_grid, supersampling_factor)
    
    # Create segmented deformable mirror
    hsm = hcipy.SegmentedDeformableMirror(segments)
    
    return {
        'pupil_grid': pupil_grid,
        'aper': aper,
        'segments': segments,
        'hsm': hsm,
        'wavelength': wavelength,
        'pupil_diameter': pupil_diameter,
        'segment_flat_to_flat': segment_flat_to_flat,
        'gap_size': gap_size,
        'num_rings': num_rings,
        'segment_point_to_point': segment_point_to_point,
        'segment_centers': segment_centers
    }

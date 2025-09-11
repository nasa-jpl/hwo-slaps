"""HCIPy telescope model setup for PSF generation.

This module contains functions to create the telescope aperture, segmented mirror,
and optical propagation setup.
"""

import numpy as np
import hcipy


def create_hcipy_telescope(config):
    """
    Create the HCIPy telescope model with aperture, segments, and propagator.
    
    This function sets up the complete optical system including the segmented
    aperture, coordinate grids, and Fraunhofer propagator for PSF generation.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary with telescope and simulation parameters.
        
    Returns
    -------
    telescope_data : dict
        Dictionary containing all telescope components:
        - pupil_grid: HCIPy grid for pupil plane
        - focal_grid: HCIPy grid for focal plane  
        - prop: Fraunhofer propagator
        - aper: Aperture function
        - segments: List of segment aperture functions
        - hsm: Segmented deformable mirror
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
    focal_length = telescope_config['focal_length']
    
    # Parameters for the simulation
    num_pix = sim_config['num_pix']
    wavelength = sim_config['wavelength']
    num_airy = sim_config['num_airy']
    sampling = sim_config['sampling']
    
    # HCIPy grids and propagator
    pupil_grid = hcipy.make_pupil_grid(dims=num_pix, diameter=pupil_diameter)
    
    focal_grid = hcipy.make_focal_grid(
        sampling,
        num_airy,
        pupil_diameter=pupil_diameter,
        reference_wavelength=wavelength,
        focal_length=focal_length,
    )
    
    prop = hcipy.FraunhoferPropagator(pupil_grid, focal_grid, focal_length)
    
    # Create segmented aperture
    aper, segments = hcipy.make_hexagonal_segmented_aperture(num_rings,
                                                             segment_flat_to_flat,
                                                             gap_size,
                                                             starting_ring=0,
                                                             return_segments=True)
    
    # Apply supersampling (required explicitly by config validation)
    supersampling_factor = telescope_config['supersampling_factor']
    aper = hcipy.evaluate_supersampled(aper, pupil_grid, supersampling_factor)
    segments = hcipy.evaluate_supersampled(segments, pupil_grid, supersampling_factor)
    
    # Create segmented deformable mirror
    hsm = hcipy.SegmentedDeformableMirror(segments)
    
    return {
        'pupil_grid': pupil_grid,
        'focal_grid': focal_grid,
        'prop': prop,
        'aper': aper,
        'segments': segments,
        'hsm': hsm,
        'wavelength': wavelength,
        'pupil_diameter': pupil_diameter,
        'segment_flat_to_flat': segment_flat_to_flat,
        'gap_size': gap_size,
        'num_rings': num_rings,
        'segment_point_to_point': segment_point_to_point
    }


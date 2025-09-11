"""
Utility functions and data structures for lensing system generation.

This module provides the core data structures and helper functions used
throughout the lensing module.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import autolens as al
from datetime import datetime


@dataclass
class LensingData:
    """
    Complete lensing system data structure.
    
    This class contains all products from the lensing system generation
    process in a unified structure with direct access to all key parameters.
    Information is organized by importance with primary data, system parameters,
    subhalo information, derived quantities, and provenance data.
    
    Parameters
    ----------
    image : `numpy.ndarray`
        The lensed source image as a 2D array.
    grid : `autolens.Grid2D`
        PyAutoLens grid object used for ray-tracing.
    tracer : `autolens.Tracer`
        PyAutoLens tracer object containing the full lensing system.
    pixel_scale : `float`
        Pixel scale in arcseconds per pixel.
    lens_redshift : `float`
        Redshift of the lens galaxy.
    source_redshift : `float`
        Redshift of the source galaxy.
    lens_einstein_radius : `float`
        Einstein radius of the main lens in arcseconds.
    cosmology_name : `str`
        Name of the cosmological model used (e.g., 'Planck15').
    subhalo_mass : `float`, optional
        Mass of the subhalo in solar masses. None if no subhalo present.
    subhalo_model : `str`, optional
        Model type for the subhalo ('PointMass', 'SIS', or 'NFW'). 
        None if no subhalo present.
    subhalo_position : `tuple` of `float`, optional
        Position of the subhalo as (y, x) in arcseconds relative to lens center.
        None if no subhalo present.
    subhalo_einstein_radius : `float`, optional
        Einstein radius of the subhalo in arcseconds. None if no subhalo present.
    subhalo_concentration : `float`, optional
        Concentration parameter for NFW subhalos. None if not NFW or no subhalo.
    lens_centre : `tuple` of `float`
        Centre position of the lens as (y, x) in arcseconds.
    lens_ellipticity : `tuple` of `float`
        Ellipticity components of the lens as (e1, e2).
    source_centre : `tuple` of `float`
        Centre position of the source as (y, x) in arcseconds.
    source_ellipticity : `tuple` of `float`
        Ellipticity components of the source as (e1, e2).
    source_intensity : `float`
        Intensity parameter of the source light profile.
    source_effective_radius : `float`
        Effective radius of the source in arcseconds.
    config : `dict`
        Complete configuration dictionary used to generate this lensing system.
    generation_timestamp : `str`
        ISO format timestamp of when the lensing system was generated.
        
    Notes
    -----
    This unified structure eliminates the need for nested dictionaries and
    provides direct access to all lensing system parameters. The subhalo
    parameters are None when no subhalo is present, which can be checked
    using the `has_subhalo` property.
    
    The coordinate convention follows PyAutoLens where the first coordinate
    is y (vertical) and the second is x (horizontal).
    
    Examples
    --------
    Access basic system information:
    
    >>> print(f"Lens at z={lensing_data.lens_redshift}")
    >>> print(f"Field of view: {lensing_data.field_of_view_arcsec}")
    >>> if lensing_data.has_subhalo:
    ...     print(f"Subhalo: {lensing_data.subhalo_mass:.1e} M_sun")
    """
    # === PRIMARY DATA ===
    image: np.ndarray
    grid: al.Grid2D
    tracer: al.Tracer
    
    # === SYSTEM PARAMETERS ===
    pixel_scale: float
    lens_redshift: float
    source_redshift: float
    lens_einstein_radius: float
    cosmology_name: str
    
    # === SUBHALO INFORMATION ===
    subhalo_mass: Optional[float] = None
    subhalo_model: Optional[str] = None
    subhalo_position: Optional[Tuple[float, float]] = None
    subhalo_einstein_radius: Optional[float] = None
    subhalo_concentration: Optional[float] = None
    
    # === GALAXY PARAMETERS ===
    lens_centre: Tuple[float, float] = (0.0, 0.0)
    lens_ellipticity: Tuple[float, float] = (0.0, 0.0)
    source_centre: Tuple[float, float] = (0.0, 0.0)
    source_ellipticity: Tuple[float, float] = (0.0, 0.0)
    source_intensity: float = 1.0
    source_effective_radius: float = 1.0
    
    # === PROVENANCE ===
    config: Optional[Dict] = None
    generation_timestamp: Optional[str] = None
    
    def __post_init__(self):
        """Set generation timestamp if not provided."""
        if self.generation_timestamp is None:
            self.generation_timestamp = datetime.now().isoformat()
    
    @property
    def grid_shape(self):
        """Shape of the coordinate grid as (y_pixels, x_pixels).
        
        Returns
        -------
        shape : `tuple` of `int`
            Grid shape as (y_pixels, x_pixels).
        """
        return self.grid.shape_native
    
    @property
    def field_of_view_arcsec(self):
        """Field of view in arcseconds as (y_fov, x_fov).
        
        Returns
        -------
        fov : `tuple` of `float`
            Field of view as (y_fov_arcsec, x_fov_arcsec).
        """
        y_fov = self.grid_shape[0] * self.pixel_scale
        x_fov = self.grid_shape[1] * self.pixel_scale
        return (y_fov, x_fov)
    
    @property
    def has_subhalo(self):
        """Whether this lensing system contains a subhalo.
        
        Returns
        -------
        has_subhalo : `bool`
            True if a subhalo is present, False otherwise.
        """
        return self.subhalo_mass is not None
    
    @property
    def total_flux(self):
        """Total flux in the lensed image.
        
        Returns
        -------
        flux : `float`
            Sum of all pixel values in the lensed image.
        """
        return float(np.sum(self.image))
    
    @property
    def peak_intensity(self):
        """Peak intensity in the lensed image.
        
        Returns
        -------
        intensity : `float`
            Maximum pixel value in the lensed image.
        """
        return float(np.max(self.image))


def get_einstein_ring_position(angle_deg, einstein_radius, offset_pixels=0, pixel_scale=0.05):
    """
    Calculate position on or near the Einstein ring.
    
    This function places a subhalo at a specified angle around the Einstein
    ring, with optional radial offset for placement studies.
    
    Parameters
    ----------
    angle_deg : `float`
        Angle in degrees (0-360) around the Einstein ring.
    einstein_radius : `float`
        Einstein radius of the lens in arcseconds.
    offset_pixels : `float`, optional
        Additional radial offset in pixels. Default is 0.
    pixel_scale : `float`, optional
        Pixel scale in arcseconds per pixel. Default is 0.05.
        
    Returns
    -------
    position : `tuple` of `float`
        Position (y, x) in arcseconds relative to lens center.
        
    Notes
    -----
    The position is returned in PyAutoLens convention where the first
    coordinate is y (vertical) and second is x (horizontal).
    
    Examples
    --------
    Place a subhalo at 45 degrees on the Einstein ring:
    
    >>> pos = get_einstein_ring_position(45.0, 1.6)
    >>> print(f"Subhalo position: y={pos[0]:.3f}, x={pos[1]:.3f}")
    """
    angle_rad = np.deg2rad(angle_deg)
    r = einstein_radius + offset_pixels * pixel_scale
    x = r * np.cos(angle_rad)
    y = r * np.sin(angle_rad)
    return (y, x)

def print_lensing_data_summary(lensing_data):
    """
    Print a comprehensive summary of lensing system data.
    
    Parameters
    ----------
    lensing_data : `LensingData`
        Lensing system data object.
        
    Examples
    --------
    Print a complete lensing system summary:
    
    >>> print_lensing_data_summary(lensing_data)
    === Lensing System Summary ===
    Pixel scale: 0.050000 arcsec/pixel
    ...
    """
    print("=== Lensing System Summary ===")
    
    # System parameters
    print(f"Pixel scale: {lensing_data.pixel_scale:.6f} arcsec/pixel")
    print(f"Grid shape: {lensing_data.grid_shape}")
    print(f"Field of view: {lensing_data.field_of_view_arcsec[0]:.2f} x {lensing_data.field_of_view_arcsec[1]:.2f} arcsec")
    print(f"Cosmology: {lensing_data.cosmology_name}")
    
    # Lens galaxy
    print(f"\n=== Lens Galaxy ===")
    print(f"Redshift: {lensing_data.lens_redshift}")
    print(f"Einstein radius: {lensing_data.lens_einstein_radius:.6f} arcsec")
    print(f"Centre: ({lensing_data.lens_centre[0]:.6f}, {lensing_data.lens_centre[1]:.6f}) arcsec")
    print(f"Ellipticity: e1={lensing_data.lens_ellipticity[0]:.3f}, e2={lensing_data.lens_ellipticity[1]:.3f}")
    
    # Source galaxy
    print(f"\n=== Source Galaxy ===")
    print(f"Redshift: {lensing_data.source_redshift}")
    print(f"Centre: ({lensing_data.source_centre[0]:.6f}, {lensing_data.source_centre[1]:.6f}) arcsec")
    print(f"Ellipticity: e1={lensing_data.source_ellipticity[0]:.3f}, e2={lensing_data.source_ellipticity[1]:.3f}")
    print(f"Effective radius: {lensing_data.source_effective_radius:.6f} arcsec")
    print(f"Intensity: {lensing_data.source_intensity:.6f}")
    
    # Subhalo information
    if lensing_data.has_subhalo:
        print(f"\n=== Subhalo Properties ===")
        print(f"Model: {lensing_data.subhalo_model}")
        print(f"Mass: {lensing_data.subhalo_mass:.2e} M_sun")
        print(f"Einstein radius: {lensing_data.subhalo_einstein_radius:.6f} arcsec")
        print(f"Position: ({lensing_data.subhalo_position[0]:.6f}, {lensing_data.subhalo_position[1]:.6f}) arcsec")
        
        # Distance from lens center
        dy = lensing_data.subhalo_position[0] - lensing_data.lens_centre[0]
        dx = lensing_data.subhalo_position[1] - lensing_data.lens_centre[1]
        distance = np.sqrt(dy**2 + dx**2)
        angle_deg = np.degrees(np.arctan2(dy, dx))
        print(f"Distance from lens: {distance:.6f} arcsec")
        print(f"Position angle: {angle_deg:.1f} degrees")
        
        # Distance relative to Einstein radius
        einstein_ratio = distance / lensing_data.lens_einstein_radius
        print(f"Distance/Einstein radius: {einstein_ratio:.3f}")
        
        # NFW-specific parameters
        if lensing_data.subhalo_model == 'NFW' and lensing_data.subhalo_concentration is not None:
            print(f"Concentration: {lensing_data.subhalo_concentration:.1f}")
    else:
        print(f"\n=== Subhalo Properties ===")
        print("No subhalo in this system")
    
    # Image statistics
    print(f"\n=== Image Statistics ===")
    print(f"Total flux: {lensing_data.total_flux:.6e}")
    print(f"Peak intensity: {lensing_data.peak_intensity:.6e}")
    print(f"Min intensity: {np.min(lensing_data.image):.6e}")
    print(f"Mean intensity: {np.mean(lensing_data.image):.6e}")
    print(f"RMS: {np.sqrt(np.mean(lensing_data.image**2)):.6e}")
    
    # Critical curves info (if available)
    if hasattr(lensing_data, 'critical_curves'):
        print(f"\n=== Critical Curves ===")
        print("Critical curve computation available")
    
    # Provenance
    if hasattr(lensing_data, 'generation_timestamp'):
        print(f"\nGenerated: {lensing_data.generation_timestamp}")
    if hasattr(lensing_data, 'config') and lensing_data.config:
        if 'run_name' in lensing_data.config:
            print(f"Run name: {lensing_data.config['run_name']}")
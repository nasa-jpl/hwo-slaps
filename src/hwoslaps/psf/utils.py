"""PSF data structures and utility functions.

This module contains data structures and utility functions for PSF generation,
including conversion to PyAutoLens kernels.

The main data structure is PSFData, which provides a unified interface to all
PSF system parameters, quality metrics, and aberration information in a flat
structure for easy access and analysis.
"""

import numpy as np
import os
from dataclasses import dataclass
from typing import Dict, Optional, Any, List
import autolens as al
from ..constants import ARCSEC_PER_RAD
from datetime import datetime


@dataclass
class PSFData:
    """Complete PSF system data structure.
    
    This class contains all products from PSF generation in a unified structure
    with direct access to all key parameters, quality metrics, and aberration
    information. Information is organized by importance with primary data,
    system parameters, telescope geometry, pre-computed quality metrics,
    aberration summaries, and provenance data.
    
    Parameters
    ----------
    psf : `hcipy.Field`
        The PSF field from HCIPy containing intensity and coordinate information.
    wavefront : `hcipy.Wavefront`
        The wavefront object containing electric field and phase information.
    telescope_data : `dict`
        Dictionary containing HCIPy telescope components (grids, propagators, etc).
    kernel : `autolens.Kernel2D`
        Pre-converted PyAutoLens kernel for immediate use in lensing simulations.
    kernel_pixel_scale : `float`
        Pixel scale of the kernel if different from PSF pixel scale.
    wavelength_nm : `float`
        Observation wavelength in nanometers.
    pupil_diameter_m : `float`
        Primary mirror diameter in meters.
    focal_length_m : `float`
        Telescope focal length in meters.
    pixel_scale_arcsec : `float`
        PSF pixel scale in arcseconds per pixel.
    sampling_factor : `float`
        Oversampling factor (pixels per lambda/D).
    requested_sampling_factor : `float`
        User-provided sampling value from configuration.
    used_sampling_factor : `float`
        Auto-calculated sampling value for integer subsampling.
    integer_subsampling_factor : `int`
        The integer factor for detector downsampling.
    num_segments : `int`
        Number of mirror segments.
    segment_flat_to_flat_m : `float`
        Flat-to-flat distance of hexagonal segments in meters.
    segment_point_to_point_m : `float`
        Point-to-point distance of hexagonal segments in meters.
    gap_size_m : `float`
        Gap size between segments in meters.
    num_rings : `int`
        Number of hexagonal rings (excluding central segment).
    fwhm_arcsec : `float`, optional
        Full Width at Half Maximum in arcseconds.
    fwhm_mas : `float`, optional
        Full Width at Half Maximum in milliarcseconds.
    strehl_ratio : `float`, optional
        Strehl ratio (peak intensity relative to perfect PSF).
    peak_intensity : `float`
        Peak intensity value of the PSF.
    total_flux : `float`
        Total integrated flux of the PSF.
    encircled_energy_50_arcsec : `float`, optional
        Radius containing 50% of PSF energy in arcseconds.
    total_rms_nm : `float`
        Total RMS wavefront error in nanometers.
    segment_piston_rms_nm : `float`
        RMS of segment piston errors in nanometers.
    segment_tiptilt_rms_urad : `float`
        RMS of segment tip/tilt errors in microradians.
    global_zernike_rms_nm : `float`
        RMS of global Zernike aberrations in nanometers.
    has_segment_pistons : `bool`
        Whether segment piston aberrations are present.
    has_segment_tiptilts : `bool`
        Whether segment tip/tilt aberrations are present.
    has_segment_hexikes : `bool`
        Whether segment-level hexike aberrations are present.
    has_global_zernikes : `bool`
        Whether global Zernike aberrations are present.
    phase_screens : `dict`, optional
        Dictionary of phase screens by type.
    phase_screen_types : `list` of `str`, optional
        List of phase screen types present.
    aberrations : `dict`, optional
        Complete aberration configuration used.
    config : `dict`, optional
        Full configuration dictionary used to generate the PSF.
    generation_timestamp : `str`
        ISO format timestamp of when the PSF was generated.
        
    Notes
    -----
    This data structure provides a comprehensive view of the PSF system with
    all relevant parameters, quality metrics, and aberration information in
    a single flat structure for easy access and analysis. The kernel field
    provides immediate access to PyAutoLens format for convolution operations.
    """
    # Primary data.
    psf: Any  # hcipy.Field - The PSF field from HCIPy.
    wavefront: Any  # hcipy.Wavefront - The wavefront object.
    telescope_data: Dict  # Dictionary with HCIPy telescope components.
    kernel: al.Kernel2D  # Pre-converted PyAutoLens kernel.
    kernel_pixel_scale: float  # Kernel pixel scale if different from PSF.
    
    # System parameters.
    wavelength_nm: float
    pupil_diameter_m: float
    focal_length_m: float
    pixel_scale_arcsec: float
    sampling_factor: float
    requested_sampling_factor: float  # User-provided value from config.
    used_sampling_factor: float       # Auto-calculated value for integer subsampling.
    integer_subsampling_factor: int   # The integer factor for the detector.
    num_segments: int
    
    # Telescope geometry.
    segment_flat_to_flat_m: float
    segment_point_to_point_m: float
    gap_size_m: float
    num_rings: int
    
    # Pre-computed quality metrics.
    fwhm_arcsec: Optional[float] = None
    fwhm_mas: Optional[float] = None
    strehl_ratio: Optional[float] = None
    peak_intensity: float = 0.0
    total_flux: float = 0.0
    encircled_energy_50_arcsec: Optional[float] = None
    
    # Aberration summary.
    total_rms_nm: float = 0.0
    segment_piston_rms_nm: float = 0.0
    segment_tiptilt_rms_urad: float = 0.0
    global_zernike_rms_nm: float = 0.0
    
    # Aberration flags.
    has_segment_pistons: bool = False
    has_segment_tiptilts: bool = False
    has_segment_hexikes: bool = False
    has_global_zernikes: bool = False
    
    # Complex data and diagnostics.
    phase_screens: Optional[Dict] = None
    phase_screen_types: Optional[List[str]] = None
    aberrations: Optional[Dict] = None
    
    # Provenance.
    config: Optional[Dict] = None
    generation_timestamp: str = ""
    # Saved artifacts (optional paths to on-disk products)
    highres_psf_npy_path: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization to compute derived values."""
        # Set timestamp if not provided.
        if not self.generation_timestamp:
            self.generation_timestamp = datetime.now().isoformat()
            
        # Extract phase screen types if phase screens provided.
        if self.phase_screens and not self.phase_screen_types:
            self.phase_screen_types = list(self.phase_screens.keys())
    
    # Derived properties.
    @property
    def wavelength_m(self):
        """Wavelength in meters.
        
        Returns
        -------
        wavelength : `float`
            Wavelength in meters.
        """
        return self.wavelength_nm * 1e-9
    
    @property
    def diffraction_limit_arcsec(self):
        """Theoretical diffraction limit (lambda/D) in arcseconds.
        
        Returns
        -------
        diffraction_limit : `float`
            Diffraction limit in arcseconds.
        """
        return (self.wavelength_m / self.pupil_diameter_m) * ARCSEC_PER_RAD
    
    @property
    def airy_disk_diameter_arcsec(self):
        """Airy disk diameter (2.44*lambda/D) in arcseconds.
        
        Returns
        -------
        airy_diameter : `float`
            Airy disk diameter in arcseconds.
        """
        return 2.44 * self.diffraction_limit_arcsec
    
    @property
    def f_number(self):
        """Telescope F-number (f/D).
        
        Returns
        -------
        f_number : `float`
            F-number of the telescope.
        """
        return self.focal_length_m / self.pupil_diameter_m
    
    @property
    def angular_resolution_mas(self):
        """Angular resolution in milliarcseconds.
        
        Returns
        -------
        resolution : `float`
            Angular resolution in milliarcseconds.
        """
        return self.diffraction_limit_arcsec * 1000
    
    @property
    def is_diffraction_limited(self):
        """Whether PSF is close to diffraction limit (Strehl > 0.8).
        
        Returns
        -------
        is_diffraction_limited : `bool` or `None`
            True if Strehl ratio > 0.8, False otherwise, None if unknown.
        """
        if self.strehl_ratio is None:
            return None
        return self.strehl_ratio > 0.8
    
    @property
    def quality_grade(self):
        """Qualitative PSF assessment based on Strehl ratio.
        
        Returns
        -------
        grade : `str`
            Quality grade: 'Excellent', 'Good', 'Fair', 'Poor', or 'Unknown'.
        """
        if self.strehl_ratio is None:
            return "Unknown"
        elif self.strehl_ratio > 0.9:
            return "Excellent"
        elif self.strehl_ratio > 0.7:
            return "Good"
        elif self.strehl_ratio > 0.5:
            return "Fair"
        else:
            return "Poor"
    
    @property
    def aberration_budget_breakdown(self):
        """Breakdown of aberration types present.
        
        Returns
        -------
        breakdown : `dict`
            Dictionary indicating which aberration types are present.
            
        Notes
        -----
        Aberrations can interfere constructively or destructively, so the
        total RMS cannot be accurately decomposed into individual contributions
        without the full phase information. This method simply indicates
        which aberration types are present rather than attempting to calculate
        incorrect percentages.
        """
        breakdown = {}
        if self.has_segment_pistons:
            breakdown['segment_pistons'] = f"{self.segment_piston_rms_nm:.1f} nm RMS"
        if self.has_segment_tiptilts:
            breakdown['segment_tiptilts'] = f"{self.segment_tiptilt_rms_urad:.1f} μrad RMS"
        if self.has_segment_hexikes:
            breakdown['segment_hexikes'] = "Present"
        if self.has_global_zernikes:
            breakdown['global_zernikes'] = f"{self.global_zernike_rms_nm:.1f} nm RMS"
            
        return breakdown
    
    @property
    def has_aberrations(self):
        """Whether any aberrations are present.
        
        Returns
        -------
        has_aberrations : `bool`
            True if any aberrations are applied, False otherwise.
        """
        return (self.has_segment_pistons or self.has_segment_tiptilts or 
                self.has_segment_hexikes or self.has_global_zernikes)

    @property
    def has_saved_highres_psf(self) -> bool:
        """Return True if a high-res PSF .npy file path is recorded and exists."""
        return bool(self.highres_psf_npy_path) and os.path.exists(self.highres_psf_npy_path)


def print_psf_data_summary(psf_data):
    """Print a comprehensive summary of PSF data.
    
    Parameters
    ----------
    psf_data : `PSFData`
        PSF data object.
        
    Examples
    --------
    Print a complete PSF system summary:
    
    >>> print_psf_data_summary(psf_data)
    === PSF System Summary ===
    Telescope diameter: 6.0 m
    ...
    """
    print("=== PSF System Summary ===")
    
    # System parameters.
    print(f"Telescope diameter: {psf_data.pupil_diameter_m:.1f} m")
    print(f"Number of segments: {psf_data.num_segments}")
    print(f"Wavelength: {psf_data.wavelength_nm:.0f} nm")
    print(f"F-number: {psf_data.f_number:.1f}")
    print(f"Pixel scale: {psf_data.pixel_scale_arcsec:.6f} arcsec/pixel")
    
    # Diverging-path sampling information.
    print(f"\n=== Diverging-Path Sampling ===")
    print(f"Requested sampling: {psf_data.requested_sampling_factor:.2f} pixels/λ/D")
    print(f"Auto-adjusted sampling: {psf_data.used_sampling_factor:.3f} pixels/λ/D")
    print(f"Integer subsampling factor: {psf_data.integer_subsampling_factor}")
    print(f"High-res pixel scale: {psf_data.pixel_scale_arcsec*1000:.3f} mas")
    print(f"Detector pixel scale: {psf_data.kernel_pixel_scale*1000:.3f} mas")
    
    # Physical scales.
    print(f"\n=== Physical Scales ===")
    print(f"Diffraction limit: {psf_data.diffraction_limit_arcsec:.6f} arcsec")
    print(f"Angular resolution: {psf_data.angular_resolution_mas:.1f} mas") 
    print(f"Airy disk diameter: {psf_data.airy_disk_diameter_arcsec:.6f} arcsec")
    
    # PSF quality.
    print(f"\n=== PSF Quality ===")
    if psf_data.fwhm_arcsec is not None:
        print(f"FWHM: {psf_data.fwhm_arcsec:.6f} arcsec ({psf_data.fwhm_mas:.1f} mas)")
    if psf_data.strehl_ratio is not None:
        print(f"Strehl ratio: {psf_data.strehl_ratio:.3f}")
        print(f"Quality grade: {psf_data.quality_grade}")
        print(f"Diffraction limited: {psf_data.is_diffraction_limited}")
    print(f"Peak intensity: {psf_data.peak_intensity:.6e}")
    print(f"Total flux: {psf_data.total_flux:.6e}")
    
    # Detailed kernel statistics.
    print(f"\n=== PyAutoLens Kernel Statistics ===")
    print(f"Kernel shape: {psf_data.kernel.shape_native}")
    print(f"Kernel pixel scale: {psf_data.kernel_pixel_scale:.6f} arcsec/pixel")
    
    # Calculate kernel statistics
    kernel_array = psf_data.kernel.native
    kernel_sum = np.sum(kernel_array)
    kernel_max = np.max(kernel_array)
    kernel_min = np.min(kernel_array)
    
    print(f"Total flux: {kernel_sum:.6f}")
    print(f"Peak value: {kernel_max:.6e}")
    print(f"Min value: {kernel_min:.6e}")
    
    # Aberrations.
    if psf_data.has_aberrations:
        print(f"\n=== Aberration Summary ===")
        print(f"Total RMS: {psf_data.total_rms_nm:.1f} nm")
        print(f"Has segment pistons: {psf_data.has_segment_pistons}")
        print(f"Has segment tip/tilts: {psf_data.has_segment_tiptilts}")
        print(f"Has segment hexikes: {psf_data.has_segment_hexikes}")
        print(f"Has global Zernikes: {psf_data.has_global_zernikes}")
        
        breakdown = psf_data.aberration_budget_breakdown
        if breakdown:
            print("Aberration types present:")
            for source, value in breakdown.items():
                print(f"  {source}: {value}")
    else:
        print(f"\n=== Aberrations ===")
        print("Perfect PSF (no aberrations)")
    
    # Phase screens.
    if psf_data.phase_screen_types:
        print(f"\nPhase screens: {', '.join(psf_data.phase_screen_types)}")
    
    # Provenance.
    print(f"\nGenerated: {psf_data.generation_timestamp}")
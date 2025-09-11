"""Utility classes and functions for observation simulation.

This module provides the ObservationData class and related utilities
for managing observation data and metadata.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
import numpy as np
import autolens as al


@dataclass
class ObservationData:
    """Complete observation data structure with unified access.
    
    This class provides a comprehensive interface to all observation
    data, including the observed image, noise map, and metadata.
    
    Attributes
    ----------
    imaging : `al.Imaging`
        PyAutoLens imaging dataset with data, noise_map, and PSF.
    noiseless_source_eps : `np.ndarray`
        Noiseless PSF-convolved source image in e-/s.
    noise_components : `dict`
        Dictionary containing individual noise components.
    config : `dict`
        Configuration used to generate the observation.
    metadata : `dict`
        Additional metadata including timestamps and run info.
    """
    
    imaging: al.Imaging
    noiseless_source_eps: np.ndarray
    noise_components: Dict[str, Any]
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def data(self) -> al.Array2D:
        """Observed data array in ADU."""
        return self.imaging.data
    
    @property
    def noise_map(self) -> al.Array2D:
        """Noise map array in ADU."""
        return self.imaging.noise_map
    
    @property
    def psf(self) -> al.Kernel2D:
        """PSF kernel used for convolution."""
        return self.imaging.psf
    
    @property
    def signal_to_noise_map(self) -> al.Array2D:
        """Signal-to-noise ratio map."""
        return self.imaging.signal_to_noise_map
    
    @property
    def exposure_time(self) -> float:
        """Exposure time in seconds."""
        return self.metadata['exposure_time']
    
    @property
    def pixel_scale(self) -> float:
        """Pixel scale in arcsec/pixel."""
        return self.metadata['pixel_scale']
    
    @property
    def detector_config(self) -> Dict[str, float]:
        """Detector configuration parameters."""
        return self.metadata['detector']
    
    @property
    def gain(self) -> float:
        """Detector gain in e-/ADU."""
        return self.detector_config['gain']
    
    @property
    def read_noise(self) -> float:
        """Read noise in e-/pixel."""
        return self.detector_config['read_noise']
    
    @property
    def dark_current(self) -> float:
        """Dark current in e-/pixel/s."""
        return self.detector_config['dark_current']
    
    @property
    def sky_background(self) -> float:
        """Sky background in e-/pixel/s."""
        return self.detector_config['sky_background']
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of the observation arrays."""
        return self.data.shape_native
    
    @property
    def peak_snr(self) -> float:
        """Peak signal-to-noise ratio."""
        return float(np.max(self.signal_to_noise_map))
    
    @property
    def total_flux_adu(self) -> float:
        """Total flux in ADU."""
        return float(np.sum(self.data))
    
    @property
    def total_flux_electrons(self) -> float:
        """Total flux in electrons."""
        return self.total_flux_adu * self.gain
    
    @property
    def sky_electrons_per_pixel(self) -> float:
        """Sky background in electrons per pixel."""
        return self.sky_background * self.exposure_time
    
    @property
    def dark_electrons_per_pixel(self) -> float:
        """Dark current in electrons per pixel."""
        return self.dark_current * self.exposure_time
    
    @property
    def source_electrons(self) -> np.ndarray:
        """Source-only contribution in electrons."""
        return self.noiseless_source_eps * self.exposure_time
    
    def get_noise_summary(self) -> Dict[str, float]:
        """Get summary of noise contributions.
        
        Returns
        -------
        summary : `dict`
            Dictionary with noise statistics by region.
        """
        # Define regions by source signal level
        source_eps = self.noiseless_source_eps
        very_dark = source_eps < 0.001
        background_dominated = (source_eps >= 0.001) & (source_eps < 0.1)
        bright = source_eps >= 0.1
        
        # Calculate mean noise in each region
        noise_e = self.noise_map.native * self.gain
        
        return {
            'dark_regions_e': float(np.mean(noise_e[very_dark])) if np.any(very_dark) else 0.0,
            'background_regions_e': float(np.mean(noise_e[background_dominated])) if np.any(background_dominated) else 0.0,
            'bright_regions_e': float(np.mean(noise_e[bright])) if np.any(bright) else 0.0,
            'mean_noise_e': float(np.mean(noise_e)),
            'min_noise_e': float(np.min(noise_e)),
            'max_noise_e': float(np.max(noise_e))
        }


def print_observation_summary(obs_data: ObservationData) -> None:
    """Print a formatted summary of the observation data.
    
    Parameters
    ----------
    obs_data : `ObservationData`
        The observation data to summarize.
    """
    print("\n=== Observation Summary ===")
    print(f"Image shape: {obs_data.shape}")
    print(f"Pixel scale: {obs_data.pixel_scale} arcsec/pixel")
    print(f"Exposure time: {obs_data.exposure_time} seconds")
    
    print("\n=== Detector Configuration ===")
    print(f"Gain: {obs_data.gain} e-/ADU")
    print(f"Read noise: {obs_data.read_noise} e-/pixel")
    print(f"Dark current: {obs_data.dark_current} e-/pixel/s")
    print(f"Sky background: {obs_data.sky_background} e-/pixel/s")
    
    print("\n=== Signal Statistics ===")
    print(f"Peak SNR: {obs_data.peak_snr:.1f}")
    print(f"Total flux: {obs_data.total_flux_adu:.2e} ADU")
    print(f"Total flux: {obs_data.total_flux_electrons:.2e} electrons")
    print(f"Max source signal: {np.max(obs_data.source_electrons):.0f} e-")
    
    print("\n=== Noise Analysis ===")
    noise_summary = obs_data.get_noise_summary()
    print(f"Dark regions: {noise_summary['dark_regions_e']:.1f} e-")
    print(f"Background-dominated: {noise_summary['background_regions_e']:.1f} e-")
    print(f"Bright regions: {noise_summary['bright_regions_e']:.1f} e-")
    print(f"Noise range: {noise_summary['min_noise_e']:.1f} - {noise_summary['max_noise_e']:.1f} e-")
    
    # Theoretical expectations
    sky_e = obs_data.sky_electrons_per_pixel
    dark_e = obs_data.dark_electrons_per_pixel
    read_e = obs_data.read_noise
    theoretical_dark = np.sqrt(sky_e + dark_e + read_e**2)
    print(f"\nTheoretical dark region noise: {theoretical_dark:.1f} e-")
    
    print(f"\nGenerated: {obs_data.metadata['generated']}")
    if 'run_name' in obs_data.metadata:
        print(f"Run name: {obs_data.metadata['run_name']}")
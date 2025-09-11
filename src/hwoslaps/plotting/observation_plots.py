"""Plotting functions for observation simulation visualization.

This module provides visualization tools for observation data.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from pathlib import Path

from ..lensing.utils import LensingData
from ..psf.utils import PSFData
from ..observation.utils import ObservationData
from .registry import plot_function


@plot_function(module='observation', description="2x2 observation process: lensed image, PSF, noise, final observation")
def plot_observation_comparison(
    lensing_data: LensingData,
    psf_data: PSFData,
    obs_data: ObservationData,
    figsize: Tuple[float, float] = (12, 12),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Create a 2x2 comparison plot of the observation process.
    
    Shows:
    - Original lensed image
    - PSF
    - Noise map
    - Final observed image
    
    Parameters
    ----------
    lensing_data : `LensingData`
        The lensing system data.
    psf_data : `PSFData`
        The PSF system data.
    obs_data : `ObservationData`
        The observation data.
    figsize : `tuple`, optional
        Figure size as (width, height).
    save_path : `str`, optional
        Path to save the figure.
        
    Returns
    -------
    fig : `matplotlib.figure.Figure`
        The created figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Original lensed image
    im1 = axes[0, 0].imshow(
        lensing_data.image,
        origin='lower',
        cmap='viridis'
    )
    axes[0, 0].set_title('Original Lensed Image')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
    
    # 2. PSF (log scale)
    psf_log = np.log10(psf_data.kernel.native + 1e-10)
    im2 = axes[0, 1].imshow(
        psf_log,
        origin='lower',
        cmap='hot'
    )
    if psf_data.fwhm_arcsec is not None:
        axes[0, 1].set_title(f'PSF (log10, FWHM={psf_data.fwhm_arcsec:.3f}")')
    else:
        axes[0, 1].set_title('PSF (log10)')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
    
    # 3. Noise map
    im3 = axes[1, 0].imshow(
        obs_data.noise_map.native,
        origin='lower',
        cmap='plasma'
    )
    axes[1, 0].set_title('Noise Map (ADU)')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
    
    # 4. Final observed image
    im4 = axes[1, 1].imshow(
        obs_data.data.native,
        origin='lower',
        cmap='viridis'
    )
    axes[1, 1].set_title('Mock Observation')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
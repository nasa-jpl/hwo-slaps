"""Generator functions for creating realistic observations.

This module implements the main observation simulation pipeline, including
PSF convolution and realistic detector noise modeling.
"""

import numpy as np
import autolens as al
from datetime import datetime
from typing import Dict, Optional

from ..lensing.utils import LensingData
from ..psf.utils import PSFData
from .utils import ObservationData
from .noise_models import (
    apply_detector_noise,
    create_noise_map,
)


def generate_observation(
    lensing_data: LensingData,
    psf_data: PSFData,
    observation_config: Optional[Dict] = None,
    full_config: Optional[Dict] = None
) -> ObservationData:
    """Generate a realistic observation from lensing and PSF data.
    
    This function takes a lensing system and PSF, applies convolution,
    and adds realistic detector noise to create a mock observation.
    
    Parameters
    ----------
    lensing_data : `LensingData`
        The lensing system data from Module 1.
    psf_data : `PSFData`
        The PSF system data from Module 2.
    observation_config : `dict`, optional
        Observation-specific configuration. If None, uses defaults.
    full_config : `dict`, optional
        Full configuration dictionary containing all module configs.
        
    Returns
    -------
    observation_data : `ObservationData`
        Complete observation data including convolved image, noise,
        and all metadata.
        
    Notes
    -----
    The observation simulation follows a two-step process:
    1. PSF convolution using PyAutoLens SimulatorImaging (noiseless)
    2. Application of realistic detector noise model
    
    The noise model includes:
    - Poisson noise (photon shot noise)
    - Read noise
    - Dark current
    - Sky background
    """
    # Strict: observation_config must be provided by pipeline validation
    if observation_config is None:
        raise ValueError("observation_config must be provided explicitly (no defaults)")
    
    # Extract parameters
    exposure_time = observation_config['exposure_time']
    detector_config = observation_config['detector']
    
    # Extract global seed from full_config
    global_seed = full_config['global_seed']
    noise_seed = global_seed
    
    # Ensure PSF kernel has odd dimensions (required by PyAutoLens)
    psf_kernel = _ensure_odd_kernel(psf_data.kernel)

    # Assert pixel scale consistency between PSF kernel and lensing image
    # This ensures physically meaningful convolution without implicit resampling.
    if hasattr(psf_data, "kernel_pixel_scale") and psf_data.kernel_pixel_scale is not None:
        if not np.isclose(psf_data.kernel_pixel_scale, lensing_data.pixel_scale, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"Pixel scale mismatch: PSF kernel_pixel_scale={psf_data.kernel_pixel_scale} arcsec/pixel "
                f"!= lensing pixel_scale={lensing_data.pixel_scale} arcsec/pixel."
            )
    
    # Convert lensed image to PyAutoLens Array2D format
    mask = al.Mask2D.all_false(
        shape_native=lensing_data.image.shape,
        pixel_scales=lensing_data.pixel_scale
    )
    lensed_image = al.Array2D(
        values=lensing_data.image,
        mask=mask
    )
    
    # Step 1: Generate noiseless PSF-convolved image
    # Use SimulatorImaging with no noise to get pure convolution
    simulator_noiseless = al.SimulatorImaging(
        exposure_time=exposure_time,
        psf=psf_kernel,
        background_sky_level=0.0,  # No background yet
        normalize_psf=False,
        add_poisson_noise_to_data=False,
        noise_seed=noise_seed
    )
    
    # Get noiseless but PSF-convolved image (units are electrons-per-second)
    noiseless_dataset = simulator_noiseless.via_image_from(image=lensed_image)
    source_only_eps = noiseless_dataset.data.native  # e-/s
    
    # Step 2: Apply realistic detector noise
    # This includes Poisson noise, read noise, dark current, and sky background
    final_image_adu, components = apply_detector_noise(
        source_eps=source_only_eps,
        exposure_time=exposure_time,
        detector_config=detector_config,
        seed=noise_seed
    )
    
    # Step 3: Create proper noise map
    # The noise map represents total uncertainty in each pixel
    noise_map_adu = create_noise_map(
        source_eps=source_only_eps,
        exposure_time=exposure_time,
        detector_config=detector_config
    )
    
    # Create PyAutoLens arrays for the final data
    data = al.Array2D(values=final_image_adu, mask=mask)
    noise_map = al.Array2D(values=noise_map_adu, mask=mask)
    
    # Create the imaging dataset
    imaging_dataset = al.Imaging(
        data=data,
        noise_map=noise_map,
        psf=psf_kernel
    )
    
    # Create metadata dictionary
    metadata = {
        'generated': datetime.now().isoformat(),
        'lensing_run': lensing_data.config.get('run_name') if lensing_data.config else None,
        'psf_run': psf_data.config.get('run_name') if psf_data.config else None,
        'exposure_time': exposure_time,
        'detector': detector_config.copy(),
        'noise_seed': noise_seed,
        'pixel_scale': lensing_data.pixel_scale,
        'field_of_view': lensing_data.field_of_view_arcsec
    }
    
    # Add run name if provided
    metadata['run_name'] = full_config['run_name']
    
    # Create and return ObservationData object
    return ObservationData(
        imaging=imaging_dataset,
        noiseless_source_eps=source_only_eps,
        noise_components=components,
        config=observation_config.copy(),
        metadata=metadata
    )


def _ensure_odd_kernel(kernel: al.Kernel2D) -> al.Kernel2D:
    """Validate the PSF kernel for observation convolution.
    
    Parameters
    ----------
    kernel : `al.Kernel2D`
        Input PSF kernel.
        
    Returns
    -------
    kernel : `al.Kernel2D`
        Validated PSF kernel.

    Raises
    ------
    ValueError
        Raised when the kernel support or flux normalization is invalid.
    """
    kernel_array = np.asarray(kernel.native, dtype=float)
    if kernel_array.ndim != 2:
        raise ValueError("PSF kernel must be a two-dimensional array")
    if kernel_array.shape[0] % 2 == 0 or kernel_array.shape[1] % 2 == 0:
        raise ValueError("PSF kernel must have odd dimensions")
    if not np.all(np.isfinite(kernel_array)):
        raise ValueError("PSF kernel values must be finite")
    if np.any(kernel_array < 0.0):
        raise ValueError("PSF kernel values must be non-negative")

    kernel_sum = float(np.sum(kernel_array))
    if not np.isclose(kernel_sum, 1.0, rtol=0.0, atol=1e-10):
        raise ValueError("PSF kernel must be normalized to unit flux")

    return kernel

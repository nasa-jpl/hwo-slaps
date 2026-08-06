"""Generator functions for creating realistic observations.

This module implements the main observation simulation pipeline, including
PSF convolution and realistic detector noise modeling.
"""

from copy import deepcopy
from datetime import datetime
from typing import Dict, Optional

import autolens as al
import numpy as np

from ..lensing.utils import LensingData
from ..psf.utils import (
    PSFData,
    make_pyauto_convolver,
    pyauto_kernel_native,
)
from .noise_models import (
    apply_detector_noise,
    create_noise_map,
)
from .utils import ObservationData


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
    1. Noiseless PSF convolution using the PyAutoLens convolver
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
    full_config = _validate_full_config(full_config)

    # Extract parameters
    exposure_time = observation_config['exposure_time']
    throughput = float(observation_config['throughput'])
    detector_config = observation_config['detector']

    # Extract global seed from full_config
    global_seed = full_config['global_seed']
    noise_seed = global_seed

    # Ensure PSF kernel has odd dimensions (required by PyAutoLens)
    psf_kernel = _ensure_odd_kernel(psf_data.kernel)
    psf_convolver = make_pyauto_convolver(psf_kernel)

    # Assert pixel scale consistency between PSF kernel and lensing image
    # Keep convolution physically meaningful without implicit resampling.
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

    # Step 1: Generate the noiseless PSF-convolved image by direct
    # convolution. SimulatorImaging is not used here because it evaluates
    # Poisson noise internally even with noise disabled, which rejects the
    # roundoff-negative pixels clamped below.
    # The convolved image is in electrons-per-second. The end-to-end system
    # throughput scales the source flux only; sky background and dark
    # current are configured as detected rates already.
    convolved_eps = psf_convolver.convolved_image_from(
        image=lensed_image,
        blurring_image=None,
    )
    source_only_eps = np.asarray(convolved_eps.native) * throughput  # e-/s

    # Convolution of the non-negative image with the non-negative kernel is
    # non-negative, but FFT evaluation leaves epsilon-scale negatives where
    # a compactly supported source is exactly zero. Anything beyond
    # roundoff scale is a genuine input error and stays loud. The stored
    # noiseless rate keeps the raw convolution output (bit-identical to
    # downstream re-convolutions of the same scene); only the noise draw
    # and noise map below use the clamped copy, since Poisson rates must
    # be non-negative.
    roundoff_tol = 1.0e-10 * float(np.max(np.abs(source_only_eps), initial=0.0))
    min_eps = float(np.min(source_only_eps, initial=0.0))
    if min_eps < -roundoff_tol:
        raise ValueError(
            "PSF-convolved source image has negative values beyond FFT "
            f"roundoff scale: min {min_eps} e-/s against tolerance "
            f"{roundoff_tol} e-/s"
        )
    source_eps_for_noise = np.maximum(source_only_eps, 0.0)

    # Step 2: Apply realistic detector noise
    # This includes Poisson noise, read noise, dark current, and sky background
    final_image_adu, components = apply_detector_noise(
        source_eps=source_eps_for_noise,
        exposure_time=exposure_time,
        detector_config=detector_config,
        seed=noise_seed
    )

    # Step 3: Create proper noise map
    # The noise map represents total uncertainty in each pixel
    noise_map_adu = create_noise_map(
        source_eps=source_eps_for_noise,
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
        psf=psf_convolver
    )

    # Create metadata dictionary
    metadata = {
        'generated': datetime.now().isoformat(),
        'lensing_run': lensing_data.config.get('run_name') if lensing_data.config else None,
        'psf_run': psf_data.config.get('run_name') if psf_data.config else None,
        'exposure_time': exposure_time,
        'throughput': throughput,
        'detector': deepcopy(detector_config),
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
        config=deepcopy(observation_config),
        metadata=metadata
    )


def _validate_full_config(full_config: Optional[Dict]) -> Dict:
    """Validate observation-level full configuration requirements.

    Parameters
    ----------
    full_config : `dict`
        Full pipeline configuration.

    Returns
    -------
    full_config : `dict`
        Validated full configuration.

    Raises
    ------
    ValueError
        Raised when required global provenance or seed values are missing.
    """
    if not isinstance(full_config, dict):
        raise ValueError("full_config must be a dict for generate_observation")
    if 'global_seed' not in full_config:
        raise ValueError("Missing required key 'global_seed' in full_config")
    global_seed = full_config['global_seed']
    if isinstance(global_seed, bool) or not isinstance(global_seed, int):
        raise ValueError("full_config.global_seed must be an int")
    if 'run_name' not in full_config:
        raise ValueError("Missing required key 'run_name' in full_config")
    run_name = full_config['run_name']
    if not isinstance(run_name, str) or not run_name:
        raise ValueError("full_config.run_name must be a non-empty string")
    return full_config


def _ensure_odd_kernel(kernel):
    """Validate the PSF kernel for observation convolution.

    Parameters
    ----------
    kernel : `object`
        Input PSF kernel.

    Returns
    -------
    kernel : `object`
        Validated PSF kernel.

    Raises
    ------
    ValueError
        Raised when the kernel support or flux normalization is invalid.
    """
    kernel_array = pyauto_kernel_native(kernel)
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

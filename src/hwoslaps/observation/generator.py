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
    DetectorPSF,
    PSFData,
    make_pyauto_convolver,
    make_pyauto_kernel,
    pyauto_kernel_native,
    pyauto_kernel_pixel_scales,
    validate_detector_kernel,
)
from .noise_models import (
    apply_detector_noise,
    create_noise_map,
)
from .forward import convolve_source_rate, predict_observation
from .utils import ObservationData


def generate_observation(
    lensing_data: LensingData,
    psf_data: PSFData | DetectorPSF,
    observation_config: Optional[Dict] = None,
    full_config: Optional[Dict] = None,
    *,
    noise_seed: Optional[int] = None,
    run_name: Optional[str] = None,
    sample_noise: bool = True,
) -> ObservationData:
    """Generate a realistic observation from lensing and PSF data.

    This function takes a lensing system and PSF, applies convolution,
    and adds realistic detector noise to create a mock observation.

    Parameters
    ----------
    lensing_data : `LensingData`
        The lensing system data from Module 1.
    psf_data : `PSFData` or `DetectorPSF`
        Generated optical PSF products or an externally supplied detector kernel.
    observation_config : `dict`, optional
        Required observation configuration; no telescope defaults are inferred.
    full_config : `dict`, optional
        Full configuration dictionary. Supplies ``global_seed`` and ``run_name``
        when their explicit counterparts are omitted.
    noise_seed : `int`, optional
        Independent seed for detector noise. Required when ``full_config`` does
        not provide ``global_seed``; explicit values take precedence.
    run_name : `str`, optional
        Provenance label. Required when not supplied by ``full_config``.
    sample_noise : `bool`, optional
        If false, use the deterministic expected detector image and noise map.
        This makes no random draws and does not require a noise seed.

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
    if not isinstance(sample_noise, bool):
        raise ValueError("sample_noise must be boolean")
    noise_seed, run_name = _resolve_observation_context(
        full_config, noise_seed, run_name, require_seed=sample_noise,
    )

    # Extract parameters
    exposure_time = observation_config['exposure_time']
    throughput = float(observation_config['throughput'])
    detector_config = observation_config['detector']

    # Ensure PSF kernel has odd dimensions (required by PyAutoLens)
    psf_kernel = _ensure_odd_kernel(psf_data.kernel)

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
    if sample_noise:
        source_only_eps = convolve_source_rate(lensing_data, psf_kernel, throughput=throughput)
        source_eps_for_noise = np.maximum(source_only_eps, 0.0)
        final_image_adu, components = apply_detector_noise(
            source_eps=source_eps_for_noise,
            exposure_time=exposure_time,
            detector_config=detector_config,
            seed=noise_seed,
        )
        noise_map_adu = create_noise_map(source_eps_for_noise, exposure_time, detector_config)
    else:
        prediction = predict_observation(lensing_data, psf_data, observation_config)
        source_only_eps = prediction.source_eps
        noise_map_adu = prediction.noise_map_adu
        final_image_adu = prediction.mean_adu
        moments = prediction.moments
        components = {
            'source_e': moments.source_e,
            'sky_e': moments.sky_e,
            'dark_e': moments.dark_e,
            'expected_e': moments.expected_e,
        }

    # Create PyAutoLens arrays for the final data
    data = al.Array2D(values=final_image_adu, mask=mask)
    noise_map = al.Array2D(values=noise_map_adu, mask=mask)

    # Create the imaging dataset. al.Imaging sum-normalizes the supplied
    # kernel in place, so wrap a private copy: the shared PSFData kernel
    # must keep the exact bytes this observation was convolved with.
    imaging_psf = make_pyauto_convolver(
        make_pyauto_kernel(
            values=np.array(pyauto_kernel_native(psf_kernel), dtype=float),
            pixel_scales=pyauto_kernel_pixel_scales(psf_kernel),
            normalize=False,
        )
    )
    imaging_dataset = al.Imaging(
        data=data,
        noise_map=noise_map,
        psf=imaging_psf
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
        'sample_noise': sample_noise,
        'pixel_scale': lensing_data.pixel_scale,
        'field_of_view': lensing_data.field_of_view_arcsec
    }

    # Add run name if provided
    metadata['run_name'] = run_name
    from ..lensing.sampling import actual_sub_size

    generation_grid = getattr(lensing_data, 'grid', None)
    metadata['generation_sub_size'] = actual_sub_size(generation_grid)

    # Create and return ObservationData object
    return ObservationData(
        imaging=imaging_dataset,
        noiseless_source_eps=source_only_eps,
        noise_components=components,
        config=deepcopy(observation_config),
        metadata=metadata
    )


def _resolve_observation_context(full_config, noise_seed, run_name, *, require_seed=True):
    """Bind randomness and provenance independently of the pipeline container."""
    if full_config is not None and not isinstance(full_config, dict):
        raise ValueError("full_config must be a dict for generate_observation")
    config = full_config if full_config is not None else {}
    if noise_seed is None and require_seed:
        if 'global_seed' not in config:
            raise ValueError("Provide noise_seed or 'global_seed' in full_config")
        noise_seed = config['global_seed']
    if noise_seed is not None and (isinstance(noise_seed, bool) or not isinstance(noise_seed, int)):
        raise ValueError("noise_seed / full_config.global_seed must be an int")
    if run_name is None:
        if 'run_name' not in config:
            raise ValueError("Provide run_name explicitly or in full_config")
        run_name = config['run_name']
    if not isinstance(run_name, str) or not run_name:
        raise ValueError("run_name / full_config.run_name must be a non-empty string")
    return noise_seed, run_name


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
    return validate_detector_kernel(kernel)

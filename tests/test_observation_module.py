import numpy as np
import autolens as al
import pytest

from hwoslaps.observation import generate_observation, ObservationData
from hwoslaps.lensing.utils import LensingData
from hwoslaps.psf.utils import PSFData


def make_lensing_data(shape=( nine := 9, nine ), pixel_scale=0.1) -> LensingData:
    # Create a simple deterministic image pattern
    y, x = np.indices(shape)
    image = (y + x).astype(float) / (shape[0] + shape[1])

    grid = al.Grid2D.uniform(shape_native=shape, pixel_scales=pixel_scale, over_sample_size=1)

    # Minimal tracer; not used by observation but required by dataclass
    tracer = al.Tracer(galaxies=[])

    return LensingData(
        image=image,
        grid=grid,
        tracer=tracer,
        pixel_scale=pixel_scale,
        lens_redshift=0.5,
        source_redshift=1.0,
        lens_einstein_radius=1.0,
        cosmology_name="Planck15",
        config={}
    )


def make_psfdata_with_kernel(kernel: al.Kernel2D, kernel_pixel_scale: float) -> PSFData:
    # Construct a full PSFData object with minimal but valid values
    return PSFData(
        psf=None,
        wavefront=None,
        telescope_data={},
        kernel=kernel,
        kernel_pixel_scale=kernel_pixel_scale,
        wavelength_nm=550.0,
        pupil_diameter_m=6.0,
        focal_length_m=120.0,
        pixel_scale_arcsec=kernel_pixel_scale,
        sampling_factor=2.0,
        requested_sampling_factor=2.0,
        used_sampling_factor=2.0,
        integer_subsampling_factor=1,
        num_segments=1,
        segment_flat_to_flat_m=1.0,
        segment_point_to_point_m=1.0,
        gap_size_m=0.0,
        num_rings=0,
        config={}
    )


def test_generate_observation_identity_psf_end_to_end():
    shape = (9, 9)
    pixel_scale = 0.1
    lensing = make_lensing_data(shape=shape, pixel_scale=pixel_scale)

    # Identity PSF (delta kernel) ensures simulator leaves image unchanged for the noiseless rate
    psf_kernel = al.Kernel2D.no_mask(values=np.array([[1.0]]), pixel_scales=pixel_scale, normalize=False)
    psf_data = make_psfdata_with_kernel(psf_kernel, kernel_pixel_scale=pixel_scale)

    exposure_time = 500.0
    detector = {
        'gain': 2.0,
        'read_noise': 3.0,
        'dark_current': 0.002,
        'sky_background': 0.5,
    }

    obs = generate_observation(
        lensing_data=lensing,
        psf_data=psf_data,
        observation_config={'exposure_time': exposure_time, 'detector': detector},
        full_config={'global_seed': 42, 'run_name': 'unit_test_observation'}
    )

    assert isinstance(obs, ObservationData)

    # Shapes
    assert obs.data.shape_native == shape
    assert obs.noise_map.shape_native == shape
    assert obs.psf.shape_native[0] % 2 == 1 and obs.psf.shape_native[1] % 2 == 1

    # Noiseless rate equals input image (identity PSF)
    np.testing.assert_allclose(obs.noiseless_source_eps, lensing.image, rtol=0, atol=1e-10)

    # Noise map matches theory
    source_e = obs.noiseless_source_eps * exposure_time
    dark_e = detector['dark_current'] * exposure_time
    sky_e = detector['sky_background'] * exposure_time
    expected_e = source_e + dark_e + sky_e
    expected_noise_map_adu = np.sqrt(expected_e + detector['read_noise'] ** 2) / detector['gain']
    np.testing.assert_allclose(obs.noise_map.native, expected_noise_map_adu, rtol=0, atol=1e-10)

    # Noise components contain expected entries, and expected_e matches
    comps = obs.noise_components
    for key in ['source_e', 'sky_e', 'dark_e', 'detected_e', 'final_e', 'expected_e']:
        assert key in comps
    np.testing.assert_allclose(comps['expected_e'], expected_e, rtol=0, atol=1e-10)

    # Metadata sanity
    assert obs.exposure_time == exposure_time
    assert obs.pixel_scale == pixel_scale
    assert obs.metadata.get('run_name') == 'unit_test_observation'


def test_pixel_scale_mismatch_raises():
    shape = (9, 9)
    lensing = make_lensing_data(shape=shape, pixel_scale=0.1)

    # PSF kernel with different pixel scale triggers assertion
    wrong_pixel_scale = 0.2
    psf_kernel = al.Kernel2D.no_mask(values=np.array([[1.0]]), pixel_scales=wrong_pixel_scale, normalize=False)
    psf_data = make_psfdata_with_kernel(psf_kernel, kernel_pixel_scale=wrong_pixel_scale)

    detector = {
        'gain': 1.0,
        'read_noise': 0.2,
        'dark_current': 0.0,
        'sky_background': 0.0,
    }
    with pytest.raises(ValueError):
        generate_observation(
            lensing_data=lensing,
            psf_data=psf_data,
            observation_config={'exposure_time': 100.0, 'detector': detector},
            full_config={'global_seed': 1, 'run_name': 'unit_test_observation'}
        )


def test_even_psf_is_trimmed_and_normalized():
    shape = (10, 10)
    pixel_scale = 0.05
    lensing = make_lensing_data(shape=shape, pixel_scale=pixel_scale)

    # Even-shaped PSF; function should trim to odd and normalize
    even_values = np.ones((4, 4), dtype=float)
    psf_kernel_even = al.Kernel2D.no_mask(values=even_values, pixel_scales=pixel_scale, normalize=False)
    psf_data = make_psfdata_with_kernel(psf_kernel_even, kernel_pixel_scale=pixel_scale)

    detector = {
        'gain': 1.0,
        'read_noise': 0.2,
        'dark_current': 0.0,
        'sky_background': 0.0,
    }
    obs = generate_observation(
        lensing_data=lensing,
        psf_data=psf_data,
        observation_config={'exposure_time': 100.0, 'detector': detector},
        full_config={'global_seed': 7, 'run_name': 'unit_test_observation'}
    )

    # PSF should be odd-shaped and normalized
    ky, kx = obs.psf.shape_native
    assert ky % 2 == 1 and kx % 2 == 1
    assert np.isclose(obs.psf.native.sum(), 1.0, rtol=0, atol=1e-12)



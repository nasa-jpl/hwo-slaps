import numpy as np
import autolens as al

from hwoslaps.lensing.utils import LensingData
from hwoslaps.observation import generate_observation, ObservationData
from hwoslaps.psf.utils import PSFData
from hwoslaps.modeling import ChiSquareSubhaloDetector


def make_uniform_lensing_data(shape=(21, 21), pixel_scale=0.1, rate_eps=1.0) -> LensingData:
    """Create a simple uniform-rate lensing image for deterministic tests."""
    image = np.full(shape, float(rate_eps))
    grid = al.Grid2D.uniform(shape_native=shape, pixel_scales=pixel_scale, over_sample_size=1)
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
        config={},
    )


def make_psfdata_with_identity_kernel(pixel_scale: float) -> PSFData:
    """Identity PSF ensures the noiseless rate equals the input image."""
    kernel = al.Kernel2D.no_mask(values=np.array([[1.0]]), pixel_scales=pixel_scale, normalize=False)
    return PSFData(
        psf=None,
        wavefront=None,
        telescope_data={},
        kernel=kernel,
        kernel_pixel_scale=pixel_scale,
        wavelength_nm=550.0,
        pupil_diameter_m=6.0,
        focal_length_m=120.0,
        pixel_scale_arcsec=pixel_scale,
        sampling_factor=2.0,
        requested_sampling_factor=2.0,
        used_sampling_factor=2.0,
        integer_subsampling_factor=1,
        num_segments=1,
        segment_flat_to_flat_m=1.0,
        segment_point_to_point_m=1.0,
        gap_size_m=0.0,
        num_rings=0,
        config={},
    )


def test_detector_uses_baseline_noise_map_and_null_hypothesis_statistics():
    shape = (25, 25)
    pixel_scale = 0.1
    exposure_time = 1000.0
    detector = {
        'gain': 2.0,
        'read_noise': 3.0,
        'dark_current': 0.01,
        'sky_background': 0.2,
    }

    # Uniform bright source to ensure many pixels pass SNR threshold
    lensing = make_uniform_lensing_data(shape=shape, pixel_scale=pixel_scale, rate_eps=1.0)
    psf_data = make_psfdata_with_identity_kernel(pixel_scale=pixel_scale)

    # Baseline and test observations (no subhalo in either). Different seeds.
    obs_baseline: ObservationData = generate_observation(
        lensing_data=lensing,
        psf_data=psf_data,
        observation_config={'exposure_time': exposure_time, 'detector': detector},
        full_config={'global_seed': 123, 'run_name': 'baseline'}
    )

    obs_test: ObservationData = generate_observation(
        lensing_data=lensing,
        psf_data=psf_data,
        observation_config={'exposure_time': exposure_time, 'detector': detector},
        full_config={'global_seed': 456, 'run_name': 'test'}
    )

    # Initialize detector with baseline and ground-truth source counts
    source_counts_ground_truth = obs_baseline.noiseless_source_eps * obs_baseline.exposure_time
    detector_model = ChiSquareSubhaloDetector(
        observation_data_no_subhalo=obs_baseline,
        source_counts_ground_truth=source_counts_ground_truth,
        snr_threshold=1.0,
        significance_levels=[1e-3, 1e-4, 1e-5],
    )

    # Variance must equal squared baseline noise map (ADU^2)
    np.testing.assert_allclose(
        detector_model.variance_2d,
        obs_baseline.noise_map.native ** 2,
        rtol=0,
        atol=1e-12,
    )

    # Detect at arbitrary position under null hypothesis (no subhalo)
    results = detector_model.detect_at_position(obs_test, subhalo_position=(0.0, 0.0))

    # Chi-square per degree-of-freedom should be near 1 under H0 (tolerant bounds)
    any_result = next(iter(results.values()))
    chi2_over_dof = any_result.chi2_value / max(any_result.dof, 1)
    assert 0.5 < chi2_over_dof < 1.5

    # SNR mask should include a reasonable fraction of pixels for bright source
    frac = np.sum(detector_model.snr_mask) / detector_model.snr_mask.size
    assert frac > 0.2



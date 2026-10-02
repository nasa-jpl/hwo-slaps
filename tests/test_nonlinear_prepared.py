"""Prepared forecast nonlinear integration without launching a sampler."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from hwoslaps.modeling.nonlinear.api import validate_nonlinear
from hwoslaps.modeling.nonlinear.autolens_runner import AutoLensFitRunner, NonlinearSearchSettings
from hwoslaps.modeling.nonlinear.output_schema import NonlinearFitSummary
from hwoslaps.modeling.nonlinear.mass_mapping import build_mass_mapping_context_explicit
from hwoslaps.psf.utils import make_pyauto_convolver, make_pyauto_kernel, pyauto_kernel_native
from hwoslaps.psf.mismatch import _kernel_sha256
from test_nonlinear_freed_mode import _config, _trial
from test_fisher_grid_map import grid_setup as grid_setup


@pytest.mark.parametrize("fit_mode", ["fixed_template", "freed"])
@pytest.mark.parametrize("fit_psf_mode", ["matched", "kernel"])
def test_prepared_validation_reuses_the_observation_and_sampling(tmp_path, monkeypatch, fit_mode, fit_psf_mode):
    """Paired fits consume one prepared image/PSF and bind the generation grid."""
    config = _config()
    config["lensing"]["grid"] = {"shape": [7, 7], "pixel_scale": 0.1, "over_sample_size": 2}
    config["modeling"] = {"fit_psf": {"mode": "matched"}}
    kernel = make_pyauto_convolver(make_pyauto_kernel(values=np.ones((3, 3)) / 9, pixel_scales=0.1))
    image = np.arange(49, dtype=float).reshape(7, 7)
    observation = SimpleNamespace(
        psf=kernel, pixel_scale=0.1, gain=1.0, exposure_time=1.0,
        sky_electrons_per_pixel=0.0, dark_electrons_per_pixel=0.0,
        noiseless_source_eps=image, noise_map=SimpleNamespace(native=np.ones((7, 7))),
        metadata={"generation_sub_size": 2},
    )
    fit_kernel = kernel
    if fit_psf_mode == "kernel":
        fit_kernel = make_pyauto_convolver(make_pyauto_kernel(values=np.ones((5, 5)) / 25, pixel_scales=0.1))
        config["modeling"]["fit_psf"] = {
            "mode": "kernel", "kernel_sha256": _kernel_sha256(pyauto_kernel_native(fit_kernel)),
            "shape_native": [5, 5], "pixel_scale_arcsec": 0.1,
        }
    original = deepcopy(config)
    prepared = SimpleNamespace(config=config, observation=observation, fit_psf=SimpleNamespace(kernel=fit_kernel))
    calls = []

    def run_model(self, **kwargs):
        calls.append(kwargs)
        return NonlinearFitSummary(
            model_role=kwargs["role"], fit_mode=kwargs["fit_mode"], status="success",
            log_likelihood_max=-2.0 if kwargs["role"] == "smooth" else -1.0,
            analysis_key=kwargs["analysis_key"],
        )

    monkeypatch.setattr(AutoLensFitRunner, "run_model", run_model)
    context = None
    if fit_mode == "freed":
        context = build_mass_mapping_context_explicit(
            subhalo_model="NFW", concentration_model="moline2017_eq7", x_sub=1.0,
            z_lens=0.2, z_source=0.6, cosmology_name="Planck15", log10_m200_range=(6.0, 9.5),
        )
    result = validate_nonlinear(
        prepared, _trial(), NonlinearSearchSettings(), output_dir=tmp_path, fit_mode=fit_mode, mass_context=context, observation=observation,
    )
    assert [call["role"] for call in calls] == ["smooth", "subhalo"]
    assert calls[0]["analysis"] is calls[1]["analysis"]
    dataset = calls[0]["analysis"].dataset
    use_mask = ~np.asarray(dataset.mask)
    np.testing.assert_array_equal(dataset.data.native[use_mask], image[use_mask])
    np.testing.assert_allclose(pyauto_kernel_native(dataset.psf), pyauto_kernel_native(fit_kernel), rtol=0, atol=1e-15)
    if fit_psf_mode == "kernel":
        assert result.psf_case.startswith("kernel:")
        assert result.dataset_metadata.psf_fit_sha256 == result.psf_case.split(":", 1)[1]
    assert result.dataset_metadata.generation_sub_size == 2
    assert result.dataset_metadata.light_profile_sub_size == 2
    assert result.metric.q == 2.0
    assert config == original


def test_freed_prepared_validation_requires_explicit_mass_support(tmp_path):
    """A new study must choose its mass prior rather than inherit an old range."""
    with pytest.raises(ValueError, match="explicit mass_context"):
        validate_nonlinear(object(), object(), output_dir=tmp_path)


def test_default_prepared_validation_injects_the_trial_with_the_real_renderer(tmp_path, monkeypatch):
    """The default fit image is the trial injection, while a null control is explicit."""
    from hwoslaps import PreparedForecast
    from hwoslaps.lensing.generator import generate_lensing_system
    from hwoslaps.modeling.nonlinear.trial import trial_from_lensing_truth
    from hwoslaps.observation.generator import generate_observation
    from hwoslaps.psf import DetectorPSF
    from nonlinear_fixtures import scene_config
    from test_observation_correctness import _observation_config

    config = scene_config()
    config["run_name"] = "nonlinear_default_injection"
    config["lensing"]["grid"].update(shape=[31, 31], pixel_scale=0.1, over_sample_size=2)
    config["lensing"]["subhalo"].update(mass=2.0e8, position={"type": "direct", "centre": [0.5, 0.3]})
    config["observation"] = _observation_config(throughput=1.0)
    config["modeling"] = {"fit_psf": {"mode": "matched"}}
    truth_psf = DetectorPSF.from_array(np.ones((3, 3)), 0.1)
    injected_scene = generate_lensing_system(config["lensing"], full_config=config)
    trial = trial_from_lensing_truth(injected_scene)
    smooth_config = deepcopy(config)
    smooth_config["lensing"]["subhalo"]["enabled"] = False
    smooth_scene = generate_lensing_system(smooth_config["lensing"], full_config=smooth_config)
    smooth_observation = generate_observation(
        smooth_scene, truth_psf, config["observation"], full_config=smooth_config, sample_noise=False,
    )
    prepared = PreparedForecast(config, smooth_scene, truth_psf, truth_psf, smooth_observation, None)
    datasets = []

    def evaluate_declared_point(self, **kwargs):
        analysis = kwargs["analysis"]
        datasets.append(analysis.dataset)
        instance = kwargs["model"].instance_from_prior_medians()
        log_likelihood = float(analysis.log_likelihood_function(instance))
        return NonlinearFitSummary(
            model_role=kwargs["role"], fit_mode=kwargs["fit_mode"], status="success",
            log_likelihood_max=log_likelihood, analysis_key=kwargs["analysis_key"],
        )

    monkeypatch.setattr(AutoLensFitRunner, "run_model", evaluate_declared_point)
    result = validate_nonlinear(prepared, trial, output_dir=tmp_path, fit_mode="fixed_template")
    expected = generate_observation(
        injected_scene, truth_psf, config["observation"], full_config=config, sample_noise=False,
    )
    use_mask = ~np.asarray(datasets[0].mask)
    np.testing.assert_allclose(datasets[0].data.native[use_mask], expected.noiseless_source_eps[use_mask], rtol=0, atol=1e-14)
    assert np.max(np.abs(datasets[0].data.native[use_mask] - smooth_observation.noiseless_source_eps[use_mask])) > 0
    assert result.metric.q > 0
    assert prepared.observation is smooth_observation


def test_profile_refinement_rejects_a_nondifferentiable_backend_before_preparation(tmp_path):
    """Unsupported tracing must fail before an expensive fit or scene render."""
    from hwoslaps.modeling.nonlinear.profile_settings import FreshProfileSettings

    with pytest.raises(ValueError, match="settings.use_jax=True"):
        validate_nonlinear(
            object(), object(), output_dir=tmp_path, fit_mode="fixed_template",
            profile_settings=FreshProfileSettings(),
        )


def test_identity_fit_kernel_is_rejected_before_simulation_or_sampler(tmp_path):
    """The pinned backend cannot render the empty blurring grid of a 1x1 PSF."""
    from hwoslaps import PreparedForecast
    from hwoslaps.psf import DetectorPSF

    identity = DetectorPSF.from_array(np.ones((1, 1)), 0.1)
    prepared = PreparedForecast({}, None, identity, identity, None, None)
    with pytest.raises(ValueError, match="1x1 fit kernels"):
        validate_nonlinear(prepared, object(), output_dir=tmp_path, fit_mode="fixed_template")


@pytest.mark.parametrize("mask_mode", ["fixed_annulus", "source_snr"])
@pytest.mark.parametrize("custom_mask", [False, True])
def test_prepared_validation_keeps_the_real_fisher_pixel_mask(grid_setup, tmp_path, monkeypatch, mask_mode, custom_mask):
    """The public fitting dataset uses the prepared likelihood's actual ROI."""
    from hwoslaps import PreparedForecast
    from hwoslaps.modeling.fisher_detector import FisherDetector
    from hwoslaps.modeling.nonlinear.trial import trial_from_lensing_truth

    config = deepcopy(grid_setup["config"])
    fisher = config["modeling"]["fisher"]
    fisher["mask_mode"] = mask_mode
    fisher["mask_annulus"] = (
        {"inner_arcsec": 0.2, "outer_arcsec": 1.2, "centre": "lens"}
        if mask_mode == "fixed_annulus" else None
    )
    detector = FisherDetector(
        observation_baseline=grid_setup["observation_baseline"],
        lensing_baseline=grid_setup["lensing_baseline"], psf_data=grid_setup["psf_data"],
        full_config=config, fisher_config=fisher,
    )
    prepared = PreparedForecast(
        config, grid_setup["lensing_baseline"], grid_setup["psf_data"],
        detector.model_psf_data, grid_setup["observation_baseline"], detector,
    )
    datasets = []

    def no_sampler(self, **kwargs):
        datasets.append(kwargs["analysis"].dataset)
        return NonlinearFitSummary(model_role=kwargs["role"], fit_mode=kwargs["fit_mode"], status="skipped")

    monkeypatch.setattr(AutoLensFitRunner, "run_model", no_sampler)
    override = None
    if custom_mask:
        override = np.zeros_like(detector.mask_2d)
        override[11:19, 13:20] = True
    result = validate_nonlinear(
        prepared, trial_from_lensing_truth(grid_setup["lensing_test"]),
        output_dir=tmp_path, fit_mode="fixed_template", observation=grid_setup["observation_test"], mask_bool_use=override,
    )
    expected = override.copy() if custom_mask else detector.mask_2d.copy()
    height, width = pyauto_kernel_native(prepared.fit_psf.kernel).shape
    for axis, half in enumerate((height // 2, width // 2)):
        if half:
            index = [slice(None), slice(None)]
            index[axis] = slice(0, half)
            expected[tuple(index)] = False
            index[axis] = slice(-half, None)
            expected[tuple(index)] = False
    assert np.any(expected) and not np.all(expected)
    np.testing.assert_array_equal(~np.asarray(datasets[0].mask), expected)
    assert result.dataset_metadata.mask_name == ("custom_minus_psf_border" if custom_mask else "fisher_minus_psf_border")
    assert result.dataset_metadata.n_unmasked_pixels == np.count_nonzero(expected)


@pytest.mark.parametrize("change", ["different_kernel", "different_shape"])
def test_matched_validation_rejects_a_different_supplied_observation_psf(tmp_path, change):
    """Only the prepared bytes or their exact backend normalization are admitted."""
    from hwoslaps import PreparedForecast
    from hwoslaps.psf import DetectorPSF

    truth = DetectorPSF.from_array(np.arange(1.0, 10.0).reshape(3, 3), 0.1)
    values = np.eye(3) if change == "different_kernel" else pyauto_kernel_native(truth.kernel).reshape(1, 9)
    wrong = DetectorPSF.from_array(values, 0.1)
    prepared = PreparedForecast({"modeling": {"fit_psf": {"mode": "matched"}}}, None, truth, truth, None, None)
    observation = SimpleNamespace(psf=wrong.kernel, pixel_scale=0.1)
    with pytest.raises(ValueError, match="requires the prepared truth PSF"):
        validate_nonlinear(
            prepared, object(), output_dir=tmp_path, fit_mode="fixed_template", observation=observation,
        )

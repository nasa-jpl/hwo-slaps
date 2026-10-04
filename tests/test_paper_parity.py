"""Paper-parity anchors: the engine reproduces the RASTI-26-183 submission numbers.

Every expected value comes from the submitted paper code (commit 41621de, tag
``rasti-26-183-submitted``) through ``tests/scripts/generate_paper_parity.py``,
which never imports this engine. The scenes are small enough for the CPU lane
yet keep the paper's physics paths: segmented-pupil optics with PSF-mode
nuisances, a delta PSF knowledge error, an image source with an external
detector kernel, three subhalo profiles, and the nonlinear likelihood the
validation searches maximized.

The engine output is bitwise identical to the paper code under the same lane
environment, so these tests assert exact equality. A failure is a change of
science rather than of rounding: ``manifest.json`` records digests of the
smooth image, noise map, Fisher mask and every nuisance derivative image, and
of the nonlinear dataset arrays, to localize it.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("autolens")

from hwoslaps import forecast, prepare_forecast, simulate

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "paper_parity"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))
NONLINEAR = "n1_nonlinear_likelihood"
FISHER_SCENES = sorted(name for name in MANIFEST["scenes"] if name != NONLINEAR)
STATISTICS = ("q_asimov", "fisher_raw", "fisher_profiled", "sigma_amplitude", "degradation")
MISMATCH_STATISTICS = (
    "amplitude_hat", "q_mismatch", "z_mismatch",
    "amplitude_spurious", "q_spurious", "z_spurious",
)
LIKELIHOOD_TERMS = ("log_likelihood", "figure_of_merit", "chi_squared", "noise_normalization")
DELTA_IDENTITY = ("delta_id", "measured_draw_rms_nm", "prior_table_sha256", "fit_kernel_sha256")


def _expected(scene):
    with np.load(FIXTURES / MANIFEST["scenes"][scene]["fixture"]) as data:
        return {key: data[key] for key in data.files}


def _array_digest(values):
    """Digest format of the generator's dataset records."""
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return hashlib.sha256(repr(array.shape).encode() + array.tobytes()).hexdigest()


@pytest.fixture(scope="module")
def prepared():
    """Prepare each scene once per backend; preparation dominates the cost."""
    cache = {}

    def get(scene, backend):
        if (scene, backend) not in cache:
            config = FIXTURES / MANIFEST["scenes"][scene]["engine_config"]
            cache[scene, backend] = prepare_forecast(config, backend=backend)
        return cache[scene, backend]

    return get


def _assert_forecast_matches_paper(scene, prepared_forecast, lane):
    entry = MANIFEST["scenes"][scene]
    expected = _expected(scene)
    result = forecast(prepared_forecast, masses=expected["masses_msun"])
    provenance = result.runtime_provenance
    assert provenance["truth_kernel"]["kernel_sha256"] == entry["truth_kernel_sha256"]
    assert provenance["fit_kernel"]["kernel_sha256"] == entry["fit_kernel_sha256"]
    assert provenance["profiled_nuisance_names"] == entry["profiled_nuisance_names"]
    if "fit_psf_delta" in entry:
        for key in DELTA_IDENTITY:
            assert provenance["fit_psf_delta"][key] == entry["fit_psf_delta"][key], key
    np.testing.assert_array_equal(result.positions_yx, expected["positions_yx"])
    mismatch = f"{lane}__q_mismatch" in expected
    assert (result.q_mismatch is not None) is mismatch
    for name in STATISTICS + (MISMATCH_STATISTICS if mismatch else ()):
        np.testing.assert_array_equal(
            getattr(result, name), expected[f"{lane}__{name}"], err_msg=f"{scene}: {name}",
        )


@pytest.mark.parametrize("scene", FISHER_SCENES)
def test_reference_forecast_reproduces_paper(scene, prepared):
    _assert_forecast_matches_paper(scene, prepared(scene, "reference"), "reference")


@pytest.mark.xtx_gpu
@pytest.mark.parametrize("scene", FISHER_SCENES)
def test_jax_forecast_reproduces_paper(scene, prepared):
    _assert_forecast_matches_paper(scene, prepared(scene, "jax"), "jax_gpu")


def _nonlinear_case(prepared_forecast, output_dir, *, use_jax):
    """Assemble the dataset, models and analysis a freed validation fit consumes.

    This follows ``validate_nonlinear`` up to its sampler: a noisy injection
    through the public simulator, the prepared delta fit kernel, the
    all-pixels-minus-PSF-border mask and the consistent-sampling objective.
    """
    from hwoslaps.lensing.sampling import configured_sub_size
    from hwoslaps.modeling.nonlinear import (
        AutoLensFitRunner, NonlinearSearchSettings, autofit_model_from_spec,
        build_mass_mapping_context, build_psf_mismatch_spec,
        smooth_model_spec_from_config, subhalo_model_spec_from_trial,
    )
    from hwoslaps.modeling.nonlinear.dataset_builder import imaging_from_observation

    spec = MANIFEST["scenes"][NONLINEAR]
    config = prepared_forecast.config
    trial = prepared_forecast.trial(spec["mass_msun"], tuple(spec["position_yx_arcsec"]))
    observation = simulate(prepared_forecast, trial=trial, sample_noise=True)
    dataset, metadata = imaging_from_observation(
        observation,
        psf_for_fit=prepared_forecast.fit_psf.kernel,
        dataset_kind=spec["dataset_kind"],
        background_treatment=spec["background_treatment"],
        psf_truth_label="prepared_truth",
        psf_fit_label=f"delta:{build_psf_mismatch_spec(config).delta_id}",
        objective_version="consistent_sampling_v2",
        generation_sub_size=configured_sub_size(config["lensing"]["grid"]),
    )
    mass_context = build_mass_mapping_context(config, log10_m200_range=tuple(spec["log10_m200_range"]))
    specs = {
        "smooth": smooth_model_spec_from_config(config),
        "freed": subhalo_model_spec_from_trial(
            config, trial=trial, fit_mode=spec["fit_mode"], mass_context=mass_context,
        ),
    }
    runner = AutoLensFitRunner(NonlinearSearchSettings(use_jax=use_jax), output_dir=output_dir)
    analysis = runner.make_analysis(dataset, model_metadata=dict(specs["freed"].metadata))
    models = {role: autofit_model_from_spec(model_spec) for role, model_spec in specs.items()}
    return trial, dataset, metadata, models, analysis


def _assert_nonlinear_inputs_match_paper(trial, dataset, metadata, models, expected):
    from hwoslaps.psf.utils import pyauto_kernel_native

    entry = MANIFEST["scenes"][NONLINEAR]
    for key in ("kappa_s", "scale_radius_arcsec", "concentration", "lens_redshift", "source_redshift"):
        assert getattr(trial, key) == entry["trial"][key], key
    recorded = entry["dataset"]
    assert metadata.psf_fit_sha256 == recorded["psf_fit_sha256"]
    assert metadata.n_unmasked_pixels == recorded["n_unmasked_pixels"]
    assert _array_digest(dataset.data.native) == recorded["data_digest"]
    assert _array_digest(dataset.noise_map.native) == recorded["noise_map_digest"]
    assert _array_digest(pyauto_kernel_native(dataset.psf)) == recorded["psf_digest"]
    assert _array_digest(np.asarray(dataset.mask).astype(float)) == recorded["mask_digest"]
    for role, model in models.items():
        paths = [".".join(path) for path in model.unique_prior_paths]
        assert paths == list(expected[f"{role}_prior_paths"])
        priors = model.priors_ordered_by_id
        np.testing.assert_array_equal([prior.lower_limit for prior in priors], expected[f"{role}_prior_lower"])
        np.testing.assert_array_equal([prior.upper_limit for prior in priors], expected[f"{role}_prior_upper"])


def _assert_likelihood_matches_paper(models, analysis, expected, lane):
    for role, model in models.items():
        instances = [model.instance_from_vector(vector=list(vector)) for vector in expected[f"{role}_vectors"]]
        fits = [analysis.fit_from(instance=instance) for instance in instances]
        np.testing.assert_array_equal(
            [float(analysis.log_likelihood_function(instance)) for instance in instances],
            expected[f"{lane}__{role}_log_likelihood_function"], err_msg=role,
        )
        for term in LIKELIHOOD_TERMS:
            np.testing.assert_array_equal(
                [float(getattr(fit, term)) for fit in fits],
                expected[f"{lane}__{role}_{term}"], err_msg=f"{role}: {term}",
            )


def test_nonlinear_likelihood_reproduces_paper(prepared, tmp_path):
    expected = _expected(NONLINEAR)
    trial, dataset, metadata, models, analysis = _nonlinear_case(
        prepared(MANIFEST["scenes"][NONLINEAR]["scene"], "reference"), tmp_path, use_jax=False,
    )
    _assert_nonlinear_inputs_match_paper(trial, dataset, metadata, models, expected)
    _assert_likelihood_matches_paper(models, analysis, expected, "reference")


@pytest.mark.xtx_gpu
def test_nonlinear_jax_objective_reproduces_paper(prepared, tmp_path):
    from hwoslaps.modeling.nonlinear.fresh_profile import make_jax_objective

    expected = _expected(NONLINEAR)
    trial, dataset, metadata, models, analysis = _nonlinear_case(
        prepared(MANIFEST["scenes"][NONLINEAR]["scene"], "jax"), tmp_path, use_jax=True,
    )
    _assert_nonlinear_inputs_match_paper(trial, dataset, metadata, models, expected)
    _assert_likelihood_matches_paper(models, analysis, expected, "jax_gpu")
    for role, model in models.items():
        objective, _, _, _ = make_jax_objective(
            analysis, model, expected[f"{role}_prior_lower"], expected[f"{role}_prior_upper"],
        )
        values = [objective(z) for z in expected[f"{role}_vectors_normalized"]]
        np.testing.assert_array_equal(
            [value for value, _ in values], expected[f"jax_gpu__{role}_half_chi2"], err_msg=role,
        )
        np.testing.assert_array_equal(
            np.vstack([gradient for _, gradient in values]),
            expected[f"jax_gpu__{role}_half_chi2_gradient"], err_msg=role,
        )

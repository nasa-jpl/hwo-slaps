"""Fit independent scenes to a preserved mock through the public API."""

from dataclasses import replace

import pytest

from hwoslaps import FitSpec, SamplerSettings, fit_observation, load_case, save_case, simulate
from hwoslaps.config.schema import parse_config
from hwoslaps.identity import array_digest
from hwoslaps.scene.cosmology import Cosmology
from hwoslaps.scene.halos import make_halo

pytestmark = pytest.mark.backend


def _candidate(config, mass, position):
    return make_halo(config.scene.subhalo, mass, position, redshift=config.scene.lens.redshift,
                     source_redshift=config.scene.source.redshift, cosmology=Cosmology(config.cosmology))


@pytest.fixture
def pair(minimal_mapping):
    minimal_mapping["forecast"]["nuisances"] = {"fixed": ["*"]}
    minimal_mapping["scene"]["lens"]["mass"]["mass"]["multipoles"] = {"m3": [0.01, -0.005]}
    truth = parse_config(minimal_mapping)
    simple = truth.replace({"scene": {"lens": {"mass": {"mass": {"multipoles": None}}}}})
    injected = _candidate(truth, 1e8, (0.0, 0.8))
    return truth, simple, simulate(truth, subhalo=injected, noise_seed=17)


def test_same_pixels_fit_with_different_scenes_and_candidate(pair, tmp_path):
    truth, simple, observation = pair
    settings = dict(fit=FitSpec(mode="fixed_template"), sampler=SamplerSettings(),
                    sampler_seed=31, output_dir=tmp_path)
    candidate = _candidate(simple, 8e7, (0.1, 0.75))
    matched = fit_observation(observation, model=truth, candidate=candidate, **settings)
    reduced = fit_observation(observation, model=simple, candidate=candidate, **settings)
    hidden = fit_observation(replace(observation, subhalo=None), model=simple,
                             candidate=candidate, **settings)

    assert matched.observation == reduced.observation
    assert reduced.observation.data_digest == array_digest(observation.data_adu)
    for key in ("data_digest", "noise_digest", "mask"):
        assert matched.data[key] == reduced.data[key] == hidden.data[key]
    assert matched.hypothesis == reduced.hypothesis == candidate
    assert matched.models["smooth"] != reduced.models["smooth"]
    assert matched.smooth.log_likelihood != pytest.approx(reduced.smooth.log_likelihood)
    assert reduced.hypothesis == candidate != reduced.observation.subhalo
    assert reduced.data["fit_scene"]["config_digest"] == simple.digest()
    assert reduced.models == hidden.models
    assert reduced.data["comparison_inputs"]["nuisance_constraints"] == hidden.data["comparison_inputs"]["nuisance_constraints"]
    assert reduced.q_signed == hidden.q_signed
    assert reduced.smooth.truth_log_likelihood is None
    assert reduced.data["reference_log_likelihoods"]["smooth"] == reduced.smooth.log_likelihood
    assert load_case(save_case(reduced, tmp_path / "fit.json")).to_mapping() == reduced.to_mapping()


def test_independent_fit_keeps_detector_and_anchor_validation(pair, tmp_path):
    truth, _, observation = pair
    settings = dict(candidate=observation.subhalo, sampler=SamplerSettings(),
                    sampler_seed=31, output_dir=tmp_path / "failed_fits")
    wrong = truth.replace({"scene": {"grid": {"shape": [43, 43]}}})
    with pytest.raises(ValueError, match="fit grid shape"):
        fit_observation(observation, model=wrong, fit=FitSpec(mode="fixed_template"), **settings)
    with pytest.raises(ValueError, match="truth_anchor requires matched"):
        fit_observation(observation, model=truth, fit=FitSpec(mode="fixed_template", h1="truth_anchor"), **settings)
    assert not settings["output_dir"].exists()


def test_fit_priors_and_reference_follow_the_fit_scene(pair, tmp_path):
    _, simple, observation = pair
    recipe = {"concentration": {"kind": "fixed", "value": 12.0}}
    fixed = ["lens.*", "source.light.light.centre_*", "source.light.light.ell_comp_*",
             "source.light.light.effective_radius"]
    model = simple.replace({"scene": {"subhalo": recipe, "source": {"light": {"light": {"intensity": 1.1}}}},
                            "forecast": {"nuisances": {"fixed": fixed}}})
    candidate = _candidate(model, 8e7, (0.1, 0.75))
    settings = dict(model=model, candidate=candidate, fit=FitSpec(mode="fixed_template"),
                    sampler=SamplerSettings(n_like_max=60), sampler_seed=31, output_dir=tmp_path)
    visible = fit_observation(observation, **settings)
    masked = fit_observation(replace(observation, subhalo=None), **settings)
    key = "source.light.light.intensity"
    assert candidate.model != observation.subhalo.model
    assert visible.fitted_parameters == masked.fitted_parameters == (key,)
    box = visible.data["comparison_inputs"]["nuisance_constraints"][key]
    assert (box["lower"] + box["upper"]) / 2 == pytest.approx(1.1)
    assert visible.data["reference_log_likelihoods"] == masked.data["reference_log_likelihoods"]
    assert visible.models == masked.models
    assert visible.q_signed is not None and visible.q_signed == masked.q_signed

"""Fit independent scenes to a preserved mock through the public API."""

from dataclasses import replace

import pytest

from hwoslaps import FitSpec, SamplerSettings, fit_observation, load_case, save_case, simulate
from hwoslaps.config.schema import parse_config
from hwoslaps.identity import array_digest
from hwoslaps.scene.cosmology import Cosmology
from hwoslaps.scene.halos import make_halo

pytestmark = pytest.mark.backend

CANDIDATE = dict(mass_msun=8e7, position_yx=(0.1, 0.75))


@pytest.fixture
def pair(minimal_mapping):
    minimal_mapping["forecast"]["nuisances"] = {"fixed": ["*"]}
    minimal_mapping["scene"]["lens"]["mass"]["mass"]["multipoles"] = {"m3": [0.01, -0.005]}
    minimal_mapping["scene"]["perturbers"] = {"halos": [{
        "type": "NFW", "mass_msun": 4e7, "centre": [-0.55, 0.35],
        "concentration": {"kind": "fixed", "value": 11.0},
    }]}
    truth = parse_config(minimal_mapping)
    simple = truth.replace({"scene": {"lens": {"mass": {"mass": {"multipoles": None}}}}})
    injected = make_halo(truth.scene.subhalo, 1e8, (0.0, 0.8), redshift=truth.scene.lens.redshift,
                         source_redshift=truth.scene.source.redshift, cosmology=Cosmology(truth.cosmology))
    return truth, simple, simulate(truth, subhalo=injected, noise_seed=17)


def test_same_pixels_fit_with_different_scenes_and_candidate(pair, tmp_path):
    truth, simple, observation = pair
    settings = dict(**CANDIDATE, fit=FitSpec(mode="fixed_template"), sampler=SamplerSettings(),
                    sampler_seed=31, output_dir=tmp_path)
    matched = fit_observation(observation, model=truth, **settings)
    reduced = fit_observation(observation, model=simple, **settings)
    hidden = fit_observation(replace(observation, subhalo=None), model=simple, **settings)

    assert matched.observation == reduced.observation
    assert reduced.observation.data_digest == array_digest(observation.data_adu)
    for key in ("data_digest", "noise_digest", "mask"):
        assert matched.data[key] == reduced.data[key] == hidden.data[key]
    candidate = reduced.hypothesis
    assert (candidate.mass_msun, candidate.position_yx_arcsec) == (8e7, (0.1, 0.75))
    assert matched.hypothesis == candidate != reduced.observation.subhalo
    assert matched.models["smooth"] != reduced.models["smooth"]
    assert matched.smooth.log_likelihood != pytest.approx(reduced.smooth.log_likelihood)
    assert reduced.data["fit_scene"]["config_digest"] == simple.digest()
    assert reduced.models == hidden.models
    assert reduced.data["comparison_inputs"]["nuisance_constraints"] == hidden.data["comparison_inputs"]["nuisance_constraints"]
    assert reduced.q_signed == hidden.q_signed
    assert reduced.smooth.truth_log_likelihood is None
    assert reduced.smooth.reference_log_likelihood == reduced.smooth.log_likelihood
    assert load_case(save_case(reduced, tmp_path / "fit.json")).to_mapping() == reduced.to_mapping()


def test_independent_fit_keeps_detector_and_anchor_validation(pair, tmp_path):
    truth, _, observation = pair
    settings = dict(**CANDIDATE, sampler=SamplerSettings(),
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
    model = simple.replace({"scene": {"subhalo": recipe, "perturbers": {"halos": []},
                                      "source": {"light": {"light": {"intensity": 1.1}}}},
                            "forecast": {"nuisances": {"fixed": fixed}}})
    settings = dict(model=model, **CANDIDATE, fit=FitSpec(mode="fixed_template"),
                    sampler=SamplerSettings(n_like_max=60), sampler_seed=31, output_dir=tmp_path)
    visible = fit_observation(observation, **settings)
    masked = fit_observation(replace(observation, subhalo=None), **settings)
    key = "source.light.light.intensity"
    assert visible.hypothesis.model == model.scene.subhalo != observation.subhalo.model
    assert not visible.data["fit_scene"]["perturbers"]
    assert visible.fitted_parameters == masked.fitted_parameters == (key,)
    box = visible.data["comparison_inputs"]["nuisance_constraints"][key]
    assert (box["lower"] + box["upper"]) / 2 == pytest.approx(1.1)
    for role in ("smooth", "subhalo"):
        reference = visible.role(role).reference_log_likelihood
        assert reference is not None and reference == masked.role(role).reference_log_likelihood
    assert visible.models == masked.models
    assert visible.q_signed is not None and visible.q_signed == masked.q_signed

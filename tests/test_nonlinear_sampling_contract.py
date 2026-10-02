"""Generation/core/ring agreement for the current imaging objective."""
from dataclasses import replace
from types import SimpleNamespace

import autolens as al
import numpy as np
import pytest

from hwoslaps.lensing.generator import _create_grid, clear_uniform_grid_cache
from hwoslaps.lensing.sampling import actual_sub_size, configured_sub_size
from hwoslaps.modeling.nonlinear.autolens_runner import analysis_key_from
from hwoslaps.modeling.nonlinear.dataset_builder import (
    imaging_from_observation, rendering_identity,
)
from hwoslaps.psf.utils import make_pyauto_convolver, make_pyauto_kernel, pyauto_kernel_native


def baseline(size=4, psf_size=3):
    psf = make_pyauto_convolver(make_pyauto_kernel(
        values=np.ones((psf_size, psf_size)) / psf_size**2,
        pixel_scales=.1, normalize=False,
    ))
    observation = SimpleNamespace(
        psf=psf, pixel_scale=.1, gain=1., exposure_time=1.,
        sky_electrons_per_pixel=0., dark_electrons_per_pixel=0.,
        noiseless_source_eps=np.arange(169.).reshape(13, 13),
        noise_map=SimpleNamespace(native=np.ones((13, 13))),
        metadata={"generation_sub_size": size},
    )
    return imaging_from_observation(observation)


@pytest.mark.parametrize("size", [2, 4])
def test_builder_preserves_observation_values_and_binds_sampling(size):
    dataset, metadata = baseline(size)
    assert actual_sub_size(dataset.grids.lp) == actual_sub_size(dataset.grids.blurring) == size
    assert metadata.generation_sub_size == size
    np.testing.assert_array_equal(dataset.data.native[1:-1, 1:-1], np.arange(169.).reshape(13, 13)[1:-1, 1:-1])
    assert np.all(np.asarray(dataset.noise_map) == 1.)
    assert metadata.objective_version == "consistent_sampling_v2"
    assert analysis_key_from(dataset, metadata, {})


def test_identity_rejects_sampling_metadata_and_runtime_grid_mismatch():
    dataset, metadata = baseline()
    with pytest.raises(ValueError, match="metadata differs"):
        analysis_key_from(dataset, replace(metadata, blurring_sub_size=1), {})
    dataset.grids._blurring = al.Grid2D.from_mask(mask=dataset.grids.blurring.mask, over_sample_size=2)
    with pytest.raises(ValueError, match="contract is violated"):
        analysis_key_from(dataset, metadata, {})


def test_generation_cache_isolates_sampling_and_preserves_default():
    clear_uniform_grid_cache()
    config = {"shape": [7, 7], "pixel_scale": .1}
    default = _create_grid(config)
    low = _create_grid(dict(config, over_sample_size=2))
    explicit = _create_grid(dict(config, over_sample_size=4))
    assert actual_sub_size(low) == 2
    assert actual_sub_size(default) == actual_sub_size(explicit) == 4
    np.testing.assert_array_equal(default.over_sampled.array, explicit.over_sampled.array)
    assert len(default.over_sampled.array) == 4 * len(low.over_sampled.array)
    low.over_sampled.array[0, 0] += 99
    fresh = _create_grid(dict(config, over_sample_size=2))
    assert fresh.over_sampled.array[0, 0] != low.over_sampled.array[0, 0]


@pytest.mark.parametrize("size", [2, 4])
def test_corrected_builder_checks_generation_and_sets_both_grids(size):
    base, _ = baseline(size)
    observation = SimpleNamespace(
        psf=base.psf, pixel_scale=.1, gain=1., exposure_time=1.,
        sky_electrons_per_pixel=0., dark_electrons_per_pixel=0.,
        noiseless_source_eps=np.arange(169.).reshape(13, 13),
        noise_map=SimpleNamespace(native=np.ones((13, 13))),
        metadata={"generation_sub_size": size},
    )
    dataset, metadata = imaging_from_observation(
        observation, objective_version="consistent_sampling_v2", generation_sub_size=size,
    )
    assert actual_sub_size(dataset.grids.lp) == actual_sub_size(dataset.grids.blurring) == size
    assert metadata.generation_sub_size == size
    assert analysis_key_from(dataset, metadata, {})
    default_dataset, default_metadata = imaging_from_observation(observation)
    assert default_metadata.generation_sub_size == size
    assert actual_sub_size(default_dataset.grids.blurring) == size
    with pytest.raises(ValueError, match="actual generation"):
        imaging_from_observation(
            observation, objective_version="consistent_sampling_v2", generation_sub_size=size + 1,
        )


@pytest.mark.parametrize("value", [True, 0, -1, 2.5, "4"])
def test_invalid_sub_size_rejected(value):
    with pytest.raises(ValueError, match="positive integer"):
        configured_sub_size({"over_sample_size": value})


def test_corrected_objective_requires_an_explicit_declared_sampling():
    from hwoslaps.config.validation import validate_nonlinear_rendering_config

    config = {"lensing": {"grid": {}},
              "nonlinear_rendering": {"objective_version": "consistent_sampling_v2"}}
    with pytest.raises(ValueError, match="over_sample_size"):
        validate_nonlinear_rendering_config(config)
    config["lensing"]["grid"]["over_sample_size"] = 2
    validate_nonlinear_rendering_config(config)
    config["nonlinear_rendering"]["objective_version"] = "unknown"
    with pytest.raises(ValueError, match="objective_version"):
        validate_nonlinear_rendering_config(config)


def test_renderer_contract_revision_is_part_of_identity(monkeypatch):
    from hwoslaps.modeling.nonlinear import dataset_builder

    new, corrected = baseline()
    before = analysis_key_from(new, corrected, {})
    monkeypatch.setattr(dataset_builder, "RENDERING_CONTRACT_REVISION", "future-incompatible-operator")
    assert analysis_key_from(new, corrected, {}) != before


def test_ray_geometry_is_part_of_corrected_identity():
    dataset, metadata = baseline()
    before = analysis_key_from(dataset, metadata, {})
    dataset.grids.lp.over_sampled.array[0, 0] += .001
    assert analysis_key_from(dataset, metadata, {}) != before


def test_no_external_blurring_pixels_has_explicit_identity():
    dataset, metadata = baseline(psf_size=1)
    contract = rendering_identity(dataset, metadata)
    assert contract["external_blurring_pixels_present"] is False
    assert contract["blurring_sub_size"] is None
    assert analysis_key_from(dataset, metadata, {})


@pytest.mark.parametrize("size", [2, 4])
def test_generated_asimov_truth_matches_corrected_fit(size):
    from hwoslaps.lensing.generator import generate_lensing_system
    from hwoslaps.observation.generator import generate_observation
    from test_lensing_generation_contracts import _small_lensing_config
    from test_observation_correctness import _make_psf_data, _observation_config

    config = _small_lensing_config()
    config["lensing"]["grid"]["over_sample_size"] = size
    lensing = generate_lensing_system(config["lensing"], full_config=config)
    psf = _make_psf_data(np.ones((3, 3)) / 9, pixel_scale=lensing.pixel_scale)
    observation = generate_observation(
        lensing, psf, observation_config=_observation_config(), full_config=config,
    )
    dataset, metadata = imaging_from_observation(
        observation, objective_version="consistent_sampling_v2", generation_sub_size=size,
    )
    assert metadata.generation_sub_size == actual_sub_size(lensing.grid) == size
    fit = al.FitImaging(dataset=dataset, tracer=lensing.tracer)
    assert float(fit.chi_squared) < 1e-16


def test_corrected_fit_samples_exactly_as_the_generator_did():
    """Fit and truth share explicit sampling and the generator default."""
    from hwoslaps.lensing.sampling import LEGACY_SUB_SIZE
    from hwoslaps.modeling.nonlinear.psf_mismatch import generation_sub_size_for

    undeclared = {"lensing": {"grid": {"pixel_scale": 0.00716, "shape": [508, 508]}}}
    assert generation_sub_size_for(undeclared, "consistent_sampling_v2") == LEGACY_SUB_SIZE
    assert generation_sub_size_for(undeclared, "consistent_sampling_v2") == configured_sub_size(
        undeclared["lensing"]["grid"]
    )
    explicit = {"lensing": {"grid": {"over_sample_size": 2}}}
    assert generation_sub_size_for(explicit, "consistent_sampling_v2") == 2
    with pytest.raises(ValueError, match="objective_version"):
        generation_sub_size_for(undeclared, "retired_objective")
    with pytest.raises(ValueError, match="over_sample_size"):
        generation_sub_size_for({"lensing": {"grid": {"over_sample_size": 0}}}, "consistent_sampling_v2")

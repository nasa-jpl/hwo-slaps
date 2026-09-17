"""Generation/core/ring agreement and objective migration regressions."""
from dataclasses import replace
from types import SimpleNamespace

import autolens as al
import numpy as np
import pytest

from hwoslaps.lensing.generator import _create_grid, clear_uniform_grid_cache
from hwoslaps.lensing.sampling import actual_sub_size, configured_sub_size
from hwoslaps.modeling.nonlinear.autolens_runner import analysis_key_from
from hwoslaps.modeling.nonlinear.dataset_builder import (
    NonlinearDatasetMetadata, consistent_sampling_copy, imaging_from_observation, rendering_identity,
)
from hwoslaps.psf.utils import make_pyauto_convolver, make_pyauto_kernel, pyauto_kernel_native


def baseline(size=4, origin=(0., 0.)):
    mask_values = np.ones((13, 13), dtype=bool)
    mask_values[2:-2, 2:-2] = False
    try:
        mask = al.Mask2D(values=mask_values, pixel_scales=.1, origin=origin)
    except TypeError:
        mask = al.Mask2D(mask=mask_values, pixel_scales=.1, origin=origin)
    psf = make_pyauto_convolver(make_pyauto_kernel(
        values=np.ones((3, 3)) / 9, pixel_scales=.1, normalize=False,
    ))
    dataset = al.Imaging(
        data=al.Array2D(values=np.arange(169.).reshape(13, 13), mask=mask),
        noise_map=al.Array2D(values=np.ones((13, 13)), mask=mask),
        psf=psf, over_sample_size_lp=size,
    )
    metadata = NonlinearDatasetMetadata(
        dataset_kind="asimov", data_units="e_per_s", background_treatment="subtract_known",
        sky_dark_background_adu=0., mask_name="test", n_unmasked_pixels=81,
        psf_truth_label="test", psf_fit_label="test",
        generation_sub_size=size,
    )
    return dataset, metadata


@pytest.mark.parametrize("size", [2, 4])
def test_migration_changes_only_sampling_and_identity(size):
    old, metadata = baseline(size, origin=(.2, -.1))
    old_key = analysis_key_from(old, metadata, {"fit_mode": "smooth"})
    new, corrected = consistent_sampling_copy(old, metadata, size)
    assert actual_sub_size(old.grids.blurring) == 1
    assert actual_sub_size(new.grids.lp) == actual_sub_size(new.grids.blurring) == size
    assert corrected.generation_sub_size == size
    for a, b in [(old.data.native, new.data.native), (old.noise_map.native, new.noise_map.native),
                 (pyauto_kernel_native(old.psf), pyauto_kernel_native(new.psf))]:
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
    assert old_key == analysis_key_from(old, metadata, {"fit_mode": "smooth"})
    assert old_key != analysis_key_from(new, corrected, {"fit_mode": "smooth"})
    with pytest.raises(ValueError, match="historical identity metadata"):
        analysis_key_from(new, metadata, {"fit_mode": "smooth"})
    np.testing.assert_array_equal(old.grids.lp.array, new.grids.lp.array)
    with pytest.raises(ValueError, match="Historical identity schema"):
        analysis_key_from(new, corrected, {}, legacy_clumpy_null=True)
    with pytest.raises(ValueError, match="historical baseline"):
        consistent_sampling_copy(new, corrected, size)


def test_migration_and_identity_fail_closed_on_sampling_mismatch():
    old, metadata = baseline()
    with pytest.raises(ValueError, match="core/generation"):
        consistent_sampling_copy(old, metadata, 2)
    new, corrected = consistent_sampling_copy(old, metadata, 4)
    with pytest.raises(ValueError, match="metadata differs"):
        analysis_key_from(new, replace(corrected, blurring_sub_size=1), {})
    new.grids._blurring = al.Grid2D.from_mask(mask=new.grids.blurring.mask, over_sample_size=1)
    with pytest.raises(ValueError, match="contract is violated"):
        analysis_key_from(new, corrected, {})
    old.grids._blurring = al.Grid2D.from_mask(mask=old.grids.blurring.mask, over_sample_size=2)
    with pytest.raises(ValueError, match="requires ring1"):
        consistent_sampling_copy(old, metadata, 4)


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
    if size != 4:
        with pytest.raises(ValueError, match="historical generation sampling"):
            imaging_from_observation(observation)
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

    old, metadata = baseline()
    new, corrected = consistent_sampling_copy(old, metadata)
    before = analysis_key_from(new, corrected, {})
    monkeypatch.setattr(dataset_builder, "RENDERING_CONTRACT_REVISION", "future-incompatible-operator")
    assert analysis_key_from(new, corrected, {}) != before


def test_ray_geometry_is_part_of_corrected_identity():
    old, metadata = baseline()
    new, corrected = consistent_sampling_copy(old, metadata.to_dict())
    before = analysis_key_from(new, corrected, {})
    new.grids.lp.over_sampled.array[0, 0] += .001
    assert analysis_key_from(new, corrected, {}) != before
    with pytest.raises(ValueError, match="metadata cannot be reconstructed"):
        consistent_sampling_copy(old, {"unexpected": True})
    missing_generation = metadata.to_dict()
    del missing_generation["generation_sub_size"]
    with pytest.raises(ValueError, match="verified generation sampling"):
        consistent_sampling_copy(old, missing_generation)


def test_no_external_blurring_pixels_has_explicit_identity():
    old, metadata = baseline()
    psf = make_pyauto_convolver(make_pyauto_kernel(
        values=np.ones((1, 1)), pixel_scales=.1, normalize=False,
    ))
    old = al.Imaging(data=old.data, noise_map=old.noise_map, psf=psf, over_sample_size_lp=4)
    new, corrected = consistent_sampling_copy(old, metadata)
    contract = rendering_identity(new, corrected)
    assert contract["external_blurring_pixels_present"] is False
    assert contract["blurring_sub_size"] is None
    assert analysis_key_from(new, corrected, {})


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

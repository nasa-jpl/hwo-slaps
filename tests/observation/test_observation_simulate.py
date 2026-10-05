"""Detector-seed order, smooth controls and prepared trial-state reuse."""

from dataclasses import replace

import numpy as np
import pytest

pytestmark = pytest.mark.backend


def test_simulation_requires_noise_seed_and_subhalo(minimal_mapping):
    from hwoslaps.simulation import simulate

    with pytest.raises(TypeError, match="noise_seed"):
        simulate(minimal_mapping, subhalo=None)
    with pytest.raises(TypeError, match="subhalo"):
        simulate(minimal_mapping, noise_seed=None)


def test_noise_seed_is_a_reproducible_detector_draw_in_paper_order(minimal_mapping):
    from hwoslaps.fisher.api import prepare_forecast
    from hwoslaps.simulation import simulate

    with prepare_forecast(minimal_mapping) as prepared:
        expected = simulate(prepared, subhalo=None, noise_seed=None)
        assert expected is prepared.observation
        first = simulate(prepared, subhalo=None, noise_seed=11)
        replay = simulate(prepared, subhalo=None, noise_seed=11)
        other = simulate(prepared, subhalo=None, noise_seed=12)
        exposure = expected.exposure
        rng = np.random.default_rng(11)
        counts = rng.poisson(expected.counts_e())
        noise = rng.normal(0.0, exposure.read_sigma_e, size=counts.shape)
        manual = (counts + noise) / exposure.detector.gain_e_per_adu
        np.testing.assert_array_equal(first.data_adu, manual)
        np.testing.assert_array_equal(first.data_adu, replay.data_adu)
        assert not np.array_equal(first.data_adu, other.data_adu)
        assert first.sampling == expected.sampling
        np.testing.assert_array_equal(first.noise_map_adu, expected.noise_map_adu)


def test_prepared_injection_equals_standalone_and_keeps_smooth_sampling(minimal_mapping):
    from hwoslaps.fisher.api import prepare_forecast
    from hwoslaps.simulation import simulate

    with prepare_forecast(minimal_mapping) as prepared:
        trial = prepared.hypothesis(1.0e8, (0.1, 0.7))
        before = prepared.mean_truth_adu.copy()
        injected = simulate(prepared, subhalo=trial, noise_seed=None)
        standalone = simulate(minimal_mapping, subhalo=trial, noise_seed=None)
        np.testing.assert_array_equal(injected.expected_adu, standalone.expected_adu)
        np.testing.assert_array_equal(injected.noise_map_adu, standalone.noise_map_adu)
        np.testing.assert_array_equal(prepared.mean_truth_adu, before)
        assert injected.config_digest == prepared.record["config_digest"]
        assert injected.sampling == prepared.observation.sampling == standalone.sampling
        assert injected.subhalo == trial
        with pytest.raises(ValueError, match="hypothesis redshift"):
            simulate(prepared, subhalo=replace(trial, redshift=0.3), noise_seed=None)
        with pytest.raises(ValueError, match="source redshift"):
            simulate(prepared, subhalo=replace(trial, source_redshift=0.8), noise_seed=None)


def test_config_seed_does_not_supply_detector_seed(minimal_mapping):
    from hwoslaps.simulation import simulate

    first = simulate(minimal_mapping, subhalo=None, noise_seed=19)
    minimal_mapping["seed"] = 983
    other_config_seed = simulate(minimal_mapping, subhalo=None, noise_seed=19)
    np.testing.assert_array_equal(first.data_adu, other_config_seed.data_adu)


def test_standalone_simulation_needs_no_forecast_section(minimal_mapping):
    from hwoslaps.config.checks import ConfigError
    from hwoslaps.fisher.api import prepare_forecast
    from hwoslaps.simulation import simulate

    del minimal_mapping["forecast"]
    expected = simulate(minimal_mapping, subhalo=None, noise_seed=None)
    assert expected.kind == "expected"
    assert expected.sampling["source"] >= 0.0
    with pytest.raises(ConfigError, match="forecast"):
        prepare_forecast(minimal_mapping)


def test_prepared_asset_mapping_cannot_relabel_injected_science(minimal_mapping, image_asset):
    from hwoslaps.fisher.api import prepare_forecast
    from hwoslaps.scene.image_source import load_image_asset
    from hwoslaps.simulation import simulate

    minimal_mapping["scene"]["source"]["light"] = {"light": {"type": "Image", "asset_path": str(image_asset),
        "centre": [-0.03, 0.08], "flux_scale": 1.0, "size_scale": 1.0, "rotation_deg": 0.0, "total_flux": 1.0}}
    with prepare_forecast(minimal_mapping) as prepared:
        trial = prepared.hypothesis(1.0e8, (0.1, 0.7))
        before = simulate(prepared, subhalo=trial, noise_seed=None)
        with pytest.raises(TypeError):
            prepared.renderer.assets[str(image_asset)] = load_image_asset(image_asset)
        with pytest.raises(ValueError):
            prepared.renderer.assets[str(image_asset)].sb[0, 0] = 1.0
        prepared.validate_identity()
        after = simulate(prepared, subhalo=trial, noise_seed=None)
        np.testing.assert_array_equal(after.expected_adu, before.expected_adu)
        assert after.config_digest == before.config_digest

"""Contracts for persistent preparation and grid caches."""

from types import SimpleNamespace
import numpy as np
import pytest
from hwoslaps.modeling.nonlinear.experimental_persistent import persistent_preparation


def test_preparation_is_detached_keyed_and_restored(monkeypatch):
    from hwoslaps.lensing import generator as lens
    from hwoslaps.psf import generator as psf
    from hwoslaps.modeling.nonlinear import autolens_runner as runner

    calls = {"lens": 0, "psf": 0, "close": 0}

    def make_lens(config, full_config):
        calls["lens"] += 1
        return SimpleNamespace(image=np.array([config["value"]], dtype=float), config=full_config)

    def make_psf(config, full_config=None):
        calls["psf"] += 1
        return SimpleNamespace(kernel=np.array([config["value"]], dtype=float), config=full_config)

    def close():
        calls["close"] += 1

    monkeypatch.setattr(lens, "generate_lensing_system", make_lens)
    monkeypatch.setattr(psf, "generate_psf_system", make_psf)
    monkeypatch.setattr(runner, "_close_training_pools", close)
    cfg = {"global_seed": 1, "lensing": {"grid": {"pixel_scale": 0.01}}, "tag": "first"}
    with persistent_preparation(max_lensing=1, max_psf=1) as stats:
        first = lens.generate_lensing_system({"value": 3}, cfg)
        first.image[0] = 100
        later = dict(cfg, tag="second")
        hit = lens.generate_lensing_system({"value": 3}, later)
        assert hit.image[0] == 3 and hit.config["tag"] == "second"
        hit.image[0] = 200
        assert lens.generate_lensing_system({"value": 3}, later).image[0] == 3
        lens.generate_lensing_system({"value": 3}, dict(cfg, global_seed=2))
        lens.generate_lensing_system({"value": 3}, cfg)
        a = psf.generate_psf_system({"value": 7}, cfg)
        a.kernel[0] = 0
        assert psf.generate_psf_system({"value": 7}, later).kernel[0] == 7
        changed = dict(cfg, lensing={"grid": {"pixel_scale": 0.02}})
        psf.generate_psf_system({"value": 7}, changed)
        runner._close_training_pools()
        assert calls["close"] == 0
    assert calls == {"lens": 3, "psf": 2, "close": 1}
    assert stats["lensing_hits"] == 2 and stats["psf_hits"] == 1
    assert lens.generate_lensing_system is make_lens
    assert runner._close_training_pools is close


def test_scope_cleanup_on_failure(monkeypatch):
    from hwoslaps.modeling.nonlinear import autolens_runner as runner
    from hwoslaps.psf import generator as psf

    original = psf.generate_psf_system
    calls = []
    monkeypatch.setattr(runner, "_close_training_pools", lambda: calls.append(True))
    with pytest.raises(ValueError):
        with persistent_preparation():
            raise ValueError("test")
    assert psf.generate_psf_system is original
    assert calls == [True]


def test_grid_cache_all_inputs_and_detached_results():
    from autoarray.operators.over_sampling import over_sample_util as util

    original = util.grid_2d_slim_over_sampled_via_mask_from
    mask = np.zeros((5, 6), dtype=bool)
    args = dict(mask_2d=mask, pixel_scales=(0.01, 0.02), sub_size=np.array([2]), origin=(0.0, 0.0))
    reference = original(**args)
    with persistent_preparation() as stats:
        first = util.grid_2d_slim_over_sampled_via_mask_from(**args)
        assert np.array_equal(reference, first)
        first[:] = 100
        second = util.grid_2d_slim_over_sampled_via_mask_from(**args)
        assert np.array_equal(reference, second)
        second[:] = 200
        assert np.array_equal(reference, util.grid_2d_slim_over_sampled_via_mask_from(**args))
        for name, value in [
            ("origin", (0.01, 0.0)),
            ("pixel_scales", (0.02, 0.02)),
            ("sub_size", np.array([3])),
        ]:
            other = dict(args, **{name: value})
            assert np.array_equal(original(**other), util.grid_2d_slim_over_sampled_via_mask_from(**other))
        other_mask = mask.copy()
        other_mask[0, 0] = True
        other = dict(args, mask_2d=other_mask)
        assert np.array_equal(original(**other), util.grid_2d_slim_over_sampled_via_mask_from(**other))
    assert stats["grid_hits"] == 2
    assert stats["grid_peak_cache_bytes"] <= 1024**3
    assert util.grid_2d_slim_over_sampled_via_mask_from is original


def test_oversized_grids_bypass_bounded_cache():
    from autoarray.operators.over_sampling import over_sample_util as util

    args = dict(mask_2d=np.zeros((5, 6), dtype=bool), pixel_scales=(0.01, 0.02), sub_size=np.array([2]))
    with persistent_preparation(max_grid_bytes=32) as stats:
        first = util.grid_2d_slim_over_sampled_via_mask_from(**args)
        second = util.grid_2d_slim_over_sampled_via_mask_from(**args)
        assert np.array_equal(first, second)
    assert stats["grid_hits"] == 0 and stats["grid_oversized_bypasses"] == 2
    assert stats["grid_peak_cache_bytes"] == 0


def test_grid_cache_byte_budget_evicts_without_changing_values():
    from autoarray.operators.over_sampling import over_sample_util as util

    args = dict(mask_2d=np.zeros((5, 6), dtype=bool), pixel_scales=(0.01, 0.02), sub_size=np.array([2]))
    original = util.grid_2d_slim_over_sampled_via_mask_from
    expected = original(**args)
    with persistent_preparation(max_grid_bytes=expected.nbytes + 1) as stats:
        util.grid_2d_slim_over_sampled_via_mask_from(**args)
        util.grid_2d_slim_over_sampled_via_mask_from(**dict(args, origin=(0.1, 0.0)))
        restored = util.grid_2d_slim_over_sampled_via_mask_from(**args)
        assert np.array_equal(expected, restored)
        assert stats["grid_cache_bytes"] <= expected.nbytes + 1
    assert stats["grid_misses"] == 3 and stats["grid_hits"] == 0

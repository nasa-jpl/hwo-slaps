"""Public construction owns detector and plane-light arrays, including replacement."""

from dataclasses import fields, replace

import numpy as np
import pytest

from hwoslaps.artifacts import load_observation, save_observation
from hwoslaps.instrument import Detector
from hwoslaps.observation.expected import Exposure
from hwoslaps.observation.noise import draw_noisy_adu
from hwoslaps.observation.observation import Observation
from hwoslaps.optics.kernels import DetectorPSF, KernelBinding
from hwoslaps.scene.spec import GridSpec


def _inputs(*, noisy, lens_light):
    source = np.arange(6, dtype=float).reshape(2, 3) + 0.5
    by_plane = {"source": source}
    if lens_light:
        by_plane["lens"] = np.full(source.shape, 0.25)
    total = source + by_plane["lens"] if lens_light else source
    exposure = Exposure(Detector(2.0, 0.5, 0.1), 10.0, 1.0, 2)
    expected = exposure.mean_adu(total)
    psfs = KernelBinding.uniform(DetectorPSF.from_array(np.ones((1, 1)), 0.1, normalize=False), list(by_plane))
    return dict(kind="noisy" if noisy else "expected", data_adu=expected + 0.1 if noisy else expected,
                expected_adu=expected, noise_map_adu=exposure.noise_map_adu(total), light_rate_e_per_s=total,
                light_rate_by_plane_e_per_s=by_plane, grid=GridSpec((2, 3), 0.1, 1), exposure=exposure,
                psfs=psfs, noise_seed=11 if noisy else None, subhalo=None, config_digest="c" * 64,
                photometry=None, sampling={name: 0.01 for name in by_plane})


def _arrays(observation):
    return {name: getattr(observation, name) for name in
            ("data_adu", "expected_adu", "noise_map_adu", "light_rate_e_per_s")} | {
                f"plane:{name}": value for name, value in observation.light_rate_by_plane_e_per_s.items()}


@pytest.mark.parametrize("noisy", [False, True], ids=["expected", "noisy"])
@pytest.mark.parametrize("lens_light", [False, True], ids=["source-only", "lens-light"])
@pytest.mark.parametrize("construction", ["direct", "replace"])
@pytest.mark.parametrize("read_only_view", [False, True], ids=["writable-input", "read-only-view"])
def test_public_observation_detaches_every_caller_array(noisy, lens_light, construction, read_only_view):
    supplied = _inputs(noisy=noisy, lens_light=lens_light)
    names = ("data_adu", "expected_adu", "noise_map_adu", "light_rate_e_per_s")
    caller_buffers = [supplied[name] for name in names] + list(supplied["light_rate_by_plane_e_per_s"].values())
    if read_only_view:
        views = {}
        for value in caller_buffers:
            view = value.view()
            view.flags.writeable = False
            views[id(value)] = view
        for name in names:
            supplied[name] = views[id(supplied[name])]
        supplied["light_rate_by_plane_e_per_s"] = {
            plane: views[id(value)] for plane, value in supplied["light_rate_by_plane_e_per_s"].items()}
    if construction == "direct":
        observation = Observation(**supplied)
    else:
        original = Observation(**_inputs(noisy=noisy, lens_light=lens_light))
        observation = replace(original, **supplied)
    originals = {name: value.copy() for name, value in _arrays(observation).items()}
    record = observation.to_mapping()
    for name, value in _arrays(observation).items():
        assert not value.flags.writeable, name
        with pytest.raises(ValueError):
            value[0, 0] += 1.0
    # Every caller buffer remains writable, including views and aliased fields.
    for value in caller_buffers:
        assert value.flags.writeable
        value[0, 0] += 1000.0
    supplied["light_rate_by_plane_e_per_s"].clear()
    supplied["sampling"]["source"] = 99.0
    assert observation.to_mapping() == record
    for name, value in _arrays(observation).items():
        np.testing.assert_array_equal(value, originals[name])
    assert (observation.data_adu is observation.expected_adu) == (not noisy)
    if not lens_light:
        assert observation.light_rate_by_plane_e_per_s["source"] is observation.light_rate_e_per_s


def test_replace_noisy_data_reproduces_review_mutation_boundary():
    original = Observation(**_inputs(noisy=True, lens_light=False))
    supplied = original.data_adu.copy()
    replaced = replace(original, data_adu=supplied)
    record = replaced.to_mapping()
    supplied[0, 0] += 1000.0
    assert replaced.to_mapping() == record
    np.testing.assert_array_equal(replaced.data_adu, original.data_adu)
    assert not replaced.data_adu.flags.writeable


@pytest.mark.parametrize("lens_light", [False, True], ids=["source-only", "lens-light"])
def test_draw_and_load_retain_values_aliases_and_read_only_arrays(tmp_path, lens_light):
    expected = Observation(**_inputs(noisy=False, lens_light=lens_light))
    noisy = expected.draw(17)
    np.testing.assert_array_equal(noisy.data_adu, draw_noisy_adu(expected.counts_e(), expected.exposure, 17))
    for field in fields(expected):
        if field.name in {"kind", "data_adu", "noise_seed"}:
            continue
        assert getattr(noisy, field.name) is getattr(expected, field.name), field.name
    assert (noisy.kind, noisy.noise_seed) == ("noisy", 17)
    for item in (expected, noisy):
        loaded = load_observation(save_observation(item, tmp_path / f"{item.kind}.npz"))
        assert loaded.to_mapping() == item.to_mapping()
        for name, value in _arrays(loaded).items():
            np.testing.assert_array_equal(value, _arrays(item)[name])
            assert not value.flags.writeable
        assert (loaded.data_adu is loaded.expected_adu) == (item.kind == "expected")
        if not lens_light:
            assert loaded.light_rate_by_plane_e_per_s["source"] is loaded.light_rate_e_per_s

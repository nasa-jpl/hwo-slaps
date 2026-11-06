"""The optional uniform-sampling heuristic must respect supported physical rendering."""

from copy import deepcopy
import json

import numpy as np
import pytest

from hwoslaps.artifacts import load_observation, save_observation
from hwoslaps.config.schema import resolve_config
from hwoslaps.fisher.api import Execution, prepare_forecast
from hwoslaps.scene.builder import build_scene, over_sampled_grid
from hwoslaps.scene.cosmology import Cosmology
from hwoslaps.scene.perturbers import realize_perturbers
from hwoslaps.scene.spec import GridSpec
from hwoslaps.simulation import simulate

pytestmark = pytest.mark.backend


def _mapping(minimal_mapping, tmp_path, *, shape, sub, kernel_size=3):
    mapping = deepcopy(minimal_mapping)
    kernel = np.arange(1.0, kernel_size**2 + 1).reshape(kernel_size, kernel_size)
    path = tmp_path / "sampling-kernel.npy"
    np.save(path, kernel/kernel.sum())
    mapping["scene"]["grid"] = {"shape": list(shape), "pixel_scale_arcsec": 0.1, "over_sample_size": sub}
    mapping["scene"]["lens"]["mass"]["mass"].update(centre=[0.007, -0.013], einstein_radius=0.18)
    mapping["scene"]["source"]["light"]["light"].update(effective_radius=0.2)
    mapping["psf"] = {"truth": {"kind": "kernel", "path": str(path), "pixel_scale_arcsec": 0.1}}
    mapping["forecast"]["nuisances"] = {"fixed": ["*"], "background_offset": False}
    mapping["forecast"]["positions"] = {"kind": "explicit", "positions_yx": [[0.13, -0.17]]}
    return mapping


def _scene(mapping):
    config = resolve_config(mapping)
    cosmology = Cosmology(config.cosmology)
    perturbers = realize_perturbers(config.scene, cosmology, seed=config.seed)
    return build_scene(config.scene, cosmology, subhalo=None, perturbers=perturbers)


def _public_observation(mapping, entrypoint):
    if entrypoint == "simulate":
        return simulate(mapping, subhalo=None, noise_seed=None)
    with prepare_forecast(mapping, execution=Execution(engine="reference")) as prepared:
        assert dict(prepared.record["sampling"]) == dict(prepared.observation.sampling)
        np.testing.assert_array_equal(simulate(prepared, subhalo=None, noise_seed=None).data_adu,
                                      prepared.observation.data_adu)
        return prepared.observation


def _assert_roundtrips(observation, tmp_path):
    for item in (observation, observation.draw(19)):
        restored = load_observation(save_observation(item, tmp_path / f"{item.kind}.npz"))
        assert dict(restored.sampling) == dict(item.sampling)
        assert restored.to_mapping() == item.to_mapping()
        np.testing.assert_array_equal(restored.data_adu, item.data_adu)


@pytest.mark.parametrize("entrypoint", ["simulate", "prepare_forecast"])
@pytest.mark.parametrize("kind", ["PointMass", "SIS"])
@pytest.mark.parametrize("sub", [1, 2], ids=["native-alignment", "subpixel-alignment"])
def test_aligned_fixed_singular_halo_keeps_finite_public_rendering(
        minimal_mapping, tmp_path, entrypoint, kind, sub):
    mapping = _mapping(minimal_mapping, tmp_path, shape=(5, 7), sub=sub)
    samples = np.asarray(over_sampled_grid(GridSpec((5, 7), 0.1, sub)).over_sampled)
    centre = samples[len(samples)//2].tolist()
    mapping["scene"]["perturbers"] = {"halos": [{"type": kind, "mass_msun": 1.0e8, "centre": centre}]}
    assert mapping["scene"]["subhalo"]["type"] == "NFW"
    # The physical native-pixel integral succeeds before any sampling diagnostic.
    scene = _scene(mapping)
    assert np.all(np.isfinite(scene.light_images["source"]))
    assert np.sum(scene.light_images["source"]) > 0.0
    observation = _public_observation(mapping, entrypoint)
    assert np.all(np.isfinite(observation.data_adu))
    assert observation.sampling["source"] is None
    _assert_roundtrips(observation, tmp_path)


@pytest.mark.parametrize("entrypoint", ["simulate", "prepare_forecast"])
def test_exterior_image_scattered_onto_detector_keeps_crop_and_unassessed_diagnostic(
        minimal_mapping, tmp_path, entrypoint):
    mapping = _mapping(minimal_mapping, tmp_path, shape=(11, 11), sub=2, kernel_size=5)
    asset = tmp_path / "compact-exterior.npz"
    sb = np.ones((9, 9), dtype=float)
    sb /= sb.sum()*0.01**2
    np.savez(asset, sb=sb, pixel_scale_arcsec=np.asarray(0.01),
             metadata_json=np.asarray(json.dumps({"format_version": 1, "provenance": {}})))
    mapping["scene"]["lens"]["light"] = {"exterior": {
        "type": "Image", "asset_path": str(asset), "centre": [0.0, 0.65], "total_flux": 1.0}}
    scene = _scene(mapping)
    np.testing.assert_array_equal(scene.light_images["lens"], 0.0)
    small = _public_observation(mapping, entrypoint)
    larger = deepcopy(mapping)
    larger["scene"]["grid"]["shape"] = [21, 21]
    big = simulate(larger, subhalo=None, noise_seed=None)
    assert np.max(small.light_rate_by_plane_e_per_s["lens"]) > 1.0e-6
    assert small.sampling["lens"] is None
    assert np.isfinite(small.sampling["source"])
    assert big.sampling["lens"] is not None
    crop = (slice(5, 16), slice(5, 16))
    for plane in ("lens", "source"):
        np.testing.assert_allclose(small.light_rate_by_plane_e_per_s[plane],
                                   big.light_rate_by_plane_e_per_s[plane][crop], rtol=2e-12, atol=2e-14)
    for name in ("expected_adu", "noise_map_adu"):
        np.testing.assert_allclose(getattr(small, name), getattr(big, name)[crop], rtol=2e-12, atol=2e-12)
    _assert_roundtrips(small, tmp_path)


@pytest.mark.parametrize("entrypoint", ["simulate", "prepare_forecast"])
def test_unresolved_physical_integral_still_raises(minimal_mapping, tmp_path, monkeypatch, entrypoint):
    from hwoslaps.scene import singular_integration

    mapping = _mapping(minimal_mapping, tmp_path, shape=(5, 7), sub=1)
    mapping["scene"]["perturbers"] = {"halos": [
        {"type": "PointMass", "mass_msun": 1.0e8, "centre": [0.0, 0.0]}]}

    def unresolved(*args, **kwargs):
        raise singular_integration.SingularIntegrationError("unresolved physical integral")

    monkeypatch.setattr(singular_integration, "correct_group_images", unresolved)
    with pytest.raises(singular_integration.SingularIntegrationError, match="unresolved physical integral"):
        _public_observation(mapping, entrypoint)

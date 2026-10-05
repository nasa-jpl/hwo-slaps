"""Scene builder: registry wiring, light per plane, grid state, sampling diagnostic, flux, refusals."""

import dataclasses
import json
import re

import numpy as np
import pytest

from hwoslaps.scene.builder import build_scene, native_sampling_variation, render_component_unlensed
from hwoslaps.scene.halos import make_halo
from hwoslaps.scene.image_source import load_image_asset
from hwoslaps.scene.spec import parse_scene, pixel_centres_yx

pytestmark = pytest.mark.backend

BULGE = {"type": "Exponential", "centre": [0.01, -0.02], "ell_comps": [0.0, 0.1], "effective_radius": 0.3,
         "intensity": 3.0}


def _grid(shape=(40, 40), scale=0.05, sub=2):
    import autolens as al

    return al.Grid2D.uniform(shape_native=shape, pixel_scales=scale, over_sample_size=sub)


def _direct_lens(redshift=0.2):
    import autolens as al

    return al.Galaxy(redshift=redshift, main=al.mp.Isothermal(centre=(0.0, 0.0), einstein_radius=0.8,
                                                              ell_comps=(0.05, 0.0)))


@pytest.mark.parametrize("source", ["exponential", "image"])
def test_registry_expansion_equals_direct_autolens_construction(scene_mapping, planck15, image_asset, source):
    import autolens as al

    from hwoslaps.scene.image_profile import ImageLightProfile

    if source == "image":
        scene_mapping["source"]["light"] = {"knots": {"type": "Image", "asset_path": str(image_asset),
                                                      "centre": [0.02, -0.03], "rotation_deg": 25.0, "total_flux": 0.3,
                                                      "flux_scale": 1.1, "size_scale": 0.9}}
        with np.load(image_asset) as data:
            light = ImageLightProfile(centre=(0.02, -0.03), rotation_deg=25.0,
                                      pixel_scale_arcsec=float(data["pixel_scale_arcsec"]), sb=data["sb"],
                                      total_flux=0.3, flux_scale=1.1, size_scale=0.9)
        direct_source = al.Galaxy(redshift=0.6, knots=light)
    else:
        direct_source = al.Galaxy(redshift=0.6, disk=al.lp.Exponential(centre=(0.02, -0.03), ell_comps=(0.1, 0.05),
                                                                       intensity=1.0, effective_radius=0.12))
    scene = build_scene(parse_scene(scene_mapping), planck15, subhalo=None)
    direct = al.Tracer(galaxies=[_direct_lens(), direct_source], cosmology=al.cosmo.Planck15())
    np.testing.assert_array_equal(scene.light_images["source"], direct.image_2d_from(grid=_grid()).native.array)
    assert not scene.light_images["source"].flags.writeable
    assert list(scene.light_images) == list(scene.light_groups) == ["source"]


def test_lens_light_image_is_the_sum_of_plane_images(scene_mapping, planck15):
    import autolens as al

    smooth_source = build_scene(parse_scene(scene_mapping), planck15, subhalo=None).light_images["source"]
    scene_mapping["lens"]["light"] = {"bulge": BULGE}
    spec = parse_scene(scene_mapping)
    scene = build_scene(spec, planck15, subhalo=None)
    unlensed = al.lp.Exponential(centre=(0.01, -0.02), ell_comps=(0.0, 0.1), intensity=3.0,
                                 effective_radius=0.3).image_2d_from(grid=_grid()).native.array
    np.testing.assert_array_equal(scene.light_images["lens"], unlensed)
    np.testing.assert_array_equal(render_component_unlensed(spec.lens.light[0], spec.grid), unlensed)
    np.testing.assert_array_equal(scene.light_images["source"], smooth_source)
    np.testing.assert_array_equal(scene.tracer.image_2d_from(grid=scene.grid).native.array,
                                  scene.light_images["lens"] + scene.light_images["source"])
    assert list(scene.light_groups) == ["lens", "source"]
    assert [type(profile).__name__ for profile in scene.light_profiles["lens"]] == ["Exponential"]


@pytest.mark.parametrize("shape, scale", [((40, 40), 0.05), ((17, 24), 0.03), ((5, 3), 0.00716), ((100, 100), 0.03)])
def test_pixel_centres_equal_the_autolens_grid(shape, scale):
    y, x = pixel_centres_yx(shape, scale)
    native = _grid(shape, scale).native.array
    np.testing.assert_array_equal(y, native[..., 0])
    np.testing.assert_array_equal(x, native[..., 1])


def test_scene_builds_share_no_grid_state(scene_mapping, planck15):
    spec = parse_scene(scene_mapping)
    first = build_scene(spec, planck15, subhalo=None)
    second = build_scene(spec, planck15, subhalo=None)
    assert first.grid is not second.grid and first.grid.over_sampled is not second.grid.over_sampled
    np.testing.assert_array_equal(first.light_images["source"], second.light_images["source"])
    expected = np.array(second.grid.over_sampled.array, copy=True)
    first.grid.over_sampled.array[0, 0] += 1.0
    third = build_scene(spec, planck15, subhalo=None)
    np.testing.assert_array_equal(third.grid.over_sampled.array, expected)
    np.testing.assert_array_equal(third.light_images["source"], second.light_images["source"])
    assert first != second and len({first, second, third}) == 3


def _sub_pixel_coordinates(shape, scale, sub):
    """Sub-pixel centres of every pixel, (pixels, sub * sub, 2), built from the pixel centres."""
    y, x = pixel_centres_yx(shape, scale)
    offsets = (np.arange(sub) + 0.5) / sub - 0.5
    sub_y = y.reshape(-1, 1, 1) - offsets.reshape(1, -1, 1) * scale
    sub_x = x.reshape(-1, 1, 1) + offsets.reshape(1, 1, -1) * scale
    return np.stack(np.broadcast_arrays(sub_y, sub_x), axis=-1).reshape(y.size, sub * sub, 2)


def _variation(samples):
    means = samples.mean(axis=1, keepdims=True)
    return np.sqrt(np.sum((samples - means) ** 2) / (np.sum(means**2) * samples.shape[1]))


def test_native_sampling_variation_is_the_relative_within_pixel_rms(scene_mapping, planck15):
    import autolens as al

    scene_mapping["grid"].update(shape=[24, 24], over_sample_size=4)
    scene_mapping["lens"]["light"] = {"bulge": dict(BULGE, effective_radius=0.08)}
    scene = build_scene(parse_scene(scene_mapping), planck15, subhalo=None)
    coordinates = _sub_pixel_coordinates((24, 24), 0.05, 4)
    flat = al.Grid2DIrregular(values=coordinates.reshape(-1, 2))
    lens_light = al.lp.Exponential(centre=(0.01, -0.02), ell_comps=(0.0, 0.1), intensity=3.0, effective_radius=0.08)
    lens_samples = np.asarray(lens_light.image_2d_from(grid=flat)).reshape(-1, 16)
    deflections = np.asarray(_direct_lens().deflections_yx_2d_from(grid=flat))
    source_light = al.lp.Exponential(centre=(0.02, -0.03), ell_comps=(0.1, 0.05), intensity=1.0, effective_radius=0.12)
    source_samples = np.asarray(source_light.image_2d_from(
        grid=al.Grid2DIrregular(values=coordinates.reshape(-1, 2) - deflections))).reshape(-1, 16)

    variation = native_sampling_variation(scene)
    assert variation["lens"] == pytest.approx(_variation(lens_samples), rel=1.0e-10)
    assert variation["source"] == pytest.approx(_variation(source_samples), rel=1.0e-10)
    assert 0.01 < variation["lens"] < 1.0 and 0.01 < variation["source"] < 1.0
    np.testing.assert_allclose(scene.light_images["lens"].ravel(), lens_samples.mean(axis=1), rtol=1.0e-12)

    scene_mapping["grid"]["over_sample_size"] = 1
    assert dict(native_sampling_variation(build_scene(parse_scene(scene_mapping), planck15, subhalo=None))) == {
        "lens": 0.0, "source": 0.0}


def test_extended_image_source_is_magnified_once(planck15, tmp_path):
    """A small source behind a circular isothermal lens follows mu = 2 theta_E / beta (SIS, beta < theta_E)."""
    scale = 0.01
    axis = (np.arange(31) - 15) * scale
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    sb = np.exp(-(xx**2 + yy**2) / (2 * 0.05**2))
    sb /= sb.sum() * scale**2
    path = tmp_path / "gaussian.npz"
    np.savez(path, sb=sb, pixel_scale_arcsec=np.asarray(scale),
             metadata_json=np.asarray(json.dumps({"format_version": 1, "provenance": {}})))
    theta_e, intrinsic = 2.0, 7.4
    beta = np.hypot(yy, 1.0 + xx)
    expected = intrinsic * np.sum(sb * scale**2 * (2 * theta_e / beta))

    def scene(total_flux):
        mapping = {"grid": {"shape": [400, 400], "pixel_scale_arcsec": 0.02, "over_sample_size": 2},
                   "lens": {"redshift": 0.2, "mass": {"main": {"type": "Isothermal", "centre": [0.0, 0.0],
                                                               "einstein_radius": theta_e, "ell_comps": [0.0, 0.0]}}},
                   "source": {"redshift": 0.6, "light": {"blob": {"type": "Image", "asset_path": str(path),
                                                                   "centre": [0.0, 1.0], "total_flux": total_flux}}},
                   "subhalo": {"type": "PointMass"}}
        return build_scene(parse_scene(mapping), planck15, subhalo=None).light_images["source"]

    image = scene(intrinsic)
    assert np.all(image[[0, -1], :] == 0.0) and np.all(image[:, [0, -1]] == 0.0)
    assert image.sum() * 0.02**2 == pytest.approx(expected, rel=0.01)
    np.testing.assert_array_equal(scene(2 * intrinsic), 2 * image)


def test_prepared_assets_render_without_reading_files(scene_mapping, planck15, image_asset):
    scene_mapping["source"]["light"] = {"knots": {"type": "Image", "asset_path": str(image_asset),
                                                  "centre": [0.0, 0.1], "total_flux": 0.4}}
    spec = parse_scene(scene_mapping)
    from_file = build_scene(spec, planck15, subhalo=None)
    assets = {str(image_asset): load_image_asset(image_asset)}
    image_asset.unlink()
    prepared = build_scene(spec, planck15, subhalo=None, assets=assets)
    np.testing.assert_array_equal(prepared.light_images["source"], from_file.light_images["source"])
    with pytest.raises(KeyError, match="not among the prepared assets"):
        build_scene(spec, planck15, subhalo=None, assets={})


def test_unrenderable_scenes_are_refused(scene_mapping, planck15):
    spec = parse_scene(scene_mapping)
    with pytest.raises(ValueError, match="cannot lens|lenses a source"):
        build_scene(spec, planck15, subhalo=make_halo(spec.subhalo, 1.0e8, (0.1, 0.2), redshift=0.2,
                                                      source_redshift=0.9, cosmology=planck15))
    for name, owner in (("id", "al.Galaxy"), ("info", "af.Model(al.Galaxy)"), ("has", "al.Galaxy")):
        renamed = dataclasses.replace(spec.lens.mass[0], name=name)
        shadowed = rf"scene\.lens\.mass\.{name}: .* shadows an attribute of {re.escape(owner)}"
        with pytest.raises(ValueError, match=shadowed):
            build_scene(dataclasses.replace(spec, lens=dataclasses.replace(spec.lens, mass=(renamed,))), planck15,
                        subhalo=None)
    scene_mapping["source"]["light"]["disk"]["intensity"] = 1.0e308
    with pytest.raises(ValueError, match="non-finite"):
        build_scene(parse_scene(scene_mapping), planck15, subhalo=None)

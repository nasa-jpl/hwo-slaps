"""Image assets (storage format, loader, preparation) and the bilinear image light profile."""

import hashlib
import json
import math
import pickle

import numpy as np
import pytest
from scipy.integrate import trapezoid

from hwoslaps.scene.image_source import ImageAsset, load_image_asset, prepare_image_asset
from hwoslaps.scene.profiles import PROFILE_TYPES


def _unit_sb(shape=(8, 10), pixel_scale=0.2):
    rows, cols = np.indices(shape, dtype=float)
    sb = np.exp(-0.5 * (((rows - 3.1) / 1.0) ** 2 + ((cols - 5.4) / 1.3) ** 2))
    return sb / (pixel_scale**2 * sb.sum())


def _first_row_set(value):
    sb = _unit_sb()
    sb[0] = value
    return sb


def _write(path, *, sb=None, pixel_scale=np.asarray(0.2), metadata_json=None, extra=None, omit=None):
    """An .npz in the version-1 asset layout, with one member replaced, added or left out."""
    members = {"sb": _unit_sb() if sb is None else sb, "pixel_scale_arcsec": pixel_scale,
               "metadata_json": np.asarray(json.dumps({"format_version": 1, "provenance": {"kind": "synthetic"}}))
               if metadata_json is None else metadata_json}
    members.update(extra or {})
    members.pop(omit, None)
    np.savez(path, **members)
    return path


def _metadata(document):
    return np.asarray(json.dumps(document))


LOADER_ROWS = [
    ("unnormalized", dict(sb=_unit_sb() * (1.0 + 1.0e-6)), "normalized"),
    ("negative-sample", dict(sb=_first_row_set(-1.0)), "non-negative"),
    ("non-finite-sample", dict(sb=_first_row_set(np.nan)), "finite"),
    ("one-dimensional", dict(sb=np.full(8, 1.0 / (0.04 * 8))), "2-D"),
    ("side-below-8", dict(sb=np.full((8, 7), 1.0 / (0.04 * 56))), "between 8 and 4096"),
    ("side-above-4096", dict(sb=np.full((8, 4097), 1.0 / (0.04 * 8 * 4097))), "between 8 and 4096"),
    ("float32-samples", dict(sb=_unit_sb().astype(np.float32)), "float64"),
    ("zero-pixel-scale", dict(pixel_scale=np.asarray(0.0)), "positive"),
    ("float32-pixel-scale", dict(pixel_scale=np.asarray(0.2, dtype=np.float32)), "float64 scalar"),
    ("integer-pixel-scale", dict(pixel_scale=np.asarray(2)), "float64 scalar"),
    ("missing-member", dict(omit="sb"), "exactly"),
    ("extra-member", dict(extra={"weights": np.ones(3)}), "exactly"),
    ("metadata-not-a-scalar", dict(metadata_json=np.asarray(["{}"])), "0-d string"),
    ("metadata-not-json", dict(metadata_json=np.asarray("{invalid")), "valid JSON"),
    ("metadata-not-an-object", dict(metadata_json=np.asarray("[]")), "JSON object"),
    ("other-format-version", dict(metadata_json=_metadata({"format_version": 2, "provenance": {}})), "format_version"),
    ("boolean-format-version", dict(metadata_json=_metadata({"format_version": True, "provenance": {}})),
     "format_version"),
    ("float-format-version", dict(metadata_json=_metadata({"format_version": 1.0, "provenance": {}})),
     "format_version"),
    ("missing-provenance", dict(metadata_json=_metadata({"format_version": 1})), "provenance"),
]


@pytest.mark.parametrize("member, fragment", [row[1:] for row in LOADER_ROWS], ids=[row[0] for row in LOADER_ROWS])
def test_image_asset_loader_validates_the_storage_format(tmp_path, member, fragment):
    path = _write(tmp_path / "asset.npz", **member)
    with pytest.raises(ValueError, match=fragment):
        load_image_asset(path)


def test_image_asset_loader_reads_the_format_and_reloads_rewritten_files(tmp_path):
    path = _write(tmp_path / "asset.npz", sb=_unit_sb() * (1.0 + 5.0e-9))
    asset = load_image_asset(path)
    assert isinstance(asset, ImageAsset)
    np.testing.assert_array_equal(asset.sb, _unit_sb() * (1.0 + 5.0e-9))
    assert asset.pixel_scale_arcsec == 0.2 and not asset.sb.flags.writeable
    assert asset.metadata == {"format_version": 1, "provenance": {"kind": "synthetic"}}
    assert asset.digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert load_image_asset(str(path)) is asset
    with pytest.raises(TypeError):
        asset.metadata["provenance"]["kind"] = "edited"

    _write(path, sb=_unit_sb(shape=(9, 10)))
    reloaded = load_image_asset(path)
    assert reloaded is not asset and reloaded.sb.shape == (9, 10)


def _evaluate(profile, points):
    import autolens as al

    return np.asarray(profile.image_2d_from(grid=al.Grid2DIrregular(values=np.asarray(points, dtype=float))),
                      dtype=float)


def _profile(**overrides):
    from hwoslaps.scene.image_profile import ImageLightProfile

    arguments = dict(centre=(0.3, -0.2), rotation_deg=0.0, pixel_scale_arcsec=0.2, sb=_unit_sb(), total_flux=1.7,
                     flux_scale=1.0, size_scale=1.0)
    return ImageLightProfile(**{**arguments, **overrides})


@pytest.mark.backend
def test_image_profile_samples_bilinearly_with_a_zero_pad():
    sb = _unit_sb()
    profile = _profile(flux_scale=1.3, size_scale=1.2)
    row_c, col_c = (sb.shape[0] - 1) / 2.0, (sb.shape[1] - 1) / 2.0
    step = 0.2 * 1.2
    y, x = 0.3 + (3 - row_c) * step, -0.2 + (4 - col_c) * step
    amplitude = 1.7 * 1.3
    centre_value, midpoint = _evaluate(profile, [(y, x), (y, x + 0.5 * step)])
    assert centre_value == pytest.approx(amplitude * sb[3, 4], rel=1.0e-14)
    assert midpoint == pytest.approx(amplitude * 0.5 * (sb[3, 4] + sb[3, 5]), rel=1.0e-14)

    ny, nx = sb.shape
    x_column = -0.2 + (3 - col_c) * step
    edge_low, outside_low, edge_high, outside_high, outside_right = _evaluate(profile, [
        (0.3 + (-0.5 - row_c) * step, x_column), (0.3 + (-1.01 - row_c) * step, x_column),
        (0.3 + (ny - 0.5 - row_c) * step, x_column), (0.3 + (ny + 0.01 - row_c) * step, x_column),
        (0.3 + (3 - row_c) * step, -0.2 + (nx + 0.01 - col_c) * step)])
    assert edge_low == pytest.approx(0.5 * amplitude * sb[0, 3], rel=1.0e-14)
    assert edge_high == pytest.approx(0.5 * amplitude * sb[ny - 1, 3], rel=1.0e-14)
    assert outside_low == outside_high == outside_right == 0.0


@pytest.mark.backend
def test_image_profile_rotates_scales_and_translates_the_image():
    sb = _unit_sb()
    row_c, col_c = (sb.shape[0] - 1) / 2.0, (sb.shape[1] - 1) / 2.0
    u, v = (7 - col_c) * 0.2, (2 - row_c) * 0.2
    assert _evaluate(_profile(rotation_deg=90.0), [(0.3 + u, -0.2 - v)])[0] == pytest.approx(1.7 * sb[2, 7], rel=1.0e-14)

    points = np.array([[0.1, -0.4], [0.35, -0.12], [0.6, 0.1]])
    base = _evaluate(_profile(), points)
    np.testing.assert_allclose(_evaluate(_profile(flux_scale=1.3), points), 1.3 * base, rtol=1.0e-12, atol=0.0)
    contracted = np.array([0.3, -0.2]) + (points - np.array([0.3, -0.2])) / 1.2
    np.testing.assert_allclose(_evaluate(_profile(size_scale=1.2), points), _evaluate(_profile(), contracted),
                               rtol=1.0e-12, atol=0.0)

    rows = np.linspace(-1.0, sb.shape[0], (sb.shape[0] + 1) * 20 + 1)
    cols = np.linspace(-1.0, sb.shape[1], (sb.shape[1] + 1) * 20 + 1)
    grid_rows, grid_cols = np.meshgrid(rows, cols, indexing="ij")
    yy, xx = 0.3 + (grid_rows - row_c) * 0.24, -0.2 + (grid_cols - col_c) * 0.24
    values = _evaluate(_profile(flux_scale=1.3, size_scale=1.2), np.column_stack((yy.ravel(), xx.ravel())))
    integral = trapezoid(trapezoid(values.reshape(yy.shape), x=xx[0], axis=1), x=yy[:, 0])
    expected = 1.7 * PROFILE_TYPES["Image"].unit_integral({"flux_scale": 1.3, "size_scale": 1.2})
    assert integral == pytest.approx(expected, rel=1.0e-6)
    assert expected == pytest.approx(1.7 * 1.3 * 1.2**2, rel=1.0e-15)


@pytest.mark.backend
def test_image_profile_matches_lenstronomy_interpolation():
    from lenstronomy.LightModel.Profiles.interpolation import Interpol

    sb = _unit_sb()
    profile = _profile(centre=(0.13, -0.27), rotation_deg=23.0, total_flux=1.0)
    points = np.array([[0.1, -0.3], [0.2, -0.1], [-0.2, -0.5]])
    theta = np.deg2rad(23.0)
    dy, dx = points[:, 0] - 0.13, points[:, 1] + 0.27
    u = dx * np.cos(theta) + dy * np.sin(theta)
    v = -dx * np.sin(theta) + dy * np.cos(theta)
    reference = Interpol().function(x=u, y=v, image=sb, center_x=0.0, center_y=0.0, phi_G=0.0, scale=0.2)
    np.testing.assert_allclose(_evaluate(profile, points), reference, rtol=0.0, atol=1.0e-12 * sb.max())


@pytest.mark.backend
def test_image_profile_survives_pickling():
    profile = _profile(rotation_deg=10.0, flux_scale=1.3, size_scale=1.2)
    points = [(0.3, -0.2), (0.1, -0.1)]
    before = _evaluate(profile, points)
    np.testing.assert_array_equal(_evaluate(pickle.loads(pickle.dumps(profile)), points), before)


SIGMA_PIXELS = 6.0


def _galaxy_frame(bin_factor=1):
    """A circular Gaussian (sigma 6 binned pixels, peak 1) at binned pixel (50, 46) of a 96 x 96 binned frame,
    on a constant 0.37 offset with Gaussian noise of 0.01 and a bright blob in a corner.

    Centring then crops binned rows 5-95 and columns 0-92 and pads two rows: a 93 x 93 asset
    whose corner blob, if kept, would sit at rows 0-2, columns 3-6."""
    shape = (96 * bin_factor, 96 * bin_factor)
    rows, cols = np.indices(shape, dtype=float)
    centre = (50 * bin_factor + (bin_factor - 1) / 2.0, 46 * bin_factor + (bin_factor - 1) / 2.0)
    galaxy = np.exp(-0.5 * ((rows - centre[0]) ** 2 + (cols - centre[1]) ** 2) / (SIGMA_PIXELS * bin_factor) ** 2)
    frame = galaxy + 0.37 + np.random.default_rng(5).normal(0.0, 0.01, shape)
    frame[3 * bin_factor:7 * bin_factor, 3 * bin_factor:7 * bin_factor] += 0.8
    return frame


@pytest.mark.parametrize("bin_factor", [1, 2])
def test_prepare_image_asset_recovers_a_synthetic_galaxy(bin_factor):
    asset = prepare_image_asset(_galaxy_frame(bin_factor), half_light_radius_arcsec=0.12, bin_factor=bin_factor,
                                provenance={"catalog_id": "synthetic"})
    assert asset.pixel_scale_arcsec**2 * asset.sb.sum() == pytest.approx(1.0, rel=1.0e-12)
    record = asset.metadata["provenance"]
    r_half = SIGMA_PIXELS * math.sqrt(2.0 * math.log(2.0))
    assert record["r_half_pixels"] == pytest.approx(r_half, rel=0.02)
    assert asset.pixel_scale_arcsec == pytest.approx(0.12 / r_half, rel=0.02)
    assert record["background"] == pytest.approx(0.37, abs=0.005)
    assert record["caller"] == {"catalog_id": "synthetic"} and record["bin_factor"] == bin_factor
    assert record["input_shape"] == (96 * bin_factor, 96 * bin_factor)
    rows, cols = np.indices(asset.sb.shape, dtype=float)
    middle = ((asset.sb.shape[0] - 1) / 2.0, (asset.sb.shape[1] - 1) / 2.0)
    assert abs((rows * asset.sb).sum() / asset.sb.sum() - middle[0]) <= 0.5
    assert abs((cols * asset.sb).sum() / asset.sb.sum() - middle[1]) <= 0.5
    model = np.exp(-0.5 * ((rows - middle[0]) ** 2 + (cols - middle[1]) ** 2) / SIGMA_PIXELS**2)
    model /= asset.pixel_scale_arcsec**2 * model.sum()
    footprint = asset.sb > 0
    assert np.linalg.norm((asset.sb - model)[footprint]) <= 0.05 * np.linalg.norm(model[footprint])
    assert asset.sb.shape == (93, 93) and asset.sb[:20, :20].max() == 0.0


def _edge_frame():
    rows, cols = np.indices((64, 64), dtype=float)
    return np.exp(-0.5 * ((rows - 2.0) ** 2 + (cols - 32.0) ** 2) / 36.0) + np.random.default_rng(3).normal(
        0.0, 0.01, (64, 64))


@pytest.mark.parametrize("image, keywords, fragment", [
    ("edge", {}, "touches the image edge"),
    ("galaxy", {"footprint_sigma": 1.0e6}, "no source footprint"),
    ("point", {}, "unresolved"),
    ("non-finite", {}, "finite 2-D array"),
    ("galaxy", {"bin_factor": 0}, "bin_factor"),
    ("galaxy", {"pixel_scale_arcsec": 0.01}, "exactly one"),
    ("galaxy", {"half_light_radius_arcsec": None}, "exactly one"),
], ids=["footprint-at-the-edge", "nothing-above-the-threshold", "unresolved-centre", "non-finite-input",
        "bin-factor-zero", "both-scales", "no-scale"])
def test_prepare_image_asset_refuses_unusable_inputs(image, keywords, fragment):
    frames = {"edge": _edge_frame, "galaxy": _galaxy_frame, "point": lambda: np.pad(np.ones((1, 1)), 20),
              "non-finite": lambda: np.where(np.indices((96, 96))[0] == 50, np.inf, _galaxy_frame())}
    with pytest.raises(ValueError, match=fragment):
        prepare_image_asset(frames[image](), **{"half_light_radius_arcsec": 0.12, **keywords})

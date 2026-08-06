"""Tests for prepared image assets and the image-source light profile."""

from __future__ import annotations

import json
import pickle

import autolens as al
import numpy as np
import pytest

from hwoslaps.lensing.image_source import (
    ImageSource,
    SourceImageAsset,
    load_source_image_asset,
)


def _normalized_sb(shape=(8, 10), pixel_scale=0.2):
    rows, cols = np.indices(shape, dtype=float)
    sb = np.exp(-0.5 * (((rows - 3.1) / 1.0) ** 2 + ((cols - 5.4) / 1.3) ** 2))
    return sb / (pixel_scale**2 * sb.sum())


def _write_asset(
    path,
    *,
    sb=None,
    pixel_scale=0.2,
    pixel_scale_dtype=np.float64,
    metadata=None,
    omit_key=None,
    extra_key=False,
):
    if sb is None:
        sb = _normalized_sb(pixel_scale=pixel_scale)
    if metadata is None:
        metadata = {"format_version": 1, "provenance": {"kind": "synthetic"}}
    values = {
        "sb": sb,
        "pixel_scale_arcsec": np.asarray(pixel_scale, dtype=pixel_scale_dtype),
        "metadata_json": np.asarray(json.dumps(metadata)),
    }
    if omit_key is not None:
        values.pop(omit_key)
    if extra_key:
        values["extra"] = np.asarray(1)
    np.savez(path, **values)
    return path


def _profile(sb=None, **overrides):
    parameters = {
        "centre": (0.3, -0.2),
        "rotation_deg": 0.0,
        "pixel_scale_arcsec": 0.2,
        "sb": _normalized_sb() if sb is None else sb,
        "total_flux": 1.7,
        "flux_scale": 1.3,
        "size_scale": 1.2,
    }
    parameters.update(overrides)
    return ImageSource(**parameters)


def _evaluate(profile, points):
    grid = al.Grid2DIrregular(values=np.asarray(points, dtype=float))
    return np.asarray(profile.image_2d_from(grid=grid), dtype=float)


def test_asset_loader_roundtrip_and_cache_identity(tmp_path):
    """Load all fields and cache one object per absolute path."""
    path = _write_asset(tmp_path / "source.npz")

    first = load_source_image_asset(path)
    second = load_source_image_asset(path.resolve())

    assert isinstance(first, SourceImageAsset)
    assert second is first
    np.testing.assert_array_equal(first.sb, _normalized_sb())
    assert first.pixel_scale_arcsec == pytest.approx(0.2)
    assert first.metadata == {
        "format_version": 1,
        "provenance": {"kind": "synthetic"},
    }
    import hashlib

    expected_hash = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    assert first.sha256_16 == expected_hash


@pytest.mark.parametrize(
    "case,fragment",
    [
        ("bad_normalization", "normalized"),
        ("negative", "non-negative"),
        ("nan", "finite"),
        ("wrong_version", "format_version"),
        ("missing_key", "exactly these keys"),
        ("extra_key", "exactly these keys"),
        ("not_2d", "2D"),
        ("too_small", "between 8 and 4096"),
        ("too_large", "between 8 and 4096"),
        ("bad_pixel_scale", "pixel_scale_arcsec"),
        ("metadata_not_scalar", "zero-dimensional"),
        ("metadata_not_dict", "JSON dict"),
        ("metadata_not_json", "valid JSON"),
        ("missing_provenance", "provenance"),
        ("bad_dtype", "coercible to float64"),
    ],
)
def test_asset_loader_rejects_invalid_assets(case, fragment, tmp_path):
    """Reject each malformed asset-format contract with a loud message."""
    path = tmp_path / f"{case}.npz"
    sb = _normalized_sb()
    metadata = {"format_version": 1, "provenance": {}}
    kwargs = {}
    if case == "bad_normalization":
        sb = sb * 2.0
    elif case == "negative":
        sb = sb.copy()
        sb[0, 0] = -1.0
    elif case == "nan":
        sb = sb.copy()
        sb[0, 0] = np.nan
    elif case == "wrong_version":
        metadata["format_version"] = 2
    elif case == "missing_key":
        kwargs["omit_key"] = "sb"
    elif case == "extra_key":
        kwargs["extra_key"] = True
    elif case == "not_2d":
        sb = np.ones(8, dtype=float)
    elif case == "too_small":
        sb = np.ones((7, 8), dtype=float) / (0.2**2 * 56)
    elif case == "too_large":
        sb = np.ones((4097, 8), dtype=float) / (0.2**2 * 4097 * 8)
    elif case == "bad_pixel_scale":
        kwargs["pixel_scale"] = 0.0
    elif case == "metadata_not_scalar":
        values = {
            "sb": sb,
            "pixel_scale_arcsec": np.asarray(0.2),
            "metadata_json": np.asarray([json.dumps(metadata)]),
        }
        np.savez(path, **values)
        with pytest.raises(ValueError, match=fragment):
            load_source_image_asset(path)
        return
    elif case == "metadata_not_dict":
        values = {
            "sb": sb,
            "pixel_scale_arcsec": np.asarray(0.2),
            "metadata_json": np.asarray(json.dumps([])),
        }
        np.savez(path, **values)
        with pytest.raises(ValueError, match=fragment):
            load_source_image_asset(path)
        return
    elif case == "metadata_not_json":
        values = {
            "sb": sb,
            "pixel_scale_arcsec": np.asarray(0.2),
            "metadata_json": np.asarray("{invalid-json"),
        }
        np.savez(path, **values)
        with pytest.raises(ValueError, match=fragment):
            load_source_image_asset(path)
        return
    elif case == "missing_provenance":
        metadata.pop("provenance")
    elif case == "bad_dtype":
        sb = np.full((8, 10), "not-a-number")
    _write_asset(path, sb=sb, metadata=metadata, **kwargs)

    with pytest.raises(ValueError, match=fragment):
        load_source_image_asset(path)


def test_asset_loader_strict_types_and_boundaries(tmp_path):
    """Enforce strict scalar types, both axis bounds, and the 1e-8 gate."""
    sb = _normalized_sb()

    int_scale_path = tmp_path / "int-scale.npz"
    sb_for_two = _normalized_sb(pixel_scale=2.0)
    np.savez(
        int_scale_path,
        sb=sb_for_two,
        pixel_scale_arcsec=np.asarray(2, dtype=np.int64),
        metadata_json=np.asarray(
            json.dumps({"format_version": 1, "provenance": {}})
        ),
    )
    with pytest.raises(ValueError, match="float64 scalar"):
        load_source_image_asset(int_scale_path)

    for tag, version in (("bool", True), ("float", 1.0)):
        path = tmp_path / f"version-{tag}.npz"
        _write_asset(
            path,
            metadata={"format_version": version, "provenance": {}},
        )
        with pytest.raises(ValueError, match="format_version"):
            load_source_image_asset(path)

    for tag, shape in (("narrow", (8, 7)), ("wide", (8, 4097))):
        path = tmp_path / f"shape-{tag}.npz"
        bad = np.ones(shape, dtype=float)
        bad /= 0.2**2 * bad.sum()
        _write_asset(path, sb=bad)
        with pytest.raises(ValueError, match="between 8 and 4096"):
            load_source_image_asset(path)

    within_path = _write_asset(tmp_path / "within.npz", sb=sb * (1.0 + 5.0e-9))
    load_source_image_asset(within_path)

    beyond_path = _write_asset(tmp_path / "beyond.npz", sb=sb * (1.0 + 1.0e-6))
    with pytest.raises(ValueError, match="normalized"):
        load_source_image_asset(beyond_path)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_asset_loader_rejects_non_float64_pixel_scale(dtype, tmp_path):
    """Reject floating pixel-scale scalars that are not native float64."""
    path = _write_asset(
        tmp_path / f"pixel-scale-{np.dtype(dtype).name}.npz",
        pixel_scale_dtype=dtype,
    )
    with pytest.raises(ValueError, match="float64 scalar"):
        load_source_image_asset(path)


def test_image_source_from_asset_copies_asset_parameters(tmp_path):
    """Construct an ImageSource from a validated immutable asset record."""
    asset = load_source_image_asset(_write_asset(tmp_path / "source.npz"))

    profile = ImageSource.from_asset(
        asset,
        centre=(0.1, -0.1),
        rotation_deg=12.0,
        total_flux=0.7,
        flux_scale=1.1,
        size_scale=0.9,
    )

    np.testing.assert_array_equal(profile.sb, asset.sb)
    assert profile.pixel_scale_arcsec == pytest.approx(asset.pixel_scale_arcsec)
    assert profile.centre == pytest.approx((0.1, -0.1))
    assert profile.rotation_deg == pytest.approx(12.0)
    assert profile.total_flux == pytest.approx(0.7)
    assert profile.flux_scale == pytest.approx(1.1)
    assert profile.size_scale == pytest.approx(0.9)


def test_image_source_matches_pixel_centres_and_midpoint():
    """Evaluate exact samples and their bilinear midpoint average."""
    sb = _normalized_sb()
    profile = _profile(sb=sb)
    row_c = (sb.shape[0] - 1) / 2.0
    col_c = (sb.shape[1] - 1) / 2.0
    row = 3
    col = 4
    y = profile.centre[0] + (row - row_c) * 0.2 * 1.2
    x = profile.centre[1] + (col - col_c) * 0.2 * 1.2
    x_mid = x + 0.5 * 0.2 * 1.2

    values = _evaluate(profile, [(y, x), (y, x_mid)])

    amplitude = 1.7 * 1.3
    assert values[0] == pytest.approx(amplitude * sb[row, col], rel=1.0e-14)
    assert values[1] == pytest.approx(
        amplitude * 0.5 * (sb[row, col] + sb[row, col + 1]),
        rel=1.0e-14,
    )


def test_image_source_zero_padding_and_outside_domain():
    """Interpolate through the one-pixel zero border and vanish beyond it."""
    sb = _normalized_sb()
    profile = _profile(sb=sb, size_scale=1.0)
    row_c = (sb.shape[0] - 1) / 2.0
    col_c = (sb.shape[1] - 1) / 2.0
    y_edge_mid = profile.centre[0] + (-0.5 - row_c) * 0.2
    x = profile.centre[1] + (3 - col_c) * 0.2
    y_outside = profile.centre[0] + (-1.01 - row_c) * 0.2

    ny, nx = sb.shape
    y_high_edge_mid = profile.centre[0] + (ny - 1 + 0.5 - row_c) * 0.2
    y_high_outside = profile.centre[0] + (ny + 0.01 - row_c) * 0.2
    x_right_outside = profile.centre[1] + (nx + 0.01 - col_c) * 0.2
    y_interior = profile.centre[0] + (3 - row_c) * 0.2

    edge_mid, outside, high_edge_mid, high_outside, right_outside = _evaluate(
        profile,
        [
            (y_edge_mid, x),
            (y_outside, x),
            (y_high_edge_mid, x),
            (y_high_outside, x),
            (y_interior, x_right_outside),
        ],
    )

    assert edge_mid == pytest.approx(0.5 * 1.7 * 1.3 * sb[0, 3])
    assert outside == 0.0
    assert high_edge_mid == pytest.approx(0.5 * 1.7 * 1.3 * sb[ny - 1, 3])
    assert high_outside == 0.0
    assert right_outside == 0.0


def test_image_source_rotation_and_centre_translation():
    """Map image pixels to rotated and translated sky coordinates exactly."""
    sb = _normalized_sb()
    profile = _profile(sb=sb, rotation_deg=90.0, size_scale=1.0)
    row_c = (sb.shape[0] - 1) / 2.0
    col_c = (sb.shape[1] - 1) / 2.0
    row = 2
    col = 7
    u = (col - col_c) * 0.2
    v = (row - row_c) * 0.2
    point = (profile.centre[0] + u, profile.centre[1] - v)

    value = _evaluate(profile, [point])[0]

    assert value == pytest.approx(1.7 * 1.3 * sb[row, col], rel=1.0e-14)


def test_image_source_flux_and_size_similarity_transforms():
    """Apply flux and stretch the image at fixed brightness."""
    sb = _normalized_sb()
    base = _profile(sb=sb, flux_scale=1.0, size_scale=1.0)
    scaled_flux = _profile(sb=sb, flux_scale=1.3, size_scale=1.0)
    scaled_size = _profile(sb=sb, flux_scale=1.0, size_scale=1.2)
    points = np.asarray([[0.1, -0.4], [0.35, -0.12], [0.6, 0.1]])
    contracted = base.centre + (points - np.asarray(base.centre)) / 1.2

    np.testing.assert_allclose(
        _evaluate(scaled_flux, points),
        1.3 * _evaluate(base, points),
        rtol=1.0e-12,
        atol=0.0,
    )
    np.testing.assert_allclose(
        _evaluate(scaled_size, points),
        _evaluate(base, contracted),
        rtol=1.0e-12,
        atol=0.0,
    )


def test_image_source_integral_matches_scaled_total_flux():
    """Integrate the zero-padded interpolant to F times f times k squared."""
    sb = _normalized_sb()
    profile = _profile(sb=sb, rotation_deg=0.0)
    subdivisions = 20
    row_coords = np.linspace(-1.0, sb.shape[0], (sb.shape[0] + 1) * subdivisions + 1)
    col_coords = np.linspace(-1.0, sb.shape[1], (sb.shape[1] + 1) * subdivisions + 1)
    rows, cols = np.meshgrid(row_coords, col_coords, indexing="ij")
    row_c = (sb.shape[0] - 1) / 2.0
    col_c = (sb.shape[1] - 1) / 2.0
    y = profile.centre[0] + (rows - row_c) * 0.2 * 1.2
    x = profile.centre[1] + (cols - col_c) * 0.2 * 1.2
    values = _evaluate(profile, np.column_stack((y.ravel(), x.ravel()))).reshape(
        rows.shape
    )

    integral_x = np.trapz(values, x=x[0], axis=1)
    integral = np.trapz(integral_x, x=y[:, 0])

    assert integral == pytest.approx(1.7 * 1.3 * 1.2**2, rel=1.0e-6)


def test_exponential_intrinsic_total_flux_formula():
    """Pin AutoGalaxy's area-preserving elliptical Exponential integral."""
    intensity = 2.0
    effective_radius = 0.11
    b1 = 1.6783469900166605
    profile = al.lp.Exponential(
        centre=(0.0, 0.0),
        ell_comps=(0.14516129, 0.25142673),
        intensity=intensity,
        effective_radius=effective_radius,
    )

    coordinates = np.linspace(-1.5, 1.5, 401)
    y, x = np.meshgrid(coordinates, coordinates, indexing="ij")
    image = np.asarray(
        profile.image_2d_from(
            grid=al.Grid2DIrregular(
                values=np.column_stack((y.ravel(), x.ravel()))
            )
        )
    ).reshape(y.shape)
    numerical = np.trapz(
        np.trapz(image, x=coordinates, axis=1),
        x=coordinates,
    )
    expected = 2.0 * np.pi * intensity * np.exp(b1) * (effective_radius / b1) ** 2

    assert numerical == pytest.approx(expected, rel=1.0e-4)


def test_image_source_matches_lenstronomy_interpol():
    """Match lenstronomy INTERPOL after mapping its pixel-grid convention."""
    interpolation = pytest.importorskip(
        "lenstronomy.LightModel.Profiles.interpolation"
    )
    sb = _normalized_sb()
    profile = _profile(
        sb=sb,
        centre=(0.13, -0.27),
        rotation_deg=23.0,
        total_flux=1.0,
        flux_scale=1.0,
        size_scale=1.0,
    )
    points = np.asarray([[0.1, -0.3], [0.2, -0.1], [-0.2, -0.5]])
    theta = np.deg2rad(profile.rotation_deg)
    dx = points[:, 1] - profile.centre[1]
    dy = points[:, 0] - profile.centre[0]
    u = dx * np.cos(theta) + dy * np.sin(theta)
    v = -dx * np.sin(theta) + dy * np.cos(theta)
    interpol = interpolation.Interpol()

    reference = interpol.function(
        x=u,
        y=v,
        image=sb,
        center_x=0.0,
        center_y=0.0,
        phi_G=0.0,
        scale=0.2,
    )

    np.testing.assert_allclose(
        _evaluate(profile, points),
        reference,
        rtol=0.0,
        atol=1.0e-12 * sb.max(),
    )


def test_image_source_pickles_without_cached_spline():
    """Exclude the spline from pickle state and rebuild it on demand."""
    profile = _profile()
    points = [(0.3, -0.2), (0.1, -0.1)]
    expected = _evaluate(profile, points)
    assert profile._spline is not None

    restored = pickle.loads(pickle.dumps(profile))

    assert restored._spline is None
    np.testing.assert_array_equal(_evaluate(restored, points), expected)
    assert restored._spline is not None


def test_image_source_rejects_radial_evaluation():
    """Reject radial evaluation because image morphology is not symmetric."""
    profile = _profile()

    with pytest.raises(NotImplementedError, match="not radially symmetric"):
        profile.image_2d_via_radii_from(np.asarray([0.1]))

"""Tests for the offline source-image preparation tool."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from hwoslaps.lensing import generate_lensing_system, load_source_image_asset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from prepare_source_image import (  # noqa: E402
    SCRIPT_VERSION,
    bin_image,
    centre_on_centroid,
    footprint_mask,
    half_light_radius_pixels,
    load_input_image,
    main,
    normalize_unit_flux,
    rescale_to_half_light,
    subtract_background,
    write_asset,
)


def _compact_source(shape=(32, 40), background=3.0):
    rows, cols = np.indices(shape, dtype=float)
    radius = np.hypot(rows - 14.0, cols - 23.0)
    source = np.where(radius <= 6.0, 20.0 * np.exp(-0.5 * (radius / 2.0) ** 2), 0.0)
    return background + source


def test_load_input_image_reads_npy_as_float64(tmp_path):
    """Load a two-dimensional NumPy input without changing its values."""
    path = tmp_path / "input.npy"
    expected = np.arange(80, dtype=np.int16).reshape(8, 10)
    np.save(path, expected)

    loaded = load_input_image(path)

    assert loaded.dtype == np.float64
    np.testing.assert_array_equal(loaded, expected)


def test_load_input_image_uses_first_two_dimensional_fits_hdu(tmp_path):
    """Select the first 2-D FITS HDU and reject files without one."""
    fits = pytest.importorskip("astropy.io.fits")
    good_path = tmp_path / "input.fits"
    expected = np.arange(80, dtype=float).reshape(8, 10)
    fits.HDUList(
        [fits.PrimaryHDU(), fits.ImageHDU(np.arange(4)), fits.ImageHDU(expected)]
    ).writeto(good_path)
    np.testing.assert_array_equal(load_input_image(good_path), expected)

    bad_path = tmp_path / "bad.fits"
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(np.arange(4))]).writeto(bad_path)
    with pytest.raises(ValueError, match="HDU.*0.*1"):
        load_input_image(bad_path)


def test_bin_image_block_means_and_records_remainder_crop():
    """Block-mean surface brightness after bottom/right remainder cropping."""
    constant = np.full((4, 6), 7.0)
    binned, crop = bin_image(constant, 2)
    assert binned.shape == (2, 3)
    assert binned.size * 4 == constant.size
    np.testing.assert_array_equal(binned, 7.0)
    assert crop == {"bottom_rows": 0, "right_columns": 0}

    values = np.arange(35, dtype=float).reshape(5, 7)
    binned, crop = bin_image(values, 2)
    expected = values[:4, :6].reshape(2, 2, 3, 2).mean(axis=(1, 3))
    np.testing.assert_array_equal(binned, expected)
    assert crop == {"bottom_rows": 1, "right_columns": 1}


def test_bin_image_one_is_identity_and_rejects_bad_factors():
    """Treat a unit bin factor as identity and reject invalid factors."""
    image = np.arange(80, dtype=float).reshape(8, 10)
    binned, crop = bin_image(image, 1)
    np.testing.assert_array_equal(binned, image)
    assert crop == {"bottom_rows": 0, "right_columns": 0}

    for bad_factor in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="positive integer"):
            bin_image(image, bad_factor)


def test_subtract_background_recovers_known_border_offset():
    """Recover and subtract a known sigma-clipped border background."""
    image = _compact_source(background=4.5)

    subtracted, background = subtract_background(image, border_frac=0.1)

    assert background == pytest.approx(4.5, abs=1.0e-12)
    assert np.median(subtracted[:3]) == pytest.approx(0.0, abs=1.0e-12)


def test_footprint_mask_keeps_largest_blob_and_zeros_corner_contaminant():
    """Keep the largest 8-connected source and remove a corner contaminant."""
    rows, cols = np.indices((32, 32), dtype=float)
    main = np.where(
        np.hypot(rows - 16.0, cols - 17.0) <= 5.0,
        np.exp(-0.5 * (np.hypot(rows - 16.0, cols - 17.0) / 2.0) ** 2),
        0.0,
    )
    image = main.copy()
    image[0:2, 0:2] = 0.4

    masked, threshold, component_size = footprint_mask(image, k_sigma=2.0)

    assert threshold == pytest.approx(0.0)
    assert component_size > 50
    assert masked[0, 0] == 0.0
    assert masked[16, 17] > 0.0


def test_footprint_mask_rejects_edge_touching_main_component():
    """Reject a largest source footprint that reaches a cutout edge."""
    image = np.zeros((24, 24), dtype=float)
    image[0:8, 5:13] = 1.0

    with pytest.raises(ValueError, match="touches the array edge"):
        footprint_mask(image)


def test_centre_on_centroid_recentres_without_interpolation():
    """Crop and zero-pad so the flux centroid reaches the array centre."""
    rows, cols = np.indices((32, 48), dtype=float)
    image = np.exp(-0.5 * (((rows - 9.2) / 2.0) ** 2 + ((cols - 31.1) / 2.5) ** 2))

    centred, shift = centre_on_centroid(image)
    out_rows, out_cols = np.indices(centred.shape, dtype=float)
    centroid_y = float((out_rows * centred).sum() / centred.sum())
    centroid_x = float((out_cols * centred).sum() / centred.sum())
    centre_y = (centred.shape[0] - 1) / 2.0
    centre_x = (centred.shape[1] - 1) / 2.0

    assert centred.shape[0] == centred.shape[1]
    assert abs(centroid_y - centre_y) <= 0.5
    assert abs(centroid_x - centre_x) <= 0.5
    assert shift != (0, 0)


def test_half_light_radius_matches_circular_gaussian():
    """Recover the analytic circular-Gaussian half-light radius."""
    sigma = 5.0
    rows, cols = np.indices((65, 65), dtype=float)
    centre = 32.0
    image = np.exp(-0.5 * (((rows - centre) / sigma) ** 2 + ((cols - centre) / sigma) ** 2))

    radius = half_light_radius_pixels(image)

    assert radius == pytest.approx(1.1774 * sigma, rel=0.02)


def test_rescale_and_normalize_define_unit_integral():
    """Set the target half-light scale and normalize the bilinear integral."""
    pixel_scale = rescale_to_half_light(0.11, 5.5)
    image = np.arange(1, 81, dtype=float).reshape(8, 10)

    normalized = normalize_unit_flux(image, pixel_scale)

    assert pixel_scale == pytest.approx(0.02)
    assert pixel_scale**2 * normalized.sum() == pytest.approx(1.0, rel=1.0e-15)


def test_write_asset_roundtrips_through_public_loader(tmp_path):
    """Write the exact NPZ schema consumed by the public asset loader."""
    pixel_scale = 0.05
    sb = normalize_unit_flux(np.ones((8, 10)), pixel_scale)
    path = write_asset(
        tmp_path / "source.npz",
        sb,
        pixel_scale,
        {"kind": "synthetic"},
    )

    asset = load_source_image_asset(path)

    np.testing.assert_array_equal(asset.sb, sb)
    assert asset.metadata == {
        "format_version": 1,
        "provenance": {"kind": "synthetic"},
    }


@pytest.mark.parametrize("suffix", [".npy", ".fits"])
def test_prepare_cli_end_to_end_records_complete_provenance(suffix, tmp_path):
    """Prepare NPY and FITS inputs with complete offline provenance."""
    image = _compact_source()
    input_path = tmp_path / f"input{suffix}"
    if suffix == ".npy":
        np.save(input_path, image)
    else:
        fits = pytest.importorskip("astropy.io.fits")
        fits.PrimaryHDU(image).writeto(input_path)
    output_path = tmp_path / f"prepared-{suffix[1:]}.npz"

    status = main(
        [
            str(input_path),
            str(output_path),
            "--target-half-light-arcsec",
            "0.11",
            "--bin",
            "1",
            "--catalog-id",
            "synthetic-42",
            "--note",
            "unit-test",
        ]
    )

    assert status == 0
    asset = load_source_image_asset(output_path)
    provenance = asset.metadata["provenance"]
    assert provenance["input_path"] == str(input_path.resolve())
    assert len(provenance["input_sha256"]) == 64
    assert provenance["script_version"] == SCRIPT_VERSION
    assert provenance["catalog_id"] == "synthetic-42"
    assert provenance["note"] == "unit-test"
    assert provenance["target_half_light_arcsec"] == pytest.approx(0.11)
    assert provenance["bin"] == 1
    assert provenance["flip_y"] is False
    assert provenance["background"] == pytest.approx(3.0)
    assert provenance["mask_component_size"] > 0
    assert len(provenance["centroid_shift"]) == 2
    assert provenance["r_half_pixels"] > 0
    assert provenance["pixel_scale_arcsec"] > 0


def test_prepare_cli_flip_y_flips_prepared_asset(tmp_path):
    """Flip the loaded row axis before applying the equivariant pipeline."""
    input_path = tmp_path / "input.npy"
    np.save(input_path, _compact_source())
    normal_path = tmp_path / "normal.npz"
    flipped_path = tmp_path / "flipped.npz"
    common = [
        str(input_path),
        None,
        "--target-half-light-arcsec",
        "0.11",
    ]
    normal_args = list(common)
    normal_args[1] = str(normal_path)
    flipped_args = list(common)
    flipped_args[1] = str(flipped_path)
    flipped_args.append("--flip-y")

    main(normal_args)
    main(flipped_args)

    normal = load_source_image_asset(normal_path)
    flipped = load_source_image_asset(flipped_path)
    np.testing.assert_allclose(flipped.sb, np.flipud(normal.sb), rtol=0.0, atol=1.0e-14)
    assert flipped.metadata["provenance"]["flip_y"] is True


def test_prepared_asset_runs_through_lensing_pipeline(tmp_path):
    """Generate a finite non-negative lensed image from a prepared asset."""
    input_path = tmp_path / "input.npy"
    output_path = tmp_path / "prepared.npz"
    np.save(input_path, _compact_source())
    main(
        [
            str(input_path),
            str(output_path),
            "--target-half-light-arcsec",
            "0.11",
        ]
    )
    light = {
        "type": "Image",
        "asset_path": str(output_path),
        "centre": [0.02, -0.03],
        "rotation_deg": 11.0,
        "total_flux": 0.4,
        "flux_scale": 1.0,
        "size_scale": 1.0,
    }
    full_config = {
        "run_name": "prepared-image-smoke",
        "global_seed": 3,
        "lensing": {
            "grid": {"shape": [25, 25], "pixel_scale": 0.08},
            "lens_galaxy": {
                "redshift": 0.2,
                "mass": {
                    "type": "Isothermal",
                    "centre": [0.0, 0.0],
                    "ell_comps": [0.05, 0.0],
                    "einstein_radius": 0.5,
                },
            },
            "source_galaxy": {"redshift": 0.8, "light": light},
            "subhalo": {"enabled": False},
            "cosmology": "Planck15",
        },
    }

    data = generate_lensing_system(full_config["lensing"], full_config=full_config)

    assert np.all(np.isfinite(data.image))
    assert np.all(data.image >= 0.0)
    assert np.any(data.image > 0.0)


def test_subtract_background_is_robust_to_border_outliers():
    """Recover the border background despite noise and bright outliers."""
    rng = np.random.default_rng(11)
    image = _compact_source(background=4.5)
    image += rng.normal(0.0, 0.05, image.shape)
    image[0, 0] = 250.0
    image[0, 5] = 180.0
    image[-1, -1] = 300.0

    _, background = subtract_background(image, border_frac=0.1)

    assert background == pytest.approx(4.5, abs=0.05)


def test_footprint_mask_threshold_scales_with_border_noise():
    """Apply a positive threshold derived from the border noise level."""
    rng = np.random.default_rng(13)
    rows, cols = np.indices((48, 48), dtype=float)
    radius = np.hypot(rows - 24.0, cols - 25.0)
    image = np.where(radius <= 7.0, 30.0 * np.exp(-0.5 * (radius / 2.5) ** 2), 0.0)
    image += rng.normal(0.0, 0.1, image.shape)
    image[2:4, 2:4] += 5.0

    masked, threshold, component_size = footprint_mask(image, k_sigma=2.0)

    assert threshold == pytest.approx(0.2, rel=0.35)
    assert component_size > 30
    assert masked[3, 3] == 0.0
    assert masked[24, 25] > 0.0


def test_write_asset_rejects_non_positive_pixel_scale(tmp_path):
    """Reject a negative pixel scale even when the squared integral holds."""
    pixel_scale = 0.05
    sb = normalize_unit_flux(np.ones((8, 10)), pixel_scale)

    with pytest.raises(ValueError, match="positive"):
        write_asset(tmp_path / "bad.npz", sb, -pixel_scale, {})


def test_half_light_radius_rejects_unresolved_central_pixel():
    """Reject a source whose central pixel holds at least half the flux."""
    image = np.zeros((17, 17), dtype=float)
    image[8, 8] = 10.0
    image[8, 9] = 1.0

    with pytest.raises(ValueError, match="unresolved"):
        half_light_radius_pixels(image)


def test_prepare_cli_bin_two_records_crop_and_bins(tmp_path):
    """Bin by two through the CLI, recording the remainder crop."""
    rows, cols = np.indices((33, 41), dtype=float)
    radius = np.hypot(rows - 15.0, cols - 21.0)
    image = 3.0 + np.where(
        radius <= 9.0, 25.0 * np.exp(-0.5 * (radius / 3.5) ** 2), 0.0
    )
    input_path = tmp_path / "input.npy"
    np.save(input_path, image)
    output_path = tmp_path / "binned.npz"

    status = main(
        [
            str(input_path),
            str(output_path),
            "--target-half-light-arcsec",
            "0.11",
            "--bin",
            "2",
        ]
    )

    assert status == 0
    asset = load_source_image_asset(output_path)
    provenance = asset.metadata["provenance"]
    assert provenance["bin"] == 2
    assert provenance["crop"] == {"bottom_rows": 1, "right_columns": 1}
    assert provenance["border_frac"] == pytest.approx(0.1)
    assert provenance["k_sigma"] == pytest.approx(2.0)
    assert provenance["mask_threshold"] >= 0.0
    assert provenance["output_path"] == str(output_path.resolve())

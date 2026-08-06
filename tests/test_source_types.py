"""Contracts for smooth, clumpy, and image-based source light types."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import autolens as al
import numpy as np
import pytest
import yaml

from hwoslaps.config.validation import validate_lensing_config
from hwoslaps.lensing.generator import (
    _create_source_galaxy,
    generate_lensing_system,
)
from hwoslaps.modeling.fisher_detector import FisherDetector


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _base_lensing_config():
    with (PROJECT_ROOT / "configs" / "master_config.yaml").open(
        "r", encoding="utf-8"
    ) as stream:
        return copy.deepcopy(yaml.safe_load(stream)["lensing"])


def _sersic_component(**overrides):
    component = {
        "centre": [-0.03, 0.08],
        "ell_comps": [0.1, -0.05],
        "intensity": 2.0,
        "effective_radius": 0.11,
        "sersic_index": 1.5,
    }
    component.update(overrides)
    return component


def _light_config(light_type, asset_path=None):
    if light_type == "Exponential":
        component = _sersic_component()
        component.pop("sersic_index")
        return {"type": light_type, **component}
    if light_type == "Sersic":
        return {"type": light_type, **_sersic_component()}
    if light_type == "Clumpy":
        return {
            "type": light_type,
            "host": _sersic_component(),
            "clumps": [
                _sersic_component(
                    centre=[0.02, 0.12],
                    intensity=0.4,
                    effective_radius=0.025,
                    sersic_index=0.8,
                )
            ],
            "flux_scale": 1.0,
            "size_scale": 1.0,
        }
    if light_type == "Image":
        return {
            "type": light_type,
            "asset_path": str(asset_path),
            "centre": [-0.03, 0.08],
            "rotation_deg": 17.0,
            "total_flux": 0.5,
            "flux_scale": 1.0,
            "size_scale": 1.0,
        }
    raise AssertionError(f"Unsupported test light type: {light_type}")


@pytest.mark.parametrize("light_type", ["Exponential", "Sersic", "Clumpy", "Image"])
def test_validation_accepts_all_source_light_types(light_type, tmp_path):
    """Accept each of the four explicitly supported source light schemas."""
    asset_path = tmp_path / "source.npz"
    asset_path.write_bytes(b"synthetic-placeholder")
    lensing = _base_lensing_config()
    lensing["source_galaxy"]["light"] = _light_config(light_type, asset_path)

    validate_lensing_config(lensing)


def test_validation_lists_all_source_light_types_for_unknown_type():
    """List the complete supported set when a source light type is invalid."""
    lensing = _base_lensing_config()
    lensing["source_galaxy"]["light"]["type"] = "Gaussian"

    with pytest.raises(
        ValueError,
        match="Exponential.*Sersic.*Clumpy.*Image",
    ):
        validate_lensing_config(lensing)


def test_validation_rejects_sersic_index_on_exponential():
    """Reject a Sersic index that an Exponential profile would ignore."""
    lensing = _base_lensing_config()
    lensing["source_galaxy"]["light"]["sersic_index"] = 1.0

    with pytest.raises(ValueError, match="sersic_index.*Exponential"):
        validate_lensing_config(lensing)


def test_validation_requires_sersic_index_on_sersic():
    """Require an explicit Sersic index for a Sersic source."""
    lensing = _base_lensing_config()
    light = _light_config("Sersic")
    light.pop("sersic_index")
    lensing["source_galaxy"]["light"] = light

    with pytest.raises(ValueError, match="Missing required key 'sersic_index'"):
        validate_lensing_config(lensing)


@pytest.mark.parametrize("sersic_index", [0.29, 10.01, 0.0, np.nan])
def test_validation_rejects_sersic_index_outside_physical_domain(sersic_index):
    """Reject non-finite or out-of-domain Sersic indices."""
    lensing = _base_lensing_config()
    lensing["source_galaxy"]["light"] = _light_config("Sersic")
    lensing["source_galaxy"]["light"]["sersic_index"] = sersic_index

    with pytest.raises(ValueError, match="sersic_index"):
        validate_lensing_config(lensing)


@pytest.mark.parametrize("sersic_index", [0.3, 10.0])
def test_validation_accepts_sersic_index_domain_boundaries(sersic_index):
    """Accept the inclusive 0.3 and 10 Sersic-index domain boundaries."""
    lensing = _base_lensing_config()
    lensing["source_galaxy"]["light"] = _light_config("Sersic")
    lensing["source_galaxy"]["light"]["sersic_index"] = sersic_index

    validate_lensing_config(lensing)


def test_validation_accepts_four_clumps():
    """Accept the maximum allowed count of four clumps."""
    lensing = _base_lensing_config()
    light = _light_config("Clumpy")
    light["clumps"] = [
        _sersic_component(centre=[0.02 * i, 0.1], intensity=0.2)
        for i in range(1, 5)
    ]
    lensing["source_galaxy"]["light"] = light

    validate_lensing_config(lensing)


@pytest.mark.parametrize(
    "light_type,field",
    [
        ("Image", "asset_path"),
        ("Image", "centre"),
        ("Image", "rotation_deg"),
        ("Image", "total_flux"),
        ("Clumpy", "host"),
        ("Clumpy", "clumps"),
    ],
)
def test_validation_requires_structured_source_fields(light_type, field, tmp_path):
    """Reject structured source configs with a missing required field."""
    asset_path = tmp_path / "source.npz"
    asset_path.write_bytes(b"synthetic-placeholder")
    lensing = _base_lensing_config()
    light = _light_config(light_type, asset_path)
    light.pop(field)
    lensing["source_galaxy"]["light"] = light

    with pytest.raises(ValueError, match=f"Missing required key '{field}'"):
        validate_lensing_config(lensing)


def test_validation_requires_every_clump_component_field():
    """Reject a clump component that is missing a required field."""
    lensing = _base_lensing_config()
    light = _light_config("Clumpy")
    light["clumps"][0].pop("intensity")
    lensing["source_galaxy"]["light"] = light

    with pytest.raises(ValueError, match="Missing required key 'intensity'"):
        validate_lensing_config(lensing)


@pytest.mark.parametrize(
    "clumps,fragment",
    [([], "zero-clump Clumpy is a Sersic"), ([{}] * 5, "at most 4")],
)
def test_validation_rejects_invalid_clump_count(clumps, fragment):
    """Reject Clumpy sources with zero or more than four clumps."""
    lensing = _base_lensing_config()
    light = _light_config("Clumpy")
    light["clumps"] = clumps
    lensing["source_galaxy"]["light"] = light

    with pytest.raises(ValueError, match=fragment):
        validate_lensing_config(lensing)


@pytest.mark.parametrize("light_type", ["Clumpy", "Image"])
def test_validation_rejects_unknown_keys(light_type, tmp_path):
    """Reject unknown top-level keys in structured source schemas."""
    asset_path = tmp_path / "source.npz"
    asset_path.write_bytes(b"synthetic-placeholder")
    lensing = _base_lensing_config()
    light = _light_config(light_type, asset_path)
    light["ignored"] = 1
    lensing["source_galaxy"]["light"] = light

    with pytest.raises(ValueError, match="unsupported keys.*ignored"):
        validate_lensing_config(lensing)


@pytest.mark.parametrize("field", ["flux_scale", "size_scale"])
@pytest.mark.parametrize("light_type", ["Clumpy", "Image"])
def test_validation_requires_joint_scale_fields(light_type, field, tmp_path):
    """Require explicit joint flux and size scales for structured sources."""
    asset_path = tmp_path / "source.npz"
    asset_path.write_bytes(b"synthetic-placeholder")
    lensing = _base_lensing_config()
    light = _light_config(light_type, asset_path)
    light.pop(field)
    lensing["source_galaxy"]["light"] = light

    with pytest.raises(ValueError, match=f"Missing required key '{field}'"):
        validate_lensing_config(lensing)


@pytest.mark.parametrize("field", ["flux_scale", "size_scale"])
@pytest.mark.parametrize("light_type", ["Clumpy", "Image"])
@pytest.mark.parametrize("bad_value", [0.0, -1.0, np.inf])
def test_validation_requires_positive_joint_scales(
    light_type, field, bad_value, tmp_path
):
    """Reject non-positive or non-finite joint source scales."""
    asset_path = tmp_path / "source.npz"
    asset_path.write_bytes(b"synthetic-placeholder")
    lensing = _base_lensing_config()
    light = _light_config(light_type, asset_path)
    light[field] = bad_value
    lensing["source_galaxy"]["light"] = light

    with pytest.raises(ValueError, match=field):
        validate_lensing_config(lensing)


@pytest.mark.parametrize("bad_value", [0.0, -1.0, np.inf])
def test_validation_requires_positive_image_total_flux(bad_value, tmp_path):
    """Reject an Image source total flux that is not positive and finite."""
    asset_path = tmp_path / "source.npz"
    asset_path.write_bytes(b"synthetic-placeholder")
    lensing = _base_lensing_config()
    light = _light_config("Image", asset_path)
    light["total_flux"] = bad_value
    lensing["source_galaxy"]["light"] = light

    with pytest.raises(ValueError, match="total_flux"):
        validate_lensing_config(lensing)


def test_validation_reports_resolved_missing_image_asset_path(tmp_path):
    """Report the resolved absolute path when an Image asset is missing."""
    asset_path = tmp_path / "missing.npz"
    lensing = _base_lensing_config()
    lensing["source_galaxy"]["light"] = _light_config("Image", asset_path)

    with pytest.raises(ValueError, match=str(asset_path.resolve())):
        validate_lensing_config(lensing)


@pytest.mark.parametrize(
    "field,bad_value,fragment",
    [
        ("asset_path", 3, "asset_path.*str"),
        ("centre", [0.0, np.inf], "centre"),
        ("rotation_deg", np.nan, "rotation_deg"),
        ("rotation_deg", True, "rotation_deg"),
    ],
)
def test_validation_checks_image_geometry_and_asset_path(
    field, bad_value, fragment, tmp_path
):
    """Reject malformed Image asset-path and geometry fields."""
    asset_path = tmp_path / "source.npz"
    asset_path.write_bytes(b"synthetic-placeholder")
    lensing = _base_lensing_config()
    light = _light_config("Image", asset_path)
    light[field] = bad_value
    lensing["source_galaxy"]["light"] = light

    with pytest.raises(ValueError, match=fragment):
        validate_lensing_config(lensing)


@pytest.mark.parametrize(
    "component_key,field,bad_value",
    [
        ("host", "centre", [0.0, np.nan]),
        ("host", "ell_comps", [1.0, 0.0]),
        ("host", "intensity", 0.0),
        ("host", "effective_radius", -0.1),
        ("host", "sersic_index", 11.0),
        ("clump", "centre", [True, 0.0]),
        ("clump", "ell_comps", [0.8, 0.8]),
        ("clump", "intensity", np.inf),
        ("clump", "effective_radius", 0.0),
        ("clump", "sersic_index", 0.2),
    ],
)
def test_validation_checks_every_clumpy_component_field(
    component_key, field, bad_value
):
    """Apply shared Sersic validation to the host and every clump."""
    lensing = _base_lensing_config()
    light = _light_config("Clumpy")
    component = light["host"] if component_key == "host" else light["clumps"][0]
    component[field] = bad_value
    lensing["source_galaxy"]["light"] = light

    with pytest.raises(ValueError, match=field):
        validate_lensing_config(lensing)


def _tiny_full_config(light):
    return {
        "run_name": "source-types-test",
        "global_seed": 7,
        "lensing": {
            "grid": {"shape": [25, 25], "pixel_scale": 0.08},
            "lens_galaxy": {
                "redshift": 0.2,
                "mass": {
                    "type": "Isothermal",
                    "centre": [0.0, 0.0],
                    "ell_comps": [0.05, -0.02],
                    "einstein_radius": 0.5,
                },
            },
            "source_galaxy": {"redshift": 0.8, "light": copy.deepcopy(light)},
            "subhalo": {"enabled": False},
            "cosmology": "Planck15",
        },
    }


def _synthetic_asset(path, pixel_scale=0.08):
    rows, cols = np.indices((12, 16), dtype=float)
    sb = np.exp(-0.5 * (((rows - 4.2) / 1.2) ** 2 + ((cols - 9.1) / 1.7) ** 2))
    sb /= pixel_scale**2 * sb.sum()
    np.savez(
        path,
        sb=sb.astype(np.float64),
        pixel_scale_arcsec=np.asarray(pixel_scale, dtype=np.float64),
        metadata_json=np.asarray(
            json.dumps(
                {"format_version": 1, "provenance": {"kind": "synthetic"}}
            )
        ),
    )
    return path


def _tracer_image(source_galaxy, grid, lens_galaxy=None):
    galaxies = [source_galaxy] if lens_galaxy is None else [lens_galaxy, source_galaxy]
    return np.asarray(al.Tracer(galaxies=galaxies).image_2d_from(grid=grid))


def test_sersic_generator_matches_direct_tracer_and_exponential_at_n_one():
    """Build Sersic truth exactly and recover Exponential when n equals one."""
    light = _light_config("Sersic")
    generated = _create_source_galaxy({"redshift": 0.8, "light": light})
    direct = al.Galaxy(
        redshift=0.8,
        light=al.lp.Sersic(
            centre=tuple(light["centre"]),
            ell_comps=tuple(light["ell_comps"]),
            intensity=light["intensity"],
            effective_radius=light["effective_radius"],
            sersic_index=light["sersic_index"],
        ),
    )
    grid = al.Grid2D.uniform(shape_native=(21, 21), pixel_scales=0.05)

    np.testing.assert_allclose(
        _tracer_image(generated, grid),
        _tracer_image(direct, grid),
        rtol=1.0e-12,
        atol=0.0,
    )

    light["sersic_index"] = 1.0
    sersic_one = _create_source_galaxy({"redshift": 0.8, "light": light})
    exponential_light = copy.deepcopy(light)
    exponential_light["type"] = "Exponential"
    exponential_light.pop("sersic_index")
    exponential = _create_source_galaxy(
        {"redshift": 0.8, "light": exponential_light}
    )
    np.testing.assert_allclose(
        _tracer_image(sersic_one, grid),
        _tracer_image(exponential, grid),
        rtol=1.0e-12,
        atol=0.0,
    )


def test_clumpy_generator_matches_sum_of_component_tracers():
    """Sum the generated Clumpy host and clumps through one common lens."""
    light = _light_config("Clumpy")
    light["clumps"].append(
        _sersic_component(
            centre=[-0.08, 0.02],
            intensity=0.25,
            effective_radius=0.018,
            sersic_index=1.1,
        )
    )
    generated = _create_source_galaxy({"redshift": 0.8, "light": light})
    lens = al.Galaxy(
        redshift=0.2,
        mass=al.mp.Isothermal(
            centre=(0.0, 0.0),
            ell_comps=(0.05, -0.02),
            einstein_radius=0.5,
        ),
    )
    grid = al.Grid2D.uniform(shape_native=(23, 23), pixel_scales=0.06)

    component_images = []
    for role in ["host", "clump_0", "clump_1"]:
        component = al.Galaxy(redshift=0.8, light=getattr(generated, role))
        component_images.append(_tracer_image(component, grid, lens))

    np.testing.assert_allclose(
        _tracer_image(generated, grid, lens),
        np.sum(component_images, axis=0),
        rtol=1.0e-10,
        atol=0.0,
    )


def test_clumpy_joint_flux_and_size_transforms():
    """Apply one flux scale and one host-centred similarity transform."""
    base_light = _light_config("Clumpy")
    base_light["clumps"].append(
        _sersic_component(
            centre=[-0.09, 0.03],
            intensity=0.2,
            effective_radius=0.02,
            sersic_index=1.4,
        )
    )
    base = _create_source_galaxy({"redshift": 0.8, "light": base_light})
    flux_light = copy.deepcopy(base_light)
    flux_light["flux_scale"] = 1.3
    scaled_flux = _create_source_galaxy({"redshift": 0.8, "light": flux_light})
    grid = al.Grid2D.uniform(shape_native=(23, 23), pixel_scales=0.05)
    np.testing.assert_allclose(
        _tracer_image(scaled_flux, grid),
        1.3 * _tracer_image(base, grid),
        rtol=1.0e-12,
        atol=0.0,
    )

    scaled_light = copy.deepcopy(base_light)
    scaled_light["size_scale"] = 1.2
    scaled = _create_source_galaxy({"redshift": 0.8, "light": scaled_light})
    manual_light = copy.deepcopy(base_light)
    host_centre = np.asarray(manual_light["host"]["centre"])
    for component in [manual_light["host"], *manual_light["clumps"]]:
        component["effective_radius"] *= 1.2
    for clump in manual_light["clumps"]:
        clump["centre"] = (
            host_centre + 1.2 * (np.asarray(clump["centre"]) - host_centre)
        ).tolist()
    manual = _create_source_galaxy({"redshift": 0.8, "light": manual_light})
    np.testing.assert_allclose(
        _tracer_image(scaled, grid),
        _tracer_image(manual, grid),
        rtol=1.0e-12,
        atol=0.0,
    )


@pytest.mark.parametrize("light_type", ["Exponential", "Sersic", "Clumpy", "Image"])
def test_truth_metadata_records_as_built_source(light_type, tmp_path):
    """Record profile type and transformed component or image-asset truth."""
    asset_path = _synthetic_asset(tmp_path / "source.npz")
    light = _light_config(light_type, asset_path)
    if light_type == "Clumpy":
        light["flux_scale"] = 1.3
        light["size_scale"] = 1.2
    config = _tiny_full_config(light)

    data = generate_lensing_system(config["lensing"], full_config=config)

    assert data.source_light_type == light_type
    if light_type == "Image":
        assert data.source_components is None
        assert data.source_image_asset["asset_path"] == str(asset_path)
        assert len(data.source_image_asset["sha256_16"]) == 16
        assert data.source_image_asset["pixel_scale_arcsec"] == pytest.approx(0.08)
        for key in (
            "rotation_deg",
            "total_flux",
            "flux_scale",
            "size_scale",
        ):
            assert data.source_image_asset[key] == pytest.approx(light[key])
        assert data.source_image_asset["metadata"]["provenance"] == {
            "kind": "synthetic"
        }
    else:
        assert data.source_image_asset is None
        expected_roles = {
            "Exponential": ["single"],
            "Sersic": ["single"],
            "Clumpy": ["host", "clump_0"],
        }[light_type]
        assert [component["role"] for component in data.source_components] == expected_roles
        if light_type == "Exponential":
            assert data.source_components[0]["sersic_index"] == pytest.approx(1.0)
        if light_type == "Sersic":
            component = data.source_components[0]
            for key in ("centre", "ell_comps", "intensity", "effective_radius"):
                assert component[key] == pytest.approx(light[key])
            assert component["sersic_index"] == pytest.approx(
                light["sersic_index"]
            )
        if light_type == "Clumpy":
            host = data.source_components[0]
            clump = data.source_components[1]
            assert host["intensity"] == pytest.approx(1.3 * light["host"]["intensity"])
            assert host["effective_radius"] == pytest.approx(
                1.2 * light["host"]["effective_radius"]
            )
            assert host["centre"] == pytest.approx(light["host"]["centre"])
            assert host["ell_comps"] == pytest.approx(light["host"]["ell_comps"])
            assert host["sersic_index"] == pytest.approx(
                light["host"]["sersic_index"]
            )
            expected_centre = np.asarray(light["host"]["centre"]) + 1.2 * (
                np.asarray(light["clumps"][0]["centre"])
                - np.asarray(light["host"]["centre"])
            )
            assert clump["centre"] == pytest.approx(expected_centre)
            assert clump["intensity"] == pytest.approx(
                1.3 * light["clumps"][0]["intensity"]
            )
            assert clump["effective_radius"] == pytest.approx(
                1.2 * light["clumps"][0]["effective_radius"]
            )
            assert clump["ell_comps"] == pytest.approx(
                light["clumps"][0]["ell_comps"]
            )
            assert clump["sersic_index"] == pytest.approx(
                light["clumps"][0]["sersic_index"]
            )


def test_exponential_generation_preserves_preexisting_fields_and_image():
    """Keep the canonical Exponential path and legacy flat fields unchanged."""
    light = _light_config("Exponential")
    config = _tiny_full_config(light)

    data = generate_lensing_system(config["lensing"], full_config=config)
    direct_source = al.Galaxy(
        redshift=0.8,
        light=al.lp.Exponential(
            centre=(-0.03, 0.08),
            ell_comps=(0.1, -0.05),
            intensity=2.0,
            effective_radius=0.11,
        ),
    )
    direct_lens = al.Galaxy(
        redshift=0.2,
        mass=al.mp.Isothermal(
            centre=(0.0, 0.0),
            ell_comps=(0.05, -0.02),
            einstein_radius=0.5,
        ),
    )
    expected_image = _tracer_image(direct_source, data.grid, direct_lens)

    np.testing.assert_array_equal(data.image, expected_image.reshape(data.image.shape))
    assert data.pixel_scale == 0.08
    assert data.lens_redshift == 0.2
    assert data.source_redshift == 0.8
    assert data.source_centre == (-0.03, 0.08)
    assert data.source_ellipticity == (0.1, -0.05)
    assert data.source_intensity == 2.0
    assert data.source_effective_radius == 0.11
    assert data.lens_centre == (0.0, 0.0)
    assert data.lens_ellipticity == (0.05, -0.02)
    assert data.lens_einstein_radius == 0.5
    assert data.cosmology_name == "Planck15"
    assert data.has_subhalo is False
    assert data.config["lensing"]["source_galaxy"]["light"] == light


@pytest.mark.parametrize("light_type", ["Exponential", "Sersic", "Clumpy", "Image"])
def test_fisher_scalar_nuisance_paths_follow_source_schema(light_type, tmp_path):
    """Route source nuisances to each source type's physical parameters."""
    asset_path = _synthetic_asset(tmp_path / "source.npz")
    detector = FisherDetector.__new__(FisherDetector)
    detector.full_config = _tiny_full_config(_light_config(light_type, asset_path))
    detector.prior_sigmas = {
        "source.intensity": 0.7,
        "source.effective_radius": 0.08,
    }
    detector.include_background_offset = False

    specs = detector._build_scalar_nuisance_specs()
    paths = {spec.name: spec.path for spec in specs}

    light_root = ("lensing", "source_galaxy", "light")
    if light_type == "Clumpy":
        assert paths["source.centre_y"] == light_root + ("host", "centre", 0)
        assert paths["source.centre_x"] == light_root + ("host", "centre", 1)
        assert paths["source.ell_comp_1"] == light_root + ("host", "ell_comps", 0)
        assert paths["source.ell_comp_2"] == light_root + ("host", "ell_comps", 1)
    else:
        assert paths["source.centre_y"] == light_root + ("centre", 0)
        assert paths["source.centre_x"] == light_root + ("centre", 1)
    if light_type in {"Clumpy", "Image"}:
        assert paths["source.intensity"] == light_root + ("flux_scale",)
        assert paths["source.effective_radius"] == light_root + ("size_scale",)
    else:
        assert paths["source.intensity"] == light_root + ("intensity",)
        assert paths["source.effective_radius"] == light_root + (
            "effective_radius",
        )
    spec_by_name = {spec.name: spec for spec in specs}
    assert spec_by_name["source.intensity"].step_mode == "multiplicative"
    assert spec_by_name["source.intensity"].step_key == "source_intensity_frac"
    assert spec_by_name["source.intensity"].prior_sigma == pytest.approx(0.7)
    assert spec_by_name["source.effective_radius"].step_mode == "multiplicative"
    assert spec_by_name["source.effective_radius"].step_key == "source_reff_frac"
    assert spec_by_name["source.effective_radius"].prior_sigma == pytest.approx(0.08)
    if light_type == "Image":
        assert "source.ell_comp_1" not in paths
        assert "source.ell_comp_2" not in paths

    lens_names = [
        "lens.centre_y",
        "lens.centre_x",
        "lens.einstein_radius",
        "lens.ell_comp_1",
        "lens.ell_comp_2",
    ]
    expected_names = {
        "Exponential": lens_names
        + [
            "source.centre_y",
            "source.centre_x",
            "source.ell_comp_1",
            "source.ell_comp_2",
            "source.intensity",
            "source.effective_radius",
        ],
        "Sersic": lens_names
        + [
            "source.centre_y",
            "source.centre_x",
            "source.ell_comp_1",
            "source.ell_comp_2",
            "source.intensity",
            "source.effective_radius",
        ],
        "Clumpy": lens_names
        + [
            "source.centre_y",
            "source.centre_x",
            "source.ell_comp_1",
            "source.ell_comp_2",
            "source.intensity",
            "source.effective_radius",
        ],
        "Image": lens_names
        + [
            "source.centre_y",
            "source.centre_x",
            "source.intensity",
            "source.effective_radius",
        ],
    }[light_type]
    assert [spec.name for spec in specs] == expected_names

    if light_type == "Exponential":
        assert [(spec.name, spec.path) for spec in specs] == [
            ("lens.centre_y", ("lensing", "lens_galaxy", "mass", "centre", 0)),
            ("lens.centre_x", ("lensing", "lens_galaxy", "mass", "centre", 1)),
            (
                "lens.einstein_radius",
                ("lensing", "lens_galaxy", "mass", "einstein_radius"),
            ),
            (
                "lens.ell_comp_1",
                ("lensing", "lens_galaxy", "mass", "ell_comps", 0),
            ),
            (
                "lens.ell_comp_2",
                ("lensing", "lens_galaxy", "mass", "ell_comps", 1),
            ),
            ("source.centre_y", light_root + ("centre", 0)),
            ("source.centre_x", light_root + ("centre", 1)),
            ("source.ell_comp_1", light_root + ("ell_comps", 0)),
            ("source.ell_comp_2", light_root + ("ell_comps", 1)),
            ("source.intensity", light_root + ("intensity",)),
            (
                "source.effective_radius",
                light_root + ("effective_radius",),
            ),
        ]


@pytest.mark.parametrize("light_type", ["Clumpy", "Image"])
def test_flux_scale_finite_difference_is_source_image_derivative(light_type, tmp_path):
    """Differentiate the joint flux scale into the baseline source image."""
    asset_path = _synthetic_asset(tmp_path / "source.npz")
    light = _light_config(light_type, asset_path)
    full_config = _tiny_full_config(light)
    detector = FisherDetector.__new__(FisherDetector)
    detector.full_config = full_config
    detector.prior_sigmas = {}
    detector.include_background_offset = False
    detector.finite_diff = {"source_intensity_frac": 1.0e-4}
    spec = next(
        item
        for item in detector._build_scalar_nuisance_specs()
        if item.name == "source.intensity"
    )
    plus = copy.deepcopy(full_config)
    minus = copy.deepcopy(full_config)
    step = detector._apply_scalar_perturbation(plus, minus, spec)
    grid = al.Grid2D.uniform(shape_native=(21, 21), pixel_scales=0.06)
    baseline = _tracer_image(
        _create_source_galaxy(full_config["lensing"]["source_galaxy"]), grid
    )
    image_plus = _tracer_image(
        _create_source_galaxy(plus["lensing"]["source_galaxy"]), grid
    )
    image_minus = _tracer_image(
        _create_source_galaxy(minus["lensing"]["source_galaxy"]), grid
    )
    derivative = (image_plus - image_minus) / (2.0 * step)
    mask = np.asarray(baseline) > np.max(np.asarray(baseline)) * 1.0e-8

    np.testing.assert_allclose(
        np.asarray(derivative)[mask],
        np.asarray(baseline)[mask],
        rtol=1.0e-8,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "spec_name,axis",
    [("source.centre_y", 0), ("source.centre_x", 1)],
)
def test_clumpy_centre_nuisance_rigidly_translates_all_components(spec_name, axis):
    """Translate the host and every clump without changing their offsets."""
    light = _light_config("Clumpy")
    light["size_scale"] = 1.2
    full_config = _tiny_full_config(light)
    detector = FisherDetector.__new__(FisherDetector)
    detector.full_config = full_config
    detector.prior_sigmas = {}
    detector.include_background_offset = False
    detector.finite_diff = {"centre_arcsec": 0.003}
    spec = next(
        item
        for item in detector._build_scalar_nuisance_specs()
        if item.name == spec_name
    )
    plus = copy.deepcopy(full_config)
    minus = copy.deepcopy(full_config)

    step = detector._apply_scalar_perturbation(plus, minus, spec)
    baseline = _create_source_galaxy(full_config["lensing"]["source_galaxy"])
    shifted_plus = _create_source_galaxy(plus["lensing"]["source_galaxy"])
    shifted_minus = _create_source_galaxy(minus["lensing"]["source_galaxy"])

    expected = np.zeros(2)
    expected[axis] = step
    for role in ("host", "clump_0"):
        base_centre = np.asarray(getattr(baseline, role).centre)
        np.testing.assert_allclose(
            np.asarray(getattr(shifted_plus, role).centre) - base_centre,
            expected,
            rtol=0.0,
            atol=1.0e-15,
        )
        np.testing.assert_allclose(
            np.asarray(getattr(shifted_minus, role).centre) - base_centre,
            -expected,
            rtol=0.0,
            atol=1.0e-15,
        )


@pytest.mark.parametrize("light_type", ["Clumpy", "Image"])
def test_size_scale_perturbation_is_symmetric_multiplicative(light_type, tmp_path):
    """Perturb the joint size scale symmetrically in both configurations."""
    asset_path = _synthetic_asset(tmp_path / "source.npz")
    full_config = _tiny_full_config(_light_config(light_type, asset_path))
    detector = FisherDetector.__new__(FisherDetector)
    detector.full_config = full_config
    detector.prior_sigmas = {}
    detector.include_background_offset = False
    detector.finite_diff = {"source_reff_frac": 0.01}
    spec = next(
        item
        for item in detector._build_scalar_nuisance_specs()
        if item.name == "source.effective_radius"
    )
    plus = copy.deepcopy(full_config)
    minus = copy.deepcopy(full_config)

    step = detector._apply_scalar_perturbation(plus, minus, spec)

    assert step == pytest.approx(0.01)
    assert plus["lensing"]["source_galaxy"]["light"]["size_scale"] == pytest.approx(
        1.01
    )
    assert minus["lensing"]["source_galaxy"]["light"]["size_scale"] == pytest.approx(
        0.99
    )

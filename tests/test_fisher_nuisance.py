"""Direct contracts for reusable scalar nuisance planning."""

import numpy as np
import pytest

from hwoslaps.modeling.fisher_nuisance import (
    ScalarNuisanceSpec,
    build_scalar_nuisance_specs,
    lookup_prior_sigma,
    select_scalar_nuisances,
)


def test_custom_scalar_directions_are_selected_in_supplied_canonical_order():
    specs = [
        ScalarNuisanceSpec(
            "source.morphology",
            ("lensing", "source_galaxy", "light", "shape"),
            "additive",
        ),
        ScalarNuisanceSpec("observation.sky_gradient", ("observation", "sky_gradient"), "additive"),
    ]
    selected, label = select_scalar_nuisances(specs, ["observation.sky_gradient", "source.morphology"])
    assert selected == specs
    assert label == "explicit"
    assert select_scalar_nuisances(specs, "source_only") == ([specs[0]], "source_only")


def test_image_nuisance_plan_uses_flux_and_size_paths_and_omits_ellipticity():
    specs = build_scalar_nuisance_specs(
        {"type": "Image"}, {"source.intensity": 0.02}, include_background_offset=False
    )
    by_name = {spec.name: spec for spec in specs}
    assert by_name["source.intensity"].path == (
        "lensing",
        "source_galaxy",
        "light",
        "flux_scale",
    )
    assert by_name["source.effective_radius"].path == (
        "lensing",
        "source_galaxy",
        "light",
        "size_scale",
    )
    assert by_name["source.intensity"].prior_sigma == 0.02
    assert "source.ell_comp_1" not in by_name
    assert "source.ell_comp_2" not in by_name
    assert "observation.background_offset_adu" not in by_name


@pytest.mark.parametrize("sigma", [0.0, -0.1, np.inf, np.nan])
def test_prior_sigma_rejects_non_positive_or_non_finite_values(sigma):
    with pytest.raises(ValueError, match="Invalid prior sigma"):
        lookup_prior_sigma({"source.morphology": sigma}, "source.morphology")


def test_absent_prior_remains_unconstrained():
    assert lookup_prior_sigma({}, "source.morphology") is None


EXPONENTIAL_NUISANCE_NAMES = [
    "lens.centre_y",
    "lens.centre_x",
    "lens.einstein_radius",
    "lens.ell_comp_1",
    "lens.ell_comp_2",
    "source.centre_y",
    "source.centre_x",
    "source.ell_comp_1",
    "source.ell_comp_2",
    "source.intensity",
    "source.effective_radius",
    "observation.background_offset_adu",
]

IMAGE_NUISANCE_NAMES = [
    name for name in EXPONENTIAL_NUISANCE_NAMES if name not in {"source.ell_comp_1", "source.ell_comp_2"}
]


def _selected_names(light_type, selector, *, include_background_offset=True):
    specs = build_scalar_nuisance_specs(
        {"type": light_type}, {}, include_background_offset=include_background_offset
    )
    selected, label = select_scalar_nuisances(specs, selector)
    return [spec.name for spec in selected], label


def test_exponential_scene_has_twelve_scalar_directions():
    """Build the documented scalar direction set for an exponential source."""
    names, label = _selected_names("Exponential", None)

    assert names == EXPONENTIAL_NUISANCE_NAMES
    assert len(names) == 12
    assert label == "all"


def test_image_scene_has_ten_scalar_directions():
    """Drop the two source ellipticity directions for an image source."""
    names, label = _selected_names("Image", None)

    assert names == IMAGE_NUISANCE_NAMES
    assert len(names) == 10
    assert label == "all"


@pytest.mark.parametrize("light_type", ["Exponential", "Image"])
@pytest.mark.parametrize("selector", [None, "all", "ALL"])
def test_nuisance_subset_all_matches_the_unfiltered_directions(light_type, selector):
    """Keep every scalar direction for the default and explicit 'all'."""
    expected = EXPONENTIAL_NUISANCE_NAMES if light_type == "Exponential" else IMAGE_NUISANCE_NAMES
    names, label = _selected_names(light_type, selector)

    assert names == expected
    assert label == "all"


@pytest.mark.parametrize("light_type", ["Exponential", "Image"])
def test_nuisance_subset_none_selects_nothing(light_type):
    """Profile no scalar direction at all for the reserved word 'none'."""
    names, label = _selected_names(light_type, "none")

    assert names == []
    assert label == "none"


@pytest.mark.parametrize("light_type", ["Exponential", "Image"])
def test_nuisance_subset_lens_only_selects_lens_directions(light_type):
    """Select every lens direction and no source or background direction."""
    names, label = _selected_names(light_type, "lens_only")

    assert names == [
        "lens.centre_y",
        "lens.centre_x",
        "lens.einstein_radius",
        "lens.ell_comp_1",
        "lens.ell_comp_2",
    ]
    assert label == "lens_only"


def test_nuisance_subset_source_only_selects_source_directions():
    """Select every source direction of an exponential-source scene."""
    names, label = _selected_names("Exponential", "source_only")

    assert names == [
        "source.centre_y",
        "source.centre_x",
        "source.ell_comp_1",
        "source.ell_comp_2",
        "source.intensity",
        "source.effective_radius",
    ]
    assert label == "source_only"


def test_nuisance_subset_source_only_drops_ellipticity_for_image_source():
    """Select the four source directions an image-source scene defines."""
    names, label = _selected_names("Image", "source_only")

    assert names == [
        "source.centre_y",
        "source.centre_x",
        "source.intensity",
        "source.effective_radius",
    ]
    assert label == "source_only"


@pytest.mark.parametrize(
    "light_type, expected_count",
    [("Exponential", 11), ("Image", 9)],
)
def test_nuisance_subset_lens_and_source_excludes_background(light_type, expected_count):
    """Select lens and source directions but never the background offset."""
    names, label = _selected_names(light_type, "lens_and_source")

    assert len(names) == expected_count
    assert "observation.background_offset_adu" not in names
    assert all(name.startswith(("lens.", "source.")) for name in names)
    assert label == "lens_and_source"


def test_nuisance_subset_explicit_list_keeps_canonical_order():
    """Select the named directions in their canonical construction order."""
    names, label = _selected_names(
        "Exponential",
        ["source.intensity", "lens.centre_y"],
    )

    assert names == ["lens.centre_y", "source.intensity"]
    assert label == "explicit"


def test_nuisance_subset_explicit_list_may_name_the_background_direction():
    """Allow the background direction to be named when it exists."""
    names, label = _selected_names(
        "Exponential",
        ["observation.background_offset_adu"],
    )

    assert names == ["observation.background_offset_adu"]
    assert label == "explicit"


def test_nuisance_subset_rejects_direction_absent_from_the_scene():
    """Reject a source ellipticity direction an image scene does not define."""
    with pytest.raises(ValueError, match="unknown direction 'source.ell_comp_1'"):
        _selected_names("Image", ["source.ell_comp_1"])


def test_nuisance_subset_unknown_name_error_lists_the_valid_names():
    """Name every valid direction when rejecting an unknown one."""
    with pytest.raises(ValueError) as excinfo:
        _selected_names("Exponential", ["lens.centre_z"])

    message = str(excinfo.value)
    for name in EXPONENTIAL_NUISANCE_NAMES:
        assert name in message


def test_nuisance_subset_rejects_background_direction_when_flag_is_off():
    """Reject the background direction when the flag leaves it undefined."""
    with pytest.raises(ValueError, match="unknown direction"):
        _selected_names("Exponential", ["observation.background_offset_adu"], include_background_offset=False)


def test_nuisance_subset_rejects_psf_mode_names():
    """Reject PSF modes, which the PSF selectors alone govern."""
    with pytest.raises(ValueError, match="must not name PSF modes"):
        _selected_names("Exponential", ["psf.global_zernikes[4]"])


def test_nuisance_subset_rejects_unknown_reserved_word():
    """Reject a reserved word outside the documented vocabulary."""
    with pytest.raises(ValueError, match="nuisance_subset must be one of"):
        _selected_names("Exponential", "lens_and_background")


def test_nuisance_subset_rejects_duplicate_names():
    """Reject a list that names the same direction twice."""
    with pytest.raises(ValueError, match="duplicate direction"):
        _selected_names("Exponential", ["lens.centre_y", "lens.centre_y"])


def test_nuisance_subset_rejects_empty_list():
    """Reject an empty list and point at the reserved word 'none'."""
    with pytest.raises(ValueError, match="must be non-empty"):
        _selected_names("Exponential", [])


@pytest.mark.parametrize("selector", [12, True, {"lens": True}])
def test_nuisance_subset_rejects_non_name_selectors(selector):
    """Reject selectors that are neither a reserved word nor a name list."""
    with pytest.raises(ValueError, match="nuisance_subset must be one of"):
        _selected_names("Exponential", selector)


def test_nuisance_subset_rejects_non_string_list_entries():
    """Reject a list entry that is not a direction name."""
    with pytest.raises(ValueError, match="entries must be nuisance"):
        _selected_names("Exponential", ["lens.centre_y", 3])

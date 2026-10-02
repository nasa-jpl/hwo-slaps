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
        ScalarNuisanceSpec(
            "observation.sky_gradient", ("observation", "sky_gradient"), "additive"
        ),
    ]
    selected, label = select_scalar_nuisances(
        specs, ["observation.sky_gradient", "source.morphology"]
    )
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

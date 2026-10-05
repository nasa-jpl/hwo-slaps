"""Fit models: free-parameter order and names, structural refusals, the AutoFit conversion."""

from __future__ import annotations

import numpy as np
import pytest

from hwoslaps.inference.fit_model import (
    FitArgument, FitComponent, FitGalaxy, FitModel, autofit_model, fixed, linked, uniform,
)

ISOTHERMAL = "autogalaxy.profiles.mass.total.isothermal:Isothermal"
GAUSSIAN = "autogalaxy.profiles.light.standard.gaussian:Gaussian"
EXPONENTIAL = "autogalaxy.profiles.light.standard.exponential:Exponential"


def _scalar(name, lower, upper, truth):
    return FitArgument(name, (uniform(lower, upper, truth=truth),), pair=False)


def _model(*, light_centre=(linked("mass", "centre", 0), linked("mass", "centre", 1)), drop=None):
    mass = FitComponent(ISOTHERMAL, (
        FitArgument("centre", (uniform(-0.005, 0.005, truth=0.0), uniform(-0.005, 0.005, truth=0.0)), pair=True),
        FitArgument("ell_comps", (uniform(0.08, 0.12, truth=0.1), fixed(0.0)), pair=True),
        _scalar("einstein_radius", 0.99, 1.01, 1.0)))
    light = FitComponent(GAUSSIAN, (FitArgument("centre", light_centre, pair=True),
                                    FitArgument("ell_comps", (fixed(0.0), fixed(0.0)), pair=True),
                                    _scalar("intensity", 0.5, 1.5, 1.0), _scalar("sigma", 0.1, 0.3, 0.2)))
    arguments = (FitArgument("centre", (fixed(0.0), fixed(0.0)), pair=True),
                 FitArgument("ell_comps", (fixed(0.05), fixed(0.0)), pair=True),
                 _scalar("intensity", 1.0, 3.0, 2.0), _scalar("effective_radius", 0.077, 0.143, 0.11))
    source = FitComponent(EXPONENTIAL, tuple(argument for argument in arguments if argument.name != drop))
    return FitModel("smooth", (FitGalaxy("lens", 0.2, (("mass", mass), ("light", light))),
                               FitGalaxy("source", 0.6, (("light", source),))))


def test_free_parameters_follow_declaration_order_and_shared_prior_naming():
    """Uniform elements in creation order; a linked prior is named by its last holder (AutoFit's rule)."""
    model = _model()
    assert model.parameter_names == (
        "galaxies.lens.light.centre.centre_0", "galaxies.lens.light.centre.centre_1",
        "galaxies.lens.mass.ell_comps.ell_comps_0", "galaxies.lens.mass.einstein_radius",
        "galaxies.lens.light.intensity", "galaxies.lens.light.sigma",
        "galaxies.source.light.intensity", "galaxies.source.light.effective_radius")
    assert model.lower.tolist() == [-0.005, -0.005, 0.08, 0.99, 0.5, 0.1, 1.0, 0.077]
    assert model.truth.tolist() == [0.0, 0.0, 0.1, 1.0, 1.0, 0.2, 2.0, 0.11]
    assert model.upper.tolist() == [0.005, 0.005, 0.12, 1.01, 1.5, 0.3, 3.0, 0.143]
    assert model.subhalo_path is None
    unlinked = _model(light_centre=(uniform(-0.01, 0.01, truth=0.0), uniform(-0.01, 0.01, truth=0.0)))
    assert unlinked.parameter_names[:2] == ("galaxies.lens.mass.centre.centre_0", "galaxies.lens.mass.centre.centre_1")
    assert unlinked.digest() != model.digest()


_COMPONENT = FitComponent(GAUSSIAN, (_scalar("intensity", 0.5, 1.5, 1.0),))
_LINK_TO_LATER = FitComponent(GAUSSIAN, (FitArgument("intensity", (linked("b", "intensity"),), pair=False),))


@pytest.mark.parametrize(("build", "message"), [
    (lambda: FitGalaxy("subhalo", 0.2, (("light", _COMPONENT),)), "other than 'subhalo'"),
    (lambda: FitGalaxy("lens", 0.2, (("a", _LINK_TO_LATER), ("b", _COMPONENT))), "not an earlier component"),
    (lambda: FitModel("smooth", (FitGalaxy("lens", 0.2, (("subhalo", _COMPONENT),)),)), "smooth role has no subhalo"),
    (lambda: FitModel("subhalo", (FitGalaxy("source", 0.6, (("subhalo", _COMPONENT),)),)),
     "galaxy lens or subhalo_plane"),
    (lambda: uniform(0.5, 1.5, truth=2.0), "truth inside"),
], ids=["galaxy-named-subhalo", "link-to-later-component", "smooth-with-subhalo", "subhalo-in-source",
        "truth-outside-box"])
def test_incoherent_fit_models_are_refused(build, message):
    with pytest.raises(ValueError, match=message):
        build()


@pytest.mark.backend
def test_autofit_model_builds_the_declared_free_priors():
    """The AutoFit collection has the declared prior paths in vector order, the truth vector lands on
    the declared attributes, and a constructor argument left unassigned is refused."""
    model = _model()
    collection = autofit_model(model)
    assert [".".join(path) for path in collection.unique_prior_paths] == list(model.parameter_names)
    instance = collection.instance_from_vector(vector=list(model.truth))
    lens, source = instance.galaxies.lens, instance.galaxies.source
    assert (lens.redshift, source.redshift) == (0.2, 0.6)
    assert tuple(lens.light.centre) == tuple(lens.mass.centre) == (0.0, 0.0)
    assert tuple(lens.mass.ell_comps) == (0.1, 0.0) and lens.mass.einstein_radius == 1.0
    assert (lens.light.intensity, lens.light.sigma) == (1.0, 0.2)
    assert (source.light.intensity, source.light.effective_radius, tuple(source.light.ell_comps)) == (2.0, 0.11,
                                                                                                    (0.05, 0.0))
    shifted = collection.instance_from_vector(vector=list(np.asarray(model.truth) + ([0.001] + [0.0] * 7)))
    assert shifted.galaxies.lens.light.centre[0] == shifted.galaxies.lens.mass.centre[0] == 0.001
    with pytest.raises(RuntimeError, match="AutoFit built 8 free priors"):
        autofit_model(_model(drop="effective_radius"))

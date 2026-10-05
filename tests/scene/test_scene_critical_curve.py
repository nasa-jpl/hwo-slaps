"""Effective Einstein radius from the tangential critical curve of the lens mass."""

import math

import pytest

from hwoslaps.scene.convert import ell_comps_from
from hwoslaps.scene.critical_curve import effective_einstein_radius
from hwoslaps.scene.spec import parse_scene

pytestmark = pytest.mark.backend

AUTOGALAXY_ROUND_AXIS_RATIO = 0.99999


def _sie_radius(theta_e, axis_ratio):
    """Area-equivalent radius of the SIE critical ellipse: the intermediate-axis radius 2 sqrt(q) theta_E / (1 + q)."""
    return 2.0 * math.sqrt(axis_ratio) * theta_e / (1.0 + axis_ratio)


@pytest.mark.parametrize("theta_e, axis_ratio, angle_deg, centre", [
    (1.0, None, 0.0, (0.0, 0.0)),
    (1.0, 0.818, 20.0, (0.0, 0.0)),
    (0.7, 0.6, 115.0, (0.0, 0.0)),
    (1.2, 0.75, 30.0, (0.3, -0.2)),
], ids=["round", "q0.818", "q0.6", "q0.75-off-centre"])
def test_effective_einstein_radius_of_analytic_lenses(scene_mapping, planck15, theta_e, axis_ratio, angle_deg, centre):
    ell_comps = (0.0, 0.0) if axis_ratio is None else ell_comps_from(axis_ratio, angle_deg)
    scene_mapping["lens"]["mass"]["main"].update(einstein_radius=theta_e, ell_comps=list(ell_comps),
                                                 centre=list(centre))
    expected = _sie_radius(theta_e, AUTOGALAXY_ROUND_AXIS_RATIO if axis_ratio is None else axis_ratio)
    assert effective_einstein_radius(parse_scene(scene_mapping), planck15) == pytest.approx(expected, rel=2.0e-5)


def test_the_critical_curve_is_the_one_around_the_lens_centre(scene_mapping, planck15):
    """The larger critical curve of a second lens component 3" away is not the lens's.

    The companion (theta_E 1.2") adds convergence and shear of 0.2 each at the main lens
    (theta_E 0.3"), so the main curve's radius lies between 0.3" and 0.3 / (1 - 0.4) = 0.5";
    the companion's own, separate curve is larger than 1.2".
    """
    scene_mapping["lens"]["mass"]["main"].update(einstein_radius=0.3, ell_comps=[0.0, 0.0])
    scene_mapping["lens"]["mass"]["companion"] = {"type": "Isothermal", "centre": [0.0, 3.0], "einstein_radius": 1.2,
                                                  "ell_comps": [0.0, 0.0]}
    assert 0.3 < effective_einstein_radius(parse_scene(scene_mapping), planck15) < 0.5


def test_a_lens_below_the_extraction_resolution_is_refused(scene_mapping, planck15):
    scene_mapping["lens"]["mass"]["main"]["einstein_radius"] = 0.02
    with pytest.raises(ValueError, match="too small to measure"):
        effective_einstein_radius(parse_scene(scene_mapping), planck15)

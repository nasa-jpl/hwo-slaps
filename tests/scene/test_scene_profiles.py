"""Profile registry: the photometric unit integral of the Exponential light profile."""

import numpy as np
import pytest
from scipy.integrate import trapezoid

from hwoslaps.scene.profiles import PROFILE_TYPES, sersic_constant

pytestmark = pytest.mark.backend


def test_exponential_unit_integral_is_the_rendered_flux_at_unit_intensity():
    import autolens as al

    values = {"centre": (0.0, 0.0), "ell_comps": (0.14516129, 0.25142673), "effective_radius": 0.11, "intensity": 1.0}
    profile = al.lp.Exponential(**values)
    assert sersic_constant(1.0) == profile.sersic_constant
    coordinates = np.linspace(-1.5, 1.5, 401)
    y, x = np.meshgrid(coordinates, coordinates, indexing="ij")
    image = np.asarray(profile.image_2d_from(grid=al.Grid2DIrregular(values=np.column_stack((y.ravel(), x.ravel())))))
    numerical = trapezoid(trapezoid(image.reshape(y.shape), x=coordinates, axis=1), x=coordinates)
    unit_integral = PROFILE_TYPES["Exponential"].unit_integral(values)
    assert numerical == pytest.approx(unit_integral, rel=1.0e-4)
    assert unit_integral == pytest.approx(2.0 * np.pi * np.exp(sersic_constant(1.0)) * (0.11 / sersic_constant(1.0)) ** 2,
                                          rel=1.0e-15)

"""Backend-free geometry conversions equal AutoGalaxy's bitwise."""

import pytest

from hwoslaps.scene.convert import ell_comps_from, multipole_components_from, shear_components_from

pytestmark = pytest.mark.backend

ANGLES_DEG = (-137.0, -45.0, 0.0, 12.5, 30.0, 45.0, 90.0, 179.9, 361.0)


@pytest.mark.parametrize("axis_ratio", (0.2, 0.55, 0.818, 0.999999, 1.0))
def test_conversions_equal_autogalaxy(axis_ratio):
    from autogalaxy import convert

    for angle in ANGLES_DEG:
        assert ell_comps_from(axis_ratio, angle) == tuple(float(v) for v in convert.ell_comps_from(axis_ratio, angle))
        magnitude = 1.0 - axis_ratio + 0.013
        assert shear_components_from(magnitude, angle) == tuple(
            float(v) for v in convert.shear_gamma_1_2_from(magnitude, angle))
        for order in (3, 4):
            assert multipole_components_from(magnitude, angle, order) == tuple(
                float(v) for v in convert.multipole_comps_from(magnitude, angle, order))

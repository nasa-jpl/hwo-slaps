"""Profile registry: parameter domains against the component tables, the Exponential unit integral."""

import copy
import math

import numpy as np
import pytest
from scipy.integrate import trapezoid

from hwoslaps.config.checks import ConfigError
from hwoslaps.scene.profiles import PROFILE_TYPES, sersic_constant


def _probes(interval):
    """(value, inside) just inside, at and just outside each finite end of ``interval``; infinite ends at infinity."""
    probes = []
    for end, is_open, inward in ((interval.lower, interval.open_lower, math.inf),
                                 (interval.upper, interval.open_upper, -math.inf)):
        if math.isinf(end):
            probes.append((end, False))
        else:
            probes += [(math.nextafter(end, inward), True), (end, not is_open), (math.nextafter(end, -inward), False)]
    return probes


def _accepted(table, values):
    try:
        table.read(values, "component")
    except ConfigError:
        return False
    return True


def test_parameter_domains_are_what_the_component_tables_accept(tmp_path):
    asset = tmp_path / "asset.npz"
    asset.write_bytes(b"reading checks only that the file exists")
    # Pair elements are zero, so the probed element alone sets a joint ellipticity.
    valid = {
        "Isothermal": {"centre": [0.0, 0.0], "einstein_radius": 1.0, "ell_comps": [0.0, 0.0]},
        "PowerLaw": {"centre": [0.0, 0.0], "einstein_radius": 1.0, "ell_comps": [0.0, 0.0], "slope": 2.0},
        "ExternalShear": {"gamma_1": 0.0, "gamma_2": 0.0},
        "Sersic": {"centre": [0.0, 0.0], "ell_comps": [0.0, 0.0], "effective_radius": 0.5, "intensity": 1.0, "sersic_index": 2.5},
        "Exponential": {"centre": [0.0, 0.0], "ell_comps": [0.0, 0.0], "effective_radius": 0.5, "intensity": 1.0},
        "Image": {"asset_path": str(asset), "centre": [0.0, 0.0], "total_flux": 1.0},
    }
    assert set(valid) == set(PROFILE_TYPES)
    for type_name, profile in PROFILE_TYPES.items():
        for definition in profile.parameters(profile.table.read(valid[type_name], type_name)):
            for value, inside in _probes(definition.domain):
                values = copy.deepcopy(valid[type_name])
                if definition.index is None:
                    values[definition.key] = value
                else:
                    values[definition.key][definition.index] = value
                assert definition.domain.contains(value) is inside, (type_name, definition.name, value)
                assert _accepted(profile.table, values) is inside, (type_name, definition.name, value)


@pytest.mark.backend
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

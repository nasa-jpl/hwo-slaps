"""Fixtures shared by the scene lane: a minimal scene mapping and the Planck15 cosmology."""

import pytest

from hwoslaps.scene.cosmology import Cosmology, parse_cosmology


@pytest.fixture
def scene_mapping():
    """A fresh scene mapping: 40 x 40 pixels of 0.05", an SIE lens, an Exponential source, an NFW hypothesis."""
    return {
        "grid": {"shape": [40, 40], "pixel_scale_arcsec": 0.05, "over_sample_size": 2},
        "lens": {"redshift": 0.2, "mass": {"main": {"type": "Isothermal", "centre": [0.0, 0.0],
                                                    "einstein_radius": 0.8, "ell_comps": [0.05, 0.0]}}},
        "source": {"redshift": 0.6, "light": {"disk": {"type": "Exponential", "centre": [0.02, -0.03],
                                                       "ell_comps": [0.1, 0.05], "effective_radius": 0.12,
                                                       "intensity": 1.0}}},
        "subhalo": {"type": "NFW", "concentration": {"kind": "moline2017_eq7", "x_sub": 1.0}},
    }


@pytest.fixture(scope="session")
def planck15():
    return Cosmology(parse_cosmology({"name": "Planck15"}))

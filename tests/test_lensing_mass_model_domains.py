"""Bugfind tests for lensing mass-model physical domains."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

TESTS_ROOT = Path(__file__).resolve().parent
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from _lensing_physics_helpers import (  # noqa: E402
    Planck15CosmologyAdapter,
    load_lensing_utils_module,
    load_mass_models_module,
)

COSMOLOGY = Planck15CosmologyAdapter()
MASS_MODELS = load_mass_models_module()
LENSING_UTILS = load_lensing_utils_module()


@pytest.mark.parametrize("bad_mass", [0.0, -1.0, -1.0e8, np.nan, np.inf, -np.inf])
def test_point_mass_einstein_radius_rejects_invalid_mass(bad_mass):
    """Reject a point-mass Einstein radius for non-positive mass."""
    with pytest.raises(ValueError):
        MASS_MODELS.einstein_radius_point_mass(bad_mass, 0.2, 2.5, COSMOLOGY)


@pytest.mark.parametrize("z_lens,z_source", [(0.2, 0.2), (0.5, 0.2)])
def test_point_mass_einstein_radius_rejects_non_physical_redshift_order(z_lens, z_source):
    """Reject a source at or in front of the point-mass lens."""
    with pytest.raises(ValueError):
        MASS_MODELS.einstein_radius_point_mass(1.0e8, z_lens, z_source, COSMOLOGY)


@pytest.mark.parametrize("bad_mass", [0.0, -1.0, -1.0e8, np.nan, np.inf, -np.inf])
def test_sis_velocity_dispersion_rejects_invalid_mass(bad_mass):
    """Reject an SIS velocity dispersion for non-positive mass."""
    with pytest.raises(ValueError):
        MASS_MODELS.sigma_v_from_m200_sis(bad_mass, 0.2, COSMOLOGY)


@pytest.mark.parametrize("z_lens,z_source", [(0.2, 0.2), (0.5, 0.2)])
def test_sis_einstein_radius_rejects_non_physical_redshift_order(z_lens, z_source):
    """Reject a source at or in front of the SIS lens."""
    with pytest.raises(ValueError):
        MASS_MODELS.einstein_radius_sis_m200(1.0e8, z_lens, z_source, COSMOLOGY)


@pytest.mark.parametrize(
    "mass_msun,concentration",
    [
        (0.0, 10.0),
        (-1.0e8, 10.0),
        (np.nan, 10.0),
        (1.0e8, 0.0),
        (1.0e8, -5.0),
        (1.0e8, np.nan),
    ],
)
def test_nfw_scale_parameters_reject_invalid_mass_or_concentration(mass_msun, concentration):
    """Reject NFW scale parameters for bad mass or concentration."""
    with pytest.raises(ValueError):
        MASS_MODELS.nfw_scale_parameters(mass_msun, concentration, 0.2, COSMOLOGY)


@pytest.mark.parametrize("bad_redshift", [True, -0.5, np.nan, np.inf])
def test_power_law_concentration_rejects_invalid_redshift(bad_redshift):
    """Reject a power-law concentration at an invalid redshift."""
    with pytest.raises(ValueError):
        MASS_MODELS.concentration_power_law(1.0e8, z=bad_redshift)


@pytest.mark.parametrize(
    "mass_msun,x_sub",
    [
        (1.0e3, 1.0),
        (1.0e5, 1.0),
        (1.0e9, 100.0),
    ],
)
def test_moline_concentration_rejects_unsupported_domain(mass_msun, x_sub):
    """Reject Moline concentrations outside the calibrated domain."""
    with pytest.raises(ValueError):
        MASS_MODELS.concentration_moline2017_eq7(
            mass_msun,
            x_sub=x_sub,
            h=COSMOLOGY.reduced_h,
        )


@pytest.mark.parametrize("offset_pixels", [-100.0, -101.0])
def test_einstein_ring_position_rejects_non_positive_final_radius(offset_pixels):
    """Reject an inward offset that drives the radius to zero."""
    with pytest.raises(ValueError):
        LENSING_UTILS.get_einstein_ring_position(
            angle_deg=0.0,
            einstein_radius=1.0,
            offset_pixels=offset_pixels,
            pixel_scale=0.01,
        )

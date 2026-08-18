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


def test_nfw_truncation_radius_scales_the_scale_radius():
    """Scale the NFW scale radius by tau in scale-ratio mode."""
    got = MASS_MODELS.nfw_truncation_radius_arcsec(
        "scale_ratio",
        0.25,
        tau=10.0,
    )
    assert got == 2.5


def test_nfw_truncation_radius_returns_the_declared_radius():
    """Return the declared truncation radius in explicit-arcsec mode."""
    got = MASS_MODELS.nfw_truncation_radius_arcsec(
        "explicit_arcsec",
        0.25,
        radius_arcsec=0.05,
    )
    assert got == 0.05


@pytest.mark.parametrize("bad_tau", [0.0, -1.0, np.nan, np.inf, -np.inf, True])
def test_nfw_truncation_radius_rejects_invalid_tau(bad_tau):
    """Reject a tau that is not finite and positive."""
    with pytest.raises(ValueError, match="tau must be a finite positive number"):
        MASS_MODELS.nfw_truncation_radius_arcsec("scale_ratio", 0.25, tau=bad_tau)


@pytest.mark.parametrize("bad_radius", [0.0, -0.05, np.nan, np.inf, -np.inf, True])
def test_nfw_truncation_radius_rejects_invalid_explicit_radius(bad_radius):
    """Reject an explicit truncation radius that is not finite and positive."""
    with pytest.raises(
        ValueError,
        match="radius_arcsec must be a finite positive number",
    ):
        MASS_MODELS.nfw_truncation_radius_arcsec(
            "explicit_arcsec",
            0.25,
            radius_arcsec=bad_radius,
        )


@pytest.mark.parametrize("bad_scale_radius", [0.0, -0.25, np.nan, np.inf])
def test_nfw_truncation_radius_rejects_invalid_scale_radius(bad_scale_radius):
    """Reject a scale radius that is not finite and positive."""
    with pytest.raises(ValueError, match="scale_radius_arcsec"):
        MASS_MODELS.nfw_truncation_radius_arcsec(
            "scale_ratio",
            bad_scale_radius,
            tau=10.0,
        )


@pytest.mark.parametrize("bad_mode", ["scale-ratio", "explicit", "", None, 10.0])
def test_nfw_truncation_radius_rejects_unknown_mode(bad_mode):
    """Reject a truncation mode outside the supported set."""
    with pytest.raises(ValueError, match="Unsupported truncation mode"):
        MASS_MODELS.nfw_truncation_radius_arcsec(bad_mode, 0.25, tau=10.0)


@pytest.mark.parametrize(
    "mode,kwargs,expected_error",
    [
        ("scale_ratio", {}, "tau is required"),
        (
            "scale_ratio",
            {"tau": 10.0, "radius_arcsec": 0.05},
            "radius_arcsec is not accepted",
        ),
        ("explicit_arcsec", {}, "radius_arcsec is required"),
        (
            "explicit_arcsec",
            {"tau": 10.0, "radius_arcsec": 0.05},
            "tau is not accepted",
        ),
    ],
)
def test_nfw_truncation_radius_rejects_cross_mode_parameters(mode, kwargs, expected_error):
    """Reject truncation parameters that the selected mode does not accept."""
    with pytest.raises(ValueError, match=expected_error):
        MASS_MODELS.nfw_truncation_radius_arcsec(mode, 0.25, **kwargs)


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

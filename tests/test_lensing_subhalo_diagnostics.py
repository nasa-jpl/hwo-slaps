"""Unit tests for subhalo physical diagnostics.

Closed-form checks run without `autolens` using analytic spherical
profiles. The profile-backed checks required by the diagnostic panel
spec run against the real PyAutoLens mass profiles when they are
installed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy import constants as const

TESTS_ROOT = Path(__file__).resolve().parent
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from _lensing_physics_helpers import (  # noqa: E402
    Planck15CosmologyAdapter,
    load_constants_module,
    load_mass_models_module,
    load_module,
)

try:
    import autolens as al
except ImportError:  # pragma: no cover - exercised only without autolens
    al = None

requires_autolens = pytest.mark.skipif(al is None, reason="autolens is not installed")

COSMOLOGY = Planck15CosmologyAdapter()
CONSTANTS = load_constants_module()
MASS_MODELS = load_mass_models_module()
DIAGNOSTICS = load_module(
    "lensing/subhalo_diagnostics.py",
    "hwoslaps.lensing.subhalo_diagnostics",
)

Z_LENS = 0.5
Z_SOURCE = 2.0
APERTURES_ARCSEC = [0.01, 0.05, 0.2, 1.0]


class _SphericalProfile:
    """Circular mass profile with a prescribed radial deflection law.

    Parameters
    ----------
    deflection_law : `callable`
        Function mapping an array of radii in arcseconds to deflection
        magnitudes in arcseconds.
    centre : `tuple` of `float`, optional
        Profile centre as (y, x) in arcseconds.
    scale_radius : `float`, optional
        Scale radius in arcseconds. Omitted when None.
    truncation_radius : `float`, optional
        Truncation radius in arcseconds. Omitted when None.
    axis_ratio : `float`, optional
        Reported axis ratio. Omitted when None.
    """

    def __init__(
        self,
        deflection_law,
        centre=(0.0, 0.0),
        scale_radius=None,
        truncation_radius=None,
        axis_ratio=1.0,
    ):
        self.centre = tuple(centre)
        self._deflection_law = deflection_law
        if scale_radius is not None:
            self.scale_radius = scale_radius
        if truncation_radius is not None:
            self.truncation_radius = truncation_radius
        if axis_ratio is not None:
            self.axis_ratio = axis_ratio

    def deflections_yx_2d_from(self, grid):
        """Return (y, x) deflections in arcseconds on an (N, 2) grid."""
        offsets = np.asarray(grid, dtype=float) - np.asarray(self.centre, dtype=float)
        radii = np.hypot(offsets[:, 0], offsets[:, 1])
        magnitude = np.asarray(self._deflection_law(radii), dtype=float)
        return magnitude[:, None] * (offsets / radii[:, None])


def _point_mass_profile(einstein_radius, centre=(0.0, 0.0)):
    """Return an analytic point-mass profile."""
    return _SphericalProfile(
        lambda radii: einstein_radius**2 / radii,
        centre=centre,
    )


def _sis_profile(einstein_radius, centre=(0.0, 0.0)):
    """Return an analytic singular isothermal sphere profile."""
    return _SphericalProfile(
        lambda radii: np.full(radii.shape, einstein_radius),
        centre=centre,
    )


def _angular_diameter_distance_lens_m(z_lens=Z_LENS):
    """Return the angular diameter distance to the lens in metres."""
    distance_mpc = float(COSMOLOGY.angular_diameter_distance(z_lens).value)
    return distance_mpc * CONSTANTS.MPC_TO_M


def _diagnostics(profile, **overrides):
    """Return diagnostics for a profile with the default panel geometry."""
    kwargs = {
        'z_lens': Z_LENS,
        'z_source': Z_SOURCE,
        'cosmology': COSMOLOGY,
        'aperture_radii_arcsec': APERTURES_ARCSEC,
    }
    kwargs.update(overrides)
    return DIAGNOSTICS.subhalo_physical_diagnostics(profile, **kwargs)


def _nfw_profile_parameters(mass_msun, concentration, z_lens=Z_LENS, z_source=Z_SOURCE):
    """Return NFW ``kappa_s``, scale radius in arcsec and ``Sigma_crit``.

    This mirrors the production `lensing.generator` NFW construction so
    the diagnostics are exercised on a physically realistic profile.
    """
    rs_kpc, rho_s = MASS_MODELS.nfw_scale_parameters(
        mass_msun,
        concentration,
        z_lens,
        COSMOLOGY,
    )
    D_l_m = _angular_diameter_distance_lens_m(z_lens)
    D_s_m = float(COSMOLOGY.angular_diameter_distance(z_source).value) * CONSTANTS.MPC_TO_M
    D_ls_m = (
        float(COSMOLOGY.angular_diameter_distance_z1z2(z_lens, z_source).value)
        * CONSTANTS.MPC_TO_M
    )
    sigma_crit = (const.c.value**2 / (4.0 * np.pi * const.G.value)) * (
        D_s_m / (D_l_m * D_ls_m)
    )
    rs_m = rs_kpc * CONSTANTS.KPC_TO_M
    kappa_s = (rho_s * rs_m) / sigma_crit
    scale_radius_arcsec = (rs_m / D_l_m) * CONSTANTS.ARCSEC_PER_RAD
    return float(kappa_s), float(scale_radius_arcsec), float(sigma_crit)


def _projected_mass_from_convergence(profile, radius_arcsec, sigma_crit, samples=200000):
    """Integrate the convergence to obtain a projected mass in solar masses.

    The integral uses the substitution ``r = R s^2`` and a midpoint rule,
    which removes the logarithmic NFW cusp from the quadrature and never
    evaluates the profile at the centre.
    """
    steps = (np.arange(samples, dtype=float) + 0.5) / samples
    radii_arcsec = radius_arcsec * steps**2
    points = np.zeros((samples, 2), dtype=float)
    points[:, 1] = np.asarray(profile.centre, dtype=float)[1] + radii_arcsec
    points[:, 0] = np.asarray(profile.centre, dtype=float)[0]
    # Convergence is over-sampled by PyAutoGalaxy and needs a real grid.
    convergence = np.asarray(
        profile.convergence_2d_from(grid=al.Grid2DIrregular(values=points)),
        dtype=float,
    )

    # 2 pi Int kappa r dr with r = R s^2 gives 4 pi R^2 Int kappa s^3 ds.
    integral_arcsec2 = (
        4.0 * np.pi * radius_arcsec**2 * float(np.sum(convergence * steps**3)) / samples
    )
    arcsec_to_m = _angular_diameter_distance_lens_m() / CONSTANTS.ARCSEC_PER_RAD
    mass_kg = sigma_crit * integral_arcsec2 * arcsec_to_m**2
    return mass_kg / DIAGNOSTICS.MSUN_TO_KG


def test_point_mass_enclosed_mass_recovers_injected_mass():
    mass_msun = 1.0e8
    einstein_radius = MASS_MODELS.einstein_radius_point_mass(
        mass_msun, Z_LENS, Z_SOURCE, COSMOLOGY
    )
    result = _diagnostics(_point_mass_profile(einstein_radius))
    for aperture in result['apertures']:
        assert aperture['enclosed_mass_2d_msun'] == pytest.approx(mass_msun, rel=1.0e-12)


def test_point_mass_diagnostics_are_offset_invariant():
    mass_msun = 1.0e8
    einstein_radius = MASS_MODELS.einstein_radius_point_mass(
        mass_msun, Z_LENS, Z_SOURCE, COSMOLOGY
    )
    centred = _diagnostics(_point_mass_profile(einstein_radius))
    offset = _diagnostics(_point_mass_profile(einstein_radius, centre=(1.3, -0.7)))
    for centred_aperture, offset_aperture in zip(
        centred['apertures'], offset['apertures']
    ):
        assert offset_aperture['deflection_arcsec'] == pytest.approx(
            centred_aperture['deflection_arcsec'], rel=1.0e-14
        )


def test_sis_enclosed_mass_matches_isothermal_closed_form():
    mass_msun = 1.0e9
    einstein_radius = MASS_MODELS.einstein_radius_sis_m200(
        mass_msun, Z_LENS, Z_SOURCE, COSMOLOGY
    )
    sigma_v_m_s = (
        MASS_MODELS.sigma_v_from_m200_sis(mass_msun, Z_LENS, COSMOLOGY) * 1000.0
    )
    arcsec_to_m = _angular_diameter_distance_lens_m() / CONSTANTS.ARCSEC_PER_RAD

    result = _diagnostics(_sis_profile(einstein_radius))
    for aperture in result['apertures']:
        radius_m = aperture['radius_arcsec'] * arcsec_to_m
        # M_2D(<R) = pi sigma_v^2 R / G for a singular isothermal sphere.
        expected_msun = (
            np.pi * sigma_v_m_s**2 * radius_m / const.G.value
        ) / DIAGNOSTICS.MSUN_TO_KG
        assert aperture['enclosed_mass_2d_msun'] == pytest.approx(
            expected_msun, rel=1.0e-12
        )
        assert aperture['deflection_arcsec'] == pytest.approx(
            einstein_radius, rel=1.0e-14
        )


def test_mean_convergence_is_deflection_over_radius():
    result = _diagnostics(_sis_profile(0.05))
    for aperture in result['apertures']:
        assert aperture['mean_convergence'] == pytest.approx(
            aperture['deflection_arcsec'] / aperture['radius_arcsec'], rel=1.0e-15
        )


def test_units_round_trip_arcsec_to_kpc_to_arcsec():
    profile = _SphericalProfile(
        lambda radii: np.full(radii.shape, 0.02),
        scale_radius=0.11,
        truncation_radius=1.1,
    )
    result = _diagnostics(profile, concentration=16.0)
    arcsec_to_kpc = result['arcsec_to_kpc']
    assert arcsec_to_kpc > 0.0

    for aperture in result['apertures']:
        assert aperture['radius_kpc'] / arcsec_to_kpc == pytest.approx(
            aperture['radius_arcsec'], rel=1.0e-15
        )
    assert result['scale_radius_kpc'] / arcsec_to_kpc == pytest.approx(
        result['scale_radius_arcsec'], rel=1.0e-15
    )
    assert result['truncation_radius_kpc'] / arcsec_to_kpc == pytest.approx(
        result['truncation_radius_arcsec'], rel=1.0e-15
    )
    assert result['r200_kpc'] == pytest.approx(
        16.0 * result['scale_radius_kpc'], rel=1.0e-15
    )


def test_diagnostics_dict_is_json_serialisable():
    profile = _SphericalProfile(
        lambda radii: np.full(radii.shape, 0.02),
        scale_radius=0.11,
        truncation_radius=1.1,
    )
    result = _diagnostics(profile, concentration=16.0)
    restored = json.loads(json.dumps(result))
    assert restored['c200'] == result['c200']
    assert restored['truncation_radius_arcsec'] == result['truncation_radius_arcsec']
    assert len(restored['apertures']) == len(APERTURES_ARCSEC)
    assert restored['apertures'][0]['enclosed_mass_2d_msun'] == pytest.approx(
        result['apertures'][0]['enclosed_mass_2d_msun'], rel=0.0, abs=0.0
    )


def test_untruncated_profile_reports_no_truncation_radius():
    result = _diagnostics(_sis_profile(0.05))
    assert result['truncation_radius_arcsec'] is None
    assert result['truncation_radius_kpc'] is None
    assert result['scale_radius_arcsec'] is None
    assert result['scale_radius_kpc'] is None
    assert result['c200'] is None
    assert result['r200_kpc'] is None


def test_scale_radius_without_concentration_leaves_r200_undefined():
    profile = _SphericalProfile(
        lambda radii: np.full(radii.shape, 0.02),
        scale_radius=0.11,
    )
    result = _diagnostics(profile)
    assert result['scale_radius_arcsec'] == pytest.approx(0.11, rel=0.0, abs=0.0)
    assert result['scale_radius_kpc'] is not None
    assert result['c200'] is None
    assert result['r200_kpc'] is None


def test_reports_the_requested_geometry():
    result = _diagnostics(_sis_profile(0.05))
    assert result['z_lens'] == pytest.approx(Z_LENS, rel=0.0, abs=0.0)
    assert result['z_source'] == pytest.approx(Z_SOURCE, rel=0.0, abs=0.0)
    assert result['sigma_crit_kg_m2'] > 0.0
    assert [aperture['radius_arcsec'] for aperture in result['apertures']] == (
        APERTURES_ARCSEC
    )


@pytest.mark.parametrize(
    "aperture_radii",
    [
        [],
        (),
        0.1,
        "0.1",
        [0.0],
        [-0.1],
        [0.1, np.nan],
        [0.1, np.inf],
        [0.1, True],
        [0.1, None],
        np.zeros((2, 2)),
    ],
)
def test_invalid_aperture_radii_raise(aperture_radii):
    with pytest.raises(ValueError):
        _diagnostics(_sis_profile(0.05), aperture_radii_arcsec=aperture_radii)


@pytest.mark.parametrize(
    "z_lens,z_source",
    [
        (0.5, 0.5),
        (2.0, 0.5),
        (0.0, 2.0),
        (-0.5, 2.0),
        (np.nan, 2.0),
        (0.5, np.inf),
    ],
)
def test_invalid_redshifts_raise(z_lens, z_source):
    with pytest.raises(ValueError):
        _diagnostics(_sis_profile(0.05), z_lens=z_lens, z_source=z_source)


@pytest.mark.parametrize("concentration", [0.0, -1.0, np.nan, np.inf, "16", True])
def test_invalid_concentration_raises(concentration):
    profile = _SphericalProfile(
        lambda radii: np.full(radii.shape, 0.02),
        scale_radius=0.11,
    )
    with pytest.raises(ValueError):
        _diagnostics(profile, concentration=concentration)


def test_concentration_rejected_for_profile_without_scale_radius():
    with pytest.raises(ValueError, match="scale_radius"):
        _diagnostics(_sis_profile(0.05), concentration=16.0)


def test_non_circular_profile_is_rejected():
    profile = _SphericalProfile(
        lambda radii: np.full(radii.shape, 0.02),
        axis_ratio=0.8,
    )
    with pytest.raises(ValueError, match="circularly symmetric"):
        _diagnostics(profile)


def test_profile_without_deflections_is_rejected():
    class _NoDeflections:
        centre = (0.0, 0.0)
        axis_ratio = 1.0

    with pytest.raises(ValueError, match="deflections_yx_2d_from"):
        _diagnostics(_NoDeflections())


@pytest.mark.parametrize("centre", [None, (0.0,), (0.0, 0.0, 0.0), 0.0, (np.nan, 0.0)])
def test_profile_with_invalid_centre_is_rejected(centre):
    profile = _sis_profile(0.05)
    profile.centre = centre
    with pytest.raises(ValueError):
        _diagnostics(profile)


def test_malformed_deflection_shape_is_rejected():
    class _BadShape:
        centre = (0.0, 0.0)
        axis_ratio = 1.0

        def deflections_yx_2d_from(self, grid):
            return np.zeros(len(grid))

    with pytest.raises(ValueError, match="shape"):
        _diagnostics(_BadShape())


def test_non_finite_deflection_is_rejected():
    with pytest.raises(ValueError, match="non-finite"):
        _diagnostics(_SphericalProfile(lambda radii: np.full(radii.shape, np.nan)))


def test_invalid_profile_scale_radius_is_rejected():
    profile = _SphericalProfile(
        lambda radii: np.full(radii.shape, 0.02),
        scale_radius=-0.1,
    )
    with pytest.raises(ValueError, match="profile.scale_radius"):
        _diagnostics(profile)


def test_invalid_profile_truncation_radius_is_rejected():
    profile = _SphericalProfile(
        lambda radii: np.full(radii.shape, 0.02),
        truncation_radius=0.0,
    )
    with pytest.raises(ValueError, match="profile.truncation_radius"):
        _diagnostics(profile)


@requires_autolens
def test_nfw_projected_mass_matches_integrated_convergence():
    """Check the deflection identity against an integrated convergence.

    Apertures start at 0.05 arcsec, roughly a third of the scale radius.
    Further inside the scale radius the PyAutoGalaxy `NFWSph` deflection
    itself loses relative precision (see the module report), so a tighter
    aperture would measure that rather than the identity.
    """
    mass_msun = 1.0e9
    concentration = MASS_MODELS.concentration_mass_relation(
        mass_msun,
        model="moline2017_eq7",
        x_sub=1.0,
        h=COSMOLOGY.reduced_h,
    )
    kappa_s, scale_radius_arcsec, sigma_crit = _nfw_profile_parameters(
        mass_msun, concentration
    )
    profile = al.mp.NFWSph(
        centre=(0.0, 0.0),
        kappa_s=kappa_s,
        scale_radius=scale_radius_arcsec,
    )
    result = _diagnostics(
        profile,
        aperture_radii_arcsec=[0.05, 0.1, 0.2, 0.5, 1.0],
        concentration=concentration,
    )

    assert result['sigma_crit_kg_m2'] == pytest.approx(sigma_crit, rel=1.0e-12)
    for aperture in result['apertures']:
        expected_msun = _projected_mass_from_convergence(
            profile, aperture['radius_arcsec'], sigma_crit
        )
        assert aperture['enclosed_mass_2d_msun'] == pytest.approx(
            expected_msun, rel=1.0e-6
        )


@requires_autolens
def test_truncated_nfw_encloses_less_mass_than_untruncated():
    mass_msun = 1.0e9
    concentration = MASS_MODELS.concentration_mass_relation(
        mass_msun,
        model="moline2017_eq7",
        x_sub=1.0,
        h=COSMOLOGY.reduced_h,
    )
    kappa_s, scale_radius_arcsec, _ = _nfw_profile_parameters(mass_msun, concentration)
    untruncated = al.mp.NFWSph(
        centre=(0.0, 0.0),
        kappa_s=kappa_s,
        scale_radius=scale_radius_arcsec,
    )
    truncated = al.mp.NFWTruncatedSph(
        centre=(0.0, 0.0),
        kappa_s=kappa_s,
        scale_radius=scale_radius_arcsec,
        truncation_radius=10.0 * scale_radius_arcsec,
    )

    untruncated_result = _diagnostics(untruncated, concentration=concentration)
    truncated_result = _diagnostics(truncated, concentration=concentration)

    assert truncated_result['truncation_radius_arcsec'] == pytest.approx(
        10.0 * scale_radius_arcsec, rel=1.0e-15
    )
    assert untruncated_result['truncation_radius_arcsec'] is None
    for truncated_aperture, untruncated_aperture in zip(
        truncated_result['apertures'], untruncated_result['apertures']
    ):
        assert (
            truncated_aperture['enclosed_mass_2d_msun']
            < untruncated_aperture['enclosed_mass_2d_msun']
        )


@requires_autolens
def test_production_cosmology_object_gives_consistent_diagnostics():
    """The PyAutoLens cosmology is the object the pipeline passes in.

    ``al.cosmo.Planck15`` and ``astropy.cosmology.Planck15`` are distinct
    Planck15 parameter sets, so the two agree to roughly 5e-6 relative in
    the critical surface density rather than exactly. The pipeline always
    passes its own cosmology object, and the module requires the geometry
    the profile was built with, so the production path is self-consistent;
    this test pins that the production object is accepted and agrees with
    an independent Planck15 to the level those two definitions allow.
    """
    profile = al.mp.NFWSph(centre=(0.0, 0.0), kappa_s=0.02, scale_radius=0.17)
    adapter_result = _diagnostics(profile, concentration=16.0)
    production_result = DIAGNOSTICS.subhalo_physical_diagnostics(
        profile,
        z_lens=Z_LENS,
        z_source=Z_SOURCE,
        cosmology=al.cosmo.Planck15(),
        aperture_radii_arcsec=APERTURES_ARCSEC,
        concentration=16.0,
    )
    assert production_result['sigma_crit_kg_m2'] == pytest.approx(
        adapter_result['sigma_crit_kg_m2'], rel=1.0e-4
    )
    assert production_result['r200_kpc'] == pytest.approx(
        adapter_result['r200_kpc'], rel=1.0e-4
    )
    for production, adapter in zip(
        production_result['apertures'], adapter_result['apertures']
    ):
        assert production['enclosed_mass_2d_msun'] == pytest.approx(
            adapter['enclosed_mass_2d_msun'], rel=1.0e-4
        )


@requires_autolens
def test_autolens_point_mass_recovers_injected_mass():
    mass_msun = 1.0e8
    einstein_radius = MASS_MODELS.einstein_radius_point_mass(
        mass_msun, Z_LENS, Z_SOURCE, COSMOLOGY
    )
    profile = al.mp.PointMass(centre=(0.4, -0.3), einstein_radius=einstein_radius)
    result = _diagnostics(profile)
    for aperture in result['apertures']:
        assert aperture['enclosed_mass_2d_msun'] == pytest.approx(mass_msun, rel=1.0e-12)

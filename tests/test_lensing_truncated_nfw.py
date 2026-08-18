"""Physics contracts for the truncated NFW subhalo model."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("autolens")

import autolens as al  # noqa: E402

from hwoslaps.lensing.generator import (  # noqa: E402
    _create_lens_galaxy,
    _create_subhalo,
)


LENS_REDSHIFT = 0.2
SOURCE_REDSHIFT = 0.6
SUBHALO_MASS_MSUN = 1.0e9
PIXEL_SCALE = 0.01
TRUNCATION_TAU_LADDER = (2.0, 5.0, 10.0, 20.0, 100.0)
RADII_ARCSEC = np.logspace(np.log10(0.005), np.log10(2.0), 96)


def _lens_galaxy():
    """Return a minimal macro lens for subhalo construction."""
    return _create_lens_galaxy(
        {
            "redshift": LENS_REDSHIFT,
            "mass": {
                "type": "Isothermal",
                "einstein_radius": 1.0,
                "centre": [0.0, 0.0],
                "ell_comps": [0.1, 0.0],
            },
        }
    )


def _subhalo_config(model, truncation=None, concentration=None):
    """Return an origin-centred subhalo configuration."""
    config = {
        "enabled": True,
        "mass": SUBHALO_MASS_MSUN,
        "model": model,
        "concentration": concentration
        or {"model": "moline2017_eq7", "x_sub": 1.0, "h": 0.6774},
        "position": {"type": "direct", "centre": [0.0, 0.0]},
    }
    if truncation is not None:
        config["truncation"] = truncation
    return config


def _build(model, truncation=None, concentration=None):
    """Build one subhalo profile and its truth record."""
    return _create_subhalo(
        _subhalo_config(model, truncation, concentration),
        LENS_REDSHIFT,
        SOURCE_REDSHIFT,
        _lens_galaxy(),
        pixel_scale=PIXEL_SCALE,
        cosmology=al.cosmo.Planck15(),
    )


def _radial_deflection(profile, radii=RADII_ARCSEC):
    """Return the radial deflection of an origin-centred profile."""
    grid = al.Grid2DIrregular(
        values=np.column_stack([np.zeros_like(radii), radii])
    )
    deflections = np.asarray(
        profile.deflections_yx_2d_from(grid=grid), dtype=float
    )
    return deflections[:, 1]


def test_large_tau_reproduces_untruncated_nfw_deflections():
    """Recover NFWSph deflections as the truncation radius grows large."""
    untruncated, _ = _build("NFW")
    truncated, _ = _build("NFWTruncated", {"mode": "scale_ratio", "tau": 1.0e6})

    alpha_untruncated = _radial_deflection(untruncated)
    alpha_truncated = _radial_deflection(truncated)

    assert np.all(alpha_untruncated > 0.0)
    # The truncated deflection is evaluated from the Baltz, Marshall and
    # Oguri (2009) closed form, whose large-tau limit involves cancelling
    # terms of order tau^2. At tau = 1e6 in double precision the residual
    # floor measured against the untruncated profile is ~1.6e-10 relative
    # (~4e-13 absolute on deflections of ~1e-3 arcsec), which is numerical
    # rather than physical. The assertion below still fails immediately if
    # the truncated branch used different lensing parameters at all.
    np.testing.assert_allclose(alpha_truncated, alpha_untruncated, rtol=1.0e-8)


def test_deflection_decreases_monotonically_as_tau_decreases():
    """Suppress the deflection at fixed M200 as the truncation tightens."""
    probe_radii = np.array([0.5, 1.0, 2.0])
    deflections = [
        _radial_deflection(
            _build("NFWTruncated", {"mode": "scale_ratio", "tau": tau})[0],
            probe_radii,
        )
        for tau in TRUNCATION_TAU_LADDER
    ]

    for tighter, looser in zip(deflections, deflections[1:]):
        assert np.all(tighter < looser)

    untruncated = _radial_deflection(_build("NFW")[0], probe_radii)
    assert np.all(deflections[-1] < untruncated)


def test_truncation_suppresses_the_deflection_at_every_sampled_radius():
    """Keep a tightly truncated profile below a loosely truncated one."""
    tight, _ = _build("NFWTruncated", {"mode": "scale_ratio", "tau": 2.0})
    loose, _ = _build("NFWTruncated", {"mode": "scale_ratio", "tau": 100.0})

    assert np.all(_radial_deflection(tight) <= _radial_deflection(loose))


def test_explicit_arcsec_matches_scale_ratio_at_the_same_radius():
    """Agree between truncation modes at an identical truncation radius."""
    ratio_profile, ratio_info = _build(
        "NFWTruncated", {"mode": "scale_ratio", "tau": 10.0}
    )
    truncation_radius = ratio_info["truncation_radius_arcsec"]
    assert truncation_radius == 10.0 * ratio_info["scale_radius_arcsec"]

    explicit_profile, explicit_info = _build(
        "NFWTruncated",
        {"mode": "explicit_arcsec", "radius_arcsec": truncation_radius},
    )

    assert explicit_info["truncation_radius_arcsec"] == truncation_radius
    np.testing.assert_array_equal(
        _radial_deflection(explicit_profile),
        _radial_deflection(ratio_profile),
    )


def test_truncated_nfw_shares_the_untruncated_lensing_parameters():
    """Differ from NFW only by truncation at a fixed mass and c200."""
    _, untruncated_info = _build("NFW")
    _, truncated_info = _build(
        "NFWTruncated", {"mode": "scale_ratio", "tau": 10.0}
    )

    assert truncated_info["concentration"] == untruncated_info["concentration"]
    assert truncated_info["kappa_s"] == untruncated_info["kappa_s"]
    assert (
        truncated_info["scale_radius_arcsec"]
        == untruncated_info["scale_radius_arcsec"]
    )


def test_truncation_provenance_is_recorded_in_the_subhalo_truth():
    """Record the truncation mode, ratio, and radius in the truth record."""
    _, ratio_info = _build("NFWTruncated", {"mode": "scale_ratio", "tau": 10.0})

    assert ratio_info["model"] == "NFWTruncated"
    assert ratio_info["truncation_mode"] == "scale_ratio"
    assert ratio_info["truncation_tau"] == 10.0
    assert (
        ratio_info["truncation_radius_arcsec"]
        == 10.0 * ratio_info["scale_radius_arcsec"]
    )
    assert (
        ratio_info["profile_parameters"]["truncation_radius"]
        == ratio_info["truncation_radius_arcsec"]
    )

    _, explicit_info = _build(
        "NFWTruncated", {"mode": "explicit_arcsec", "radius_arcsec": 0.05}
    )

    assert explicit_info["truncation_mode"] == "explicit_arcsec"
    assert explicit_info["truncation_tau"] is None
    assert explicit_info["truncation_radius_arcsec"] == 0.05


def test_untruncated_nfw_truth_carries_no_truncation_fields():
    """Leave the truncation truth absent for an untruncated NFW subhalo."""
    _, info = _build("NFW")

    assert "truncation_mode" not in info
    assert "truncation_tau" not in info
    assert "truncation_radius_arcsec" not in info
    assert "truncation_radius" not in info["profile_parameters"]


def test_untruncated_nfw_rejects_a_truncation_block():
    """Reject a truncation block supplied for an untruncated NFW subhalo."""
    with pytest.raises(ValueError, match="truncation is supported only when"):
        _build("NFW", {"mode": "scale_ratio", "tau": 10.0})


def test_truncated_nfw_requires_a_truncation_block():
    """Reject a truncated NFW subhalo with no truncation block."""
    with pytest.raises(ValueError, match="truncation must be a dict"):
        _build("NFWTruncated")


def test_truncated_nfw_accepts_the_explicit_concentration_model():
    """Combine a declared concentration with a truncated NFW subhalo."""
    _, info = _build(
        "NFWTruncated",
        {"mode": "scale_ratio", "tau": 10.0},
        concentration={"model": "explicit", "c200": 15.0},
    )

    assert info["concentration"] == 15.0
    assert info["concentration_model"] == "explicit"
    assert info["truncation_radius_arcsec"] > 0.0


def test_concentration_offset_matches_the_equivalent_explicit_concentration():
    """Reproduce an offset concentration with a directly declared c200."""
    _, offset_info = _build(
        "NFW",
        concentration={
            "model": "moline2017_eq7",
            "x_sub": 1.0,
            "h": 0.6774,
            "offset_dex": 0.3,
        },
    )
    _, explicit_info = _build(
        "NFW",
        concentration={"model": "explicit", "c200": offset_info["concentration"]},
    )

    assert offset_info["concentration_offset_dex"] == 0.3
    assert (
        offset_info["concentration"]
        == offset_info["concentration_pre_offset"] * 10.0**0.3
    )
    assert explicit_info["kappa_s"] == offset_info["kappa_s"]
    assert (
        explicit_info["scale_radius_arcsec"]
        == offset_info["scale_radius_arcsec"]
    )


def test_absent_concentration_offset_leaves_the_profile_unchanged():
    """Reproduce the un-offset profile exactly at offset_dex zero."""
    _, default_info = _build("NFW")
    _, zero_offset_info = _build(
        "NFW",
        concentration={
            "model": "moline2017_eq7",
            "x_sub": 1.0,
            "h": 0.6774,
            "offset_dex": 0.0,
        },
    )

    assert default_info["concentration_offset_dex"] is None
    assert default_info["concentration_pre_offset"] == default_info["concentration"]
    assert zero_offset_info["concentration"] == default_info["concentration"]
    assert zero_offset_info["kappa_s"] == default_info["kappa_s"]
    assert (
        zero_offset_info["scale_radius_arcsec"]
        == default_info["scale_radius_arcsec"]
    )

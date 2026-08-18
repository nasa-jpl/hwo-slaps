"""Unit tests for NFW concentration models in lensing mass models."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

TESTS_ROOT = Path(__file__).resolve().parent
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from _lensing_physics_helpers import load_mass_models_module  # noqa: E402

mass_models = load_mass_models_module()


def _moline_expected(mass_msun: float, x_sub: float, h: float) -> float:
    log_mass_term = np.log10((mass_msun * h) / 1.0e8)
    return float(
        19.9
        * (
            1.0
            + (-0.195 * log_mass_term)
            + (0.089 * log_mass_term)**2
            + (0.089 * log_mass_term)**3
        )
        * (1.0 + (-0.54 * np.log10(x_sub)))
    )


def test_moline_eq7_anchor_value():
    """Return 19.9 at the Moline eq. 7 normalization anchor."""
    h = 0.6774
    mass_anchor = 1.0e8 / h
    value = mass_models.concentration_moline2017_eq7(mass_anchor, x_sub=1.0, h=h)
    assert value == pytest.approx(19.9, rel=1e-10)


def test_moline_eq7_paper_regression_points():
    """Reproduce published Moline eq. 7 values at reference points."""
    test_points = [
        (1.0e6, 1.0, 0.6774, 28.915897079765447),
        (1.0e8, 1.0, 0.6774, 20.560847592361515),
        (1.0e9, 1.0, 0.6774, 16.792762422153395),
        (5.0e9, 0.5, 0.6774, 16.720675479468554),
        (1.0e10, 0.3, 0.70, 17.138469176898678),
        (1.0e12, 1.0, 0.6774, 8.136344788704443),
    ]
    for mass_msun, x_sub, h, expected in test_points:
        got = mass_models.concentration_moline2017_eq7(mass_msun, x_sub=x_sub, h=h)
        assert got == pytest.approx(expected, rel=1e-12)
        assert got == pytest.approx(_moline_expected(mass_msun, x_sub, h), rel=1e-12)


def test_moline_eq7_radial_monotonicity():
    """Decrease Moline concentration with increasing host radius."""
    mass_msun = 1.0e9
    h = 0.6774
    c_inner = mass_models.concentration_moline2017_eq7(mass_msun, x_sub=0.3, h=h)
    c_mid = mass_models.concentration_moline2017_eq7(mass_msun, x_sub=1.0, h=h)
    c_outer = mass_models.concentration_moline2017_eq7(mass_msun, x_sub=1.5, h=h)
    assert c_inner > c_mid > c_outer


@pytest.mark.parametrize(
    "mass_msun,x_sub,h",
    [
        (0.0, 1.0, 0.6774),
        (-1.0, 1.0, 0.6774),
        (1.0e9, 0.0, 0.6774),
        (1.0e9, -0.5, 0.6774),
        (1.0e9, 1.0, 0.0),
        (1.0e9, 1.0, -0.1),
        (1.0e5, 1.0, 0.6774),
        (1.0e13, 1.0, 0.6774),
        (1.0e9, 1.6, 0.6774),
        (np.inf, 1.0, 0.6774),
        (1.0e9, np.inf, 0.6774),
        (1.0e9, 1.0, np.inf),
    ],
)
def test_moline_eq7_rejects_invalid_inputs(mass_msun, x_sub, h):
    """Reject Moline inputs that are non-positive or out of domain."""
    with pytest.raises(ValueError):
        mass_models.concentration_moline2017_eq7(mass_msun, x_sub=x_sub, h=h)


def test_dispatch_requires_moline_arguments():
    """Require both x_sub and h for the Moline dispatch path."""
    with pytest.raises(ValueError, match="x_sub is required"):
        mass_models.concentration_mass_relation(1.0e9, model="moline2017_eq7", h=0.6774)
    with pytest.raises(ValueError, match="h is required"):
        mass_models.concentration_mass_relation(1.0e9, model="moline2017_eq7", x_sub=1.0)


def test_power_law_parity():
    """Reproduce the closed-form power-law concentration relation."""
    points = [
        (1.0e7, 0.2),
        (1.0e8, 0.5),
        (1.0e9, 1.0),
        (1.0e10, 2.0),
    ]
    for mass_msun, redshift in points:
        expected = 19.9 * (mass_msun / 1.0e8) ** (-0.195) * (1.0 + redshift) ** (-0.54)
        got = mass_models.concentration_mass_relation(
            mass_msun,
            model="power_law",
            z=redshift,
        )
        assert got == pytest.approx(expected, rel=1e-12)


def test_explicit_model_returns_the_declared_concentration():
    """Pass a declared c200 through without consulting any relation."""
    for c200 in (2.5, 12.0, 40.0):
        got = mass_models.concentration_mass_relation(
            1.0e9,
            model="explicit",
            c200=c200,
        )
        assert got == c200


def test_explicit_model_ignores_the_relation_mass_domain():
    """Accept a mass outside the Moline domain for the explicit model."""
    got = mass_models.concentration_mass_relation(
        1.0e15,
        model="explicit",
        c200=15.0,
    )
    assert got == 15.0


def test_explicit_dispatch_requires_c200():
    """Require an explicit c200 for the explicit dispatch path."""
    with pytest.raises(ValueError, match="c200 is required"):
        mass_models.concentration_mass_relation(1.0e9, model="explicit")


@pytest.mark.parametrize("bad_c200", [0.0, -1.0, np.nan, np.inf, -np.inf, True])
def test_explicit_model_rejects_non_positive_or_non_finite_c200(bad_c200):
    """Reject an explicit c200 that is not finite and positive."""
    with pytest.raises(ValueError):
        mass_models.concentration_mass_relation(
            1.0e9,
            model="explicit",
            c200=bad_c200,
        )


def test_unsupported_concentration_model_is_rejected():
    """Reject a concentration model outside the supported dispatch set."""
    with pytest.raises(ValueError, match="Unsupported concentration model"):
        mass_models.concentration_mass_relation(1.0e9, model="nfw_generic")


def test_absent_concentration_offset_is_an_exact_identity():
    """Leave the concentration untouched when no offset is configured."""
    base = mass_models.concentration_moline2017_eq7(1.0e9, x_sub=1.0, h=0.6774)
    assert mass_models.apply_concentration_offset_dex(base, None) == base


def test_zero_concentration_offset_is_an_exact_identity():
    """Reproduce the un-offset concentration exactly at offset_dex 0."""
    for model_value in (
        mass_models.concentration_moline2017_eq7(1.0e9, x_sub=1.0, h=0.6774),
        mass_models.concentration_power_law(1.0e9, z=0.2),
        mass_models.concentration_mass_relation(1.0e9, model="explicit", c200=12.0),
    ):
        assert (
            mass_models.apply_concentration_offset_dex(model_value, 0.0)
            == model_value
        )


@pytest.mark.parametrize("offset_dex", [0.5, 1.0, 2.0, -0.25, -1.0])
def test_concentration_offset_applies_the_exact_dex_factor(offset_dex):
    """Scale the concentration by exactly ten to the offset."""
    base = mass_models.concentration_moline2017_eq7(1.0e9, x_sub=1.0, h=0.6774)
    got = mass_models.apply_concentration_offset_dex(base, offset_dex)
    assert got == base * 10.0**offset_dex


def test_explicit_concentration_composes_with_the_offset():
    """Compose the explicit model with a one-dex offset."""
    base = mass_models.concentration_mass_relation(
        1.0e9,
        model="explicit",
        c200=10.0,
    )
    assert mass_models.apply_concentration_offset_dex(base, 1.0) == 100.0


@pytest.mark.parametrize(
    "bad_offset",
    [np.nan, np.inf, -np.inf, True, [0.5], object()],
)
def test_concentration_offset_rejects_non_numeric_or_non_finite_values(bad_offset):
    """Reject an offset that is not a finite non-boolean number."""
    with pytest.raises(ValueError, match="offset_dex must be a finite number"):
        mass_models.apply_concentration_offset_dex(19.9, bad_offset)


@pytest.mark.parametrize("extreme_offset", [400.0, -400.0])
def test_concentration_offset_rejects_a_non_finite_positive_result(extreme_offset):
    """Reject an offset that overflows or underflows the concentration."""
    with pytest.raises(ValueError, match="Offset concentration must be"):
        mass_models.apply_concentration_offset_dex(19.9, extreme_offset)


@pytest.mark.parametrize("bad_c200", [0.0, -1.0, np.nan, np.inf, True])
def test_concentration_offset_rejects_an_invalid_base_concentration(bad_c200):
    """Reject a pre-offset concentration that is not finite and positive."""
    with pytest.raises(ValueError, match="c200 must be a finite positive number"):
        mass_models.apply_concentration_offset_dex(bad_c200, 0.5)

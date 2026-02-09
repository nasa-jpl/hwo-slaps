"""Unit tests for NFW concentration models in lensing mass models."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src" / "hwoslaps"


def _bootstrap_hwoslaps_namespace():
    if "hwoslaps" not in sys.modules:
        pkg = types.ModuleType("hwoslaps")
        pkg.__path__ = [str(SRC_ROOT)]
        sys.modules["hwoslaps"] = pkg
    if "hwoslaps.lensing" not in sys.modules:
        pkg = types.ModuleType("hwoslaps.lensing")
        pkg.__path__ = [str(SRC_ROOT / "lensing")]
        sys.modules["hwoslaps.lensing"] = pkg


def _load_module(relative_path: str, module_name: str):
    module_path = SRC_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_mass_models():
    _bootstrap_hwoslaps_namespace()
    _load_module("constants.py", "hwoslaps.constants")
    return _load_module("lensing/mass_models.py", "hwoslaps.lensing.mass_models")


mass_models = _load_mass_models()


def _moline_expected(mass_msun: float, x_sub: float, h: float) -> float:
    log_mass_term = np.log10((mass_msun * h) / 1.0e8)
    return float(
        19.9
        * (1.0 + (-0.195 * log_mass_term) + (0.089 * log_mass_term**2) + (0.089 * log_mass_term**3))
        * (1.0 + (-0.54 * np.log10(x_sub)))
    )


def test_moline_eq7_anchor_value():
    h = 0.6774
    mass_anchor = 1.0e8 / h
    value = mass_models.concentration_moline2017_eq7(mass_anchor, x_sub=1.0, h=h)
    assert value == pytest.approx(19.9, rel=1e-10)


def test_moline_eq7_coefficient_regression_points():
    test_points = [
        (1.0e8, 1.0, 0.6774),
        (1.0e9, 1.0, 0.6774),
        (5.0e9, 0.5, 0.6774),
        (1.0e10, 0.3, 0.70),
        (2.5e7, 2.0, 0.6774),
    ]
    for mass_msun, x_sub, h in test_points:
        expected = _moline_expected(mass_msun, x_sub, h)
        got = mass_models.concentration_moline2017_eq7(mass_msun, x_sub=x_sub, h=h)
        assert got == pytest.approx(expected, rel=1e-12)


def test_moline_eq7_radial_monotonicity():
    mass_msun = 1.0e9
    h = 0.6774
    c_inner = mass_models.concentration_moline2017_eq7(mass_msun, x_sub=0.3, h=h)
    c_mid = mass_models.concentration_moline2017_eq7(mass_msun, x_sub=1.0, h=h)
    c_outer = mass_models.concentration_moline2017_eq7(mass_msun, x_sub=3.0, h=h)
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
        (np.inf, 1.0, 0.6774),
        (1.0e9, np.inf, 0.6774),
        (1.0e9, 1.0, np.inf),
    ],
)
def test_moline_eq7_rejects_invalid_inputs(mass_msun, x_sub, h):
    with pytest.raises(ValueError):
        mass_models.concentration_moline2017_eq7(mass_msun, x_sub=x_sub, h=h)


def test_dispatch_requires_moline_arguments():
    with pytest.raises(ValueError, match="x_sub is required"):
        mass_models.concentration_mass_relation(1.0e9, model="moline2017_eq7", h=0.6774)
    with pytest.raises(ValueError, match="h is required"):
        mass_models.concentration_mass_relation(1.0e9, model="moline2017_eq7", x_sub=1.0)


def test_power_law_parity():
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

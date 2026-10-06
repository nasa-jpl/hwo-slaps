"""Analytic spectral shapes and table frame/conversion rules."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from hwoslaps.config.checks import ConfigError
from hwoslaps.spectra.sed import build_sed, parse_sed


@pytest.mark.parametrize("mapping,expected", [({"kind": "flat_fnu"}, [1.0, 1.0, 1.0]),
    ({"kind": "flat_flambda"}, [0.25, 1.0, 4.0]), ({"kind": "power_law", "index": -2.0}, [0.25, 1.0, 4.0])])
def test_analytic_sed_shapes_follow_frequency_convention(mapping, expected):
    sed = build_sed(parse_sed(mapping, "sed"), redshift=0.5)
    np.testing.assert_array_equal(sed.fnu(np.array([0.5, 1.0, 2.0]) / 1.0e6), expected)
    assert sed.knots_m().size == 0


@pytest.mark.parametrize("frame,redshift", [("observed", 0.5), ("rest", 0.5), ("rest", 0.3), ("rest", 0.7)])
def test_table_flambda_conversion_and_observed_knots(spectral_file, frame, redshift):
    table = spectral_file([300.0, 400.0, 500.0], 1.0 / (np.array([300.0, 400.0, 500.0]) / 1.0e9)**2)
    sed = build_sed(parse_sed({"kind": "table", **table, "quantity": "flambda", "frame": frame}, "sed"), redshift=redshift)
    factor = 1.0+redshift if frame == "rest" else 1.0
    observed = np.array([300.0, 400.0, 500.0]) / 1.0e9 * factor
    np.testing.assert_allclose(sed.knots_m(), observed, rtol=1.0e-15)
    np.testing.assert_allclose(sed.fnu(observed), 1.0, rtol=1.0e-12)
    with pytest.raises(ValueError, match="support"):
        sed.fnu(np.array([200.0, 600.0]) / 1.0e9 * factor)


def test_sed_identity_uses_content_keys_and_rest_redshift(spectral_file):
    table = spectral_file([200.0, 400.0, 800.0], [0.2, 1.0, 0.4])
    copied = Path(table["path"]).with_name("copy.npz")
    shutil.copyfile(table["path"], copied)
    first = build_sed(parse_sed({"kind": "table", **table, "quantity": "fnu", "frame": "rest"}, "sed"), redshift=0.5)
    second = build_sed(parse_sed({"kind": "table", **table, "path": str(copied), "quantity": "fnu", "frame": "rest"}, "sed"), redshift=0.5)
    different_z = build_sed(first.spec, redshift=0.8)
    assert first.digest() == second.digest()
    assert first.digest() != different_z.digest()
    with pytest.raises(TypeError):
        first.file_digests[str(copied)] = "changed"


def test_rest_frame_and_blackbody_are_refused_for_analytic_kinds():
    with pytest.raises(ConfigError, match="unknown"):
        parse_sed({"kind": "power_law", "index": 1.0, "frame": "rest"}, "sed")
    with pytest.raises(ConfigError, match="kind"):
        parse_sed({"kind": "blackbody", "temperature_k": 5000.0}, "sed")


@pytest.mark.parametrize("quantity", ["fnu", "flambda"])
def test_log_fnu_preserves_linear_table_interpolation_without_intermediate_overflow(spectral_file, quantity):
    table = spectral_file([400.0, 600.0], [0.0, 1.0e308])
    sed = build_sed(parse_sed({"kind": "table", **table, "quantity": quantity}, "sed"), redshift=0.0)
    wavelengths = np.array([400.0, 450.0, 500.0, 600.0])/1.0e9
    actual = sed.log_fnu(wavelengths)
    assert actual[0] == -np.inf
    expected = np.log(1.0e308) + np.log(np.array([0.25, 0.5, 1.0]))
    if quantity == "flambda":
        expected += 2.0*np.log(wavelengths[1:])
    np.testing.assert_allclose(actual[1:], expected, rtol=1.0e-15)
    with pytest.raises(ValueError, match="support"):
        sed.log_fnu(np.array([399.0, 601.0])/1.0e9)

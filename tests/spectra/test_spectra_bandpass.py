"""Independent throughput products, integration and wavelength-bin geometry."""

import math

import numpy as np
import pytest

from hwoslaps.config.checks import ConfigError
from hwoslaps.spectra.bandpass import bin_integrals, build_bandpass, integrate_dlnlambda, parse_bandpass
from hwoslaps.spectra.photometry import band_mean_throughput


@pytest.mark.parametrize("suffix,unit,power", [("yaml", "nm", 1), ("csv", "angstrom", 2), ("npz", "um", 2)])
def test_tabulated_bandpass_integrates_analytic_ramp_powers(spectral_file, suffix, unit, power):
    table = spectral_file([400.0, 450.0, 500.0, 550.0, 600.0], [0.20, 0.225, 0.25, 0.275, 0.30], suffix=suffix, unit=unit)
    band = build_bandpass(parse_bandpass({"kind": "table", **table, "power": power, "support_nm": [450.0, 550.0]}, "band"))
    low, high, slope = 450.0, 550.0, 0.0005
    if power == 1:
        expected = slope * (high - low)
    else:
        expected = slope**2 * (high**2 - low**2) / 2.0
    assert integrate_dlnlambda(band.throughput, band.wavelengths_m) == pytest.approx(expected, rel=1.0e-9)
    assert band.throughput_at(500.0 / 1.0e9) == pytest.approx(0.25**power, rel=1.0e-12)
    assert not band.throughput_at(650.0 / 1.0e9)


def test_product_response_applies_power_to_each_surface(spectral_file):
    table = spectral_file([400.0, 450.0, 500.0, 550.0, 600.0], [0.45, 0.475, 0.5, 0.525, 0.55])
    band = build_bandpass(parse_bandpass({"kind": "product", "support_nm": [450.0, 550.0],
        "factors": [{"kind": "table", **table, "power": 2}, {"kind": "constant", "value": 0.8}]}, "band"))
    assert band.throughput_at(500.0 / 1.0e9) == pytest.approx(0.2, rel=1.0e-12)
    low, high, a, b = 450.0, 550.0, 0.25, 0.0005
    expected = 0.8 * (a*a*math.log(high/low) + 2*a*b*(high-low) + b*b*(high*high-low*low)/2) / math.log(high/low)
    assert band_mean_throughput(band) == pytest.approx(expected, rel=1.0e-9)


@pytest.mark.parametrize("support,response,error", [([200.0, 550.0], [0.2, 0.5], "cover"),
                                                    ([450.0, 550.0], [0.2, 1.2], "unit|interval|\\[0, 1\\]")])
def test_bandpass_tables_refuse_extrapolation_or_nonphysical_response(spectral_file, support, response, error):
    table = spectral_file([450.0, 550.0], response)
    with pytest.raises(ValueError, match=error):
        build_bandpass(parse_bandpass({"kind": "table", **table, "support_nm": support}, "band"))


def test_product_top_hat_must_cover_its_support():
    with pytest.raises(ConfigError) as error:
        parse_bandpass({"kind": "product", "support_nm": [450.0, 550.0],
            "factors": [{"kind": "top_hat", "min_nm": 460.0, "max_nm": 540.0, "throughput": 0.8}]}, "band")
    assert error.value.path == "band.factors[0]"


def test_nodes_and_clipped_bins_match_hand_geometry():
    band = build_bandpass(parse_bandpass({"kind": "top_hat", "min_nm": 400.0, "max_nm": 600.0, "throughput": 1.0}, "band"))
    np.testing.assert_allclose(band.nodes(2), np.array([450.0, 550.0]) / 1.0e9, rtol=0.0, atol=1.0e-22)
    np.testing.assert_allclose(band.bin_edges(np.array([300.0, 450.0, 550.0, 800.0]) / 1.0e9),
                               np.array([400.0, 400.0, 500.0, 600.0, 600.0]) / 1.0e9, rtol=0.0, atol=1.0e-22)
    with pytest.raises(ValueError, match="integer"):
        band.nodes(True)
    values = np.array([1.0, 3.0, 2.0])
    wavelengths = np.exp([0.0, 1.0, 2.0])
    np.testing.assert_allclose(bin_integrals(values, wavelengths, np.exp([0.0, 0.5, 0.5, 2.0])), [0.75, 0.0, 3.75], atol=1.0e-14)


def test_repeated_factor_path_refuses_two_loaded_file_epochs(spectral_file):
    import sys
    from pathlib import Path
    from hwoslaps.spectra.tables import read_table

    table = spectral_file([400.0, 600.0], [0.4, 0.6])
    path = Path(table["path"])
    original = path.read_bytes()
    previous = sys.getprofile()
    changed = []
    def publish(frame, event, returned):
        if not changed and event == "return" and frame.f_code is read_table.__code__:
            np.savez(path, wave=[400.0, 600.0], response=[0.3, 0.8])
            changed.append(True)
    sys.setprofile(publish)
    try:
        with pytest.raises(ValueError, match="changed between factor reads"):
            build_bandpass(parse_bandpass({"kind": "product", "support_nm": [450.0, 550.0],
                "factors": [{"kind": "table", **table}, {"kind": "table", **table}]}, "band"))
        assert changed == [True]
    finally:
        sys.setprofile(previous)
        path.write_bytes(original)

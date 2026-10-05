"""Independent AB equations, narrow-line integrals and bin photon conservation."""

import math

import numpy as np
import pytest

from hwoslaps.spectra.bandpass import bin_integrals, build_bandpass, parse_bandpass
from hwoslaps.spectra.photometry import (
    ab_scale_jy, ab_to_fnu_jy, detected_flux_per_m2, effective_wavelength_m, rate_from_ab,
    sky_rate_e_per_s_per_pixel, synthetic_ab_mag,
)
from hwoslaps.spectra.sed import build_sed, parse_sed


def band():
    return build_bandpass(parse_bandpass({"kind": "top_hat", "min_nm": 450.0, "max_nm": 550.0, "throughput": 0.21}, "band"))


def test_flat_fnu_top_hat_rates_match_paper_closed_form():
    response = band()
    assert ab_to_fnu_jy(0.0) == 3631.0
    assert ab_to_fnu_jy(5.0) == pytest.approx(36.31, rel=1.0e-12)
    area = 33.606448937520405
    expected = area * 0.21 * (3631.0 * 10.0**(-0.4 * 24.845)) * 1.0e-26 / 6.62607015e-34 * math.log(550.0/450.0)
    assert expected == pytest.approx(8.951505744562876, rel=1.0e-12)
    assert rate_from_ab(24.845, response, area) == pytest.approx(expected, rel=1.0e-12)
    assert sky_rate_e_per_s_per_pixel(23.0, response, area, 0.00716) == pytest.approx(0.002510279845963486, rel=1.0e-12)


@pytest.mark.parametrize("mapping", [{"kind": "flat_fnu"}, {"kind": "flat_flambda"}, {"kind": "power_law", "index": 0.7}])
def test_instrument_band_magnitude_fixes_rate_for_any_sed(mapping):
    response = band()
    sed = build_sed(parse_sed(mapping, "sed"), redshift=0.6)
    expected = 33.6 * 0.21 * 3631.0 * 10.0**(-0.4 * 24.0) * 1.0e-26 / 6.62607015e-34 * math.log(550.0/450.0)
    assert rate_from_ab(24.0, response, 33.6, sed=sed) == pytest.approx(expected, rel=1.0e-12)
    assert synthetic_ab_mag(sed, ab_scale_jy(sed, 24.0, response), response) == pytest.approx(24.0, abs=1.0e-12)


@pytest.mark.parametrize("index", [-2.0, 0.7])
def test_reference_band_colour_matches_power_law_closed_form(index):
    response = band()
    reference = build_bandpass(parse_bandpass({"kind": "top_hat", "min_nm": 700.0, "max_nm": 950.0, "throughput": 0.8}, "reference"))
    sed = build_sed(parse_sed({"kind": "power_law", "index": index}, "sed"), redshift=0.4)
    def mean(low, high):
        return (low**(-index) - high**(-index)) / (index * math.log(high/low))
    expected = 33.6 * 3631.0 * 10.0**(-0.4 * 24.0) * 1.0e-26 / 6.62607015e-34 * 0.21 * math.log(550.0/450.0) * mean(450.0, 550.0) / mean(700.0, 950.0)
    assert rate_from_ab(24.0, response, 33.6, sed=sed, reference_band=reference) == pytest.approx(expected, rel=1.0e-9)


@pytest.mark.parametrize("kind", ["flat_fnu", "flat_flambda"])
def test_photon_weighted_effective_wavelength_closed_forms(kind):
    response = band()
    sed = build_sed(parse_sed({"kind": kind}, "sed"), redshift=0.0)
    low, high = 450.0 / 1.0e9, 550.0 / 1.0e9
    expected = (high-low)/math.log(high/low) if kind == "flat_fnu" else (2.0/3.0) * (high**3-low**3)/(high**2-low**2)
    assert effective_wavelength_m(sed, response) == pytest.approx(expected, rel=1.0e-9)


@pytest.mark.parametrize("frame", ["observed", "rest"])
def test_narrow_line_between_dense_nodes_is_integrated(spectral_file, frame):
    response = band()
    grid_nm = response.wavelengths_m * 1.0e9
    index = int(np.searchsorted(grid_nm, 500.0))
    left, right = grid_nm[index-1], grid_nm[index]
    width = right-left
    knots = np.array([left+0.2*width, left+0.5*width, left+0.8*width])
    factor = 1.5 if frame == "rest" else 1.0
    table = spectral_file(np.array([400.0, *knots, 600.0])/factor, [0.0, 0.0, 1.0, 0.0, 0.0])
    sed = build_sed(parse_sed({"kind": "table", **table, "quantity": "fnu", "frame": frame}, "sed"), redshift=0.5)
    assert not np.any(sed.fnu(response.wavelengths_m))
    low, peak, high = knots
    # Integrating the two linear-in-lambda flanks in log wavelength analytically.
    expected = 0.21 * ((peak-low-low*math.log1p((peak-low)/low))/(peak-low)
                       + (high*math.log1p((high-peak)/peak)-(high-peak))/(high-peak))
    actual = detected_flux_per_m2(sed, 1.0, response) * 6.62607015e-34 / 1.0e-26
    assert actual == pytest.approx(expected, rel=1.0e-6)


def test_bin_rates_partition_one_integrand_and_keep_narrow_line_photons(spectral_file):
    qe = spectral_file([400.0, 500.0, 600.0], [0.4, 0.5, 0.6], name="qe")
    response = build_bandpass(parse_bandpass({"kind": "product", "support_nm": [450.0, 550.0],
        "factors": [{"kind": "table", **qe}, {"kind": "constant", "value": 0.832}]}, "band"))
    knots = np.array([400.0, 499.998, 500.0, 500.002, 600.0])
    table = spectral_file(knots, 0.001*knots+np.array([0.0, 0.0, 1.0, 0.0, 0.0]), name="continuum_line")
    sed = build_sed(parse_sed({"kind": "table", **table, "quantity": "fnu"}, "sed"), redshift=0.0)
    wavelengths, throughput = response.integration_grid(sed)
    integrand = throughput * sed.fnu(wavelengths)
    edges = np.array([450.0, 470.1, 490.2, 499.999, 520.3, 535.4, 550.0]) / 1.0e9
    rates = bin_integrals(integrand, wavelengths, edges)
    total = detected_flux_per_m2(sed, 1.0, response) * 6.62607015e-34 / 1.0e-26
    assert rates.sum() == pytest.approx(total, rel=1.0e-13)
    line_table = spectral_file(knots, [0.0, 0.0, 1.0, 0.0, 0.0], name="line_only")
    line_sed = build_sed(parse_sed({"kind": "table", **line_table, "quantity": "fnu"}, "sed"), redshift=0.0)
    line_grid, line_throughput = response.integration_grid(line_sed)
    whole_line = bin_integrals(line_throughput*line_sed.fnu(line_grid), line_grid,
        np.array([450.0, 460.0, 480.0, 495.0, 505.0, 520.0, 540.0, 550.0])/1.0e9)
    line_total = detected_flux_per_m2(line_sed, 1.0, response) * 6.62607015e-34 / 1.0e-26
    np.testing.assert_array_equal(whole_line[[0, 1, 2, 4, 5, 6]], 0.0)
    assert whole_line[3] == pytest.approx(line_total, rel=1.0e-13)

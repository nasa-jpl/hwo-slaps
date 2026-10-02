"""Independent photometric equations, with explicit observing inputs."""

import math
import pytest

from hwoslaps.observation.photometry import (
    ab_mag_to_fnu_jy, photon_rate_per_m2, sky_rate_e_per_pix_s, effective_read_noise,
    solve_arc_snr_scale,
)


def test_ab_zero_point_and_magnitude_difference():
    assert ab_mag_to_fnu_jy(0) == 3631.0
    assert ab_mag_to_fnu_jy(5) == pytest.approx(36.31)


def test_flat_fnu_photon_integral_and_sky_pixel_area():
    expected = 1e-26 / 6.62607015e-34 * math.log(550 / 450)
    assert photon_rate_per_m2(1, 450e-9, 550e-9) == pytest.approx(expected, rel=1e-12)
    sky_flux = 3631 * 10**(-0.4 * 23)
    sky = sky_flux * expected * 33.6 * 0.21 * 0.00716**2
    assert sky_rate_e_per_pix_s(23, 33.6, 0.21, 0.00716, 450e-9, 550e-9) == pytest.approx(sky, rel=1e-12)
    with pytest.raises(ValueError, match='must exceed'):
        photon_rate_per_m2(1, 550e-9, 450e-9)


def test_independent_detector_reads_add_variance():
    assert effective_read_noise(3, 4) == 6
    with pytest.raises(ValueError, match='positive integer'):
        effective_read_noise(3, True)


def test_arc_snr_normalization_solves_independent_monotone_response():
    scale, record = solve_arc_snr_scale(math.sqrt, 1.0, 5.0)
    assert scale == pytest.approx(25.0, rel=1e-6)
    assert record['achieved_arc_snr'] == pytest.approx(5.0, rel=1e-6)
    assert record['bracket_low_scale_factor'] <= scale <= record['bracket_high_scale_factor']
    assert record['relative_residual'] <= record['relative_tolerance']
    same, exact = solve_arc_snr_scale(lambda _: 7.0, 0.25, 7.0)
    assert same == pytest.approx(0.25)
    assert exact['bracket_steps'] == exact['solver_iterations'] == 0


def test_arc_snr_normalization_rejects_unreachable_or_unphysical_response():
    with pytest.raises(ValueError, match='not bracketed'):
        solve_arc_snr_scale(lambda scale: 12 - 1 / (1 + scale), 1, 25, max_bracket_steps=3)
    with pytest.raises(ValueError, match='positive and finite'):
        solve_arc_snr_scale(lambda scale: -scale, 1, 10)

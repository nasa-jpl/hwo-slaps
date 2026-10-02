"""Explicit flat-f_nu photometry helpers; no telescope or source defaults."""

import math
from numbers import Integral

AB_ZERO_POINT_JY = 3631.0
JANSKY_TO_SI = 1.0e-26
PLANCK_H = 6.62607015e-34


def _positive(value, name):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f'{name} must be positive and finite')
    return number


def ab_mag_to_fnu_jy(m_ab):
    """Convert a finite AB magnitude to flux density in janskys."""
    magnitude = float(m_ab)
    if not math.isfinite(magnitude):
        raise ValueError('AB magnitude must be finite')
    return AB_ZERO_POINT_JY * 10.0 ** (-0.4 * magnitude)


def photon_rate_per_m2(fnu_jy, lam_min_m, lam_max_m):
    """Integrate a flat-f_nu spectrum through an ideal top-hat band.

    Wavelength edges are metres; the result is photons per second per square
    metre. This is the existing analytic approximation, not an arbitrary SED
    or wavelength-dependent response integrator.
    """
    flux = float(fnu_jy)
    if not math.isfinite(flux) or flux < 0:
        raise ValueError('fnu_jy must be finite and non-negative')
    low = _positive(lam_min_m, 'lam_min_m')
    high = _positive(lam_max_m, 'lam_max_m')
    if high <= low:
        raise ValueError('lam_max_m must exceed lam_min_m')
    return flux * JANSKY_TO_SI / PLANCK_H * math.log(high / low)


def detected_source_rate_e_per_s(m_ab, area_m2, system_qe, lam_min_m, lam_max_m):
    """Convert a band AB magnitude to detected electrons per second."""
    area = _positive(area_m2, 'area_m2')
    efficiency = _positive(system_qe, 'system_qe')
    if efficiency > 1:
        raise ValueError('system_qe must not exceed one')
    return photon_rate_per_m2(ab_mag_to_fnu_jy(m_ab), lam_min_m, lam_max_m) * area * efficiency


def sky_rate_e_per_pix_s(sky_mag_per_arcsec2, area_m2, system_qe,
                       pixel_scale_arcsec, lam_min_m, lam_max_m):
    """Convert sky AB surface brightness to detected electrons per pixel/s."""
    scale = _positive(pixel_scale_arcsec, 'pixel_scale_arcsec')
    return detected_source_rate_e_per_s(
        sky_mag_per_arcsec2, area_m2, system_qe, lam_min_m, lam_max_m,
    ) * scale**2


def effective_read_noise(per_read_e, n_reads):
    """Combine independent read-noise contributions in quadrature."""
    per_read = _positive(per_read_e, 'per_read_e')
    if isinstance(n_reads, bool) or not isinstance(n_reads, Integral) or n_reads < 1:
        raise ValueError('n_reads must be a positive integer')
    return per_read * math.sqrt(n_reads)


def solve_arc_snr_scale(response, initial_scale, target_arc_snr, *, bracket_factor=10.0,
                        max_bracket_steps=12, log_scale_tolerance=1e-9, relative_tolerance=1e-6):
    """Solve one monotone arc-S/N response for its target scale factor.

    The achieved arc S/N grows strictly monotonically with the source
    scale factor, linearly where the blank-pixel variance dominates and
    as its square root where source shot noise does, so the root is
    unique. The solve runs Brent's method on ``log(scale)`` after a
    geometric bracket search, and every failure is loud: a bracket that
    does not close, a solve that does not converge, and an achieved arc
    S/N outside :data:`relative_tolerance` all raise.

    Parameters
    ----------
    response : `callable`
        Function mapping one scale factor to an achieved arc S/N.
    initial_scale : `float`
        Scale factor the bracket search starts from.
    target_arc_snr : `float`
        Requested achieved arc S/N.

    Returns
    -------
    scale : `float`
        Scale factor whose achieved arc S/N is the requested value.
    record : `dict`
        Provenance record of the requested and achieved values, the
        bracket, and the solver effort.
    """
    from scipy.optimize import brentq

    bracket_factor = _positive(bracket_factor, 'bracket_factor')
    if bracket_factor <= 1:
        raise ValueError('bracket_factor must exceed one')
    if isinstance(max_bracket_steps, bool) or not isinstance(max_bracket_steps, Integral) or max_bracket_steps < 1:
        raise ValueError('max_bracket_steps must be a positive integer')
    log_scale_tolerance = _positive(log_scale_tolerance, 'log_scale_tolerance')
    relative_tolerance = _positive(relative_tolerance, 'relative_tolerance')
    target = _positive(target_arc_snr, 'target_arc_snr')
    start = _positive(initial_scale, 'initial_scale')
    evaluations = 0

    def objective(log_scale):
        """Return the log ratio of achieved to requested arc S/N."""
        nonlocal evaluations
        evaluations += 1
        scale = math.exp(log_scale)
        achieved = response(scale)
        value = float(achieved)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f'The arc S/N response returned {achieved!r} at scale factor '
                f'{scale}; it must be positive and finite.'
            )
        return math.log(value / target)

    log_start = math.log(start)
    start_value = objective(log_start)
    log_low = log_high = log_start
    low_value = high_value = start_value
    steps = 0
    if start_value != 0.0:
        log_step = math.log(bracket_factor)
        while low_value * high_value > 0.0:
            if steps >= max_bracket_steps:
                raise ValueError(
                    f'Arc S/N target {target} is not bracketed by scale '
                    f'factors {math.exp(log_low)} to {math.exp(log_high)} '
                    f'after {steps} steps of factor {bracket_factor}; '
                    f'the achieved values there are '
                    f'{target * math.exp(low_value)} and '
                    f'{target * math.exp(high_value)}.'
                )
            steps += 1
            if start_value < 0.0:
                log_high += log_step
                high_value = objective(log_high)
            else:
                log_low -= log_step
                low_value = objective(log_low)
        log_scale, result = brentq(
            objective,
            log_low,
            log_high,
            xtol=log_scale_tolerance,
            full_output=True,
            disp=False,
        )
        if not result.converged:
            raise ValueError(
                f'The Brent solve for arc S/N target {target} did not '
                f'converge: {result.flag}.'
            )
        iterations = int(result.iterations)
    else:
        log_scale = log_start
        iterations = 0

    scale = math.exp(log_scale)
    achieved = float(response(scale))
    evaluations += 1
    residual = abs(achieved / target - 1.0)
    if residual > relative_tolerance:
        raise ValueError(
            f'The solved scale factor {scale} achieves an arc S/N of '
            f'{achieved} against the requested {target}, a relative miss of '
            f'{residual} beyond the accepted {relative_tolerance}.'
        )
    record = {
        'requested_arc_snr': target,
        'achieved_arc_snr': achieved,
        'relative_residual': residual,
        'initial_scale_factor': start,
        'bracket_low_scale_factor': math.exp(log_low),
        'bracket_high_scale_factor': math.exp(log_high),
        'bracket_steps': steps,
        'solver': 'scipy.optimize.brentq on log(scale factor)',
        'solver_iterations': iterations,
        'forward_model_evaluations': evaluations,
        'log_scale_tolerance': log_scale_tolerance,
        'relative_tolerance': relative_tolerance,
    }
    return scale, record

"""Shared spherical-halo values, with no angular core or coordinate relocation.

NFW/BMO n=1 have a removable vector value at zero, but their derivatives with
respect to the centre are not finite there. PointMass/SIS have no central vector
value: callers must integrate the image or reject an unresolved singular sample.
"""
from __future__ import annotations

from fractions import Fraction
from math import comb

import numpy as np

_REGULAR_WIDTH = 0.02
_F_REGULAR = (1., -2 / 3, 7 / 15, -12 / 35, 83 / 315, -146 / 693, 523 / 3003,
              -952 / 6435, 14051 / 109395)
_G_REGULAR = (1 / 3, -2 / 5, 13 / 35, -20 / 63, 61 / 231, -94 / 429, 1181 / 6435,
              -1896 / 12155, 6223 / 46189)
_SMALL_RADIUS = 0.1
# F(x) = sum_n x^(2n) [a_n log(2/x) + b_n].  Derive exact
# rational coefficients once; this is a convergent series, not fitted anchors.
_A = tuple(Fraction(comb(2 * n, n), 4**n) for n in range(9))
_B = tuple(sum((-_A[k] * _A[n-k] / (2 * k) for k in range(1, n+1)), Fraction())
           for n in range(1, 9))


def polynomial(offset, coefficients):
    value = coefficients[-1]
    for coefficient in reversed(coefficients[:-1]):
        value = coefficient + offset * value
    return value


def nfw_f(radius, xp=np):
    """Real NFW F, regular at one, with safe inactive branches for autodiff."""
    radius = xp.real(xp.asarray(radius))
    regular = xp.abs(radius - 1.) <= _REGULAR_WIDTH
    lower = xp.where((radius < 1.) & ~regular, radius, 0.5)
    upper = xp.where((radius > 1.) & ~regular, radius, 2.)
    below = xp.arccosh(1. / lower) / xp.sqrt(1. - lower**2)
    above = xp.arccos(1. / upper) / xp.sqrt(upper**2 - 1.)
    return xp.where(regular, polynomial(radius - 1., _F_REGULAR), xp.where(radius < 1., below, above))


def nfw_g(radius, xp=np):
    radius = xp.real(xp.asarray(radius))
    regular = xp.abs(radius - 1.) <= _REGULAR_WIDTH
    safe = xp.where(regular, 2., radius)
    ordinary = (1. - nfw_f(safe, xp)) / (safe**2 - 1.)
    return xp.where(regular, polynomial(radius - 1., _G_REGULAR), ordinary)


def _nfw_mass_over_radius_squared_small(radius, xp):
    logarithm = xp.log(2. / radius)
    coefficients = tuple(float(a) * logarithm + float(b) for a, b in zip(_A[1:], _B))
    return polynomial(radius**2, coefficients)


def nfw_mass_over_radius_squared(radius, xp=np):
    """[log(x/2)+F(x)]/x^2 for positive x, without central cancellation."""
    small = radius <= _SMALL_RADIUS
    series_radius = xp.where(small, radius, 0.05)
    ordinary_radius = xp.where(small, 0.5, radius)
    ordinary = (xp.log(ordinary_radius / 2.) + nfw_f(ordinary_radius, xp)) / ordinary_radius**2
    return xp.where(small, _nfw_mass_over_radius_squared_small(series_radius, xp), ordinary)


def tnfw_mass_over_radius_squared(radius, tau, xp=np):
    """BMO n=1 projected mass / x^2, normalized by 4 pi rho_s r_s^3.

    The central branch cancels the large logarithms algebraically before
    evaluation. It retains the exact dependence on x/tau and uses the convergent
    NFW F-log series through x^16. The ordinary branch is regular at x=1.
    """
    small = radius <= _SMALL_RADIUS * xp.minimum(1., tau)
    x = xp.where(small, radius, 0.05 * xp.minimum(1., tau))
    squared, t2 = x**2, tau**2
    root = xp.sqrt(t2 + squared)
    root_sum = root + tau
    # log((sqrt(tau^2+x^2)+tau)/(2 tau))/x^2, formed using
    # log1p(z)/z with a removable value to avoid underflow at tiny x.
    z = squared / (2. * tau * root_sum)
    safe_z = xp.where(z == 0., 1., z)
    log_ratio_over_squared = xp.where(z == 0., 1., xp.log1p(z) / safe_z) / (2. * tau * root_sum)
    factor = (t2 - 1.) / (tau * root_sum)
    central = t2 / (t2 + 1.)**2 * (
        (t2 + 2. * squared - 1.) * _nfw_mass_over_radius_squared_small(x, xp)
        + (2. - factor) * xp.log(2. / x) - factor * xp.log(tau)
        - root * (t2 - 1.) / tau * log_ratio_over_squared - xp.pi / root_sum)
    x = xp.where(small, 0.5, radius)
    root = xp.sqrt(t2 + x**2)
    k = xp.log(x / (root + tau))
    ordinary = t2 / (t2 + 1.)**2 * (
        (t2 + 2. * x**2 - 1.) * nfw_f(x, xp) + xp.pi * tau
        + (t2 - 1.) * xp.log(tau) + root * ((t2 - 1.) / tau * k - xp.pi)) / x**2
    # At large radius the two O(x) terms in the projected mass cancel.
    # Rationalize that difference, and use K=-asinh(tau/x), so the finite
    # total-mass limit remains accurate even for a compact halo far away.
    x = xp.where(radius > 2., radius, 3.)
    root = xp.sqrt(t2 + x**2)
    nfw_root = xp.sqrt(x**2 - 1.)
    cancelling_terms = (-2.*x**2*xp.arcsin(1./x)/nfw_root
        + xp.pi*((1.-t2)*x**2+t2)/(nfw_root*(x**2 + root*nfw_root)))
    far = t2/(t2+1.)**2 * ((t2-1.)*(nfw_f(x, xp) + xp.log(tau)
        - root/tau*xp.arcsinh(tau/x)) + xp.pi*tau + cancelling_terms)/x**2
    return xp.where(small, central, xp.where(radius > 2., far, ordinary))


def spherical_deflections(kind, delta_yx, parameters, xp=np):
    """Analytic positive-radius deflections and declared central value policy.

    No finite positive radius is clipped. Genuine central singularities return
    NaN to require an image-integration adapter; they never acquire a core.
    """
    delta = xp.asarray(delta_yx)
    squared = xp.sum(delta**2, axis=-1)
    radius = xp.sqrt(squared)
    positive_radius = xp.where(squared > 0., radius, 1.)
    if kind == "PointMass":
        factor = parameters["einstein_radius"]**2 / xp.where(squared > 0., squared, 1.)
    elif kind == "SIS":
        factor = parameters["einstein_radius"] / positive_radius
    else:
        eta = positive_radius / parameters["scale_radius"]
        mass = (nfw_mass_over_radius_squared(eta, xp) if kind == "NFW" else
                tnfw_mass_over_radius_squared(eta, parameters["truncation_radius"] / parameters["scale_radius"], xp))
        factor = 4. * parameters["kappa_s"] * mass
    values = delta * factor[..., None]
    if kind in ("PointMass", "SIS"):
        return xp.where((squared > 0.)[..., None], values, xp.nan)
    return values

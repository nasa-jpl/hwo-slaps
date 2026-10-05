"""Inference lane fixtures: synthetic refinement objectives and a raw backend dataset.

``box_objective`` builds the optimiser's production input type, ``BoxObjective``, from a
physical half chi-square and its gradient, so refinement runs without a backend.
``raw_imaging`` is an AutoLens ``Imaging`` built from numpy arrays (a Gaussian image, a constant
noise map and the root ``tiny_gaussian_kernel``) and ``light_model`` the fit model of its H0, for
the sampler and backend tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from hwoslaps.inference.fit_model import FitArgument, FitComponent, FitGalaxy, FitModel, fixed, uniform
from hwoslaps.inference.objective import BoxObjective, ScalarCheck

RAW_SHAPE = (15, 15)
RAW_PIXEL_SCALE = 0.1
RAW_NOISE = 0.1


def make_box_objective(lower, upper, half_chi2, gradient, *, noise_normalization=0.0):
    """``BoxObjective`` of a physical half chi-square ``f(x)`` with gradient ``df/dx``.

    The residual is ``[sqrt(2 f)]`` and the direct likelihood is ``-f - N / 2``, the identities
    the AutoLens objective satisfies.
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    widths = upper - lower

    def to_x(z):
        return lower + np.asarray(z, dtype=float) * widths

    def value_and_gradient(z):
        x = to_x(z)
        return float(half_chi2(x)), np.asarray(gradient(x), dtype=float) * widths

    def residual(z):
        return np.array([np.sqrt(2.0 * float(half_chi2(to_x(z))))])

    def direct_check(z, value):
        direct = -float(half_chi2(to_x(z))) - 0.5 * noise_normalization
        implied = -float(value) - 0.5 * noise_normalization
        return ScalarCheck(direct_log_likelihood=direct, implied_log_likelihood=implied,
                           direct_log_likelihood_error=abs(direct - implied))

    return BoxObjective(lower=lower, upper=upper, value_and_gradient=value_and_gradient, residual=residual,
                        direct_check=direct_check)


@pytest.fixture
def box_objective():
    """Factory: ``box_objective(lower, upper, half_chi2, gradient, noise_normalization=0.0)``."""
    return make_box_objective


def make_raw_imaging(kernel):
    """A 15 x 15 AutoLens ``Imaging`` at 0.1": a unit-peak Gaussian of sigma 2 pixels, noise 0.1."""
    import autolens as al

    y, x = np.mgrid[:RAW_SHAPE[0], :RAW_SHAPE[1]].astype(float) - 7.0
    mask = al.Mask2D.all_false(shape_native=RAW_SHAPE, pixel_scales=RAW_PIXEL_SCALE)
    data = al.Array2D(values=np.exp(-(x**2 + y**2) / 8.0), mask=mask)
    noise_map = al.Array2D(values=np.full(RAW_SHAPE, RAW_NOISE), mask=mask)
    psf = al.Convolver(kernel=al.Array2D.no_mask(values=kernel, pixel_scales=RAW_PIXEL_SCALE))
    return al.Imaging(data=data, noise_map=noise_map, psf=psf)


@pytest.fixture
def raw_imaging(tiny_gaussian_kernel):
    return make_raw_imaging(tiny_gaussian_kernel)


def make_light_model():
    """H0 of the raw imaging: one Gaussian light profile, centre, intensity and sigma free."""
    centre = FitArgument("centre", (uniform(-0.05, 0.05, truth=0.0), uniform(-0.05, 0.05, truth=0.0)), pair=True)
    light = FitComponent("autogalaxy.profiles.light.standard.gaussian:Gaussian", (
        centre, FitArgument("ell_comps", (fixed(0.0), fixed(0.0)), pair=True),
        FitArgument("intensity", (uniform(0.5, 1.5, truth=1.0),), pair=False),
        FitArgument("sigma", (uniform(0.1, 0.3, truth=0.2),), pair=False)))
    return FitModel("smooth", (FitGalaxy("lens", 0.5, (("light", light),)),))


@pytest.fixture
def light_model():
    return make_light_model()

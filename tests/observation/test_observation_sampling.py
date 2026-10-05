"""Native-pixel convolution against an over-sampled reference: the sampling budget and its threshold.

The forward model bins the over-sampled light to native pixels and then convolves with the
pixel-integrated kernel. That is exact for light constant within each pixel. This oracle
measures the route against a reference that never bins: the light point-sampled on a grid
REFERENCE_FACTOR times finer, convolved with the continuous pixel-integrated Gaussian and read
at the native pixel centres. The budget is on the raw information ``sum (X / sigma)^2`` of the
smooth image (every case) and of the subhalo template (lensed cases): relative error at most
1e-2. ``MAX_NATIVE_SAMPLING_VARIATION`` is the measured threshold: the largest within-pixel
variation (``native_sampling_variation``) among passing cases below the smallest variation of
any failing case, rounded down to two significant figures. Every lensed case of that table lies
above the threshold, so the P1 scene at half its pixel scale supplies lensed cases below it, where
the subhalo template, whose information the threshold protects, must meet the budget.
"""

import math
from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal
from pathlib import Path

import numpy as np
import pytest
from scipy.special import erf

from hwoslaps.instrument import Detector
from hwoslaps.observation.expected import Exposure, convolve_light
from hwoslaps.observation.observation import MAX_NATIVE_SAMPLING_VARIATION
from hwoslaps.optics.kernels import DetectorPSF, KernelBinding
from hwoslaps.scene.builder import build_scene, native_sampling_variation
from hwoslaps.scene.cosmology import Cosmology, parse_cosmology
from hwoslaps.scene.halos import make_halo
from hwoslaps.scene.spec import LightGroup, parse_scene

pytestmark = pytest.mark.backend

PIXEL = 0.03
OVER_SAMPLE = 4
REFERENCE_FACTOR = 16
BUDGET = 1.0e-2
FWHMS = (0.5, 1.0, 2.0)
PHASES = tuple((k / 8, (3 * k % 8) / 8) for k in range(8))
EXPONENTIAL_RADII = (0.5, 1.0, 2.0, 4.0, 8.0)
SUBHALO_MASSES = (1.0e8, 1.0e9)
P1_FIELD_ARCSEC = 3.0
HALF_PIXEL_PHASES = PHASES
HALF_PIXEL_REFERENCE_FACTOR = REFERENCE_FACTOR // 2   # the fine grid of the 0.03" lensed rows, 0.001875"
EXPOSURE = Exposure(Detector(1.0, 0.2, 0.002), exposure_time_s=900.0, sky_rate_e_per_s=1.0)
ASSET = Path(__file__).resolve().parents[1] / "fixtures" / "paper_parity" / "source_image.npz"
P1_SOURCE_ELLIPTICITY = [0.14516129, 0.25142673]
NFW = {"type": "NFW", "concentration": {"kind": "moline2017_eq7", "x_sub": 1.0, "h": None}}
COSMOLOGY = Cosmology(parse_cosmology({"name": "Planck15"}))


def _shifted(centre, phase, pixel=PIXEL):
    return [centre[0] + phase[0] * pixel, centre[1] + phase[1] * pixel]


def _unlensed_mapping(light, phase):
    """A 64 x 64 scene whose only lens-plane light is ``light``; the wide source ring is not compared."""
    return {
        "grid": {"shape": [64, 64], "pixel_scale_arcsec": PIXEL, "over_sample_size": OVER_SAMPLE},
        "lens": {"redshift": 0.2,
                 "mass": {"mass": {"type": "Isothermal", "centre": _shifted([0.0, 0.0], phase),
                                   "einstein_radius": 0.5, "ell_comps": [0.0, 0.0]}},
                 "light": {"light": {**light, "centre": _shifted([0.0, 0.0], phase)}}},
        "source": {"redshift": 0.6,
                   "light": {"light": {"type": "Exponential", "centre": _shifted([0.0, 0.0], phase),
                                       "ell_comps": [0.0, 0.0], "intensity": 1.0, "effective_radius": 0.3}}},
        "subhalo": NFW,
    }


def _lensed_mapping(phase, pixel=PIXEL):
    """The P1 scene in the final schema (a 3" field, 100 x 100 at 0.03"), at ``pixel`` arcsec per pixel
    and shifted by ``phase`` pixels."""
    size = round(P1_FIELD_ARCSEC / pixel)
    return {
        "grid": {"shape": [size, size], "pixel_scale_arcsec": pixel, "over_sample_size": OVER_SAMPLE},
        "lens": {"redshift": 0.2,
                 "mass": {"mass": {"type": "Isothermal", "centre": _shifted([0.0, 0.0], phase, pixel),
                                   "einstein_radius": 1.0, "ell_comps": [0.1, 0.0]}}},
        "source": {"redshift": 0.6,
                   "light": {"light": {"type": "Exponential", "centre": _shifted([-0.03, 0.08], phase, pixel),
                                       "ell_comps": P1_SOURCE_ELLIPTICITY, "intensity": 2.0,
                                       "effective_radius": 0.11}}},
        "subhalo": NFW,
    }


def _fine(mapping, factor):
    grid = mapping["grid"]
    return {**mapping, "grid": {"shape": [n * factor for n in grid["shape"]],
                                "pixel_scale_arcsec": grid["pixel_scale_arcsec"] / factor, "over_sample_size": 1}}


def _pixel_gaussian(offset_pixels, fwhm_pixels):
    """Integral over one pixel of a unit Gaussian of the given FWHM, centred ``offset`` pixels away."""
    width = fwhm_pixels / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    scale = width * math.sqrt(2.0)
    return 0.5 * (erf((offset_pixels + 0.5) / scale) - erf((offset_pixels - 0.5) / scale))


def _engine_kernel(fwhm_pixels, pixel):
    g = _pixel_gaussian(np.arange(-7, 8, dtype=float), fwhm_pixels)
    return DetectorPSF.from_array(np.outer(g, g), pixel, normalize=True)


def _reference_rate(fine_image, fwhm_pixels, factor):
    """``R = A_y I A_x^T`` with ``A[j, z]`` the pixel-integrated Gaussian at native centre j from sample z."""
    def matrix(native):
        centres = np.arange(native) + 0.5
        samples = (np.arange(native * factor) + 0.5) / factor
        return _pixel_gaussian(centres[:, None] - samples[None, :], fwhm_pixels) / factor
    ny, nx = fine_image.shape[0] // factor, fine_image.shape[1] // factor
    return matrix(ny) @ fine_image @ matrix(nx).T


@dataclass(frozen=True)
class Render:
    native: np.ndarray          # the compared group's native light image (engine route input)
    fine: np.ndarray            # the compared group's point samples on the fine grid (reference input)
    variation: float            # native_sampling_variation of the compared group


def _render(mapping, group, subhalo, factor=REFERENCE_FACTOR):
    scene = build_scene(parse_scene(mapping), COSMOLOGY, subhalo=subhalo)
    fine = build_scene(parse_scene(_fine(mapping, factor)), COSMOLOGY, subhalo=subhalo)
    return Render(np.asarray(scene.light_images[group]), np.asarray(fine.light_images[group]),
                  float(native_sampling_variation(scene)[group]))


def _subhalo(mass, phase, pixel):
    model = parse_scene(_lensed_mapping(phase, pixel)).subhalo
    return make_halo(model, mass, tuple(_shifted([1.0, 0.0], phase, pixel)), redshift=0.2, source_redshift=0.6,
                     cosmology=COSMOLOGY)


def _engine_rate(image, group_plane, fwhm_pixels, pixel=PIXEL):
    groups = {group_plane: LightGroup(plane=group_plane, sed=None, components=("light",))}
    binding = KernelBinding.uniform(_engine_kernel(fwhm_pixels, pixel), [group_plane])
    return convolve_light({group_plane: image}, groups, binding, pixel)[group_plane]


def _information_metrics(engine, reference, sigma):
    information = float(np.sum((reference / sigma) ** 2))
    eps = abs(float(np.sum((engine / sigma) ** 2)) / information - 1.0)
    rho = float(np.sum(((engine - reference) / sigma) ** 2)) / information
    return eps, rho


@dataclass(frozen=True)
class Case:
    name: str
    variation: float
    eps_m: float
    rho_m: float
    eps_s: float | None = None
    rho_s: float | None = None

    @property
    def passes(self) -> bool:
        return self.eps_m <= BUDGET and (self.eps_s is None or self.eps_s <= BUDGET)

    @property
    def residuals_within_budget(self) -> bool:
        return self.rho_m <= BUDGET and (self.rho_s is None or self.rho_s <= BUDGET)


def _unlensed_cases(name, light, phase, fwhms=FWHMS, factor=REFERENCE_FACTOR):
    render = _render(_unlensed_mapping(light, phase), "lens", None, factor)
    cases = []
    for fwhm in fwhms:
        reference = _reference_rate(render.fine, fwhm, factor)
        sigma = EXPOSURE.noise_map_adu(reference)
        eps, rho = _information_metrics(EXPOSURE.signal_adu(_engine_rate(render.native, "lens", fwhm)),
                                        EXPOSURE.signal_adu(reference), sigma)
        cases.append(Case(f"{name} FWHM {fwhm} phase {phase}", render.variation, eps, rho))
    return cases


def _lensed_cases(phase, pixel=PIXEL, factor=REFERENCE_FACTOR):
    mapping = _lensed_mapping(phase, pixel)
    smooth = _render(mapping, "source", None, factor)
    halos = {mass: _render(mapping, "source", _subhalo(mass, phase, pixel), factor) for mass in SUBHALO_MASSES}
    cases = []
    for fwhm in FWHMS:
        reference = _reference_rate(smooth.fine, fwhm, factor)
        engine = _engine_rate(smooth.native, "source", fwhm, pixel)
        sigma = EXPOSURE.noise_map_adu(reference)
        eps_m, rho_m = _information_metrics(EXPOSURE.signal_adu(engine), EXPOSURE.signal_adu(reference), sigma)
        for mass, halo in halos.items():
            template_engine = EXPOSURE.signal_adu(_engine_rate(halo.native, "source", fwhm, pixel) - engine)
            template_reference = EXPOSURE.signal_adu(_reference_rate(halo.fine, fwhm, factor) - reference)
            eps_s, rho_s = _information_metrics(template_engine, template_reference, sigma)
            cases.append(Case(f"P1 {pixel}\" NFW {mass:.0e} FWHM {fwhm} phase {phase}", smooth.variation,
                              eps_m, rho_m, eps_s, rho_s))
    return cases


def _unlensed_lights():
    lights = {f"Exponential r_e {radius} px": {"type": "Exponential", "ell_comps": P1_SOURCE_ELLIPTICITY,
                                                "intensity": 1.0, "effective_radius": radius * PIXEL}
              for radius in EXPONENTIAL_RADII}
    lights["Image P3 asset"] = {"type": "Image", "asset_path": str(ASSET), "rotation_deg": 30.0,
                                "total_flux": 0.29}
    return lights


def _table():
    cases = []
    for phase in PHASES:
        for name, light in _unlensed_lights().items():
            cases.extend(_unlensed_cases(name, light, phase))
        cases.extend(_lensed_cases(phase))
    return cases


def _floor_two_figures(value):
    exponent = math.floor(math.log10(value)) - 1
    digits = Decimal(repr(value)).scaleb(-exponent).to_integral_value(rounding=ROUND_FLOOR)
    return float(digits.scaleb(exponent))


def _threshold(cases):
    smallest_failing = min(case.variation for case in cases if not case.passes)
    below = [case.variation for case in cases if case.passes and case.variation < smallest_failing]
    assert below, f"no passing case lies below the smallest failing variation {smallest_failing!r}"
    return max(below)


def _print_table(cases):
    print(f"\n{'case':64s} {'variation':>10s} {'eps_m':>10s} {'eps_s':>10s} {'rho_m':>10s} {'rho_s':>10s}")
    for case in cases:
        print(f"{case.name:64s} {case.variation:10.4g} {case.eps_m:10.3g} "
              f"{case.eps_s if case.eps_s is not None else float('nan'):10.3g} {case.rho_m:10.3g} "
              f"{case.rho_s if case.rho_s is not None else float('nan'):10.3g}")


def test_reference_grid_resolves_the_most_compact_cases():
    for name in ("Exponential r_e 0.5 px", "Image P3 asset"):
        light = _unlensed_lights()[name]
        (coarse,) = _unlensed_cases(name, light, PHASES[0], fwhms=(0.5,))
        (finer,) = _unlensed_cases(name, light, PHASES[0], fwhms=(0.5,), factor=2 * REFERENCE_FACTOR)
        assert abs(coarse.eps_m - finer.eps_m) <= BUDGET / 10, name
        assert abs(coarse.rho_m - finer.rho_m) <= BUDGET / 10, name


def test_native_pixel_convolution_meets_the_sampling_budget():
    cases = _table()
    _print_table(cases)
    failing = [case for case in cases if not case.passes]
    assert failing, "no case fails the budget, so the threshold is not set by a measured failure"
    threshold = _threshold(cases)
    print(f"v* = {threshold!r}; MAX_NATIVE_SAMPLING_VARIATION = {MAX_NATIVE_SAMPLING_VARIATION!r}")
    assert MAX_NATIVE_SAMPLING_VARIATION == _floor_two_figures(threshold)
    for case in cases:
        if case.variation <= MAX_NATIVE_SAMPLING_VARIATION:
            assert case.passes, case
            assert case.residuals_within_budget, case


def test_subhalo_template_meets_the_budget_below_the_threshold():
    # The quantity the threshold protects, measured below it: the table's lensed cases all lie above
    # the threshold, so this checks the P1 scene at half its pixel scale (200 x 200 at 0.015").
    cases = [case for phase in HALF_PIXEL_PHASES
             for case in _lensed_cases(phase, PIXEL / 2, HALF_PIXEL_REFERENCE_FACTOR)]
    _print_table(cases)
    for case in cases:
        assert case.variation <= MAX_NATIVE_SAMPLING_VARIATION, case
        assert case.passes, case
        assert case.residuals_within_budget, case

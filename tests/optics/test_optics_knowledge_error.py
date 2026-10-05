"""Wavefront draws at an exact aperture RMS and PSF knowledge errors (optics.knowledge_error)."""

import math
from importlib import resources

import numpy as np
import pytest

from hwoslaps.optics.knowledge_error import (
    RMS_RELATIVE_TOLERANCE, WavefrontDrawSpec, draw_knowledge_error, draw_wavefront,
)
from hwoslaps.optics.mode_priors import ModeWeightPriorSpec, PowerLawPriorSpec, load_prior
from hwoslaps.optics.providers import build_psf_provider, parse_psf
from hwoslaps.optics.pupils import build_pupil, parse_pupil
from hwoslaps.optics.wavefront import WavefrontBasis, parse_wavefront_selection

pytestmark = pytest.mark.backend

DRIFT = ModeWeightPriorSpec("packaged", "jwst_wss_drift_v1", None, None)
STATIC = ModeWeightPriorSpec("packaged", "jwst_wss_static_v1", None, None)
SEED = 20261005


def _within_tolerance(measured, amplitude):
    return abs(measured - amplitude) <= RMS_RELATIVE_TOLERANCE * max(1.0, amplitude)


def _optics(truth, scale=0.03):
    return build_psf_provider(parse_psf({"truth": truth}).truth, pixel_scale_arcsec=scale)


def _circular_truth(circular_pupil):
    return {"kind": "optical", "pupil": circular_pupil, "focal_length_m": 10.0, "wavelength_nm": 500.0,
            "detector_oversampling": 1, "kernel_shape": [11, 11]}


@pytest.mark.parametrize("family", ["combined", "global", "segment"])
def test_draw_has_exact_aperture_rms_on_a_segmented_pupil(p1_truth, family):
    truth = _optics(p1_truth)
    spec = WavefrontDrawSpec(DRIFT, 10.0, SEED, family)
    error = draw_knowledge_error(truth, spec)
    assert _within_tolerance(error.draw.measured_rms_nm, 10.0) and _within_tolerance(error.effective_rms_nm, 10.0)
    assert error.truth == truth.coefficients
    for mode, value in error.draw.coefficients.entries:
        assert error.model.value(mode) == truth.coefficients.value(mode) + value
    families = {mode.family for mode, _ in error.draw.coefficients.entries}
    assert families == {"combined": {"segment_hexikes", "zernikes"}, "global": {"zernikes"},
                        "segment": {"segment_hexikes"}}[family]
    doubled = draw_wavefront(truth.basis, WavefrontDrawSpec(DRIFT, 20.0, SEED, family))
    for (mode, single), (twin, double) in zip(error.draw.coefficients.entries, doubled.coefficients.entries):
        assert mode == twin and double == pytest.approx(2.0 * single, rel=1e-12)


def test_draw_on_a_circular_pupil_with_a_power_law_prior(circular_pupil):
    truth = _optics(_circular_truth(circular_pupil), scale=0.05)
    prior = ModeWeightPriorSpec("power_law", None, None, PowerLawPriorSpec(2.0, (4, 21), None, 0.0))
    draw = draw_wavefront(truth.basis, WavefrontDrawSpec(prior, 25.0, 7, "global"))
    assert _within_tolerance(draw.measured_rms_nm, 25.0)
    assert {mode.family for mode, _ in draw.coefficients.entries} == {"zernikes"} and draw.dark_segments == ()
    assert [mode.noll for mode, _ in draw.coefficients.entries] == list(range(4, 22))


def test_zero_amplitude_keeps_the_truth_and_still_loads_the_prior(p1_truth, tmp_path):
    truth = _optics(p1_truth)
    error = draw_knowledge_error(truth, WavefrontDrawSpec(DRIFT, 0.0, SEED, "combined"))
    assert error.model == truth.coefficients and error.draw.coefficients.is_empty
    assert error.draw.prior_digest == load_prior(DRIFT)[1]
    missing = ModeWeightPriorSpec("path", None, tmp_path / "missing.yaml", None)
    with pytest.raises(FileNotFoundError):
        draw_knowledge_error(truth, WavefrontDrawSpec(missing, 0.0, SEED, "combined"))


@pytest.mark.parametrize("truth_nm, message", [(1e12, "destroyed it in floating point"),
                                               (1e20, "erased it in floating point")])
def test_draw_gates_reject_erased_draws(p1_truth, truth_nm, message):
    prior, _ = load_prior(DRIFT)
    huge = {"segment_hexikes": {s: {n: truth_nm for n in prior.segment_weights} for s in range(19)},
            "zernikes": {n: truth_nm for n in prior.global_weights}}
    truth = _optics({**p1_truth, "wavefront": huge})
    with pytest.raises(ValueError, match=message):
        draw_knowledge_error(truth, WavefrontDrawSpec(DRIFT, 10.0, SEED, "combined"))


def test_segment_draws_need_a_segmented_pupil(circular_pupil):
    truth = _optics(_circular_truth(circular_pupil), scale=0.05)
    with pytest.raises(ValueError, match="segment draw needs segment hexikes"):
        draw_wavefront(truth.basis, WavefrontDrawSpec(DRIFT, 10.0, SEED, "segment"))


def test_draw_identity_changes_with_each_input(p1_truth, tmp_path):
    reference_truth = _optics(p1_truth)
    spec = WavefrontDrawSpec(DRIFT, 10.0, SEED, "combined")
    reference = draw_knowledge_error(reference_truth, spec).digest()
    assert draw_knowledge_error(reference_truth, spec).digest() == reference

    first, second = tmp_path / "a" / "drift.yaml", tmp_path / "b" / "drift.yaml"
    for path in (first, second):
        path.parent.mkdir()
    packaged = resources.files("hwoslaps.optics").joinpath("priors", "jwst_wss_drift_v1.yaml").read_bytes()
    first.write_bytes(packaged)
    second.write_bytes(packaged)
    at_first = WavefrontDrawSpec(ModeWeightPriorSpec("path", None, first, None), 10.0, SEED, "combined")
    at_second = WavefrontDrawSpec(ModeWeightPriorSpec("path", None, second, None), 10.0, SEED, "combined")
    assert draw_knowledge_error(reference_truth, at_first).digest() == reference
    assert draw_knowledge_error(reference_truth, at_second).digest() == reference

    edited = tmp_path / "edited.yaml"
    edited.write_bytes(packaged.replace(b"segment_variance_fraction: 0.", b"segment_variance_fraction: 0.1", 1))
    assert edited.read_bytes() != packaged
    variants = {
        "prior bytes": (reference_truth, WavefrontDrawSpec(ModeWeightPriorSpec("path", None, edited, None),
                                                           10.0, SEED, "combined")),
        "amplitude": (reference_truth, WavefrontDrawSpec(DRIFT, 11.0, SEED, "combined")),
        "seed": (reference_truth, WavefrontDrawSpec(DRIFT, 10.0, SEED + 1, "combined")),
        "family": (reference_truth, WavefrontDrawSpec(DRIFT, 10.0, SEED, "global")),
        "truth coefficients": (_optics({**p1_truth, "wavefront": {"zernikes": {4: 5.0}}}), spec),
        "pixel scale": (_optics(p1_truth, scale=0.029), spec),
    }
    digests = {name: draw_knowledge_error(truth, variant).digest() for name, (truth, variant) in variants.items()}
    assert all(digest != reference for digest in digests.values()), digests
    assert len(set(digests.values())) == len(digests)


def _independent_rms_nm(psf, coefficients):
    wavelength = psf.wavelengths_m[0]
    assert np.max(np.abs(psf.basis.opd_nm(coefficients))) * 1e-9 < 0.4 * wavelength
    field = np.asarray(psf.pupil_wavefront(coefficients=coefficients).electric_field)
    phase = np.angle(field[psf.pupil.illuminated_mask])
    opd = phase * wavelength / (2 * math.pi) * 1e9
    return float(np.sqrt(np.mean((opd - opd.mean()) ** 2)))


def test_obscured_segmented_pupil_draws(p1_truth):
    bare = _optics(p1_truth)
    assert bare.pupil.active_segments == tuple(range(19)) and bare.pupil.dark_segments == ()

    obscured = _optics({**p1_truth, "wavefront": {}, "pupil": {**p1_truth["pupil"], "obscuration_ratio": 0.25}})
    assert obscured.pupil.active_segments == tuple(range(1, 19))
    draw = draw_wavefront(obscured.basis, WavefrontDrawSpec(STATIC, 20.0, SEED, "combined"))
    assert draw.dark_segments == (0,)
    assert 0 not in {segment for segment, _, _ in draw.coefficients.segment_hexikes()}
    assert 0 not in draw.orthonormal_segment
    assert _within_tolerance(draw.measured_rms_nm, 20.0)
    assert _independent_rms_nm(obscured, draw.coefficients) == pytest.approx(20.0, rel=1e-9)
    every = parse_wavefront_selection({"segment_hexikes": {"segments": "all", "nolls": [1, 2]}}, "modes")
    assert sorted({mode.segment for mode in obscured.basis.select(every)}) == list(range(1, 19))
    named = parse_wavefront_selection({"segment_hexikes": {"segments": [0], "nolls": [1]}}, "modes")
    with pytest.raises(ValueError, match="segment 0 is dark"):
        obscured.basis.select(named)

    spiders = _optics({**p1_truth, "wavefront": {}, "pupil": {
        **p1_truth["pupil"], "spiders": {"count": 3, "width_m": 0.1, "angle_deg": 90.0}}})
    assert spiders.pupil.active_segments == tuple(range(19))
    assert _within_tolerance(draw_wavefront(spiders.basis, WavefrontDrawSpec(
        STATIC, 20.0, SEED, "combined")).measured_rms_nm, 20.0)

    coarse = build_pupil(parse_pupil({**p1_truth["pupil"], "rings": 1, "diameter_m": 5.0, "pixels": 8}, "pupil"))
    first = coarse.active_segments[0]
    pixels = int(np.count_nonzero(coarse.illuminated_mask & (np.asarray(coarse.segments[first]) > 0.5)))
    assert 0 < pixels < 10
    with pytest.raises(ValueError, match=f"segment {first} has {pixels} illuminated pixels, fewer than its 10"):
        draw_wavefront(WavefrontBasis(coarse, reference_wavelength_m=5e-7),
                       WavefrontDrawSpec(STATIC, 20.0, SEED, "segment"))

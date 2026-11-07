"""Wavefront coefficients, the nuisance mode grammar, and the mode normalization (optics.wavefront)."""

import math

import numpy as np
import pytest

from hwoslaps.config.checks import ConfigError
from hwoslaps.optics.providers import build_psf_provider, parse_psf
from hwoslaps.optics.pupils import parse_pupil
from hwoslaps.optics.wavefront import (
    WavefrontCoefficients, WavefrontMode, parse_wavefront_selection, select_modes,
)


@pytest.mark.parametrize("mapping, path", [
    ({"zernikes": {True: 1.0}}, "wf.zernikes.True"),
    ({"zernikes": {0: 1.0}}, "wf.zernikes.0"),
    ({"segment_hexikes": {-1: {2: 1.0}}}, "wf.segment_hexikes.-1"),
    ({"zernikes": {4: float("nan")}}, "wf.zernikes.4"),
    ({"zernikes": {4: True}}, "wf.zernikes.4"),
    ({"segment_hexikes": {3: {}}}, "wf.segment_hexikes.3"),
    ({"pistons": {0: 1.0}}, "wf.pistons"),
    ({"segment_hexikes": {"0": {1: 1.0}}}, "wf.segment_hexikes.0"),
    ({"zernikes": {"4": 1.0}}, "wf.zernikes.4"),
    ({"segment_hexikes": {0: {0: 1.0}}}, "wf.segment_hexikes.0.0"),
    ({"segment_hexikes": {0: {2: [1.0]}}}, "wf.segment_hexikes.0.2"),
], ids=["bool-key", "noll-0", "negative-segment", "nan", "bool-value", "empty-segment", "unknown-family",
        "string-segment", "string-noll", "segment-noll-0", "coefficient-list"])
def test_coefficients_parse_rejects_with_the_key_path(mapping, path):
    with pytest.raises(ConfigError) as caught:
        WavefrontCoefficients.from_mapping(mapping, "wf")
    assert caught.value.path == path


def test_coefficients_keep_canonical_order_zeros_and_union_arithmetic():
    shuffled = WavefrontCoefficients.from_mapping(
        {"zernikes": {8: 1.0, 4: -0.0}, "segment_hexikes": {7: {6: 2.0}, 0: {7: 0.0, 1: 3.0}}}, "wf")
    ordered = WavefrontCoefficients.from_mapping(
        {"segment_hexikes": {0: {1: 3.0, 7: 0.0}, 7: {6: 2.0}}, "zernikes": {4: 0.0, 8: 1.0}}, "wf")
    assert shuffled == ordered and shuffled.digest() == ordered.digest()
    assert [mode.name for mode, _ in shuffled.entries] == [
        "segment_hexikes[0][1]", "segment_hexikes[0][7]", "segment_hexikes[7][6]", "zernikes[4]", "zernikes[8]"]
    assert shuffled.hexike_mode_count() == 7 and shuffled.max_zernike_noll() == 8
    assert math.copysign(1.0, shuffled.value(WavefrontMode("zernikes", 4))) == 1.0
    assert shuffled.to_mapping() == {"segment_hexikes": {0: {1: 3.0, 7: 0.0}, 7: {6: 2.0}},
                                     "zernikes": {4: 0.0, 8: 1.0}}

    other = WavefrontCoefficients.from_mapping({"segment_hexikes": {0: {1: 0.5}}, "zernikes": {5: 4.0}}, "wf")
    assert shuffled.plus(other).to_mapping() == {"segment_hexikes": {0: {1: 3.5, 7: 0.0}, 7: {6: 2.0}},
                                                 "zernikes": {4: 0.0, 5: 4.0, 8: 1.0}}
    assert shuffled.minus(other).to_mapping() == {"segment_hexikes": {0: {1: 2.5, 7: 0.0}, 7: {6: 2.0}},
                                                  "zernikes": {4: 0.0, 5: -4.0, 8: 1.0}}
    assert shuffled.scaled(2.0).to_mapping()["segment_hexikes"][0][1] == 6.0
    replaced = shuffled.replace(WavefrontMode("segment_hexikes", 2, 3), 1.5).replace(WavefrontMode("zernikes", 8), -1.0)
    assert [mode.name for mode, _ in replaced.entries][2:4] == ["segment_hexikes[3][2]", "segment_hexikes[7][6]"]
    assert replaced.value(WavefrontMode("zernikes", 8)) == -1.0 and replaced.value(WavefrontMode("zernikes", 6)) == 0.0
    with pytest.raises(ValueError, match="more than once"):
        WavefrontCoefficients(((WavefrontMode("zernikes", 4), 1.0), (WavefrontMode("zernikes", 4), 2.0)))


def test_selection_grammar_and_order(p1_pupil, circular_pupil):
    p1 = parse_pupil(p1_pupil, "pupil")
    path = "forecast.nuisances.wavefront.modes"
    paper = parse_wavefront_selection({"segment_hexikes": {"segments": [3, 0], "nolls": [2, 1]},
                                       "zernikes": {"nolls": [5, 4]}}, path)
    assert [mode.name for mode in select_modes(paper, p1, path)] == [
        "segment_hexikes[0][1]", "segment_hexikes[0][2]", "segment_hexikes[3][1]", "segment_hexikes[3][2]",
        "zernikes[4]", "zernikes[5]"]

    every = parse_wavefront_selection({"segment_hexikes": {"segments": "all", "nolls": [1, 2, 3]}}, path)
    modes = select_modes(every, p1, path)
    assert len(modes) == 19 * 3 and [(m.segment, m.noll) for m in modes[:4]] == [(0, 1), (0, 2), (0, 3), (1, 1)]
    active = select_modes(every, p1, path, active_segments=(1, 2, 3))
    assert sorted({mode.segment for mode in active}) == [1, 2, 3]

    outside = parse_wavefront_selection({"segment_hexikes": {"segments": [19], "nolls": [1]}}, path)
    with pytest.raises(ConfigError, match="segment 19 does not exist"):
        select_modes(outside, p1, path)
    dark = parse_wavefront_selection({"segment_hexikes": {"segments": [0, 1], "nolls": [1]}}, path)
    with pytest.raises(ConfigError, match="segment 0 is dark"):
        select_modes(dark, p1, path, active_segments=(1, 2, 3))
    with pytest.raises(ConfigError, match="circular pupil has no segments"):
        select_modes(paper, parse_pupil(circular_pupil, "pupil"), path)

    for mapping, where in [({"zernikes": {"nolls": [1, 4]}}, f"{path}.zernikes.nolls"),
                           ({}, path),
                           ({"zernikes": {"nolls": [4, 4]}}, f"{path}.zernikes.nolls[1]"),
                           ({"segment_hexikes": {"segments": "every", "nolls": [1]}},
                            f"{path}.segment_hexikes.segments"),
                           ({"segment_hexikes": {"segments": ["zero"], "nolls": [1]}},
                            f"{path}.segment_hexikes.segments[0]")]:
        with pytest.raises(ConfigError) as caught:
            parse_wavefront_selection(mapping, path)
        assert caught.value.path == where


@pytest.mark.backend
def test_aperture_rms_follows_mode_normalization(circular_pupil, paper_pupil):
    def provider(pupil, wavefront, shape, scale, oversampling):
        truth = {"kind": "optical", "pupil": pupil, "focal_length_m": 144.0, "wavelength_nm": 500.0,
                 "detector_oversampling": oversampling, "kernel_shape": [shape, shape], "wavefront": wavefront}
        return build_psf_provider(parse_psf({"truth": truth}).truth, pixel_scale_arcsec=scale)

    circular = provider(circular_pupil, {"zernikes": {7: 30.0}}, 11, 0.05, 1)
    assert circular.basis.aperture_rms_nm(circular.coefficients) == pytest.approx(30.0, rel=2e-3)
    piston = WavefrontCoefficients.from_mapping({"zernikes": {1: 30.0}}, "wavefront")
    assert circular.basis.aperture_rms_nm(piston) == pytest.approx(0.0, abs=1e-9)

    hexike = provider(paper_pupil, {"segment_hexikes": {3: {4: 30.0}}}, 51, 0.00716, 3)
    pupil = hexike.pupil
    in_segment = np.count_nonzero(pupil.illuminated_mask & (np.asarray(pupil.segments[3]) > 0.5))
    expected = 30.0 * math.sqrt(in_segment / np.count_nonzero(pupil.illuminated_mask))
    # HCIPy normalizes a hexike on its point-sampled hexagon and multiplies it by the grey segment
    # mask: on the illuminated segment pixels its RMS is 0.6% below one at 512 pixels (0.8% at 1024).
    assert hexike.basis.aperture_rms_nm(hexike.coefficients) == pytest.approx(expected, rel=1e-2)

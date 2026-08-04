"""Tests for named PSF perturbation family draws.

Unit tests pin the draw and normalization contracts; integration tests check
that family draws pass config validation and produce the requested measured
pupil RMS through the real PSF generator.
"""

from __future__ import annotations

import contextlib
import copy
import io
from pathlib import Path

import numpy as np
import pytest
import yaml

from hwoslaps.psf.families import (
    SEGMENT_PISTON_NOLLS,
    SEGMENT_TIPTILT_NOLLS,
    SPIE_GLOBAL_ZERNIKE_NOLLS,
    SPIE_SEGMENT_HEXIKE_NOLLS,
    draw_global_zernike_family,
    draw_segment_hexike_family,
    draw_segment_piston_family,
    draw_segment_tiptilt_family,
    measure_aperture_rms_nm,
    renormalize_to_aperture_rms,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NUM_SEGMENTS = 19
SEGMENTS = list(range(NUM_SEGMENTS))


def _flat_coefficients(segment_hexikes: dict) -> np.ndarray:
    return np.array(
        [
            segment_hexikes[segment][mode]
            for segment in sorted(segment_hexikes)
            for mode in sorted(segment_hexikes[segment])
        ],
        dtype=float,
    )


def test_piston_family_draw_contract():
    """Draw pistons with the common term removed and exact RMS."""
    target = 25.0
    draw = draw_segment_piston_family(np.random.default_rng(7), SEGMENTS, target)

    assert sorted(draw) == SEGMENTS
    for mode_dict in draw.values():
        assert tuple(mode_dict) == SEGMENT_PISTON_NOLLS

    coeffs = _flat_coefficients(draw)
    # Common piston is removed, and the piston-removed aperture RMS is exact.
    assert float(np.mean(coeffs)) == pytest.approx(0.0, abs=1e-12)
    assert float(np.sqrt(np.mean(coeffs**2))) == pytest.approx(target, rel=1e-12)


def test_tiptilt_family_draw_contract():
    """Draw tip/tilts normalized to target*sqrt(n_segments)."""
    target = 40.0
    draw = draw_segment_tiptilt_family(np.random.default_rng(11), SEGMENTS, target)

    assert sorted(draw) == SEGMENTS
    for mode_dict in draw.values():
        assert tuple(sorted(mode_dict)) == SEGMENT_TIPTILT_NOLLS

    coeffs = _flat_coefficients(draw)
    # Flattened vector is normalized to target*sqrt(n_segments).
    assert float(np.linalg.norm(coeffs)) == pytest.approx(
        target * np.sqrt(NUM_SEGMENTS), rel=1e-12
    )


def test_hexike_family_matches_reference_normalization():
    """Match the hexike draw against the study-ensemble reference."""
    target = 10.0
    seed = 20260527
    draw = draw_segment_hexike_family(
        np.random.default_rng(seed),
        SEGMENTS,
        SPIE_SEGMENT_HEXIKE_NOLLS,
        target,
    )

    # Reference implementation of the study-ensemble semantics: a raw
    # standard-normal matrix whose flattened vector is rescaled to
    # target*sqrt(n_segments).
    raw = np.random.default_rng(seed).standard_normal(
        (NUM_SEGMENTS, len(SPIE_SEGMENT_HEXIKE_NOLLS))
    )
    expected = raw * (target * np.sqrt(NUM_SEGMENTS) / np.linalg.norm(raw))

    for seg_idx, segment in enumerate(SEGMENTS):
        for mode_idx, mode in enumerate(SPIE_SEGMENT_HEXIKE_NOLLS):
            assert draw[segment][mode] == pytest.approx(
                expected[seg_idx, mode_idx], rel=1e-12
            )


def test_global_zernike_family_draw_contract():
    """Draw global Zernikes whose coefficient norm is the target."""
    target = 15.0
    modes = (4, 5, 6, 7, 8, 9, 10, 11)
    draw = draw_global_zernike_family(np.random.default_rng(3), modes, target)

    assert tuple(sorted(draw)) == modes
    coeffs = np.array([draw[mode] for mode in modes])
    assert float(np.linalg.norm(coeffs)) == pytest.approx(target, rel=1e-12)


def test_family_draws_are_deterministic_per_seed():
    """Reproduce a draw for one seed and differ across seeds."""
    draw_a = draw_segment_piston_family(np.random.default_rng(5), SEGMENTS, 20.0)
    draw_b = draw_segment_piston_family(np.random.default_rng(5), SEGMENTS, 20.0)
    draw_c = draw_segment_piston_family(np.random.default_rng(6), SEGMENTS, 20.0)

    np.testing.assert_array_equal(_flat_coefficients(draw_a), _flat_coefficients(draw_b))
    assert not np.array_equal(_flat_coefficients(draw_a), _flat_coefficients(draw_c))


def test_zero_target_returns_empty_draw():
    """Return an empty draw when the target RMS is zero."""
    rng = np.random.default_rng(1)
    assert draw_segment_piston_family(rng, SEGMENTS, 0.0) == {}
    assert draw_segment_tiptilt_family(rng, SEGMENTS, 0.0) == {}
    assert draw_global_zernike_family(rng, (4, 5), 0.0) == {}


@pytest.mark.parametrize(
    "bad_call",
    [
        lambda rng: draw_segment_hexike_family(rng, SEGMENTS, (), 10.0),
        lambda rng: draw_segment_hexike_family(rng, SEGMENTS, (0, 1), 10.0),
        lambda rng: draw_segment_hexike_family(rng, [], (1,), 10.0),
        lambda rng: draw_segment_hexike_family(rng, SEGMENTS, (1,), -1.0),
        lambda rng: draw_segment_hexike_family(rng, SEGMENTS, (1,), float("nan")),
        lambda rng: draw_segment_piston_family(rng, [0], 10.0),
    ],
)
def test_invalid_draw_inputs_raise(bad_call):
    """Reject empty mode lists, bad Noll indices, and bad targets."""
    with pytest.raises(ValueError):
        bad_call(np.random.default_rng(2))


# ---------------------------------------------------------------------------
# Integration against config validation and the real PSF generator.
# ---------------------------------------------------------------------------

pytest.importorskip("hcipy")
pytest.importorskip("autolens")

from hwoslaps.config.validation import validate_or_raise  # noqa: E402
from hwoslaps.psf.generator import generate_psf_system  # noqa: E402


@pytest.fixture()
def compact_config() -> dict:
    """Load master_config.yaml shrunk to a fast, unaberrated PSF."""
    with (PROJECT_ROOT / "configs" / "master_config.yaml").open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg = copy.deepcopy(cfg)
    cfg["plotting"]["enabled"] = False
    cfg["psf"]["hres_psf"]["num_pix"] = 128
    cfg["psf"]["hres_psf"]["num_airy"] = 6
    cfg["psf"]["hres_psf"]["sampling"] = 5
    cfg["psf"]["hres_psf"]["save_highres_psf_npy"] = False
    cfg["psf"]["kernel"]["shape_native"] = [15, 15]

    aberr = cfg["psf"]["aberrations"]
    aberr["enable_segment_pistons"] = False
    aberr["enable_segment_tiptilts"] = False
    aberr["enable_segment_hexikes"] = False
    aberr["enable_global_zernikes"] = False
    aberr["segment_pistons"] = {}
    aberr["segment_tiptilts"] = {}
    aberr["segment_hexikes"] = {}
    aberr["global_zernikes"] = {}
    return cfg


def _quiet_generate(config: dict):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return generate_psf_system(config["psf"], full_config=config)


def _config_with_family_draw(compact_config: dict, segment_hexikes: dict) -> dict:
    cfg = copy.deepcopy(compact_config)
    aberr = cfg["psf"]["aberrations"]
    aberr["enable_segment_hexikes"] = True
    aberr["segment_hexikes"] = segment_hexikes
    return cfg


def test_family_draws_pass_config_validation(compact_config):
    """Accept piston and tip/tilt family draws as valid config."""
    rng = np.random.default_rng(13)
    cfg = _config_with_family_draw(
        compact_config, draw_segment_piston_family(rng, SEGMENTS, 20.0)
    )
    validate_or_raise(cfg)

    cfg = _config_with_family_draw(
        compact_config, draw_segment_tiptilt_family(rng, SEGMENTS, 20.0)
    )
    validate_or_raise(cfg)


@pytest.mark.parametrize(
    "draw_family",
    [draw_segment_piston_family, draw_segment_tiptilt_family],
    ids=["segment_piston", "segment_tiptilt"],
)
def test_family_draw_reproduces_target_pupil_rms(compact_config, draw_family):
    """Reproduce the target pupil RMS through the real generator."""
    target_rms_nm = 30.0
    draw = draw_family(np.random.default_rng(17), SEGMENTS, target_rms_nm)
    cfg = _config_with_family_draw(compact_config, draw)

    psf_data = _quiet_generate(cfg)

    # The measured piston-removed pupil RMS should match the draw target to
    # within pixelated-hexagon discretization error.
    assert psf_data.total_rms_nm == pytest.approx(target_rms_nm, rel=0.05)


# ---------------------------------------------------------------------------
# Exact aperture-RMS renormalization.
# ---------------------------------------------------------------------------

from hwoslaps.psf.telescope_models import create_hcipy_telescope  # noqa: E402


def _config_with_global_draw(compact_config: dict, global_zernikes: dict) -> dict:
    cfg = copy.deepcopy(compact_config)
    aberr = cfg["psf"]["aberrations"]
    aberr["enable_global_zernikes"] = True
    aberr["global_zernikes"] = global_zernikes
    return cfg


def test_renormalized_global_draw_hits_target_through_generator(compact_config):
    """Hit the target aperture RMS after renormalizing a global draw."""
    target_rms_nm = 25.0
    raw = draw_global_zernike_family(np.random.default_rng(23),
                                     SPIE_GLOBAL_ZERNIKE_NOLLS, target_rms_nm)
    telescope_data = create_hcipy_telescope(compact_config["psf"])
    _, scaled = renormalize_to_aperture_rms(telescope_data, target_rms_nm,
                                            global_zernikes=raw)

    # The raw coefficient-space draw under-realizes the target on the
    # segmented aperture; renormalization must land exactly on it.
    raw_measured = measure_aperture_rms_nm(telescope_data, global_zernikes=raw)
    assert raw_measured < 0.98 * target_rms_nm

    psf_data = _quiet_generate(_config_with_global_draw(compact_config, scaled))
    assert psf_data.total_rms_nm == pytest.approx(target_rms_nm, rel=1e-5)


def test_renormalized_combined_draw_hits_target_through_generator(compact_config):
    """Hit the target RMS for a combined segment plus global draw."""
    target_rms_nm = 25.0
    split = target_rms_nm / np.sqrt(2.0)
    rng = np.random.default_rng(29)
    seg_raw = draw_segment_hexike_family(rng, SEGMENTS,
                                         SPIE_SEGMENT_HEXIKE_NOLLS, split)
    glob_raw = draw_global_zernike_family(rng, SPIE_GLOBAL_ZERNIKE_NOLLS, split)
    telescope_data = create_hcipy_telescope(compact_config["psf"])
    seg, glob = renormalize_to_aperture_rms(telescope_data, target_rms_nm,
                                            segment_hexikes=seg_raw,
                                            global_zernikes=glob_raw)

    cfg = _config_with_family_draw(compact_config, seg)
    cfg = _config_with_global_draw(cfg, glob)
    psf_data = _quiet_generate(cfg)
    assert psf_data.total_rms_nm == pytest.approx(target_rms_nm, rel=1e-5)


def test_renormalize_zero_target_returns_empty():
    """Return empty coefficient dicts for a zero target RMS."""
    seg, glob = renormalize_to_aperture_rms(None, 0.0,
                                            segment_hexikes={0: {1: 5.0}},
                                            global_zernikes={4: 5.0})
    assert seg == {}
    assert glob == {}


def test_renormalize_rejects_zero_measured_and_bad_targets(compact_config):
    """Reject renormalization with no coefficients or a bad target."""
    telescope_data = create_hcipy_telescope(compact_config["psf"])
    with pytest.raises(ValueError):
        renormalize_to_aperture_rms(telescope_data, 10.0)
    with pytest.raises(ValueError):
        renormalize_to_aperture_rms(telescope_data, -1.0,
                                    global_zernikes={4: 5.0})
    with pytest.raises(ValueError):
        renormalize_to_aperture_rms(telescope_data, float("nan"),
                                    global_zernikes={4: 5.0})

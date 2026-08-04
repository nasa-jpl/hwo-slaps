"""Tests for shape-weighted PSF perturbation family draws."""

from __future__ import annotations

import contextlib
import copy
import io
from pathlib import Path

import numpy as np
import pytest
import yaml

from hwoslaps.psf.families import (
    ModeWeightPrior,
    draw_global_zernike_family,
    draw_segment_hexike_family,
    draw_segment_piston_family,
    draw_weighted_combined_family,
    draw_weighted_global_zernike_family,
    draw_weighted_segment_hexike_family,
    load_mode_weight_prior,
    make_power_law_prior,
    noll_to_radial_order,
    renormalize_to_aperture_rms,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEGMENTS = list(range(19))


def _flat_segment_coefficients(segment_hexikes):
    """Flatten a nested segment coefficient dictionary in sorted order."""
    return np.array([
        segment_hexikes[segment][mode]
        for segment in sorted(segment_hexikes)
        for mode in sorted(segment_hexikes[segment])
    ])


@pytest.mark.parametrize(
    'noll, expected',
    [
        (1, 0),
        (2, 1),
        (3, 1),
        (4, 2),
        (6, 2),
        (7, 3),
        (10, 3),
        (11, 4),
        (15, 4),
        (16, 5),
        (21, 5),
        (22, 6),
        (28, 6),
        (29, 7),
        (36, 7),
        (37, 8),
        (45, 8),
        (46, 9),
        (55, 9),
    ],
)
def test_noll_to_radial_order_required_values(noll, expected):
    """Map representative complete-order boundaries correctly."""
    assert noll_to_radial_order(noll) == expected


@pytest.mark.parametrize('noll', [0, -1, True, 1.0])
def test_noll_to_radial_order_rejects_invalid_inputs(noll):
    """Reject nonpositive, Boolean, and noninteger Noll inputs."""
    with pytest.raises(ValueError, match='noll'):
        noll_to_radial_order(noll)


@pytest.mark.parametrize('alpha', [0.0, 1.0, 2.0])
def test_power_law_prior_ratios_and_normalization(alpha):
    """Preserve radial power-law ratios while normalizing each side."""
    prior = make_power_law_prior(
        alpha,
        global_mode_range=(4, 11),
        segment_mode_range=(1, 6),
    )
    assert np.linalg.norm(list(prior.global_weights.values())) == pytest.approx(1.0)
    assert np.linalg.norm(list(prior.segment_weights.values())) == pytest.approx(1.0)

    assert prior.global_weights[4] / prior.global_weights[11] == pytest.approx(
        (2 / 4)**(-alpha)
    )
    assert prior.segment_weights[1] / prior.segment_weights[4] == pytest.approx(
        (1 / 3)**(-alpha)
    )
    if alpha == 0.0:
        assert len(set(prior.global_weights.values())) == 1
        assert len(set(prior.segment_weights.values())) == 1


@pytest.mark.parametrize(
    'keywords',
    [
        {'global_mode_range': (1, 55)},
        {'global_mode_range': (6, 4)},
        {'segment_mode_range': (3, 2)},
        {'global_mode_range': None, 'segment_mode_range': None},
        {'segment_variance_fraction': -0.1},
        {'segment_variance_fraction': 1.1},
    ],
)
def test_power_law_prior_rejects_invalid_ranges_and_fraction(keywords):
    """Reject invalid ranges, omitted sides, and variance fractions."""
    with pytest.raises(ValueError):
        make_power_law_prior(1.0, **keywords)


@pytest.mark.parametrize('alpha', [-1.0, np.inf, -np.inf, np.nan])
def test_power_law_prior_rejects_invalid_alpha(alpha):
    """Reject negative and nonfinite power-law indices."""
    with pytest.raises(ValueError, match='alpha'):
        make_power_law_prior(alpha)


def test_mode_weight_prior_loads_and_normalizes_yaml(tmp_path):
    """Load a valid table with normalized weights and preserved metadata."""
    path = tmp_path / 'prior.yaml'
    document = {
        'name': 'flight_prior',
        'segment_variance_fraction': 0.4,
        'global_weights': {4: 3.0, 5: 4.0},
        'segment_weights': {1: 5.0, 2: 12.0},
        'metadata': {'source': 'offline-test'},
    }
    path.write_text(yaml.safe_dump(document), encoding='utf-8')

    prior = load_mode_weight_prior(path)

    assert prior.name == 'flight_prior'
    assert prior.segment_variance_fraction == pytest.approx(0.4)
    assert prior.global_weights == pytest.approx({4: 0.6, 5: 0.8})
    assert prior.segment_weights == pytest.approx({1: 5 / 13, 2: 12 / 13})
    assert prior.metadata == {'source': 'offline-test'}


def test_mode_weight_prior_loading_is_idempotent(tmp_path):
    """Reload already normalized weights without changing their values."""
    first_path = tmp_path / 'first.yaml'
    first_path.write_text(
        yaml.safe_dump({
            'name': 'idempotent',
            'segment_variance_fraction': 0.5,
            'global_weights': {4: 2.0, 5: 7.0},
        }),
        encoding='utf-8',
    )
    first = load_mode_weight_prior(first_path)
    second_path = tmp_path / 'second.yaml'
    second_path.write_text(
        yaml.safe_dump({
            'name': first.name,
            'segment_variance_fraction': first.segment_variance_fraction,
            'global_weights': first.global_weights,
            'metadata': first.metadata,
        }),
        encoding='utf-8',
    )

    second = load_mode_weight_prior(second_path)

    assert second.global_weights == pytest.approx(first.global_weights, rel=1e-15)


@pytest.mark.parametrize(
    'document, offending_field',
    [
        ({
            'name': 'bad',
            'segment_variance_fraction': 0.5,
            'global_weights': {4: 1.0},
            'unknown': 1,
        }, 'unknown'),
        ({
            'segment_variance_fraction': 0.5,
            'global_weights': {4: 1.0},
        }, 'name'),
        ({'name': 'bad', 'segment_variance_fraction': 0.5}, 'global_weights'),
        ({
            'name': 'bad',
            'segment_variance_fraction': 0.5,
            'global_weights': {4: -1.0},
        }, 'global_weights'),
        ({
            'name': 'bad',
            'segment_variance_fraction': 0.5,
            'segment_weights': {1: 0.0, 2: 0.0},
        }, 'segment_weights'),
        ({
            'name': 'bad',
            'segment_variance_fraction': 0.5,
            'global_weights': {'4': 1.0},
        }, 'global_weights'),
        ({
            'name': 'bad',
            'segment_variance_fraction': 0.5,
            'segment_weights': {True: 1.0},
        }, 'segment_weights'),
        ({
            'name': 'bad',
            'segment_variance_fraction': 0.5,
            'global_weights': {3: 1.0},
        }, 'global_weights'),
        ({
            'name': 'bad',
            'segment_variance_fraction': 1.1,
            'global_weights': {4: 1.0},
        }, 'segment_variance_fraction'),
    ],
)
def test_mode_weight_prior_loader_rejects_malformed_tables(
    tmp_path, document, offending_field
):
    """Reject malformed YAML tables with the offending field named."""
    path = tmp_path / 'bad.yaml'
    path.write_text(yaml.safe_dump(document), encoding='utf-8')
    with pytest.raises(ValueError, match=offending_field):
        load_mode_weight_prior(path)


def test_weighted_global_draw_contract():
    """Return an ordered global draw with the exact coefficient norm."""
    prior = ModeWeightPrior('global', {8: 2.0, 4: 1.0, 6: 3.0}, {}, 0.0)
    draw = draw_weighted_global_zernike_family(
        np.random.default_rng(1), prior, 17.0
    )

    assert tuple(draw) == (4, 6, 8)
    assert np.linalg.norm(list(draw.values())) == pytest.approx(17.0, rel=1e-12)
    assert draw_weighted_global_zernike_family(
        np.random.default_rng(1), prior, 0.0
    ) == {}


def test_weighted_global_draw_rejects_empty_side():
    """Reject a nonzero global draw from a segment-only prior."""
    prior = ModeWeightPrior('segment-only', {}, {1: 1.0}, 1.0)
    with pytest.raises(ValueError, match='global_weights'):
        draw_weighted_global_zernike_family(
            np.random.default_rng(1), prior, 10.0
        )


def test_weighted_global_draw_sample_variances_follow_weights():
    """Recover squared-weight variance fractions over many fixed-seed draws."""
    prior = ModeWeightPrior('three-mode', {4: 1.0, 5: 0.95, 6: 0.9}, {}, 0.0)
    rng = np.random.default_rng(934)
    draws = np.array([
        list(draw_weighted_global_zernike_family(rng, prior, 1.0).values())
        for _ in range(6000)
    ])
    observed = np.var(draws, axis=0)
    expected = np.array(list(prior.global_weights.values()))**2
    np.testing.assert_allclose(observed, expected, rtol=0.15, atol=0.0)


def test_flat_weighted_global_draw_matches_existing_family():
    """Reproduce the legacy global family for flat weights and one seed."""
    prior = make_power_law_prior(
        0.0, global_mode_range=(4, 11), segment_mode_range=None
    )
    weighted = draw_weighted_global_zernike_family(
        np.random.default_rng(44), prior, 20.0
    )
    existing = draw_global_zernike_family(
        np.random.default_rng(44), range(4, 12), 20.0
    )

    assert weighted == pytest.approx(existing, rel=1e-12)


def test_weighted_segment_draw_norm_and_piston_handling():
    """Normalize segments while removing only the common piston column."""
    prior = ModeWeightPrior('segment', {}, {1: 1.0, 2: 0.7, 3: 0.4}, 1.0)
    target = 13.0
    draw = draw_weighted_segment_hexike_family(
        np.random.default_rng(19), SEGMENTS, prior, target
    )
    matrix = np.array([
        [draw[segment][mode] for mode in (1, 2, 3)]
        for segment in SEGMENTS
    ])

    assert np.linalg.norm(matrix) == pytest.approx(
        target * np.sqrt(len(SEGMENTS)), rel=1e-12
    )
    assert np.mean(matrix[:, 0]) == pytest.approx(0.0, abs=1e-12)
    assert abs(np.mean(matrix[:, 1])) > 1e-3
    assert abs(np.mean(matrix[:, 2])) > 1e-3
    assert draw_weighted_segment_hexike_family(
        np.random.default_rng(19), SEGMENTS, prior, 0.0
    ) == {}


def test_weighted_segment_draw_rejects_invalid_geometry_and_side():
    """Reject empty segments, empty segment weights, and one-segment piston."""
    segment_prior = ModeWeightPrior('segment', {}, {1: 1.0}, 1.0)
    global_prior = ModeWeightPrior('global', {4: 1.0}, {}, 0.0)
    with pytest.raises(ValueError, match='segments'):
        draw_weighted_segment_hexike_family(
            np.random.default_rng(1), [], segment_prior, 10.0
        )
    with pytest.raises(ValueError, match='segment_weights'):
        draw_weighted_segment_hexike_family(
            np.random.default_rng(1), SEGMENTS, global_prior, 10.0
        )
    with pytest.raises(ValueError, match='at least two'):
        draw_weighted_segment_hexike_family(
            np.random.default_rng(1), [0], segment_prior, 10.0
        )


def test_flat_weighted_segment_hexikes_match_existing_family():
    """Reproduce the legacy mode-2--6 segment family for flat weights."""
    prior = make_power_law_prior(
        0.0, global_mode_range=None, segment_mode_range=(2, 6)
    )
    weighted = draw_weighted_segment_hexike_family(
        np.random.default_rng(72), SEGMENTS, prior, 20.0
    )
    existing = draw_segment_hexike_family(
        np.random.default_rng(72), SEGMENTS, range(2, 7), 20.0
    )

    np.testing.assert_allclose(
        _flat_segment_coefficients(weighted),
        _flat_segment_coefficients(existing),
        rtol=1e-12,
        atol=0.0,
    )


def test_flat_weighted_segment_pistons_match_existing_family():
    """Reproduce the legacy piston family for a flat mode-1-only prior."""
    prior = make_power_law_prior(
        0.0, global_mode_range=None, segment_mode_range=(1, 1)
    )
    weighted = draw_weighted_segment_hexike_family(
        np.random.default_rng(73), SEGMENTS, prior, 20.0
    )
    existing = draw_segment_piston_family(
        np.random.default_rng(73), SEGMENTS, 20.0
    )

    np.testing.assert_allclose(
        _flat_segment_coefficients(weighted),
        _flat_segment_coefficients(existing),
        rtol=1e-12,
        atol=0.0,
    )


@pytest.mark.parametrize('fraction', [0.5, 0.2])
def test_weighted_combined_draw_splits_coefficient_variance(fraction):
    """Split combined coefficient norms according to the prior fraction."""
    prior = ModeWeightPrior(
        'combined', {4: 1.0, 5: 0.5}, {1: 1.0, 2: 0.5}, fraction
    )
    target = 30.0
    segment, global_modes = draw_weighted_combined_family(
        np.random.default_rng(81), SEGMENTS, prior, target
    )

    assert np.linalg.norm(_flat_segment_coefficients(segment)) == pytest.approx(
        target * np.sqrt(fraction) * np.sqrt(len(SEGMENTS)), rel=1e-12
    )
    assert np.linalg.norm(list(global_modes.values())) == pytest.approx(
        target * np.sqrt(1.0 - fraction), rel=1e-12
    )


@pytest.mark.parametrize('fraction', [0.0, 1.0])
def test_weighted_combined_draw_skips_zero_budget_rng(fraction):
    """Skip a zero-budget side without consuming random numbers."""
    prior = ModeWeightPrior(
        'edge-split', {4: 1.0, 5: 0.5}, {1: 1.0, 2: 0.5}, fraction
    )
    actual_rng = np.random.default_rng(82)
    reference_rng = np.random.default_rng(82)

    segment, global_modes = draw_weighted_combined_family(
        actual_rng, SEGMENTS, prior, 30.0
    )
    if fraction == 0.0:
        reference = draw_weighted_global_zernike_family(
            reference_rng, prior, 30.0
        )
        assert segment == {}
        assert global_modes == pytest.approx(reference)
    else:
        reference = draw_weighted_segment_hexike_family(
            reference_rng, SEGMENTS, prior, 30.0
        )
        np.testing.assert_allclose(
            _flat_segment_coefficients(segment),
            _flat_segment_coefficients(reference),
            rtol=1e-15,
        )
        assert global_modes == {}
    assert actual_rng.bit_generator.state == reference_rng.bit_generator.state


def test_weighted_combined_draw_order_and_zero_target_rng():
    """Draw segments first and leave RNG untouched for a zero total target."""
    prior = ModeWeightPrior(
        'combined', {4: 1.0, 5: 0.5}, {1: 1.0, 2: 0.5}, 0.5
    )
    actual_rng = np.random.default_rng(83)
    reference_rng = np.random.default_rng(83)
    actual = draw_weighted_combined_family(
        actual_rng, SEGMENTS, prior, 30.0
    )
    segment_reference = draw_weighted_segment_hexike_family(
        reference_rng, SEGMENTS, prior, 30.0 / np.sqrt(2.0)
    )
    global_reference = draw_weighted_global_zernike_family(
        reference_rng, prior, 30.0 / np.sqrt(2.0)
    )

    np.testing.assert_allclose(
        _flat_segment_coefficients(actual[0]),
        _flat_segment_coefficients(segment_reference),
        rtol=1e-15,
    )
    assert actual[1] == pytest.approx(global_reference)
    assert actual_rng.bit_generator.state == reference_rng.bit_generator.state

    zero_rng = np.random.default_rng(84)
    untouched_state = copy.deepcopy(zero_rng.bit_generator.state)
    assert draw_weighted_combined_family(
        zero_rng, SEGMENTS, prior, 0.0
    ) == ({}, {})
    assert zero_rng.bit_generator.state == untouched_state


# ---------------------------------------------------------------------------
# Integration through the real HCIPy PSF generator.
# ---------------------------------------------------------------------------

pytest.importorskip('hcipy')
pytest.importorskip('autolens')

from hwoslaps.psf.generator import generate_psf_system  # noqa: E402
from hwoslaps.psf.telescope_models import create_hcipy_telescope  # noqa: E402


@pytest.fixture()
def compact_config():
    """Load the master config reduced to a fast unaberrated PSF."""
    with (PROJECT_ROOT / 'configs' / 'master_config.yaml').open(
        'r', encoding='utf-8'
    ) as stream:
        config = yaml.safe_load(stream)
    config = copy.deepcopy(config)
    config['plotting']['enabled'] = False
    config['psf']['hres_psf']['num_pix'] = 128
    config['psf']['hres_psf']['num_airy'] = 6
    config['psf']['hres_psf']['sampling'] = 5
    config['psf']['hres_psf']['save_highres_psf_npy'] = False
    config['psf']['kernel']['shape_native'] = [15, 15]
    aberrations = config['psf']['aberrations']
    for family in (
        'segment_pistons',
        'segment_tiptilts',
        'segment_hexikes',
        'global_zernikes',
    ):
        aberrations[f'enable_{family}'] = False
        aberrations[family] = {}
    return config


def _generate_with_draw(compact_config, segment_hexikes, global_zernikes):
    """Generate a PSF quietly with both supplied aberration dictionaries."""
    config = copy.deepcopy(compact_config)
    aberrations = config['psf']['aberrations']
    if segment_hexikes:
        aberrations['enable_segment_hexikes'] = True
        aberrations['segment_hexikes'] = segment_hexikes
    if global_zernikes:
        aberrations['enable_global_zernikes'] = True
        aberrations['global_zernikes'] = global_zernikes
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        return generate_psf_system(config['psf'], full_config=config)


@pytest.mark.parametrize('family', ['global', 'segment', 'combined'])
def test_weighted_draw_renormalizes_to_exact_aperture_rms(
    compact_config, family
):
    """Realize full-range weighted draws at exact physical aperture RMS."""
    prior = make_power_law_prior(1.0)
    target = 25.0
    rng = np.random.default_rng(91)
    if family == 'global':
        segment_raw = {}
        global_raw = draw_weighted_global_zernike_family(rng, prior, target)
    elif family == 'segment':
        segment_raw = draw_weighted_segment_hexike_family(
            rng, SEGMENTS, prior, target
        )
        global_raw = {}
    else:
        segment_raw, global_raw = draw_weighted_combined_family(
            rng, SEGMENTS, prior, target
        )
    telescope_data = create_hcipy_telescope(compact_config['psf'])
    segment, global_modes = renormalize_to_aperture_rms(
        telescope_data,
        target,
        segment_hexikes=segment_raw,
        global_zernikes=global_raw,
    )

    psf_data = _generate_with_draw(compact_config, segment, global_modes)

    assert psf_data.total_rms_nm == pytest.approx(target, rel=1e-5)

"""Tests for JWST-prior aperture-basis consistency."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml

from hwoslaps.psf.aberration_models import (
    _make_global_zernike_basis,
    _make_segmented_hexike_surface,
    apply_global_zernikes,
)
from hwoslaps.psf.families import (
    draw_weighted_global_zernike_family,
    draw_weighted_segment_hexike_family,
    load_mode_weight_prior,
    renormalize_to_aperture_rms,
    realize_weighted_draw,
)
from hwoslaps.psf.opd_basis import (
    ApertureBasisTransform,
    build_raw_noll_basis,
    fit_orthonormal_basis,
    noll_to_zernike,
    orthonormalize_basis,
)
from hwoslaps.psf.telescope_models import create_hcipy_telescope


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / 'scripts' / 'derive_jwst_mode_weight_tables.py'
DRIFT_PATH = (
    PROJECT_ROOT / 'configs' / 'psf_priors' / 'jwst_wss_drift_v1.yaml'
)


def _load_derivation_script():
    """Load the offline derivation module without invoking its CLI."""
    spec = importlib.util.spec_from_file_location(
        'derive_jwst_weights_basis_test', SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _orthonormal_values(raw_values):
    """Return the sign-fixed unit-mean-square QR basis for raw columns."""
    q_matrix, r_matrix = np.linalg.qr(raw_values, mode='reduced')
    signs = np.where(np.diag(r_matrix) < 0.0, -1.0, 1.0)
    return np.sqrt(raw_values.shape[0]) * q_matrix * signs[np.newaxis, :]


def _compact_eac1_config():
    """Return the real EAC1 geometry at a fast pupil sampling."""
    with (PROJECT_ROOT / 'configs' / 'master_config.yaml').open(
        'r', encoding='utf-8'
    ) as stream:
        config = copy.deepcopy(yaml.safe_load(stream)['psf'])
    config['hres_psf']['num_pix'] = 128
    return config


@pytest.fixture(scope='module')
def eac1_basis_data():
    """Build one EAC1 telescope and one cached full-range transform."""
    telescope_data = create_hcipy_telescope(_compact_eac1_config())
    prior = load_mode_weight_prior(DRIFT_PATH)
    global_modes = tuple(prior.global_weights)
    segment_modes = tuple(prior.segment_weights)
    transform = ApertureBasisTransform(
        telescope_data, global_modes, segment_modes
    )
    aperture_mask = np.asarray(telescope_data['aper']) > 0.5
    global_basis = _make_global_zernike_basis(
        telescope_data, max(global_modes)
    )
    global_raw = np.asarray([
        np.asarray(global_basis[mode - 1]) for mode in global_modes
    ])
    global_orthonormal = orthonormalize_basis(global_raw, aperture_mask)
    return {
        'telescope': telescope_data,
        'prior': prior,
        'transform': transform,
        'aperture_mask': aperture_mask,
        'global_raw': global_raw,
        'global_orthonormal': global_orthonormal,
    }


def test_promoted_functions_preserve_derivation_round_trip():
    """Keep the promoted pure functions identical to offline behavior."""
    derivation = _load_derivation_script()
    rows, columns = np.indices((9, 11), dtype=float)
    mask = ((rows - 4.0) / 3.5)**2 + ((columns - 5.0) / 4.5)**2 <= 1.0
    modes = (4, 5, 6, 7)
    raw = build_raw_noll_basis(columns, rows, mask, modes)
    basis = orthonormalize_basis(raw, mask)
    expected = np.array([2.0, -1.5, 0.25, 3.0])
    opd = np.tensordot(expected, basis, axes=1)

    coefficients, model = fit_orthonormal_basis(opd, mask, basis)
    script_coefficients, script_model = derivation.fit_orthonormal_basis(
        opd, mask, derivation.orthonormalize_basis(raw, mask)
    )

    np.testing.assert_allclose(coefficients, expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(model, opd, rtol=1e-13, atol=1e-13)
    np.testing.assert_array_equal(script_coefficients, coefficients)
    np.testing.assert_array_equal(script_model, model)
    assert derivation._noll_to_zernike(11) == noll_to_zernike(11)


def test_identity_limit_for_already_orthonormal_basis():
    """Return identical coefficients when sampled raw modes are orthonormal."""
    rng = np.random.default_rng(101)
    q_matrix, _ = np.linalg.qr(rng.standard_normal((80, 5)), mode='reduced')
    values = np.sqrt(80.0) * q_matrix
    modes = (1, 2, 3, 4, 5)
    transform = ApertureBasisTransform.from_sampled_values(
        global_mode_nolls=modes, global_values=values
    )
    coefficients = {
        mode: float(value)
        for mode, value in zip(modes, rng.standard_normal(len(modes)))
    }

    assert transform.global_to_raw(coefficients) == pytest.approx(
        coefficients, rel=1e-13, abs=1e-13
    )


def test_eac1_transform_algebra_global_and_segment(eac1_basis_data):
    """Realize V c_raw = B c_ortho on both real EAC1 basis blocks."""
    data = eac1_basis_data
    transform = data['transform']
    rng = np.random.default_rng(102)
    global_ortho = {
        mode: float(value)
        for mode, value in zip(
            transform.global_mode_nolls,
            rng.standard_normal(len(transform.global_mode_nolls)),
        )
    }
    global_raw = transform.global_to_raw(global_ortho)
    raw_values = data['global_raw'][:, data['aperture_mask']].T
    ortho_values = data['global_orthonormal'][:, data['aperture_mask']].T
    np.testing.assert_allclose(
        raw_values @ np.array(list(global_raw.values())),
        ortho_values @ np.array(list(global_ortho.values())),
        rtol=1e-10,
        atol=1e-11,
    )

    telescope_data = data['telescope']
    segment_modes = transform.segment_mode_nolls
    surface = _make_segmented_hexike_surface(
        telescope_data, max(segment_modes)
    )
    recovered_pistons = []
    segment_ortho = draw_weighted_segment_hexike_family(
        rng,
        range(len(telescope_data['segments'])),
        data['prior'],
        1.0,
    )
    segment_raw = transform.segment_to_raw(segment_ortho)

    aperture_mask = data['aperture_mask']
    for segment, segment_field in enumerate(telescope_data['segments']):
        segment_mask = aperture_mask & (np.asarray(segment_field) > 0.5)
        columns = []
        for mode in segment_modes:
            surface.flatten()
            surface.set_segment_coefficients(
                segment, {mode: 0.5}, indexing='noll'
            )
            columns.append(np.asarray(surface.opd)[segment_mask])
        raw_segment_values = np.column_stack(columns)
        ortho_segment_values = _orthonormal_values(raw_segment_values)
        raw_coefficients = np.array([
            segment_raw[segment][mode] for mode in segment_modes
        ])
        ortho_coefficients = np.array([
            segment_ortho[segment][mode] for mode in segment_modes
        ])
        realized = raw_segment_values @ raw_coefficients
        expected = ortho_segment_values @ ortho_coefficients
        np.testing.assert_allclose(
            realized, expected, rtol=1e-10, atol=1e-11
        )
        recovered = np.mean(
            ortho_segment_values * realized[:, np.newaxis], axis=0
        )
        recovered_pistons.append(recovered[0])
    assert np.mean(recovered_pistons) == pytest.approx(0.0, abs=1e-12)


def _predicted_conditioned_moments(weights, basis_means, sample_count, seed):
    """Predict exact-normalization moments by vectorized Monte Carlo."""
    rng = np.random.default_rng(seed)
    weighted = rng.standard_normal((sample_count, len(weights))) * weights
    coefficients = weighted / np.linalg.norm(weighted, axis=1)[:, np.newaxis]
    physical_rms = np.sqrt(
        1.0 - (coefficients @ basis_means)**2
    )
    squared = (coefficients / physical_rms[:, np.newaxis])**2
    return np.mean(squared, axis=0), np.var(squared, axis=0, ddof=1)


@pytest.fixture(scope='module')
def global_round_trip_measurement(eac1_basis_data):
    """Measure fixed and pre-fix weighted global paths on EAC1."""
    data = eac1_basis_data
    telescope_data = data['telescope']
    prior = data['prior']
    transform = data['transform']
    basis = data['global_orthonormal']
    aperture_mask = data['aperture_mask']
    rng = np.random.default_rng(20260806)
    fixed = []
    old = []
    draw_count = 256
    for _ in range(draw_count):
        orthonormal_draw = draw_weighted_global_zernike_family(
            rng, prior, 1.0
        )
        _, fixed_raw = realize_weighted_draw(
            telescope_data,
            transform,
            1.0,
            global_coefficients=orthonormal_draw,
        )
        _, old_raw = renormalize_to_aperture_rms(
            telescope_data,
            1.0,
            global_zernikes=orthonormal_draw,
        )
        for destination, raw_coefficients in (
            (fixed, fixed_raw),
            (old, old_raw),
        ):
            phase = apply_global_zernikes(
                raw_coefficients, telescope_data, telescope_data['wavelength']
            )
            opd_nm = (
                np.asarray(phase)
                * telescope_data['wavelength']
                / (2.0 * np.pi)
                * 1.0e9
            )
            coefficients, _ = fit_orthonormal_basis(
                opd_nm, aperture_mask, basis
            )
            destination.append(coefficients)
    return {
        'draw_count': draw_count,
        'fixed': np.asarray(fixed),
        'old': np.asarray(old),
        'weights': np.array(list(prior.global_weights.values())),
        'basis_means': np.mean(basis[:, aperture_mask], axis=1),
    }


def _assert_round_trip_acceptance(measurement, recovered):
    """Apply the decisive production-table shape acceptance criteria."""
    weights = measurement['weights']
    observed_squared = recovered**2
    observed_moments = np.mean(observed_squared, axis=0)
    predicted, predicted_variance = _predicted_conditioned_moments(
        weights,
        measurement['basis_means'],
        200_000,
        106,
    )
    standard_error = np.sqrt(
        predicted_variance / measurement['draw_count']
    )
    np.testing.assert_array_less(
        np.abs(observed_moments - predicted),
        3.0 * standard_error,
    )
    observed_shape = np.sqrt(observed_moments)
    cosine = np.dot(observed_shape, weights) / (
        np.linalg.norm(observed_shape) * np.linalg.norm(weights)
    )
    assert cosine >= 0.995


def test_weighted_global_round_trip_acceptance(global_round_trip_measurement):
    """Recover the drift-table shape through the full runtime path."""
    _assert_round_trip_acceptance(
        global_round_trip_measurement,
        global_round_trip_measurement['fixed'],
    )


def test_round_trip_acceptance_rejects_pre_fix_behavior(
    global_round_trip_measurement,
):
    """Prove the decisive acceptance rejects direct raw-Noll application."""
    with pytest.raises(AssertionError):
        _assert_round_trip_acceptance(
            global_round_trip_measurement,
            global_round_trip_measurement['old'],
        )


def test_drift_prior_exact_normalization_moments():
    """Pin the documented F3 conditioning effect for the drift table."""
    prior = load_mode_weight_prior(DRIFT_PATH)
    weights = np.array(list(prior.global_weights.values()))
    first, first_variance = _predicted_conditioned_moments(
        weights, np.zeros_like(weights), 200_000, 107
    )
    replicate, replicate_variance = _predicted_conditioned_moments(
        weights, np.zeros_like(weights), 200_000, 108
    )
    standard_error = np.sqrt(
        (first_variance + replicate_variance) / 200_000
    )
    np.testing.assert_array_less(
        np.abs(first - replicate), 3.0 * standard_error
    )
    realized_fractions = first / np.sum(first)
    naive_fractions = weights**2 / np.sum(weights**2)
    difference = np.sum(np.abs(realized_fractions - naive_fractions))
    assert 0.05 <= difference <= 0.10

"""Offline tests for the JWST WSS mode-weight derivation numerics."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from hwoslaps.psf.families import load_mode_weight_prior


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / 'scripts' / 'derive_jwst_mode_weight_tables.py'
SPEC = importlib.util.spec_from_file_location('derive_jwst_weights', SCRIPT_PATH)
DERIVATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DERIVATION)


def _synthetic_segmented_problem():
    """Construct a three-segment aperture with exactly recoverable content."""
    aperture = np.zeros((8, 14), dtype=bool)
    segment_masks = []
    for column_start in (1, 5, 9):
        mask = np.zeros_like(aperture)
        mask[2:6, column_start:column_start + 3] = True
        aperture |= mask
        segment_masks.append(mask)
    segment_masks = np.asarray(segment_masks)

    global_raw = np.zeros((1,) + aperture.shape)
    global_raw[0, aperture] = 1.0
    segment_raw = []
    for mask in segment_masks:
        rows, columns = np.indices(aperture.shape, dtype=float)
        local_x = columns - np.mean(columns[mask])
        segment_raw.append(np.asarray([
            np.where(mask, 1.0, 0.0),
            np.where(mask, local_x, 0.0),
        ]))
    segment_raw = np.asarray(segment_raw)
    return aperture, segment_masks, global_raw, segment_raw


def test_sequential_decomposition_recovers_synthetic_coefficients():
    """Recover injected global and per-segment orthonormal coefficients."""
    aperture, segment_masks, global_raw, segment_raw = (
        _synthetic_segmented_problem()
    )
    global_basis = DERIVATION.orthonormalize_basis(global_raw, aperture)
    segment_bases = np.asarray([
        DERIVATION.orthonormalize_basis(raw, mask)
        for raw, mask in zip(segment_raw, segment_masks)
    ])
    expected_global = np.array([5.0])
    expected_segment = np.array([
        [3.0, 2.0],
        [-1.0, -4.0],
        [-2.0, 1.0],
    ])
    opd = np.tensordot(expected_global, global_basis, axes=1)
    for coefficients, basis in zip(expected_segment, segment_bases):
        opd += np.tensordot(coefficients, basis, axes=1)

    global_coefficients, segment_coefficients, residual = (
        DERIVATION.decompose_opd_map(
            opd, aperture, global_raw, segment_masks, segment_raw
        )
    )

    np.testing.assert_allclose(
        np.abs(global_coefficients), np.abs(expected_global), rtol=0.05
    )
    np.testing.assert_allclose(
        np.abs(segment_coefficients), np.abs(expected_segment), rtol=0.05
    )
    assert np.linalg.norm(residual[aperture]) < 1e-10

    global_weights, segment_weights, fraction = (
        DERIVATION.aggregate_mode_statistics(
            global_coefficients[np.newaxis, :],
            segment_coefficients[np.newaxis, :, :],
            np.full(3, 1 / 3),
        )
    )
    expected_segment_variance = np.mean(np.sum(expected_segment**2, axis=1))
    expected_fraction = expected_segment_variance / (
        expected_segment_variance + np.sum(expected_global**2)
    )
    assert global_weights == pytest.approx(np.abs(expected_global), rel=0.05)
    assert segment_weights == pytest.approx(
        np.sqrt(np.mean(expected_segment**2, axis=0)), rel=0.05
    )
    assert fraction == pytest.approx(expected_fraction, rel=0.05)


def test_k_step_differences_return_expected_maps_and_pairs():
    """Difference every valid k-separated pair in a synthetic OPD series."""
    series = np.arange(6, dtype=float)[:, np.newaxis, np.newaxis] * np.ones(
        (1, 2, 3)
    )

    differences, pairs = DERIVATION.difference_opd_series(series, step=2)

    assert differences.shape == (4, 2, 3)
    np.testing.assert_array_equal(differences, np.full((4, 2, 3), 2.0))
    np.testing.assert_array_equal(
        pairs, np.array([[0, 2], [1, 3], [2, 4], [3, 5]])
    )


def test_weight_aggregation_matches_hand_computed_rms():
    """Aggregate per-mode RMS and area-weighted variance by definition."""
    global_coefficients = np.array([[3.0, 4.0], [0.0, 8.0]])
    segment_coefficients = np.array([
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0]],
    ])
    areas = np.array([0.25, 0.75])

    global_weights, segment_weights, fraction = (
        DERIVATION.aggregate_mode_statistics(
            global_coefficients, segment_coefficients, areas
        )
    )

    np.testing.assert_allclose(
        global_weights,
        np.sqrt(np.mean(global_coefficients**2, axis=0)),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        segment_weights,
        np.sqrt(np.mean(segment_coefficients**2, axis=(0, 1))),
        rtol=1e-12,
    )
    segment_variance = np.sum(segment_coefficients**2, axis=2) @ areas
    global_variance = np.sum(global_coefficients**2, axis=1)
    expected_fraction = np.mean(
        segment_variance / (segment_variance + global_variance)
    )
    assert fraction == pytest.approx(expected_fraction, rel=1e-12)


def test_weight_table_writer_round_trips_through_public_loader(tmp_path):
    """Write safe YAML that the public prior loader reads identically."""
    document = DERIVATION.make_weight_document(
        'offline-derived',
        [4, 5],
        [3.0, 4.0],
        [1, 2],
        [5.0, 12.0],
        0.37,
        {'decomposition_method': DERIVATION.DECOMPOSITION_METHOD},
    )
    path = tmp_path / 'derived.yaml'

    DERIVATION.write_weight_table(path, document)
    prior = load_mode_weight_prior(path)

    assert prior.name == 'offline-derived'
    assert prior.global_weights == pytest.approx({4: 0.6, 5: 0.8})
    assert prior.segment_weights == pytest.approx({1: 5 / 13, 2: 12 / 13})
    assert prior.segment_variance_fraction == pytest.approx(0.37)
    assert prior.metadata['decomposition_method'] == DERIVATION.DECOMPOSITION_METHOD

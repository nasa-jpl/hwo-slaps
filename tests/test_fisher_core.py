"""Unit tests for the Fisher / Asimov statistical core.

These tests avoid AutoLens / HCIPy entirely and validate the linear-algebra
identities that the detector relies on.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "src" / "hwoslaps" / "modeling" / "fisher_core.py"

import sys
spec = importlib.util.spec_from_file_location("hwoslaps_fisher_core", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

ProfileLikelihoodWorkspace = module.ProfileLikelihoodWorkspace
Whitener = module.Whitener
compute_asimov_detectability = module.compute_asimov_detectability
evaluate_signal_bank = module.evaluate_signal_bank
compute_spurious_amplitude = module.compute_spurious_amplitude
scan_systematic_modes = module.scan_systematic_modes
sidak_local_p = module.sidak_local_p
sidak_local_z = module.sidak_local_z
global_p_from_local = module.global_p_from_local
detectable_area = module.detectable_area


def test_no_nuisance_recovers_raw_information_exactly():
    signal = np.array([1.0, -2.0, 0.5, 3.0])
    result = compute_asimov_detectability(signal, sigma=np.ones(signal.size))

    expected_raw = float(signal @ signal)
    assert result.fisher_raw == pytest.approx(expected_raw)
    assert result.fisher_profiled == pytest.approx(expected_raw)
    assert result.degradation == pytest.approx(1.0)
    assert result.absorbed_fraction == pytest.approx(0.0)
    assert result.z_asimov_local == pytest.approx(np.sqrt(expected_raw))


def test_profiled_information_never_exceeds_raw():
    signal = np.array([1.0, 2.0, -1.0, 0.0])
    nuisance = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, -0.5],
        ]
    )
    result = compute_asimov_detectability(signal, nuisance_jacobian=nuisance, sigma=np.ones(signal.size))

    assert result.fisher_profiled >= 0.0
    assert result.fisher_profiled <= result.fisher_raw + 1.0e-12
    assert 0.0 <= result.degradation <= 1.0


def test_nuisance_basis_change_leaves_profiled_information_invariant():
    rng = np.random.default_rng(42)
    signal = rng.normal(size=7)
    nuisance = rng.normal(size=(7, 3))
    transform = np.array(
        [
            [2.0, -1.0, 0.3],
            [0.0, 1.5, 0.2],
            [0.0, 0.0, 0.7],
        ]
    )
    nuisance_rot = nuisance @ transform

    result_1 = compute_asimov_detectability(signal, nuisance_jacobian=nuisance, sigma=np.ones(signal.size))
    result_2 = compute_asimov_detectability(signal, nuisance_jacobian=nuisance_rot, sigma=np.ones(signal.size))

    assert result_1.fisher_profiled == pytest.approx(result_2.fisher_profiled, rel=1e-12, abs=1e-12)
    assert result_1.z_asimov_local == pytest.approx(result_2.z_asimov_local, rel=1e-12, abs=1e-12)


def test_stronger_nuisance_priors_can_only_help_information():
    signal = np.array([1.0, -0.5, 2.0, 0.25, -1.0])
    nuisance = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
            [1.0, -1.0],
            [-0.5, 0.25],
        ]
    )

    free = compute_asimov_detectability(signal, nuisance_jacobian=nuisance, sigma=np.ones(signal.size))
    weak_prior = compute_asimov_detectability(
        signal,
        nuisance_jacobian=nuisance,
        sigma=np.ones(signal.size),
        prior_precision=np.array([0.1, 0.2]),
    )
    strong_prior = compute_asimov_detectability(
        signal,
        nuisance_jacobian=nuisance,
        sigma=np.ones(signal.size),
        prior_precision=np.array([100.0, 100.0]),
    )

    assert free.fisher_profiled <= weak_prior.fisher_profiled + 1.0e-12
    assert weak_prior.fisher_profiled <= strong_prior.fisher_profiled + 1.0e-12
    assert strong_prior.fisher_profiled <= strong_prior.fisher_raw + 1.0e-12


def test_dense_covariance_matches_explicit_whitening():
    signal = np.array([0.5, -1.2, 2.0])
    nuisance = np.array(
        [
            [1.0, 0.2],
            [0.0, 1.0],
            [0.3, -0.4],
        ]
    )
    covariance = np.array(
        [
            [4.0, 1.0, 0.0],
            [1.0, 3.0, 0.5],
            [0.0, 0.5, 2.0],
        ]
    )

    direct = compute_asimov_detectability(signal, nuisance_jacobian=nuisance, covariance=covariance)

    whitener = Whitener.from_covariance(covariance)
    signal_w = whitener.apply(signal)
    nuisance_w = whitener.apply(nuisance)
    workspace = ProfileLikelihoodWorkspace(nuisance_whitened=nuisance_w)
    whitened = workspace.evaluate_signal(signal_w)

    assert direct.fisher_profiled == pytest.approx(whitened.fisher_profiled, rel=1e-12, abs=1e-12)
    assert direct.fisher_raw == pytest.approx(whitened.fisher_raw, rel=1e-12, abs=1e-12)
    assert direct.z_asimov_local == pytest.approx(whitened.z_asimov_local, rel=1e-12, abs=1e-12)


def test_signal_bank_matches_individual_calls():
    rng = np.random.default_rng(1)
    signals = rng.normal(size=(5, 8))
    nuisance = rng.normal(size=(8, 2))
    sigma = 0.5 + rng.random(8)

    bank = evaluate_signal_bank(signals, nuisance_jacobian=nuisance, sigma=sigma)
    singles = [
        compute_asimov_detectability(signal, nuisance_jacobian=nuisance, sigma=sigma)
        for signal in signals
    ]

    np.testing.assert_allclose(bank.fisher_profiled, [s.fisher_profiled for s in singles])
    np.testing.assert_allclose(bank.z_asimov_local, [s.z_asimov_local for s in singles])
    np.testing.assert_allclose(bank.degradation, [s.degradation for s in singles])


def test_spurious_amplitude_matches_closed_form_without_nuisance():
    signal = np.array([1.0, 2.0, -1.0])
    bias = np.array([0.3, -0.2, 0.5])

    result = compute_spurious_amplitude(signal, bias, sigma=np.ones(signal.size))
    expected_amp = float(signal @ bias) / float(signal @ signal)
    expected_z = abs(expected_amp) * np.sqrt(float(signal @ signal))

    assert result.amplitude_spurious == pytest.approx(expected_amp)
    assert result.z_spurious == pytest.approx(expected_z)


def test_spurious_amplitude_is_undefined_for_zero_information_signal():
    signal = np.zeros(3)
    bias = np.array([0.3, -0.2, 0.5])

    result = compute_spurious_amplitude(signal, bias, sigma=np.ones(signal.size))

    assert result.fisher_profiled == pytest.approx(0.0)
    assert np.isinf(result.sigma_amplitude_profiled)
    assert np.isnan(result.amplitude_spurious)
    assert np.isnan(result.z_spurious)
    assert np.isnan(result.numerator)


def test_systematic_mode_scan_reports_expected_tolerances_and_rms():
    signal = np.array([1.0, 0.0, 0.0])
    modes = np.eye(3)
    mode_sigmas = np.array([0.5, 2.0, 1.0])
    systematic_cov = np.diag(mode_sigmas**2)

    scan = scan_systematic_modes(
        signal=signal,
        systematic_modes=modes,
        sigma=np.ones(3),
        mode_names=["x", "y", "z"],
        mode_sigmas=mode_sigmas,
        z_tolerance=1.0,
        systematic_covariance=systematic_cov,
    )

    coupling_x = scan.couplings[0]
    coupling_y = scan.couplings[1]
    coupling_z = scan.couplings[2]

    assert coupling_x.mode_name == "x"
    assert coupling_x.amplitude_per_unit == pytest.approx(1.0)
    assert coupling_x.z_per_unit == pytest.approx(1.0)
    assert coupling_x.one_sigma_z == pytest.approx(0.5)
    assert coupling_x.tolerance_for_zmax == pytest.approx(1.0)

    assert coupling_y.z_per_unit == pytest.approx(0.0)
    assert np.isinf(coupling_y.tolerance_for_zmax)
    assert coupling_z.z_per_unit == pytest.approx(0.0)

    assert scan.rms_spurious_amplitude == pytest.approx(0.5)
    assert scan.rms_spurious_z == pytest.approx(0.5)


def test_systematic_mode_scan_is_undefined_for_zero_information_signal():
    signal = np.zeros(3)
    modes = np.eye(3)
    mode_sigmas = np.array([0.5, 2.0, 1.0])
    systematic_cov = np.diag(mode_sigmas**2)

    scan = scan_systematic_modes(
        signal=signal,
        systematic_modes=modes,
        sigma=np.ones(3),
        mode_names=["x", "y", "z"],
        mode_sigmas=mode_sigmas,
        z_tolerance=1.0,
        systematic_covariance=systematic_cov,
    )

    assert scan.fisher_profiled == pytest.approx(0.0)
    assert np.isinf(scan.sigma_amplitude_profiled)
    assert np.isnan(scan.rms_spurious_amplitude)
    assert np.isnan(scan.rms_spurious_z)
    for coupling in scan.couplings:
        assert np.isnan(coupling.amplitude_per_unit)
        assert np.isnan(coupling.z_per_unit)
        assert np.isnan(coupling.one_sigma_z)
        assert np.isnan(coupling.tolerance_for_zmax)


def test_sidak_and_global_p_are_consistent():
    global_p = 2.7e-3
    n_eff = 20
    local_p = sidak_local_p(global_p, n_eff)

    assert 0.0 < local_p < global_p
    assert global_p_from_local(local_p, n_eff) == pytest.approx(global_p)
    assert sidak_local_z(global_p, n_eff) == pytest.approx(module.norm.isf(local_p))


def test_detectable_area_counts_cells_above_threshold():
    values = np.array([[1.0, 3.2], [5.1, 4.9]])
    area = detectable_area(values, cell_area=0.25, threshold=5.0)
    assert area == pytest.approx(0.25)


def test_singular_nuisance_is_handled_by_pseudoinverse():
    signal = np.array([1.0, -1.0, 0.0, 0.5])
    nuisance = np.array(
        [
            [1.0, 2.0],
            [0.0, 0.0],
            [1.0, 2.0],
            [0.5, 1.0],
        ]
    )
    result = compute_asimov_detectability(signal, nuisance_jacobian=nuisance, sigma=np.ones(signal.size))

    assert result.nuisance_rank == 1
    assert result.fisher_profiled >= 0.0
    assert result.fisher_profiled <= result.fisher_raw + 1.0e-12

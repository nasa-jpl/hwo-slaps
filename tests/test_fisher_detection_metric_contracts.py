"""Scientific contracts for Fisher detectability metric outputs."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = PROJECT_ROOT / "src" / "hwoslaps" / "modeling" / "fisher_core.py"
SCDD_DELTA_LOG_L_THRESHOLD = 5.0
SCDD_Q_THRESHOLD = 2.0 * SCDD_DELTA_LOG_L_THRESHOLD

core_spec = importlib.util.spec_from_file_location(
    "hwoslaps_fisher_detection_metric_contracts_core",
    CORE_PATH,
)
core_module = importlib.util.module_from_spec(core_spec)
sys.modules[core_spec.name] = core_module
core_spec.loader.exec_module(core_module)
compute_asimov_detectability = core_module.compute_asimov_detectability
evaluate_signal_bank = core_module.evaluate_signal_bank


def _load_master_config() -> dict:
    config_path = PROJECT_ROOT / "configs" / "master_config.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_master_config_uses_scdd_baseline_redshifts():
    """Pin the SCDD baseline lens and source redshifts in the config."""
    config = _load_master_config()

    assert config["lensing"]["lens_galaxy"]["redshift"] == pytest.approx(0.2)
    assert config["lensing"]["source_galaxy"]["redshift"] == pytest.approx(0.6)


def test_scdd_threshold_maps_to_fisher_q_and_local_z():
    """Map the SCDD threshold between q, Z, and delta log-likelihood."""
    signal = np.array([np.sqrt(SCDD_Q_THRESHOLD), 0.0])

    result = compute_asimov_detectability(signal=signal, sigma=np.ones(signal.size))
    delta_log_l_equiv = result.q_asimov_local / 2.0

    assert result.fisher_profiled == pytest.approx(SCDD_Q_THRESHOLD)
    assert result.q_asimov_local == pytest.approx(result.fisher_profiled)
    assert result.z_asimov_local == pytest.approx(np.sqrt(SCDD_Q_THRESHOLD))
    assert delta_log_l_equiv == pytest.approx(SCDD_DELTA_LOG_L_THRESHOLD)


def test_scdd_metric_uses_profiled_not_raw_fisher_information():
    """Report the SCDD metric from profiled, not raw, information."""
    signal = np.array([2.0, 0.0])
    nuisance = np.array([[1.0], [0.0]])
    prior_precision = np.array([[1.0]])

    result = compute_asimov_detectability(
        signal=signal,
        nuisance_jacobian=nuisance,
        sigma=np.ones(signal.size),
        prior_precision=prior_precision,
    )

    assert result.fisher_raw == pytest.approx(4.0)
    assert result.fisher_profiled == pytest.approx(2.0)
    assert result.q_asimov_local == pytest.approx(result.fisher_profiled)
    assert result.z_asimov_local == pytest.approx(np.sqrt(result.fisher_profiled))
    assert result.q_asimov_local / 2.0 == pytest.approx(1.0)


def test_detectable_ring_fraction_uses_scdd_q_threshold():
    """Count ring positions as detected using the SCDD q threshold."""
    signals = np.array(
        [
            [2.0, 0.0],
            [np.sqrt(SCDD_Q_THRESHOLD), 0.0],
            [np.sqrt(12.0), 0.0],
            [0.0, 0.0],
        ]
    )

    bank = evaluate_signal_bank(signal_bank=signals, sigma=np.ones(signals.shape[1]))
    detected = bank.q_asimov_local >= SCDD_Q_THRESHOLD
    detectable_ring_fraction = float(np.count_nonzero(detected) / detected.size)

    np.testing.assert_allclose(bank.q_asimov_local, bank.fisher_profiled)
    np.testing.assert_allclose(bank.z_asimov_local, np.sqrt(bank.q_asimov_local))
    assert detectable_ring_fraction == pytest.approx(0.5)


def test_sparse_nonlinear_calibration_qfit_definition_matches_scdd_threshold():
    """Define q_fit as twice the log-likelihood gap, as SCDD does."""
    log_l_smooth = -105.0
    log_l_subhalo = -100.0

    q_fit = 2.0 * (log_l_subhalo - log_l_smooth)

    assert q_fit == pytest.approx(SCDD_Q_THRESHOLD)
    assert q_fit / 2.0 == pytest.approx(SCDD_DELTA_LOG_L_THRESHOLD)

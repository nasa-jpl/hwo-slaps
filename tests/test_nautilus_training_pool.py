"""Canaries for the scoped Nautilus emulator-training worker pool."""

from __future__ import annotations

import numpy as np
import pytest

from nautilus.neural import NeuralNetworkEmulator

from hwoslaps.modeling.nonlinear import autolens_runner

WORKERS_ENV = "HWOSLAPS_NAUTILUS_TRAINING_WORKERS"


@pytest.fixture
def restored_training_seam(monkeypatch):
    """Restore the emulator seam and pool registry after each test."""
    original_train = NeuralNetworkEmulator.__dict__["train"]
    original_pools = dict(autolens_runner._TRAINING_POOLS)
    autolens_runner._close_training_pools()
    yield monkeypatch
    NeuralNetworkEmulator.train = original_train
    autolens_runner._close_training_pools()
    autolens_runner._TRAINING_POOLS.update(original_pools)


def _training_data(n_points=200, n_dim=6, seed=3):
    rng = np.random.default_rng(seed)
    return rng.random((n_points, n_dim)), rng.random(n_points)


def _weights(emulator):
    return [
        array
        for network in emulator.neural_networks
        for array in list(network.coefs_) + list(network.intercepts_)
    ]


def test_scope_restores_original_train_after_enabled_run(restored_training_seam):
    before = NeuralNetworkEmulator.train.__func__
    restored_training_seam.setenv(WORKERS_ENV, "2")
    with autolens_runner._nautilus_training_pool_scope(number_of_cores=1):
        assert NeuralNetworkEmulator.train.__func__ is not before
    assert NeuralNetworkEmulator.train.__func__ is before
    assert autolens_runner._TRAINING_POOLS == {}


def test_enable_then_disable_trains_serially(restored_training_seam):
    x, y = _training_data()
    restored_training_seam.setenv(WORKERS_ENV, "2")
    with autolens_runner._nautilus_training_pool_scope(number_of_cores=1):
        assert autolens_runner._training_pool_for_current_env() is not None
        restored_training_seam.setenv(WORKERS_ENV, "1")
        assert NeuralNetworkEmulator.train(x, y, n_networks=2) is not None


def test_worker_count_change_does_not_reuse_old_pool(restored_training_seam):
    restored_training_seam.setenv(WORKERS_ENV, "2")
    with autolens_runner._nautilus_training_pool_scope(number_of_cores=1):
        first = autolens_runner._training_pool_for_current_env()
        restored_training_seam.setenv(WORKERS_ENV, "3")
        second = autolens_runner._training_pool_for_current_env()
        assert first is not second
        assert first is not None
        assert second is not None


def test_number_of_cores_above_one_does_not_patch(restored_training_seam):
    before = NeuralNetworkEmulator.train.__func__
    restored_training_seam.setenv(WORKERS_ENV, "4")
    with autolens_runner._nautilus_training_pool_scope(number_of_cores=2):
        assert NeuralNetworkEmulator.train.__func__ is before
    assert autolens_runner._TRAINING_POOLS == {}


def test_explicit_pool_argument_wins(restored_training_seam):
    x, y = _training_data()
    restored_training_seam.setenv(WORKERS_ENV, "2")
    used = []

    class RecordingPool:
        def map(self, func, iterable):
            used.append(True)
            return list(map(func, iterable))

    with autolens_runner._nautilus_training_pool_scope(number_of_cores=1):
        NeuralNetworkEmulator.train(x, y, n_networks=2, pool=RecordingPool())
    assert used == [True]


@pytest.mark.xtx_gpu
def test_pooled_weights_match_serial(restored_training_seam):
    x, y = _training_data()
    serial = NeuralNetworkEmulator.train(x, y, n_networks=2)

    restored_training_seam.setenv(WORKERS_ENV, "2")
    with autolens_runner._nautilus_training_pool_scope(number_of_cores=1):
        pooled = NeuralNetworkEmulator.train(x, y, n_networks=2)

    serial_weights, pooled_weights = _weights(serial), _weights(pooled)
    assert len(serial_weights) == len(pooled_weights)
    for expected, actual in zip(serial_weights, pooled_weights):
        assert np.array_equal(expected, actual)
    assert np.array_equal(serial.predict(x), pooled.predict(x))


@pytest.mark.parametrize("invalid", [0, -1, True, 1.5, "2"])
def test_training_worker_settings_reject_invalid_counts(invalid):
    with pytest.raises(ValueError, match="nautilus_training_workers"):
        autolens_runner.NonlinearSearchSettings(nautilus_training_workers=invalid)


def test_explicit_training_workers_override_environment(restored_training_seam):
    restored_training_seam.setenv(WORKERS_ENV, "8")
    assert autolens_runner._training_runtime_provenance(1, 2) == {
        "requested": 2, "effective": 2, "start_method": "spawn",
    }
    assert autolens_runner._training_runtime_provenance(2, 2) == {
        "requested": 2, "effective": 1, "start_method": "serial",
    }
    before = NeuralNetworkEmulator.train.__func__
    with autolens_runner._nautilus_training_pool_scope(1, 1):
        assert NeuralNetworkEmulator.train.__func__ is before
    assert autolens_runner._TRAINING_POOLS == {}


def test_explicit_pool_workers_do_not_follow_environment_changes(restored_training_seam):
    created = []

    class Pool:
        def __init__(self, workers):
            created.append(workers)

        def close(self):
            pass

        def join(self):
            pass

    restored_training_seam.setattr(autolens_runner, "_NetworkTrainingPool", Pool)
    restored_training_seam.setenv(WORKERS_ENV, "8")
    with autolens_runner._nautilus_training_pool_scope(1, 2):
        first = autolens_runner._training_pool_for_current_env(2)
        restored_training_seam.setenv(WORKERS_ENV, "3")
        second = autolens_runner._training_pool_for_current_env(2)
        assert first is second
    assert created == [2]

"""Execution-policy contracts without importing the rendering stack."""

from concurrent.futures import Future
from copy import deepcopy

import pytest

from hwoslaps.modeling import fisher_runtime


def test_reference_worker_override_preserves_numerical_configuration():
    config = {"engine": "reference", "num_workers": 1, "grid": {"spacing_arcsec": 0.1}}
    original = deepcopy(config)
    assert fisher_runtime.grid_num_workers(config, " 4 ") == 4
    assert fisher_runtime.grid_runtime_provenance(config, "4") == {
        "fisher_grid_workers_requested": 1,
        "fisher_grid_workers_effective": 4,
        "fisher_grid_start_method": "spawn",
    }
    assert config == original


def test_jax_worker_policy_ignores_reference_override_and_records_engine():
    config = {"engine": "jax", "num_workers": 3}
    assert fisher_runtime.grid_num_workers(config, "invalid") == 3
    assert fisher_runtime.grid_runtime_provenance(config, "invalid") == {
        "fisher_grid_workers_requested": 3,
        "fisher_grid_workers_effective": 1,
        "fisher_grid_start_method": "jax",
    }


@pytest.mark.parametrize("override", ["0", "-1", "invalid"])
def test_reference_worker_override_rejects_invalid_counts(override):
    with pytest.raises(ValueError):
        fisher_runtime.grid_num_workers({"num_workers": 1}, override)


class _ImmediateExecutor:
    """Deterministically reproduce an exhausted completed batch."""

    def __init__(self, **kwargs):
        pass

    def submit(self, func, value):
        result = Future()
        result.set_result(func(value))
        return result

    def shutdown(self, **kwargs):
        pass


def test_supervised_map_must_not_drop_inputs_when_entire_batch_finishes(monkeypatch):
    monkeypatch.setattr(fisher_runtime, "ProcessPoolExecutor", _ImmediateExecutor)
    values = list(range(10))
    assert (
        list(fisher_runtime.supervised_ordered_map(abs, values, num_workers=2))
        == values
    )

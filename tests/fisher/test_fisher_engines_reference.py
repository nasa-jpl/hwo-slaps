"""Real spawned reference evaluation and ordered worker failure semantics."""

import os
from concurrent.futures.process import BrokenProcessPool
from copy import deepcopy

import numpy as np
import pytest

pytestmark = pytest.mark.backend


def worker_initializer():
    pass


def worker_square(value):
    return value * value


def worker_failure(value):
    if value == 2:
        raise ValueError("deliberate worker failure")
    return value


def worker_death(value):
    if value == 2:
        os._exit(17)
    return value


@pytest.mark.parametrize("mismatched", [False, True])
def test_pooled_reference_engine_equals_serial_bitwise(minimal_mapping, tiny_gaussian_kernel, tmp_path, mismatched):
    from hwoslaps.fisher.api import Execution, forecast, prepare_forecast

    if mismatched:
        kernel = tiny_gaussian_kernel.copy()
        kernel[3, 3] *= 1.1
        np.save(tmp_path / "model.npy", kernel / kernel.sum())
        minimal_mapping["psf"]["model"] = {"kind": "kernel", "path": str(tmp_path / "model.npy"), "pixel_scale_arcsec": 0.05}
    with prepare_forecast(minimal_mapping) as serial, prepare_forecast(minimal_mapping, execution=Execution(reference_workers=2)) as pool:
        expected = forecast(serial, masses_msun=[1.0e8, 3.0e8])
        actual = forecast(pool, masses_msun=[1.0e8, 3.0e8])
    for name in ("fisher_raw", "fisher_profiled", "amplitude_hat", "amplitude_spurious"):
        if getattr(expected, name) is not None:
            np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))


@pytest.mark.parametrize("worker,error", [(worker_square, None), (worker_failure, ValueError), (worker_death, BrokenProcessPool)])
def test_ordered_process_map_surfaces_failures_and_keeps_order(worker, error):
    from hwoslaps.fisher.engines.reference import ordered_process_map

    def run():
        return list(ordered_process_map(worker, range(9), workers=2, initializer=worker_initializer, initargs=()))
    if error is None:
        assert run() == [0, 1, 4, 9, 16, 25, 36, 49, 64]
    else:
        with pytest.raises(error):
            run()

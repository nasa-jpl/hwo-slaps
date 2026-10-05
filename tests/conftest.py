"""Collection policy and the fixtures shared across test lanes.

Tests that need GPUs run only where cards are assigned: ``xtx_gpu`` items are
skipped without a JAX GPU and ``xtx_multi_gpu`` items without two. The gpu and
multi launchers pass ``--require-gpu``, so a missing card fails there instead.
No global backend configuration or JIT setting is changed here.
"""

import numpy as np
import pytest

MARKERS = (
    "backend: needs a scientific backend (PyAutoLens stack, hcipy, jax, nautilus, numba or matplotlib)",
    "xtx_gpu: needs the pinned CUDA JAX runtime and one GPU",
    "xtx_multi_gpu: needs two GPUs assigned by the orchestrator",
)


def pytest_configure(config):
    for marker in MARKERS:
        config.addinivalue_line("markers", marker)


def _visible_gpus():
    try:
        import jax
        return len(jax.devices("gpu"))
    except (ImportError, RuntimeError):
        return 0


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    """Skip GPU-owned contracts that the visible devices cannot run.

    Runs after marker deselection, so a lane that deselects GPU tests never imports JAX.
    """
    needs = []
    for item in items:
        if item.get_closest_marker("xtx_multi_gpu"):
            needs.append((item, 2))
        elif item.get_closest_marker("xtx_gpu"):
            needs.append((item, 1))
    if not needs:
        return
    available = _visible_gpus()
    for item, count in needs:
        if available < count:
            item.add_marker(pytest.mark.skip(reason=f"requires {count} JAX GPU(s), found {available}"))


@pytest.fixture
def tiny_gaussian_kernel():
    """A 7 x 7 Gaussian detector kernel, sigma 1 pixel, unit sum, float64."""
    y, x = np.mgrid[-3:4, -3:4].astype(np.float64)
    kernel = np.exp(-(x**2 + y**2) / 2.0)
    return kernel / kernel.sum()

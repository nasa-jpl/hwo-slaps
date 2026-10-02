"""Collection policy without global backend configuration or JIT mutation."""

import pytest


def pytest_collection_modifyitems(config, items):
    """Skip explicitly GPU-owned contracts when the backend is unavailable."""
    gpu_tests = [item for item in items if item.get_closest_marker("xtx_gpu")]
    if not gpu_tests:
        return
    try:
        import jax
        available = bool(jax.devices("gpu"))
    except (ImportError, RuntimeError):
        available = False
    if not available:
        for item in gpu_tests:
            item.add_marker(pytest.mark.skip(reason="requires a JAX GPU backend"))

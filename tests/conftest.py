"""Pytest configuration for lensing test compatibility."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import pytest


def pytest_configure() -> None:
    """Preload `autoarray` config and disable numba JIT for stable imports."""
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

    try:
        from autoconf import conf as autoconf_conf
    except Exception:
        return

    autoarray_spec = importlib.util.find_spec("autoarray")
    if autoarray_spec is None or autoarray_spec.origin is None:
        return

    config_path = Path(autoarray_spec.origin).resolve().parent / "config"
    if not config_path.exists():
        return

    autoconf_conf.instance.push(str(config_path), keep_first=True)


def pytest_collection_modifyitems(config, items):
    """Skip xtx_gpu tests when JAX or a GPU runtime is unavailable."""
    for item in items:
        if not item.get_closest_marker("xtx_gpu"):
            continue

        reason = None
        try:
            import jax
        except Exception as exc:
            reason = f"requires jax for xtx_gpu tests: {type(exc).__name__}"
        else:
            try:
                if not jax.devices("gpu"):
                    reason = "requires a JAX GPU device"
            except RuntimeError as exc:
                reason = f"jax.devices('gpu') failed: {type(exc).__name__}"
        if reason:
            item.add_marker(pytest.mark.skip(reason=reason))

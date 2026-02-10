"""Pytest configuration for lensing test compatibility."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


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

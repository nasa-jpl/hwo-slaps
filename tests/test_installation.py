#!/usr/bin/env python
"""Basic import tests for HWO-SLAPS core dependencies and package.

This test file replaces a previous script-style checker with proper pytest
tests to avoid parameterized function collection errors and to integrate with
CI.
"""

import importlib


def _can_import(module_name: str) -> bool:
    """Return True if the given module can be imported, False otherwise."""
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


def test_core_dependency_imports():
    """Core dependencies should import successfully in the test environment."""
    core_modules = [
        "autolens",
        "hcipy",
        "numpy",
        "scipy",
        "matplotlib",
        "yaml",
        "astropy",
    ]
    failures = [name for name in core_modules if not _can_import(name)]
    assert not failures, f"Failed to import core modules: {', '.join(failures)}"


def test_hwoslaps_package_importable():
    """The hwoslaps package and pipeline module should be importable."""
    assert _can_import("hwoslaps")
    assert _can_import("hwoslaps.pipeline")


def test_hcipy_hexike_api_symbols_present():
    """HCIPy installation should expose the hexike API used by hwoslaps."""
    import hcipy

    required_symbols = [
        "make_hexike_basis",
        "SegmentedHexikeSurface",
        "make_segment_hexike_surface_from_hex_aperture",
    ]
    missing = [name for name in required_symbols if not hasattr(hcipy, name)]
    assert not missing, f"Missing required HCIPy hexike API symbols: {', '.join(missing)}"

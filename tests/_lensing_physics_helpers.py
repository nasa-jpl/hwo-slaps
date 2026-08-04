"""Helpers for lensing physics tests.

This module provides small utilities for loading project modules without
importing the top-level package, plus a cosmology adapter used by physics
equation tests that run without `autolens`.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any, Dict

import yaml
from astropy.cosmology import Planck15

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src" / "hwoslaps"


def bootstrap_hwoslaps_namespace() -> None:
    """Create minimal package namespaces for direct module loading."""
    package_paths = {
        "hwoslaps": SRC_ROOT,
        "hwoslaps.config": SRC_ROOT / "config",
        "hwoslaps.lensing": SRC_ROOT / "lensing",
        "hwoslaps.modeling": SRC_ROOT / "modeling",
        "hwoslaps.observation": SRC_ROOT / "observation",
        "hwoslaps.plotting": SRC_ROOT / "plotting",
    }
    for module_name, path in package_paths.items():
        if module_name not in sys.modules:
            pkg = types.ModuleType(module_name)
            pkg.__path__ = [str(path)]
            sys.modules[module_name] = pkg


def load_module(relative_path: str, module_name: str):
    """Load a module from ``src/hwoslaps`` by file path."""
    if module_name in sys.modules:
        return sys.modules[module_name]

    module_path = SRC_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_constants_module():
    """Load ``hwoslaps.constants`` without importing package ``__init__``."""
    bootstrap_hwoslaps_namespace()
    return load_module("constants.py", "hwoslaps.constants")


def load_mass_models_module():
    """Load ``hwoslaps.lensing.mass_models`` without the package init."""
    bootstrap_hwoslaps_namespace()
    load_constants_module()
    return load_module("lensing/mass_models.py", "hwoslaps.lensing.mass_models")


def load_validation_module():
    """Load ``hwoslaps.config.validation`` without the package init."""
    bootstrap_hwoslaps_namespace()
    return load_module("config/validation.py", "hwoslaps.config.validation")


def load_lensing_utils_module():
    """Load ``hwoslaps.lensing.utils``.

    When `autolens` is unavailable, this function injects a tiny temporary
    stub module so pure helper functions (for example
    ``get_einstein_ring_position``) can still be imported for core tests.
    """
    bootstrap_hwoslaps_namespace()

    injected_stub = False
    if "autolens" not in sys.modules and importlib.util.find_spec("autolens") is None:
        autolens_stub = types.ModuleType("autolens")
        autolens_stub.Grid2D = type("Grid2D", (), {})
        autolens_stub.Tracer = type("Tracer", (), {})
        sys.modules["autolens"] = autolens_stub
        injected_stub = True

    try:
        module = load_module("lensing/utils.py", "hwoslaps.lensing.utils")
    finally:
        if injected_stub:
            sys.modules.pop("autolens", None)
    return module


def load_master_config() -> Dict[str, Any]:
    """Return the parsed master configuration."""
    config_path = PROJECT_ROOT / "configs" / "master_config.yaml"
    with config_path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def load_lensing_anchor_fixture() -> Dict[str, Any]:
    """Return frozen lensing-physics regression anchors."""
    anchor_path = PROJECT_ROOT / "tests" / "fixtures" / "lensing_physics_anchors.json"
    with anchor_path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


class Planck15CosmologyAdapter:
    """Adapter exposing the lensing mass-model cosmology interface."""

    def __init__(self):
        self._cosmology = Planck15

    def H(self, redshift: float):
        """Return Hubble parameter at ``redshift``."""
        return self._cosmology.H(redshift)

    def angular_diameter_distance(self, redshift: float):
        """Return angular diameter distance for ``redshift``."""
        return self._cosmology.angular_diameter_distance(redshift)

    def angular_diameter_distance_z1z2(self, z1: float, z2: float):
        """Return angular diameter distance between two redshifts."""
        return self._cosmology.angular_diameter_distance_z1z2(z1, z2)

    @property
    def reduced_h(self) -> float:
        """Return reduced Hubble constant ``h``."""
        return float(self._cosmology.H0.value) / 100.0

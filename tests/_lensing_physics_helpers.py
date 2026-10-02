"""Helpers for lensing physics tests.

This module owns fixture files and a lazy Astropy cosmology adapter. Tests
import the real package normally; no test namespace replaces package modules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
        from astropy.cosmology import Planck15

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

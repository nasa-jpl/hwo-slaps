"""Integration test for NFW concentration provenance in lensing outputs."""

from __future__ import annotations

import copy
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import yaml


pytest.importorskip("autolens")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src" / "hwoslaps"


def _bootstrap_hwoslaps_namespace():
    if "hwoslaps" not in sys.modules:
        pkg = types.ModuleType("hwoslaps")
        pkg.__path__ = [str(SRC_ROOT)]
        sys.modules["hwoslaps"] = pkg
    if "hwoslaps.lensing" not in sys.modules:
        pkg = types.ModuleType("hwoslaps.lensing")
        pkg.__path__ = [str(SRC_ROOT / "lensing")]
        sys.modules["hwoslaps.lensing"] = pkg


def _load_module(relative_path: str, module_name: str):
    if module_name in sys.modules:
        return sys.modules[module_name]
    module_path = SRC_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_lensing_modules():
    _bootstrap_hwoslaps_namespace()
    _load_module("constants.py", "hwoslaps.constants")
    mass_models = _load_module("lensing/mass_models.py", "hwoslaps.lensing.mass_models")
    _load_module("lensing/utils.py", "hwoslaps.lensing.utils")
    generator = _load_module("lensing/generator.py", "hwoslaps.lensing.generator")
    return generator, mass_models


def _load_master_config():
    config_path = PROJECT_ROOT / "configs" / "master_config.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_nfw_concentration_provenance_is_recorded():
    generator, mass_models = _load_lensing_modules()

    config = _load_master_config()
    cfg = copy.deepcopy(config)
    cfg["lensing"]["grid"]["shape"] = [120, 120]
    cfg["lensing"]["subhalo"]["enabled"] = True
    cfg["lensing"]["subhalo"]["model"] = "NFW"
    cfg["lensing"]["subhalo"]["concentration"] = {
        "model": "moline2017_eq7",
        "x_sub": 1.2,
        "h": None,
    }

    lensing_data = generator.generate_lensing_system(cfg["lensing"], full_config=cfg)

    assert lensing_data.subhalo_model == "NFW"
    assert lensing_data.subhalo_concentration is not None
    assert lensing_data.subhalo_concentration_model == "moline2017_eq7"
    assert lensing_data.subhalo_concentration_source == "Moline2017 Eq7 Table2"
    assert lensing_data.subhalo_concentration_x_sub == pytest.approx(1.2)
    assert lensing_data.subhalo_concentration_h is not None
    assert lensing_data.subhalo_concentration_h > 0

    expected = mass_models.concentration_mass_relation(
        cfg["lensing"]["subhalo"]["mass"],
        model="moline2017_eq7",
        x_sub=lensing_data.subhalo_concentration_x_sub,
        h=lensing_data.subhalo_concentration_h,
    )
    assert lensing_data.subhalo_concentration == pytest.approx(expected, rel=1e-12)

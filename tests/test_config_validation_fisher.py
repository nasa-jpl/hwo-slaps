"""Validation tests for Fisher detection configuration schema."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src" / "hwoslaps"


def _load_module(relative_path: str, module_name: str):
    module_path = SRC_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validation = _load_module("config/validation.py", "hwoslaps_config_validation_fisher")


def _load_master_config():
    config_path = PROJECT_ROOT / "configs" / "master_config.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _with_valid_fisher_block(config: dict) -> dict:
    fisher_config = copy.deepcopy(config)
    fisher_config["modeling"]["detection"] = "fisher"
    fisher_config["modeling"]["fisher"] = {
        "mode": "both",
        "snr_threshold": 3.0,
        "include_background_offset": True,
        "finite_diff": {
            "centre_arcsec": 1.0e-3,
            "einstein_radius_arcsec": 1.0e-3,
            "ell_comp": 1.0e-3,
            "source_intensity_frac": 1.0e-2,
            "source_reff_frac": 1.0e-2,
        },
        "map": {
            "num_angles": 24,
            "offset_pixels": 0.0,
            "explicit_positions_yx": None,
        },
    }
    return fisher_config


def test_fisher_detection_requires_fisher_block():
    config = _load_master_config()
    bad_config = copy.deepcopy(config)
    bad_config["modeling"]["detection"] = "fisher"
    bad_config["modeling"].pop("fisher", None)

    with pytest.raises(ValueError, match="Missing required key 'fisher'"):
        validation.validate_or_raise(bad_config)


def test_fisher_rejects_invalid_mode():
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"]["mode"] = "invalid"

    with pytest.raises(ValueError, match="modeling.fisher.mode must be one of"):
        validation.validate_or_raise(config)


def test_fisher_requires_all_finite_diff_fields():
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"]["finite_diff"].pop("centre_arcsec")

    with pytest.raises(ValueError, match="Missing required key 'centre_arcsec'"):
        validation.validate_or_raise(config)


@pytest.mark.parametrize("bad_step", [0.0, -1.0, float("nan"), True])
def test_fisher_rejects_invalid_finite_diff_values(bad_step):
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"]["finite_diff"]["ell_comp"] = bad_step

    with pytest.raises(ValueError, match="modeling.fisher.finite_diff.ell_comp"):
        validation.validate_or_raise(config)


@pytest.mark.parametrize("bad_num_angles", [0, -4, 2.5, True])
def test_fisher_rejects_invalid_map_num_angles(bad_num_angles):
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"]["map"]["num_angles"] = bad_num_angles

    with pytest.raises(ValueError, match="modeling.fisher.map.num_angles must be a positive integer"):
        validation.validate_or_raise(config)


def test_fisher_rejects_invalid_explicit_positions_shape():
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"]["map"]["explicit_positions_yx"] = [[0.1], [0.2, 0.3, 0.4]]

    with pytest.raises(ValueError, match="entries must be \\[y, x\\] pairs"):
        validation.validate_or_raise(config)


def test_fisher_rejects_non_finite_explicit_positions():
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"]["map"]["explicit_positions_yx"] = [[0.1, float("inf")]]

    with pytest.raises(ValueError, match="entries must be finite"):
        validation.validate_or_raise(config)


def test_valid_fisher_config_passes_validation():
    config = _with_valid_fisher_block(_load_master_config())
    validation.validate_or_raise(config)

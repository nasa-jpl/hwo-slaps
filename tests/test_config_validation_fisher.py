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
            "type": "ring",
            "ring": {
                "num_angles": 24,
                "offset_pixels": 0.0,
            },
            "grid": {
                "spacing_arcsec": 0.05,
                "half_width_arcsec": 1.5,
                "annulus": None,
            },
            "explicit_positions_yx": None,
            "detection_q_threshold": 10.0,
            "num_workers": 1,
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


def test_modeling_rejects_non_fisher_detection_mode():
    config = _load_master_config()
    config["modeling"]["detection"] = "legacy"

    with pytest.raises(ValueError, match="modeling.detection must be 'fisher'"):
        validation.validate_or_raise(config)


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
    config["modeling"]["fisher"]["map"]["ring"]["num_angles"] = bad_num_angles

    with pytest.raises(ValueError, match="modeling.fisher.map.ring.num_angles must be a positive integer"):
        validation.validate_or_raise(config)


def test_fisher_map_requires_type():
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"]["map"].pop("type")

    with pytest.raises(ValueError, match="Missing required key 'type'"):
        validation.validate_or_raise(config)


def test_fisher_map_rejects_unknown_type():
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"]["map"]["type"] = "spiral"

    with pytest.raises(ValueError, match="modeling.fisher.map.type must be one of"):
        validation.validate_or_raise(config)


def test_fisher_map_rejects_legacy_flat_keys():
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"]["map"]["num_angles"] = 24

    with pytest.raises(ValueError, match="modeling.fisher.map contains unsupported keys"):
        validation.validate_or_raise(config)


def test_fisher_map_ring_type_requires_ring_block():
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"]["map"].pop("ring")

    with pytest.raises(ValueError, match="Missing required key 'ring'"):
        validation.validate_or_raise(config)


def test_fisher_map_grid_type_requires_grid_block():
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"]["map"]["type"] = "grid"
    config["modeling"]["fisher"]["map"].pop("grid")

    with pytest.raises(ValueError, match="Missing required key 'grid'"):
        validation.validate_or_raise(config)


@pytest.mark.parametrize("bad_spacing", [0.0, -0.1, float("nan"), True])
def test_fisher_map_rejects_invalid_grid_spacing(bad_spacing):
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"]["map"]["grid"]["spacing_arcsec"] = bad_spacing

    with pytest.raises(ValueError, match="modeling.fisher.map.grid.spacing_arcsec"):
        validation.validate_or_raise(config)


def test_fisher_map_rejects_half_width_below_spacing():
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"]["map"]["grid"]["half_width_arcsec"] = 0.01

    with pytest.raises(ValueError, match="half_width_arcsec must be >= spacing_arcsec"):
        validation.validate_or_raise(config)


def test_fisher_map_rejects_inverted_annulus():
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"]["map"]["grid"]["annulus"] = {
        "r_min_arcsec": 1.0,
        "r_max_arcsec": 0.5,
    }

    with pytest.raises(ValueError, match="r_max_arcsec must be > r_min_arcsec"):
        validation.validate_or_raise(config)


def test_fisher_map_rejects_unknown_annulus_key():
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"]["map"]["grid"]["annulus"] = {
        "r_min_arcsec": 0.5,
        "r_max_arcsec": 1.5,
        "radius": 1.0,
    }

    with pytest.raises(ValueError, match="annulus contains unsupported keys"):
        validation.validate_or_raise(config)


def test_fisher_map_explicit_type_requires_positions():
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"]["map"]["type"] = "explicit"
    config["modeling"]["fisher"]["map"]["explicit_positions_yx"] = []

    with pytest.raises(ValueError, match="must be non-empty when map.type is 'explicit'"):
        validation.validate_or_raise(config)


@pytest.mark.parametrize("bad_threshold", [0.0, -10.0, float("inf"), True])
def test_fisher_map_rejects_invalid_detection_q_threshold(bad_threshold):
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"]["map"]["detection_q_threshold"] = bad_threshold

    with pytest.raises(ValueError, match="modeling.fisher.map.detection_q_threshold"):
        validation.validate_or_raise(config)


@pytest.mark.parametrize("bad_workers", [0, -2, 1.5, True])
def test_fisher_map_rejects_invalid_num_workers(bad_workers):
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"]["map"]["num_workers"] = bad_workers

    with pytest.raises(ValueError, match="modeling.fisher.map.num_workers must be a positive integer"):
        validation.validate_or_raise(config)


def test_fisher_map_grid_config_passes_validation():
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"]["map"]["type"] = "grid"
    config["modeling"]["fisher"]["map"]["grid"]["annulus"] = {
        "r_min_arcsec": 0.5,
        "r_max_arcsec": 1.5,
    }
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


def test_master_config_fisher_defaults_pass_validation():
    validation.validate_or_raise(_load_master_config())


def test_fisher_rejects_unknown_key():
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"]["unexpected_option"] = True

    with pytest.raises(ValueError, match="modeling.fisher contains unsupported keys"):
        validation.validate_or_raise(config)


def test_fisher_rejects_invalid_mask_mode():
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"]["mask_mode"] = "bad"

    with pytest.raises(ValueError, match="modeling.fisher.mask_mode"):
        validation.validate_or_raise(config)


def test_fisher_rejects_nonpositive_psf_mode_step():
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"]["psf_mode_steps"] = {"segment_hexikes": 0.0}

    with pytest.raises(ValueError, match="modeling.fisher.psf_mode_steps"):
        validation.validate_or_raise(config)


def test_fisher_rejects_nonpositive_psf_mode_prior_sigma():
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"]["psf_mode_prior_sigmas"] = {"segment_hexikes": 0.0}

    with pytest.raises(ValueError, match="modeling.fisher.psf_mode_prior_sigmas"):
        validation.validate_or_raise(config)


def test_fisher_rejects_legacy_psf_mode_selection_alias():
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"]["psf_mode_selection"] = {"segment_pistons": {"segments": [0]}}

    with pytest.raises(ValueError, match="psf_mode_selection is not supported"):
        validation.validate_or_raise(config)


def test_fisher_rejects_malformed_psf_basis_selector():
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"]["psf_basis"] = {"segment_pistons": {"segments": ["zero"]}}

    with pytest.raises(ValueError, match="segment id"):
        validation.validate_or_raise(config)


def test_fisher_psf_features_require_psf_basis():
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"]["include_psf_nuisance"] = True
    config["modeling"]["fisher"]["compute_psf_mode_scan"] = False

    with pytest.raises(ValueError, match="psf_basis is required"):
        validation.validate_or_raise(config)


def test_fisher_scan_requires_explicit_scan_selection():
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"].update({
        "include_psf_nuisance": True,
        "compute_psf_mode_scan": True,
        "psf_basis": {"segment_hexikes": {"segments": [0], "mode_nolls": [1, 2]}},
    }
    )

    with pytest.raises(ValueError, match="scan_psf_mode_selection is required"):
        validation.validate_or_raise(config)


def test_valid_fisher_psf_options_pass_validation():
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"].update({
        "mask_mode": "source_snr",
        "include_psf_nuisance": True,
        "compute_psf_mode_scan": True,
        "mode_scan_z_tolerance": 1.0,
        "psf_mode_steps": {
            "segment_pistons": 1.0,
            "segment_tiptilts": 0.1,
            "segment_hexikes": 1.0,
            "global_zernikes": 1.0,
        },
        "psf_basis": {
            "segment_pistons": {"segments": [0, 1]},
            "segment_tiptilts": {"segments": [0]},
            "segment_hexikes": {"segments": [0, 1], "mode_nolls": [1, 2]},
            "global_zernikes": {"mode_nolls": [4, 5]},
        },
        "fit_psf_mode_selection": {
            "segment_hexikes": {"segments": [0, 1], "mode_nolls": [1]},
        },
        "scan_psf_mode_selection": {
            "global_zernikes": {"mode_nolls": [4, 5]},
        },
    })
    validation.validate_or_raise(config)

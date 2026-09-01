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
            "slope": 1.0e-3,
            "multipole_comp": 1.0e-3,
            "shear_comp": 1.0e-3,
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
    """Reject fisher detection with no modeling.fisher block."""
    config = _load_master_config()
    bad_config = copy.deepcopy(config)
    bad_config["modeling"]["detection"] = "fisher"
    bad_config["modeling"].pop("fisher", None)

    with pytest.raises(ValueError, match="Missing required key 'fisher'"):
        validation.validate_or_raise(bad_config)


def test_modeling_rejects_non_fisher_detection_mode():
    """Reject any modeling.detection value other than 'fisher'."""
    config = _load_master_config()
    config["modeling"]["detection"] = "legacy"

    with pytest.raises(ValueError, match="modeling.detection must be 'fisher'"):
        validation.validate_or_raise(config)


def test_fisher_rejects_invalid_mode():
    """Reject an unrecognised modeling.fisher.mode value."""
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"]["mode"] = "invalid"

    with pytest.raises(ValueError, match="modeling.fisher.mode must be one of"):
        validation.validate_or_raise(config)


def test_fisher_requires_all_finite_diff_fields():
    """Reject a finite_diff block missing a required step field."""
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"]["finite_diff"].pop("centre_arcsec")

    with pytest.raises(ValueError, match="Missing required key 'centre_arcsec'"):
        validation.validate_or_raise(config)


@pytest.mark.parametrize("bad_step", [0.0, -1.0, float("nan"), True])
def test_fisher_rejects_invalid_finite_diff_values(bad_step):
    """Reject non-positive or non-finite finite_diff step sizes."""
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"]["finite_diff"]["ell_comp"] = bad_step

    with pytest.raises(ValueError, match="modeling.fisher.finite_diff.ell_comp"):
        validation.validate_or_raise(config)


@pytest.mark.parametrize("bad_num_angles", [0, -4, 2.5, True])
def test_fisher_rejects_invalid_map_num_angles(bad_num_angles):
    """Reject ring num_angles that is not a positive integer."""
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"]["map"]["ring"]["num_angles"] = bad_num_angles

    with pytest.raises(ValueError, match="modeling.fisher.map.ring.num_angles must be a positive integer"):
        validation.validate_or_raise(config)


def test_fisher_map_requires_type():
    """Reject a map block with no type field."""
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"]["map"].pop("type")

    with pytest.raises(ValueError, match="Missing required key 'type'"):
        validation.validate_or_raise(config)


def test_fisher_map_rejects_unknown_type():
    """Reject a map type outside the supported set."""
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"]["map"]["type"] = "spiral"

    with pytest.raises(ValueError, match="modeling.fisher.map.type must be one of"):
        validation.validate_or_raise(config)


def test_fisher_map_rejects_legacy_flat_keys():
    """Reject legacy flat map keys left beside the nested blocks."""
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"]["map"]["num_angles"] = 24

    with pytest.raises(ValueError, match="modeling.fisher.map contains unsupported keys"):
        validation.validate_or_raise(config)


def test_fisher_map_ring_type_requires_ring_block():
    """Reject map type 'ring' with no ring block."""
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"]["map"].pop("ring")

    with pytest.raises(ValueError, match="Missing required key 'ring'"):
        validation.validate_or_raise(config)


def test_fisher_map_grid_type_requires_grid_block():
    """Reject map type 'grid' with no grid block."""
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"]["map"]["type"] = "grid"
    config["modeling"]["fisher"]["map"].pop("grid")

    with pytest.raises(ValueError, match="Missing required key 'grid'"):
        validation.validate_or_raise(config)


@pytest.mark.parametrize("bad_spacing", [0.0, -0.1, float("nan"), True])
def test_fisher_map_rejects_invalid_grid_spacing(bad_spacing):
    """Reject non-positive or non-finite grid spacing."""
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"]["map"]["grid"]["spacing_arcsec"] = bad_spacing

    with pytest.raises(ValueError, match="modeling.fisher.map.grid.spacing_arcsec"):
        validation.validate_or_raise(config)


def test_fisher_map_rejects_half_width_below_spacing():
    """Reject a grid half-width smaller than the grid spacing."""
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"]["map"]["grid"]["half_width_arcsec"] = 0.01

    with pytest.raises(ValueError, match="half_width_arcsec must be >= spacing_arcsec"):
        validation.validate_or_raise(config)


def test_fisher_map_rejects_inverted_annulus():
    """Reject an annulus whose outer radius is below the inner one."""
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"]["map"]["grid"]["annulus"] = {
        "r_min_arcsec": 1.0,
        "r_max_arcsec": 0.5,
    }

    with pytest.raises(ValueError, match="r_max_arcsec must be > r_min_arcsec"):
        validation.validate_or_raise(config)


def test_fisher_map_rejects_unknown_annulus_key():
    """Reject an annulus block carrying an unsupported key."""
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"]["map"]["grid"]["annulus"] = {
        "r_min_arcsec": 0.5,
        "r_max_arcsec": 1.5,
        "radius": 1.0,
    }

    with pytest.raises(ValueError, match="annulus contains unsupported keys"):
        validation.validate_or_raise(config)


def test_fisher_map_explicit_type_requires_positions():
    """Reject map type 'explicit' with an empty position list."""
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"]["map"]["type"] = "explicit"
    config["modeling"]["fisher"]["map"]["explicit_positions_yx"] = []

    with pytest.raises(ValueError, match="must be non-empty when map.type is 'explicit'"):
        validation.validate_or_raise(config)


@pytest.mark.parametrize("bad_threshold", [0.0, -10.0, float("inf"), True])
def test_fisher_map_rejects_invalid_detection_q_threshold(bad_threshold):
    """Reject a non-positive or non-finite detection q threshold."""
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"]["map"]["detection_q_threshold"] = bad_threshold

    with pytest.raises(ValueError, match="modeling.fisher.map.detection_q_threshold"):
        validation.validate_or_raise(config)


@pytest.mark.parametrize("bad_workers", [0, -2, 1.5, True])
def test_fisher_map_rejects_invalid_num_workers(bad_workers):
    """Reject map num_workers that is not a positive integer."""
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"]["map"]["num_workers"] = bad_workers

    with pytest.raises(ValueError, match="modeling.fisher.map.num_workers must be a positive integer"):
        validation.validate_or_raise(config)


def test_fisher_map_grid_config_passes_validation():
    """Accept a grid map config with a well-formed annulus."""
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"]["map"]["type"] = "grid"
    config["modeling"]["fisher"]["map"]["grid"]["annulus"] = {
        "r_min_arcsec": 0.5,
        "r_max_arcsec": 1.5,
    }
    validation.validate_or_raise(config)


def test_fisher_rejects_invalid_explicit_positions_shape():
    """Reject explicit positions that are not [y, x] pairs."""
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"]["map"]["explicit_positions_yx"] = [[0.1], [0.2, 0.3, 0.4]]

    with pytest.raises(ValueError, match="entries must be \\[y, x\\] pairs"):
        validation.validate_or_raise(config)


def test_fisher_rejects_non_finite_explicit_positions():
    """Reject explicit positions holding non-finite coordinates."""
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"]["map"]["explicit_positions_yx"] = [[0.1, float("inf")]]

    with pytest.raises(ValueError, match="entries must be finite"):
        validation.validate_or_raise(config)


def test_valid_fisher_config_passes_validation():
    """Accept a fully populated, well-formed fisher block."""
    config = _with_valid_fisher_block(_load_master_config())
    validation.validate_or_raise(config)


def test_master_config_fisher_defaults_pass_validation():
    """Accept the fisher defaults shipped in master_config.yaml."""
    validation.validate_or_raise(_load_master_config())


def test_fisher_rejects_unknown_key():
    """Reject an unrecognised key inside modeling.fisher."""
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"]["unexpected_option"] = True

    with pytest.raises(ValueError, match="modeling.fisher contains unsupported keys"):
        validation.validate_or_raise(config)


def test_fisher_rejects_invalid_mask_mode():
    """Reject an unrecognised modeling.fisher.mask_mode value."""
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"]["mask_mode"] = "bad"

    with pytest.raises(ValueError, match="modeling.fisher.mask_mode"):
        validation.validate_or_raise(config)


def test_fisher_accepts_psf_border_mask():
    """Accept the PSF-border mask used for support-matched validation."""
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"]["mask_mode"] = "psf_border"

    validation.validate_or_raise(config)


def test_fisher_accepts_fixed_annulus_mask():
    """Accept the fixed-annulus mask with a well-formed annulus block."""
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"]["mask_mode"] = "fixed_annulus"
    config["modeling"]["fisher"]["mask_annulus"] = {
        "inner_arcsec": 0.6,
        "outer_arcsec": 1.4,
        "centre": "lens",
    }

    validation.validate_or_raise(config)


def test_fisher_fixed_annulus_requires_annulus_block():
    """Reject the fixed-annulus mask with no annulus block."""
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"]["mask_mode"] = "fixed_annulus"

    with pytest.raises(ValueError, match="mask_annulus is required"):
        validation.validate_or_raise(config)


@pytest.mark.parametrize("mask_mode", ["source_snr", "all_pixels", "psf_border"])
def test_fisher_rejects_annulus_block_for_other_mask_modes(mask_mode):
    """Reject an annulus block supplied for a mask mode that ignores it."""
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"]["mask_mode"] = mask_mode
    config["modeling"]["fisher"]["mask_annulus"] = {
        "inner_arcsec": 0.6,
        "outer_arcsec": 1.4,
    }

    with pytest.raises(ValueError, match="mask_annulus is only accepted"):
        validation.validate_or_raise(config)


@pytest.mark.parametrize(
    "annulus, match",
    [
        ({"outer_arcsec": 1.4}, "Missing required key 'inner_arcsec'"),
        ({"inner_arcsec": 0.6}, "Missing required key 'outer_arcsec'"),
        (
            {"inner_arcsec": -0.1, "outer_arcsec": 1.4},
            "inner_arcsec must be non-negative",
        ),
        (
            {"inner_arcsec": float("nan"), "outer_arcsec": 1.4},
            "inner_arcsec must be finite",
        ),
        (
            {"inner_arcsec": True, "outer_arcsec": 1.4},
            "inner_arcsec must be numeric",
        ),
        (
            {"inner_arcsec": 0.6, "outer_arcsec": 0.0},
            "outer_arcsec must be positive",
        ),
        (
            {"inner_arcsec": 0.6, "outer_arcsec": float("inf")},
            "outer_arcsec must be finite",
        ),
        (
            {"inner_arcsec": 1.4, "outer_arcsec": 1.4},
            "outer_arcsec must be > inner_arcsec",
        ),
        (
            {"inner_arcsec": 0.6, "outer_arcsec": 1.4, "centre": "source"},
            "centre must be one of",
        ),
        (
            {"inner_arcsec": 0.6, "outer_arcsec": 1.4, "centre": 3},
            "mask_annulus.centre",
        ),
        (
            {"inner_arcsec": 0.6, "outer_arcsec": 1.4, "radius": 1.0},
            "unsupported keys: radius",
        ),
    ],
)
def test_fisher_rejects_malformed_annulus_block(annulus, match):
    """Reject every malformed fixed-annulus declaration."""
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"]["mask_mode"] = "fixed_annulus"
    config["modeling"]["fisher"]["mask_annulus"] = annulus

    with pytest.raises(ValueError, match=match):
        validation.validate_or_raise(config)


@pytest.mark.parametrize(
    "subset",
    [
        "all",
        "none",
        "lens_only",
        "source_only",
        "lens_and_source",
        ["lens.centre_y", "source.intensity"],
    ],
)
def test_fisher_accepts_valid_nuisance_subsets(subset):
    """Accept every reserved subset word and an explicit name list."""
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"]["nuisance_subset"] = subset

    validation.validate_or_raise(config)


@pytest.mark.parametrize(
    "subset, match",
    [
        ("lens_and_background", "nuisance_subset must be one of"),
        ([], "must be non-empty"),
        (["lens.centre_y", "lens.centre_y"], "duplicate direction"),
        (["lens.centre_y", 3], r"nuisance_subset\[1\]"),
        (["  "], r"nuisance_subset\[0\]"),
        (True, "must be a reserved word"),
        ({"lens": True}, "must be a reserved word"),
    ],
)
def test_fisher_rejects_malformed_nuisance_subset(subset, match):
    """Reject unknown words, empty lists, duplicates and mistyped entries."""
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"]["nuisance_subset"] = subset

    with pytest.raises(ValueError, match=match):
        validation.validate_or_raise(config)


def test_fisher_rejects_nonpositive_psf_mode_step():
    """Reject a non-positive PSF mode finite-difference step."""
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"]["psf_mode_steps"] = {"segment_hexikes": 0.0}

    with pytest.raises(ValueError, match="modeling.fisher.psf_mode_steps"):
        validation.validate_or_raise(config)


def test_fisher_rejects_nonpositive_psf_mode_prior_sigma():
    """Reject a non-positive PSF mode prior sigma."""
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"]["psf_mode_prior_sigmas"] = {"segment_hexikes": 0.0}

    with pytest.raises(ValueError, match="modeling.fisher.psf_mode_prior_sigmas"):
        validation.validate_or_raise(config)


def test_fisher_rejects_legacy_psf_mode_selection_alias():
    """Reject the removed psf_mode_selection alias."""
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"]["psf_mode_selection"] = {"segment_pistons": {"segments": [0]}}

    with pytest.raises(ValueError, match="psf_mode_selection is not supported"):
        validation.validate_or_raise(config)


def test_fisher_rejects_malformed_psf_basis_selector():
    """Reject a psf_basis selector with a non-integer segment id."""
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"]["psf_basis"] = {"segment_pistons": {"segments": ["zero"]}}

    with pytest.raises(ValueError, match="segment id"):
        validation.validate_or_raise(config)


def test_fisher_psf_features_require_psf_basis():
    """Reject PSF nuisance fitting with no psf_basis defined."""
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fisher"]["include_psf_nuisance"] = True
    config["modeling"]["fisher"]["compute_psf_mode_scan"] = False

    with pytest.raises(ValueError, match="psf_basis is required"):
        validation.validate_or_raise(config)


def test_fisher_scan_requires_explicit_scan_selection():
    """Reject a PSF mode scan with no scan_psf_mode_selection."""
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
    """Accept a complete set of PSF nuisance and scan options."""
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


@pytest.mark.parametrize(
    "fit_psf",
    [
        {"mode": "matched"},
        {"mode": "explicit"},
    ],
)
def test_fit_psf_accepts_supported_modes(fit_psf):
    """Accept matched and explicit fit-PSF configurations."""
    config = _with_valid_fisher_block(_load_master_config())
    if fit_psf["mode"] == "explicit":
        fit_psf["psf"] = copy.deepcopy(config["psf"])
    config["modeling"]["fit_psf"] = fit_psf

    validation.validate_or_raise(config)


def _valid_aberrations(config: dict) -> dict:
    """Return a valid bare aberrations block."""
    return copy.deepcopy(config["psf"]["aberrations"])


def _valid_delta() -> dict:
    """Return one complete fit-PSF delta configuration."""
    return {
        "prior_table": "not-checked-at-validation.yaml",
        "amplitude_rms_nm": 0.0,
        "seed": 11,
        "family": "combined",
    }


@pytest.mark.parametrize(
    "map_key,malformed",
    [
        ("segment_pistons", {"0": 1.0}),
        ("segment_tiptilts", {0: [1.0]}),
        ("segment_hexikes", {0: {0: 1.0}}),
        ("global_zernikes", {"4": 1.0}),
    ],
)
def test_psf_aberrations_validate_disabled_present_maps(map_key, malformed):
    """Reject malformed staged coefficients even when their flag is false."""
    config = _with_valid_fisher_block(_load_master_config())
    aberrations = {
        "enable_segment_pistons": False,
        "enable_segment_tiptilts": False,
        "enable_segment_hexikes": False,
        "enable_global_zernikes": False,
        "segment_pistons": {},
        "segment_tiptilts": {},
        "segment_hexikes": {},
        "global_zernikes": {},
    }
    aberrations[map_key] = malformed
    config["psf"]["aberrations"] = aberrations

    with pytest.raises(ValueError, match=rf"psf.aberrations.{map_key}"):
        validation.validate_or_raise(config)


def test_psf_aberrations_accept_disabled_empty_maps():
    """Accept explicitly empty coefficient maps for disabled families."""
    config = _with_valid_fisher_block(_load_master_config())
    config["psf"]["aberrations"] = {
        "enable_segment_pistons": False,
        "enable_segment_tiptilts": False,
        "enable_segment_hexikes": False,
        "enable_global_zernikes": False,
        "segment_pistons": {},
        "segment_tiptilts": {},
        "segment_hexikes": {},
        "global_zernikes": {},
    }

    validation.validate_or_raise(config)


@pytest.mark.parametrize("bad_fit_psf", ["matched", None, []])
def test_fit_psf_rejects_non_dict_block(bad_fit_psf):
    """Reject a non-dictionary modeling.fit_psf block."""
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fit_psf"] = bad_fit_psf

    with pytest.raises(ValueError, match="modeling.fit_psf"):
        validation.validate_or_raise(config)


def test_fit_psf_rejects_unknown_key():
    """Reject an unsupported key inside modeling.fit_psf."""
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fit_psf"] = {"mode": "matched", "extra": True}

    with pytest.raises(ValueError, match="modeling.fit_psf contains unsupported keys"):
        validation.validate_or_raise(config)


def test_fit_psf_requires_mode():
    """Reject a modeling.fit_psf block without a mode."""
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fit_psf"] = {}

    with pytest.raises(ValueError, match="mode.*modeling.fit_psf"):
        validation.validate_or_raise(config)


@pytest.mark.parametrize("bad_mode", ["offset", 1, None])
def test_fit_psf_rejects_invalid_mode(bad_mode):
    """Reject a non-string or unsupported modeling.fit_psf mode."""
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fit_psf"] = {"mode": bad_mode}

    with pytest.raises(ValueError, match="modeling.fit_psf.mode"):
        validation.validate_or_raise(config)


def test_explicit_fit_psf_requires_psf():
    """Reject explicit fit-PSF mode without a PSF block."""
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fit_psf"] = {"mode": "explicit"}

    with pytest.raises(ValueError, match="psf.*modeling.fit_psf"):
        validation.validate_or_raise(config)


def test_matched_fit_psf_rejects_psf():
    """Reject a silently ignored PSF block in matched mode."""
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fit_psf"] = {
        "mode": "matched",
        "psf": copy.deepcopy(config["psf"]),
    }

    with pytest.raises(ValueError, match="modeling.fit_psf.psf"):
        validation.validate_or_raise(config)


def test_explicit_fit_psf_rejects_non_dict_psf():
    """Reject a non-dictionary explicit fit-PSF payload."""
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fit_psf"] = {"mode": "explicit", "psf": []}

    with pytest.raises(ValueError, match="modeling.fit_psf.psf"):
        validation.validate_or_raise(config)


def test_explicit_fit_psf_delegates_inner_validation():
    """Reject an invalid inner PSF block with the full fit-PSF path."""
    config = _with_valid_fisher_block(_load_master_config())
    fit_psf = copy.deepcopy(config["psf"])
    fit_psf["hres_psf"].pop("wavelength")
    config["modeling"]["fit_psf"] = {"mode": "explicit", "psf": fit_psf}

    with pytest.raises(ValueError, match="modeling.fit_psf.psf"):
        validation.validate_or_raise(config)


def test_fit_psf_delta_accepts_complete_block_and_optional_family_default():
    """Accept complete delta blocks with explicit or default family."""
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fit_psf"] = {
        "mode": "delta",
        "delta": _valid_delta(),
    }
    validation.validate_or_raise(config)

    config["modeling"]["fit_psf"]["delta"].pop("family")
    validation.validate_or_raise(config)

    config["modeling"]["fit_psf"]["delta"]["amplitude_rms_nm"] = -0.0
    validation.validate_or_raise(config)


@pytest.mark.parametrize(
    "bad_amplitude",
    [-1.0, float("nan"), float("inf"), True, [0.0]],
)
def test_fit_psf_delta_rejects_invalid_scalar_amplitude(bad_amplitude):
    """Reject negative, non-finite, boolean, and list delta amplitudes."""
    config = _with_valid_fisher_block(_load_master_config())
    delta = _valid_delta()
    delta["amplitude_rms_nm"] = bad_amplitude
    config["modeling"]["fit_psf"] = {"mode": "delta", "delta": delta}

    with pytest.raises(
        ValueError,
        match="modeling.fit_psf.delta.amplitude_rms_nm",
    ):
        validation.validate_or_raise(config)


@pytest.mark.parametrize("bad_seed", [-1, 1.0, True])
def test_fit_psf_delta_rejects_invalid_seed(bad_seed):
    """Reject delta seeds that are not non-negative integer scalars."""
    config = _with_valid_fisher_block(_load_master_config())
    delta = _valid_delta()
    delta["seed"] = bad_seed
    config["modeling"]["fit_psf"] = {"mode": "delta", "delta": delta}

    with pytest.raises(ValueError, match="modeling.fit_psf.delta.seed"):
        validation.validate_or_raise(config)


@pytest.mark.parametrize("bad_family", ["all", "", 1, None])
def test_fit_psf_delta_rejects_invalid_family(bad_family):
    """Reject delta families outside the supported enum."""
    config = _with_valid_fisher_block(_load_master_config())
    delta = _valid_delta()
    delta["family"] = bad_family
    config["modeling"]["fit_psf"] = {"mode": "delta", "delta": delta}

    with pytest.raises(ValueError, match="modeling.fit_psf.delta.family"):
        validation.validate_or_raise(config)


@pytest.mark.parametrize(
    "missing_key",
    ["prior_table", "amplitude_rms_nm", "seed"],
)
def test_fit_psf_delta_requires_generation_fields(missing_key):
    """Require every delta generation input except optional family."""
    config = _with_valid_fisher_block(_load_master_config())
    delta = _valid_delta()
    delta.pop(missing_key)
    config["modeling"]["fit_psf"] = {"mode": "delta", "delta": delta}

    with pytest.raises(ValueError, match=missing_key):
        validation.validate_or_raise(config)


@pytest.mark.parametrize("bad_prior", [None, 4, []])
def test_fit_psf_delta_rejects_non_string_prior_table(bad_prior):
    """Require a string delta prior-table path."""
    config = _with_valid_fisher_block(_load_master_config())
    delta = _valid_delta()
    delta["prior_table"] = bad_prior
    config["modeling"]["fit_psf"] = {"mode": "delta", "delta": delta}

    with pytest.raises(ValueError, match="modeling.fit_psf.delta.prior_table"):
        validation.validate_or_raise(config)


def test_fit_psf_delta_rejects_unknown_key():
    """Reject unsupported keys in the strict delta block."""
    config = _with_valid_fisher_block(_load_master_config())
    delta = _valid_delta()
    delta["replicates"] = 2
    config["modeling"]["fit_psf"] = {"mode": "delta", "delta": delta}

    with pytest.raises(ValueError, match="modeling.fit_psf.delta"):
        validation.validate_or_raise(config)


@pytest.mark.parametrize("mode,payload", [
    ("matched", "delta"),
    ("matched", "psf"),
    ("explicit", "delta"),
    ("delta", "psf"),
])
def test_fit_psf_modes_reject_mutually_exclusive_payloads(mode, payload):
    """Reject every payload that is incompatible with the selected mode."""
    config = _with_valid_fisher_block(_load_master_config())
    fit_psf = {"mode": mode}
    if mode == "explicit":
        fit_psf["psf"] = copy.deepcopy(config["psf"])
    elif mode == "delta":
        fit_psf["delta"] = _valid_delta()
    if payload == "psf":
        fit_psf["psf"] = copy.deepcopy(config["psf"])
    else:
        fit_psf["delta"] = _valid_delta()
    config["modeling"]["fit_psf"] = fit_psf

    with pytest.raises(ValueError, match=f"modeling.fit_psf.{payload}"):
        validation.validate_or_raise(config)


def test_fit_psf_delta_requires_delta_block():
    """Reject delta mode without its delta block."""
    config = _with_valid_fisher_block(_load_master_config())
    config["modeling"]["fit_psf"] = {"mode": "delta"}

    with pytest.raises(ValueError, match="delta.*modeling.fit_psf"):
        validation.validate_or_raise(config)

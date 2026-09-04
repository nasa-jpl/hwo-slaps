"""Loader and validation for the canonical DesignFreeze artifact.

``configs/design/design_freeze_v1.yaml`` is the single machine-readable
design authority required by Sol Pro go-time finding P0-4. Everything
that generates or consumes a production campaign reads it through this
module rather than restating a value of its own.

The loader is fail-closed. It requires every block the campaign depends
on, cross-checks the pinned module constants against the modules that
declare them, and verifies the digests of the artifacts the freeze binds
by hash: the observing reference, the pre-registration document and the
five template assets. The verification runs inside `load_design_freeze`
itself, so no caller can hold a validated freeze whose bound bytes have
moved without having asked for that explicitly.

The freeze file's own digest is deliberately NOT pinned in code. The
freeze is expected to be amended at ratification, and a self-pinned hash
would turn every legitimate amendment into a test failure. What is
pinned is the rule: the digest travels in the campaign manifest and in
every artifact built from it, so any change of the design is visible in
the provenance chain even though it is not frozen in a test.
"""

from __future__ import annotations

import datetime
import hashlib
import math
from pathlib import Path
from typing import Any, Optional

import yaml


__all__ = [
    "DESIGN_FREEZE_SCHEMA_VERSION",
    "DEFAULT_DESIGN_FREEZE_PATH",
    "REQUIRED_BLOCKS",
    "REQUIRED_PROVISIONAL_ITEMS",
    "DesignFreezeError",
    "design_freeze_digest",
    "file_sha256",
    "load_design_freeze",
    "repo_root",
    "template_levels",
    "validate_design_freeze",
    "verify_bound_artifacts",
]


DESIGN_FREEZE_SCHEMA_VERSION = 1
"""Supported ``schema_version`` of the freeze document (`int`)."""

REQUIRED_BLOCKS = (
    "freeze",
    "provisional_items",
    "foreground_free_ceiling",
    "claim_labels",
    "stage0",
    "strata",
    "observing",
    "templates",
    "selection",
    "aperture",
    "grid_sizing",
    "mass_ladder",
    "seeds",
    "derived",
    "reporting",
    "parent_design_source",
    "parent_design",
    "psf_knowledge_error",
    "nonlinear_validation",
)
"""Top-level blocks every freeze document must carry (`tuple` of `str`)."""

REQUIRED_NONLINEAR_VALIDATION_KEYS = (
    "declared",
    "arms",
    "injection_rule",
    "fit",
    "seeds",
    "smoke_gate",
    "success_criteria",
    "member_sets",
    "campaigns",
)
"""Keys the nonlinear-validation block must carry (`tuple`)."""

REQUIRED_NONLINEAR_FIT_KEYS = (
    "kernel_shape_native",
    "fit_psf",
    "n_live_smooth",
    "n_live_subhalo_search",
    "n_live_subhalo_fixed",
    "maxcall",
    "jax_n_batch",
    "number_of_cores",
    "log10_m200_range",
    "nautilus_training_workers",
)
"""Fit settings the nonlinear-validation protocol must declare (`tuple`)."""

REQUIRED_PROVISIONAL_ITEMS = ()
"""Identifiers of the items awaiting ratification (`tuple` of `str`).

Empty since the 2026-08-23 ratification: every item the version-1
freeze listed was ruled by George that evening and the rulings are
recorded in the freeze's ``ratifications`` block. A freeze document
naming any provisional item now fails validation, because nothing is
open.
"""


class DesignFreezeError(ValueError):
    """Raised for any design-freeze validation or verification failure."""


def repo_root() -> Path:
    """Return the repository root.

    Returns
    -------
    root : `pathlib.Path`
        Directory holding ``configs`` and ``src``.
    """
    return Path(__file__).resolve().parents[3]


DEFAULT_DESIGN_FREEZE_PATH = repo_root()/"configs"/"design"/"design_freeze_v1.yaml"
"""Committed freeze artifact (`pathlib.Path`)."""


def file_sha256(path) -> str:
    """Return the full SHA-256 hex digest of one file.

    Parameters
    ----------
    path : path-like
        File to digest.

    Returns
    -------
    digest : `str`
        Sixty-four hexadecimal characters.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_mapping(value: Any, path: str) -> dict:
    """Return a mapping value or raise naming its path."""
    if not isinstance(value, dict):
        raise DesignFreezeError(f"{path} must be a mapping")
    return value


def _required(mapping: dict, key: str, path: str) -> Any:
    """Return a required member or raise naming its path."""
    if key not in mapping:
        raise DesignFreezeError(f"Missing required key '{key}' in {path}")
    return mapping[key]


def _require_positive_int(value: Any, path: str) -> int:
    """Return a strictly positive integer or raise."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise DesignFreezeError(f"{path} must be a positive integer, got {value!r}")
    return int(value)


def _require_positive_float(value: Any, path: str) -> float:
    """Return a strictly positive finite float or raise."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DesignFreezeError(f"{path} must be a number, got {value!r}")
    number = float(value)
    if not number > 0.0 or number != number or number == float("inf"):
        raise DesignFreezeError(f"{path} must be positive and finite, got {value!r}")
    return number


def _require_nonnegative_float(value: Any, path: str) -> float:
    """Return a non-negative finite float or raise."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DesignFreezeError(f"{path} must be a number, got {value!r}")
    number = float(value)
    if not number >= 0.0 or number == float("inf"):
        raise DesignFreezeError(
            f"{path} must be non-negative and finite, got {value!r}"
        )
    return number


def _require_iso_date(value: Any, path: str) -> str:
    """Require an ISO calendar date string and return it."""
    if not isinstance(value, str) or not value.strip():
        raise DesignFreezeError(f"{path} must be a non-empty ISO date string")
    try:
        datetime.date.fromisoformat(value.strip())
    except ValueError as exc:
        raise DesignFreezeError(
            f"{path} must be an ISO calendar date, got {value!r}"
        ) from exc
    return value


def _require_sha256(value: Any, path: str) -> str:
    """Return a full hexadecimal SHA-256 digest string or raise."""
    if not isinstance(value, str) or len(value) != 64:
        raise DesignFreezeError(f"{path} must be a 64-character sha256 digest")
    if any(character not in "0123456789abcdef" for character in value):
        raise DesignFreezeError(f"{path} must be lowercase hexadecimal")
    return value


def _validate_module_constants(freeze: dict) -> None:
    """Cross-check the pinned constants against the modules that own them."""
    from hwoslaps.analysis import selection_score
    from hwoslaps.lensing import critical_curve

    selection = freeze["selection"]
    constants = _require_mapping(
        _required(selection, "module_constants", "selection"),
        "selection.module_constants",
    )
    expected = {
        "FLOOR_THETA_E_ARCSEC": selection_score.FLOOR_THETA_E_ARCSEC,
        "FLOOR_ARC_SNR": selection_score.FLOOR_ARC_SNR,
        "SELECTED_TIER_SIZE": selection_score.SELECTED_TIER_SIZE,
        "GOLDEN_TIER_SIZE": selection_score.GOLDEN_TIER_SIZE,
        "APERTURE_THETA_E_MULTIPLE": selection_score.APERTURE_THETA_E_MULTIPLE,
    }
    for name, module_value in expected.items():
        frozen = _required(constants, name, "selection.module_constants")
        if float(frozen) != float(module_value):
            raise DesignFreezeError(
                f"selection.module_constants.{name} is {frozen} but "
                f"hwoslaps.analysis.selection_score.{name} is {module_value}"
            )

    aperture = freeze["aperture"]
    factor = _require_positive_float(
        _required(aperture, "theta_e_factor", "aperture"),
        "aperture.theta_e_factor",
    )
    for module_name, module_value in (
        (
            "hwoslaps.lensing.critical_curve.DEFAULT_APERTURE_THETA_E_FACTOR",
            critical_curve.DEFAULT_APERTURE_THETA_E_FACTOR,
        ),
        (
            "hwoslaps.analysis.selection_score.APERTURE_THETA_E_MULTIPLE",
            selection_score.APERTURE_THETA_E_MULTIPLE,
        ),
    ):
        if float(module_value) != factor:
            raise DesignFreezeError(
                f"aperture.theta_e_factor is {factor} but {module_name} is "
                f"{module_value}; the freeze is the single source of truth"
            )

    margin = _required(aperture, "computational_margin_fraction", "aperture")
    if float(margin) != float(
        critical_curve.DEFAULT_COMPUTATIONAL_MARGIN_FRACTION
    ):
        raise DesignFreezeError(
            f"aperture.computational_margin_fraction is {margin} but "
            "hwoslaps.lensing.critical_curve."
            "DEFAULT_COMPUTATIONAL_MARGIN_FRACTION is "
            f"{critical_curve.DEFAULT_COMPUTATIONAL_MARGIN_FRACTION}"
        )

    algorithm = _require_mapping(
        _required(aperture, "theta_e_algorithm", "aperture"),
        "aperture.theta_e_algorithm",
    )
    for key, module_value in (
        ("algorithm_id", critical_curve.ALGORITHM_ID),
        ("choice_rule_id", critical_curve.CHOICE_RULE_ID),
    ):
        frozen = _required(algorithm, key, "aperture.theta_e_algorithm")
        if frozen != module_value:
            raise DesignFreezeError(
                f"aperture.theta_e_algorithm.{key} is {frozen!r} but the "
                f"module declares {module_value!r}"
            )


def _validate_extraction_settings(freeze: dict) -> None:
    """Validate the frozen ``theta_E`` extraction grid and contour guards.

    Every Stage 0 job configuration carries these settings and the
    runner consumes exactly them, so they are part of the design rather
    than a module default a runner may inherit. The values themselves
    are the freeze's to choose; what is checked is that each one exists
    and is usable by the extraction.
    """
    aperture = _require_mapping(freeze["aperture"], "aperture")
    algorithm = _require_mapping(
        _required(aperture, "theta_e_algorithm", "aperture"),
        "aperture.theta_e_algorithm",
    )
    grid_path = "aperture.theta_e_algorithm.extraction_grid"
    grid = _require_mapping(
        _required(algorithm, "extraction_grid", "aperture.theta_e_algorithm"),
        grid_path,
    )
    _require_positive_float(
        _required(grid, "pixel_scale_arcsec", grid_path),
        f"{grid_path}.pixel_scale_arcsec",
    )
    _require_positive_float(
        _required(grid, "half_width_factor", grid_path),
        f"{grid_path}.half_width_factor",
    )
    guards_path = "aperture.theta_e_algorithm.guards"
    guards = _require_mapping(
        _required(algorithm, "guards", "aperture.theta_e_algorithm"), guards_path
    )
    _require_positive_float(
        _required(guards, "closure_tolerance_pixels", guards_path),
        f"{guards_path}.closure_tolerance_pixels",
    )
    _require_nonnegative_float(
        _required(guards, "border_margin_pixels", guards_path),
        f"{guards_path}.border_margin_pixels",
    )
    vertices = _require_positive_int(
        _required(guards, "min_contour_vertices", guards_path),
        f"{guards_path}.min_contour_vertices",
    )
    if vertices < 4:
        raise DesignFreezeError(
            f"{guards_path}.min_contour_vertices is {vertices}; the extraction "
            "rejects anything below 4 because a polygon needs three vertices "
            "and a repeated closing one"
        )


def _validate_templates(freeze: dict) -> None:
    """Validate the template bank block for shape and self-consistency."""
    templates = _require_mapping(freeze["templates"], "templates")
    count = _require_positive_int(
        _required(templates, "count", "templates"), "templates.count"
    )
    per_level = _require_positive_int(
        _required(templates, "per_level", "templates"), "templates.per_level"
    )
    _require_positive_float(
        _required(
            templates, "rate_contract_production_tolerance", "templates"
        ),
        "templates.rate_contract_production_tolerance",
    )
    levels = _required(templates, "levels", "templates")
    if not isinstance(levels, list) or len(levels) != count:
        raise DesignFreezeError(
            f"templates.levels must list exactly templates.count = {count} entries"
        )
    identifiers = []
    for index, level in enumerate(levels):
        path = f"templates.levels[{index}]"
        block = _require_mapping(level, path)
        identifiers.append(str(_required(block, "id", path)))
        _required(block, "morphology_class", path)
        _required(block, "asset_path", path)
        _require_sha256(_required(block, "sha256", path), f"{path}.sha256")
        _require_positive_float(
            _required(block, "canonical_total_flux", path),
            f"{path}.canonical_total_flux",
        )
        _require_positive_float(
            _required(block, "asset_pixel_scale_arcsec", path),
            f"{path}.asset_pixel_scale_arcsec",
        )
    if len(set(identifiers)) != len(identifiers):
        raise DesignFreezeError("templates.levels ids must be unique")
    n_systems = _require_positive_int(
        _required(freeze["stage0"], "n_systems", "stage0"), "stage0.n_systems"
    )
    if per_level*count != n_systems:
        raise DesignFreezeError(
            f"templates.per_level {per_level} times templates.count {count} is "
            f"{per_level*count}, which does not fill stage0.n_systems {n_systems}"
        )
    design_levels = list(
        freeze["parent_design"]["distributions"]["source_template"]["levels"]
    )
    if design_levels != identifiers:
        raise DesignFreezeError(
            "templates.levels ids do not match the embedded parent design "
            f"levels: freeze {identifiers}, design {design_levels}"
        )


def _validate_seeds(freeze: dict) -> None:
    """Validate the declared seed streams against the parent design."""
    seeds = _require_mapping(freeze["seeds"], "seeds")
    entropy = _required(seeds, "entropy", "seeds")
    if isinstance(entropy, bool) or not isinstance(entropy, int):
        raise DesignFreezeError("seeds.entropy must be an integer")
    streams = _require_mapping(_required(seeds, "streams", "seeds"), "seeds.streams")
    for name in (
        "parent_design",
        "primary_noise",
        "rank_stability_noise",
        "template_permutation",
        "bootstrap",
        "null_noise",
    ):
        block = _require_mapping(
            _required(streams, name, "seeds.streams"), f"seeds.streams.{name}"
        )
        key = _required(block, "spawn_key", f"seeds.streams.{name}")
        if not isinstance(key, list) or not key:
            raise DesignFreezeError(
                f"seeds.streams.{name}.spawn_key must be a non-empty list"
            )
    replicates = _require_positive_int(
        _required(
            streams["rank_stability_noise"],
            "replicates",
            "seeds.streams.rank_stability_noise",
        ),
        "seeds.streams.rank_stability_noise.replicates",
    )
    indices = _required(
        streams["rank_stability_noise"],
        "replicate_indices",
        "seeds.streams.rank_stability_noise",
    )
    if list(indices) != list(range(replicates)):
        raise DesignFreezeError(
            "seeds.streams.rank_stability_noise.replicate_indices must list "
            f"0 .. {replicates - 1} explicitly"
        )
    null_noise = _require_mapping(
        streams["null_noise"], "seeds.streams.null_noise"
    )
    null_spawn_key = _required(
        null_noise, "spawn_key", "seeds.streams.null_noise"
    )
    if null_spawn_key != [6]:
        raise DesignFreezeError(
            "seeds.streams.null_noise.spawn_key must be exactly [6]"
        )
    null_replicates = _require_positive_int(
        _required(null_noise, "replicates", "seeds.streams.null_noise"),
        "seeds.streams.null_noise.replicates",
    )
    null_indices = _required(
        null_noise, "replicate_indices", "seeds.streams.null_noise"
    )
    if not isinstance(null_indices, list):
        raise DesignFreezeError(
            "seeds.streams.null_noise.replicate_indices must be a list"
        )
    if any(
        isinstance(index, bool) or not isinstance(index, int)
        for index in null_indices
    ):
        raise DesignFreezeError(
            "seeds.streams.null_noise.replicate_indices must contain integers"
        )
    if null_indices != list(range(1, null_replicates + 1)):
        raise DesignFreezeError(
            "seeds.streams.null_noise.replicate_indices must list "
            f"1 .. {null_replicates} explicitly"
        )
    direction_stream = _require_mapping(
        _required(
            streams,
            "psf_knowledge_direction",
            "seeds.streams",
        ),
        "seeds.streams.psf_knowledge_direction",
    )
    if _required(
        direction_stream,
        "spawn_key",
        "seeds.streams.psf_knowledge_direction",
    ) != [7]:
        raise DesignFreezeError(
            "seeds.streams.psf_knowledge_direction.spawn_key must be exactly [7]"
        )
    directions = _require_positive_int(
        _required(
            direction_stream,
            "directions",
            "seeds.streams.psf_knowledge_direction",
        ),
        "seeds.streams.psf_knowledge_direction.directions",
    )
    direction_indices = _required(
        direction_stream,
        "direction_indices",
        "seeds.streams.psf_knowledge_direction",
    )
    if (
        not isinstance(direction_indices, list)
        or any(
            isinstance(direction, bool) or not isinstance(direction, int)
            for direction in direction_indices
        )
        or direction_indices != list(range(1, directions + 1))
    ):
        raise DesignFreezeError(
            "seeds.streams.psf_knowledge_direction.direction_indices must "
            f"list 1 .. {directions} explicitly"
        )
    order = _required(seeds, "draw_order", "seeds")
    design_order = list(freeze["parent_design"]["seeds"]["draw_order"])
    if list(order) != design_order:
        raise DesignFreezeError(
            "seeds.draw_order does not match the embedded parent design draw "
            f"order: freeze {list(order)}, design {design_order}"
        )
    design_entropy = freeze["parent_design"]["seeds"]["entropy"]
    if int(design_entropy) != int(entropy):
        raise DesignFreezeError(
            f"seeds.entropy {entropy} does not match the embedded parent "
            f"design entropy {design_entropy}"
        )


def _validate_psf_knowledge_error(freeze: dict) -> None:
    """Validate the frozen PSF knowledge-error protocol block.

    Parameters
    ----------
    freeze : `dict`
        Freeze document whose PSF knowledge-error block is checked.

    Raises
    ------
    DesignFreezeError
        Raised when the residual rungs, direction stream, prior binding,
        gates, ratio floor, member set or Fisher campaign is malformed.
    """
    block = _require_mapping(
        freeze["psf_knowledge_error"], "psf_knowledge_error"
    )
    for key in (
        "declared_v5",
        "rulings_v5",
        "purpose",
        "truth_state",
        "residual_model",
        "member_set",
        "mass_rungs_rule",
        "estimands",
        "reporting",
        "success_criteria",
        "campaigns",
        "kernel_shape_native",
    ):
        _required(block, key, "psf_knowledge_error")
    if block["truth_state"] != "science35":
        raise DesignFreezeError(
            "psf_knowledge_error.truth_state must be 'science35'"
        )
    if block["rulings_v5"] != [f"D-K{index}" for index in range(1, 11)]:
        raise DesignFreezeError(
            "psf_knowledge_error.rulings_v5 must list D-K1 through D-K10"
        )
    _require_iso_date(block["declared_v5"], "psf_knowledge_error.declared_v5")
    criteria = block["success_criteria"]
    if (
        not isinstance(criteria, list)
        or not criteria
        or any(
            not isinstance(item, str) or not item.strip() for item in criteria
        )
    ):
        raise DesignFreezeError(
            "psf_knowledge_error.success_criteria must be a non-empty list "
            "of non-empty strings"
        )
    amendment = freeze["freeze"].get("amendment_v5")
    if not isinstance(amendment, str) or not amendment.strip():
        raise DesignFreezeError(
            "freeze.amendment_v5 must be a non-empty paragraph for a "
            "version-5 freeze"
        )

    residual = _require_mapping(
        block["residual_model"], "psf_knowledge_error.residual_model"
    )
    rungs = _required(
        residual,
        "amplitude_rms_nm_rungs",
        "psf_knowledge_error.residual_model",
    )
    if not isinstance(rungs, list) or not rungs:
        raise DesignFreezeError(
            "psf_knowledge_error.residual_model.amplitude_rms_nm_rungs "
            "must be a non-empty list"
        )
    rung_values = []
    for index, value in enumerate(rungs):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise DesignFreezeError(
                "psf_knowledge_error.residual_model."
                f"amplitude_rms_nm_rungs[{index}] must be a finite number"
            )
        number = float(value)
        if not math.isfinite(number):
            raise DesignFreezeError(
                "psf_knowledge_error.residual_model."
                f"amplitude_rms_nm_rungs[{index}] must be finite"
            )
        if index and number <= rung_values[-1]:
            raise DesignFreezeError(
                "psf_knowledge_error.residual_model."
                "amplitude_rms_nm_rungs must be strictly increasing"
            )
        rung_values.append(number)
    if rung_values[0] != 0.0:
        raise DesignFreezeError(
            "psf_knowledge_error.residual_model."
            "amplitude_rms_nm_rungs must start at 0.0"
        )
    endpoint = _require_nonnegative_float(
        _required(
            residual,
            "endpoint_anchor_nm",
            "psf_knowledge_error.residual_model",
        ),
        "psf_knowledge_error.residual_model.endpoint_anchor_nm",
    )
    if endpoint != rung_values[-1]:
        raise DesignFreezeError(
            "psf_knowledge_error.residual_model.endpoint_anchor_nm must "
            "equal the last amplitude RMS rung"
        )
    if residual.get("mode") != "delta":
        raise DesignFreezeError(
            "psf_knowledge_error.residual_model.mode must be 'delta'"
        )
    if residual.get("family") != "combined":
        raise DesignFreezeError(
            "psf_knowledge_error.residual_model.family must be 'combined'"
        )
    directions = _require_positive_int(
        _required(
            residual,
            "directions",
            "psf_knowledge_error.residual_model",
        ),
        "psf_knowledge_error.residual_model.directions",
    )
    direction_indices = _required(
        residual,
        "direction_indices",
        "psf_knowledge_error.residual_model",
    )
    if (
        not isinstance(direction_indices, list)
        or any(
            isinstance(direction, bool) or not isinstance(direction, int)
            for direction in direction_indices
        )
        or direction_indices != list(range(1, directions + 1))
    ):
        raise DesignFreezeError(
            "psf_knowledge_error.residual_model.direction_indices must "
            f"list 1 .. {directions} explicitly"
        )

    prior_path = _required(
        residual,
        "prior_table",
        "psf_knowledge_error.residual_model",
    )
    if not isinstance(prior_path, str) or not prior_path:
        raise DesignFreezeError(
            "psf_knowledge_error.residual_model.prior_table must be a "
            "non-empty repo-relative path"
        )
    root = repo_root().resolve()
    resolved_prior = (root/prior_path).resolve()
    try:
        resolved_prior.relative_to(root)
    except ValueError as exc:
        raise DesignFreezeError(
            "psf_knowledge_error.residual_model.prior_table must be under "
            f"the repository root, got {prior_path!r}"
        ) from exc
    if not resolved_prior.is_file():
        raise DesignFreezeError(
            "psf_knowledge_error residual prior table does not exist: "
            f"{resolved_prior}"
        )
    prior_digest = _require_sha256(
        _required(
            residual,
            "prior_table_sha256",
            "psf_knowledge_error.residual_model",
        ),
        "psf_knowledge_error.residual_model.prior_table_sha256",
    )
    observed_digest = file_sha256(resolved_prior)
    if observed_digest != prior_digest:
        raise DesignFreezeError(
            "psf_knowledge_error residual prior table sha256 "
            f"{observed_digest} does not match bound {prior_digest}"
        )

    gates = _require_mapping(
        _required(block, "gates", "psf_knowledge_error"),
        "psf_knowledge_error.gates",
    )
    retention = _require_positive_float(
        _required(gates, "retention_q10_min", "psf_knowledge_error.gates"),
        "psf_knowledge_error.gates.retention_q10_min",
    )
    spurious = _require_positive_float(
        _required(gates, "spurious_q90_max", "psf_knowledge_error.gates"),
        "psf_knowledge_error.gates.spurious_q90_max",
    )
    if retention >= 1.0 or spurious >= 1.0:
        raise DesignFreezeError(
            "psf_knowledge_error.gates values must lie strictly between 0 "
            "and 1"
        )
    if retention <= 0.5:
        raise DesignFreezeError(
            "psf_knowledge_error.gates.retention_q10_min must be greater "
            "than 0.5"
        )
    if spurious >= 0.5:
        raise DesignFreezeError(
            "psf_knowledge_error.gates.spurious_q90_max must be less than "
            "0.5"
        )
    sensitivity = _required(gates, "sensitivity", "psf_knowledge_error.gates")
    if not isinstance(sensitivity, list) or not sensitivity:
        raise DesignFreezeError(
            "psf_knowledge_error.gates.sensitivity must be a non-empty list"
        )
    for index, pair in enumerate(sensitivity):
        if not isinstance(pair, list) or len(pair) != 2:
            raise DesignFreezeError(
                "psf_knowledge_error.gates.sensitivity["
                f"{index}] must be a two-element list"
            )
        pair_retention = _require_positive_float(
            pair[0], f"psf_knowledge_error.gates.sensitivity[{index}][0]"
        )
        pair_spurious = _require_positive_float(
            pair[1], f"psf_knowledge_error.gates.sensitivity[{index}][1]"
        )
        if pair_retention >= 1.0 or pair_spurious >= 1.0:
            raise DesignFreezeError(
                "psf_knowledge_error.gates.sensitivity values must lie "
                "strictly between 0 and 1"
            )
        if pair_retention <= 0.5 or pair_spurious >= 0.5:
            raise DesignFreezeError(
                "psf_knowledge_error.gates.sensitivity must use retention "
                "above 0.5 and spurious below 0.5"
            )

    floor = _require_mapping(
        _required(block, "ratio_floor", "psf_knowledge_error"),
        "psf_knowledge_error.ratio_floor",
    )
    floor_cells = _require_positive_int(
        _required(floor, "cells", "psf_knowledge_error.ratio_floor"),
        "psf_knowledge_error.ratio_floor.cells",
    )
    floor_area = _require_positive_float(
        _required(floor, "arcsec2", "psf_knowledge_error.ratio_floor"),
        "psf_knowledge_error.ratio_floor.arcsec2",
    )
    if abs(floor_area - floor_cells*0.0025) > 1.0e-12:
        raise DesignFreezeError(
            "psf_knowledge_error.ratio_floor.arcsec2 must equal cells "
            "times 0.0025 to 1e-12"
        )
    _required(floor, "rule", "psf_knowledge_error.ratio_floor")

    member_set = _require_mapping(
        _required(block, "member_set", "psf_knowledge_error"),
        "psf_knowledge_error.member_set",
    )
    for key in ("name", "source", "source_campaign_uuid", "n_systems", "tier"):
        _required(member_set, key, "psf_knowledge_error.member_set")
    if member_set["tier"] != "selected":
        raise DesignFreezeError(
            "psf_knowledge_error.member_set.tier must be 'selected'"
        )
    member_count = _require_positive_int(
        member_set["n_systems"], "psf_knowledge_error.member_set.n_systems"
    )
    selected_count = _require_positive_int(
        freeze["strata"]["selected"]["size"], "strata.selected.size"
    )
    if member_count != selected_count:
        raise DesignFreezeError(
            "psf_knowledge_error.member_set.n_systems must equal "
            f"strata.selected.size {selected_count}"
        )

    campaigns = _require_mapping(
        _required(block, "campaigns", "psf_knowledge_error"),
        "psf_knowledge_error.campaigns",
    )
    campaign = _require_mapping(
        _required(
            campaigns,
            "psf_knowledge_fisher_v1",
            "psf_knowledge_error.campaigns",
        ),
        "psf_knowledge_error.campaigns.psf_knowledge_fisher_v1",
    )
    campaign_path = "psf_knowledge_error.campaigns.psf_knowledge_fisher_v1"
    if campaign.get("block") != "psf_knowledge_error":
        raise DesignFreezeError(
            f"{campaign_path}.block must be 'psf_knowledge_error'"
        )
    if campaign.get("member_set") != member_set["name"]:
        raise DesignFreezeError(
            f"{campaign_path}.member_set must equal "
            "psf_knowledge_error.member_set.name"
        )
    if campaign.get("phases") != ["maps"]:
        raise DesignFreezeError(f"{campaign_path}.phases must be exactly ['maps']")
    smoke = _require_mapping(
        _required(campaign, "smoke_rule", campaign_path),
        f"{campaign_path}.smoke_rule",
    )
    smoke_deltas = _required(smoke, "deltas", f"{campaign_path}.smoke_rule")
    if not isinstance(smoke_deltas, list):
        raise DesignFreezeError(f"{campaign_path}.smoke_rule.deltas must be a list")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in smoke_deltas
    ):
        raise DesignFreezeError(
            f"{campaign_path}.smoke_rule.deltas must contain numbers"
        )
    smoke_values = [float(value) for value in smoke_deltas]
    if any(value not in rung_values for value in smoke_values):
        raise DesignFreezeError(
            f"{campaign_path}.smoke_rule.deltas must be a subset of the "
            "declared amplitude RMS rungs"
        )
    if 0.0 not in smoke_values or not any(value > 0.0 for value in smoke_values):
        raise DesignFreezeError(
            f"{campaign_path}.smoke_rule.deltas must contain 0 and a positive rung"
        )
    smoke_direction = _required(smoke, "direction", f"{campaign_path}.smoke_rule")
    if (
        isinstance(smoke_direction, bool)
        or not isinstance(smoke_direction, int)
        or smoke_direction not in direction_indices
    ):
        raise DesignFreezeError(
            f"{campaign_path}.smoke_rule.direction must be one of the "
            "declared direction indices"
        )
    smoke_members = _required(smoke, "members", f"{campaign_path}.smoke_rule")
    if smoke_members != ["smallest_image", "largest_image"]:
        raise DesignFreezeError(
            f"{campaign_path}.smoke_rule.members must be exactly "
            "['smallest_image', 'largest_image']"
        )
    kernel_shape = _required(block, "kernel_shape_native", "psf_knowledge_error")
    if kernel_shape != [999, 999]:
        raise DesignFreezeError(
            "psf_knowledge_error.kernel_shape_native must be exactly [999, 999]"
        )


def validate_design_freeze(document: dict) -> dict:
    """Validate one loaded freeze document.

    Parameters
    ----------
    document : `dict`
        Parsed contents of a design freeze YAML file.

    Returns
    -------
    freeze : `dict`
        The same mapping, once every required block has been checked.

    Raises
    ------
    DesignFreezeError
        Raised for an unsupported schema version, a missing block, a
        malformed digest, a template bank that does not fill the pool, a
        seed declaration that disagrees with the embedded parent design,
        a provisional-item list that is not exactly the declared set, an
        unusable ``theta_E`` extraction grid or contour guard, or a
        pinned constant that disagrees with the module owning it.
    """
    freeze = _require_mapping(document, "design freeze")
    version = _required(freeze, "schema_version", "design freeze")
    if version != DESIGN_FREEZE_SCHEMA_VERSION:
        raise DesignFreezeError(
            f"schema_version {version!r} is not the supported version "
            f"{DESIGN_FREEZE_SCHEMA_VERSION}"
        )
    missing = [block for block in REQUIRED_BLOCKS if block not in freeze]
    if missing:
        raise DesignFreezeError(
            "design freeze is missing required blocks: " + ", ".join(missing)
        )
    if freeze["foreground_free_ceiling"] is not True:
        raise DesignFreezeError(
            "foreground_free_ceiling must be true: D-F6 is ceiling-only and "
            "every central result carries the source-only ceiling label"
        )

    items = freeze["provisional_items"]
    if not isinstance(items, list):
        raise DesignFreezeError("provisional_items must be a list")
    identifiers = tuple(
        str(_required(_require_mapping(item, "provisional_items entry"), "id", "item"))
        for item in items
    )
    if identifiers != REQUIRED_PROVISIONAL_ITEMS:
        raise DesignFreezeError(
            "provisional_items must name exactly "
            f"{list(REQUIRED_PROVISIONAL_ITEMS)}, got {list(identifiers)}"
        )

    stage0 = _require_mapping(freeze["stage0"], "stage0")
    _require_positive_int(
        _required(stage0, "n_systems", "stage0"), "stage0.n_systems"
    )
    _require_positive_float(
        _required(stage0, "pixel_scale_arcsec", "stage0"),
        "stage0.pixel_scale_arcsec",
    )
    _require_positive_float(
        _required(stage0, "exposure_time_s", "stage0"), "stage0.exposure_time_s"
    )

    strata = _require_mapping(freeze["strata"], "strata")
    sizes = {}
    for tier in ("parent", "selected", "golden"):
        block = _require_mapping(_required(strata, tier, "strata"), f"strata.{tier}")
        sizes[tier] = _require_positive_int(
            _required(block, "size", f"strata.{tier}"), f"strata.{tier}.size"
        )
    if not sizes["golden"] <= sizes["selected"] <= sizes["parent"]:
        raise DesignFreezeError(
            "strata sizes must satisfy golden <= selected <= parent, got "
            f"{sizes}"
        )
    if strata["parent"].get("mode") != "stratified_representative_subsample":
        raise DesignFreezeError(
            "strata.parent.mode must be stratified_representative_subsample: "
            "the score selects only the 12 and the 5"
        )

    grid_sizing = _require_mapping(freeze["grid_sizing"], "grid_sizing")
    _require_positive_int(
        _required(grid_sizing, "max_side_px", "grid_sizing"),
        "grid_sizing.max_side_px",
    )
    _require_positive_float(
        _required(grid_sizing, "pixel_scale_arcsec", "grid_sizing"),
        "grid_sizing.pixel_scale_arcsec",
    )

    observing = _require_mapping(freeze["observing"], "observing")
    reference = _require_mapping(
        _required(observing, "reference", "observing"), "observing.reference"
    )
    _require_sha256(
        _required(reference, "sha256", "observing.reference"),
        "observing.reference.sha256",
    )
    if "golden_anchor" in observing:
        golden_anchor = _require_mapping(
            observing["golden_anchor"], "observing.golden_anchor"
        )
        anchor_path = _required(golden_anchor, "path", "observing.golden_anchor")
        if not isinstance(anchor_path, str) or not anchor_path:
            raise DesignFreezeError(
                "observing.golden_anchor.path must be a repo-relative path, "
                f"got {anchor_path!r}"
            )
        _require_sha256(
            _required(golden_anchor, "sha256", "observing.golden_anchor"),
            "observing.golden_anchor.sha256",
        )
    detector = _require_mapping(
        _required(observing, "detector", "observing"), "observing.detector"
    )
    read_noise = _required(detector, "read_noise_per_read_e", "observing.detector")
    if float(read_noise) != 0.2:
        raise DesignFreezeError(
            f"observing.detector.read_noise_per_read_e is {read_noise}; P0-1 "
            "rules 0.2 e- per read central with no 2.5 e- bracket in the design"
        )
    throughput = _require_mapping(
        _required(observing, "throughput", "observing"), "observing.throughput"
    )
    if float(_required(throughput, "baseline", "observing.throughput")) != 0.21:
        raise DesignFreezeError(
            "observing.throughput.baseline must be the ruled 0.21 XeLiF baseline"
        )
    arms = _require_mapping(
        _required(
            _require_mapping(
                _required(observing, "r_arms", "observing"), "observing.r_arms"
            ),
            "arms",
            "observing.r_arms",
        ),
        "observing.r_arms.arms",
    )
    for arm in ("R0", "R1", "R2", "R3"):
        block = _require_mapping(
            _required(arms, arm, "observing.r_arms.arms"),
            f"observing.r_arms.arms.{arm}",
        )
        for key in ("label", "meaning", "source_magnitude_ab", "throughput"):
            _required(block, key, f"observing.r_arms.arms.{arm}")

    selection = _require_mapping(freeze["selection"], "selection")
    pre_registration = _require_mapping(
        _required(selection, "pre_registration", "selection"),
        "selection.pre_registration",
    )
    _require_sha256(
        _required(pre_registration, "sha256", "selection.pre_registration"),
        "selection.pre_registration.sha256",
    )
    if "committed_path" in pre_registration:
        committed_path = pre_registration["committed_path"]
        if not isinstance(committed_path, str) or not committed_path:
            raise DesignFreezeError(
                "selection.pre_registration.committed_path must be a "
                f"repo-relative path, got {committed_path!r}"
            )
    score = _require_mapping(
        _required(selection, "score", "selection"), "selection.score"
    )
    if _required(score, "expression", "selection.score") != "score = z(log S) + z(log C)":
        raise DesignFreezeError(
            "selection.score.expression must restate the frozen score "
            "'score = z(log S) + z(log C)'"
        )
    floor_cuts = _require_mapping(
        _required(selection, "floor_cuts", "selection"), "selection.floor_cuts"
    )
    for key, value in (
        ("theta_e_arcsec_min", 0.5),
        ("arc_snr_min", 20.0),
    ):
        frozen = float(_required(floor_cuts, key, "selection.floor_cuts"))
        if frozen != value:
            raise DesignFreezeError(
                f"selection.floor_cuts.{key} is {frozen}, not the Collett 2015 "
                f"value {value}"
            )

    mass_ladder = _require_mapping(freeze["mass_ladder"], "mass_ladder")
    coarse = _require_mapping(
        _required(mass_ladder, "coarse", "mass_ladder"), "mass_ladder.coarse"
    )
    for key in ("step_dex", "low", "high"):
        _required(coarse, key, "mass_ladder.coarse")
    _required(mass_ladder, "refine", "mass_ladder")
    _required(mass_ladder, "extend_down", "mass_ladder")
    _required(mass_ladder, "extend_up", "mass_ladder")

    claim_labels = _require_mapping(freeze["claim_labels"], "claim_labels")
    for key in (
        "central_result",
        "central_result_rule",
        "counts",
        "counts_rule",
        "ensemble",
        "ensemble_rule",
        "templates",
        "templates_rule",
    ):
        _required(claim_labels, key, "claim_labels")

    parent_source = _require_mapping(
        freeze["parent_design_source"], "parent_design_source"
    )
    _require_sha256(
        _required(parent_source, "sha256", "parent_design_source"),
        "parent_design_source.sha256",
    )
    _require_mapping(freeze["parent_design"], "parent_design")

    _validate_templates(freeze)
    _validate_seeds(freeze)
    _validate_psf_knowledge_error(freeze)
    _validate_extraction_settings(freeze)
    _validate_module_constants(freeze)
    _validate_nonlinear_validation(freeze)
    return freeze


def _validate_nonlinear_validation(freeze: dict) -> None:
    """Validate the version-4 and version-5 nonlinear protocol block.

    Parameters
    ----------
    freeze : `dict`
        Freeze document whose ``nonlinear_validation`` block is checked.

    Raises
    ------
    DesignFreezeError
        Raised for missing protocol keys, missing fit settings, a
        malformed arm declaration, duplicate arm indices, invalid
        replicate declarations, or a campaign that names an undeclared
        member set or arm.
    """
    block = _require_mapping(
        freeze["nonlinear_validation"], "nonlinear_validation"
    )
    if int(freeze["freeze"]["version"]) >= 5:
        for key in ("declared_v5", "rulings_v5"):
            _required(block, key, "nonlinear_validation")
        _require_iso_date(
            block["declared_v5"], "nonlinear_validation.declared_v5"
        )
        if block["rulings_v5"] != [f"D-K{index}" for index in range(1, 11)]:
            raise DesignFreezeError(
                "nonlinear_validation.rulings_v5 must list D-K1 through D-K10"
            )
        criteria = block.get("success_criteria")
        if (
            not isinstance(criteria, list)
            or not criteria
            or any(
                not isinstance(item, str) or not item.strip()
                for item in criteria
            )
        ):
            raise DesignFreezeError(
                "nonlinear_validation.success_criteria must be a non-empty "
                "list of non-empty strings"
            )
    missing = [
        key for key in REQUIRED_NONLINEAR_VALIDATION_KEYS if key not in block
    ]
    if missing:
        raise DesignFreezeError(
            "nonlinear_validation is missing required keys: "
            + ", ".join(missing)
        )

    fit = _require_mapping(block["fit"], "nonlinear_validation.fit")
    missing = [key for key in REQUIRED_NONLINEAR_FIT_KEYS if key not in fit]
    if missing:
        raise DesignFreezeError(
            "nonlinear_validation.fit is missing required settings: "
            + ", ".join(missing)
        )

    null_noise = _require_mapping(
        freeze["seeds"]["streams"].get("null_noise"),
        "seeds.streams.null_noise",
    )
    null_replicate_indices = _required(
        null_noise,
        "replicate_indices",
        "seeds.streams.null_noise",
    )
    knowledge = _require_mapping(
        freeze["psf_knowledge_error"], "psf_knowledge_error"
    )
    knowledge_residual = _require_mapping(
        knowledge["residual_model"], "psf_knowledge_error.residual_model"
    )
    knowledge_rungs = [
        float(value)
        for value in knowledge_residual["amplitude_rms_nm_rungs"]
    ]
    direction_stream = _require_mapping(
        freeze["seeds"]["streams"]["psf_knowledge_direction"],
        "seeds.streams.psf_knowledge_direction",
    )
    direction_indices = list(direction_stream["direction_indices"])

    arms = _require_mapping(block["arms"], "nonlinear_validation.arms")
    if not arms:
        raise DesignFreezeError("nonlinear_validation.arms is empty")
    indices = []
    replicate_by_arm = {}
    delta_pairs = {}
    for name, declaration in arms.items():
        arm = _require_mapping(
            declaration, f"nonlinear_validation.arms.{name}"
        )
        for key in (
            "arm_index",
            "dataset_kind",
            "subhalo_in_truth",
            "fit_mode",
            "rung",
            "sample",
        ):
            if key not in arm:
                raise DesignFreezeError(
                    f"nonlinear_validation.arms.{name} is missing {key!r}"
                )
        if arm["dataset_kind"] not in ("asimov", "noisy"):
            raise DesignFreezeError(
                f"nonlinear_validation.arms.{name}.dataset_kind must be "
                "'asimov' or 'noisy'"
            )
        if arm["fit_mode"] not in ("freed", "fixed_template"):
            raise DesignFreezeError(
                f"nonlinear_validation.arms.{name}.fit_mode must be "
                "'freed' or 'fixed_template'"
            )
        if arm["rung"] not in ("top", "below"):
            raise DesignFreezeError(
                f"nonlinear_validation.arms.{name}.rung must be "
                "'top' or 'below'"
            )
        if arm["sample"] not in ("all", "non_censored", "golden"):
            raise DesignFreezeError(
                f"nonlinear_validation.arms.{name}.sample must be "
                "'all', 'non_censored' or 'golden'"
            )
        if "fit_psf_delta" in arm:
            delta_path = f"nonlinear_validation.arms.{name}.fit_psf_delta"
            delta = _require_mapping(arm["fit_psf_delta"], delta_path)
            amplitude = _require_positive_float(
                _required(delta, "amplitude_rms_nm", delta_path),
                f"{delta_path}.amplitude_rms_nm",
            )
            if amplitude not in knowledge_rungs:
                raise DesignFreezeError(
                    f"{delta_path}.amplitude_rms_nm must be a positive "
                    "declared PSF knowledge-error rung"
                )
            directions = _required(delta, "directions", delta_path)
            if not isinstance(directions, list) or not directions:
                raise DesignFreezeError(
                    f"{delta_path}.directions must be a non-empty list"
                )
            if any(
                isinstance(direction, bool)
                or not isinstance(direction, int)
                or direction not in direction_indices
                for direction in directions
            ):
                raise DesignFreezeError(
                    f"{delta_path}.directions must use the declared "
                    "direction indices"
                )
            if len(set(directions)) != len(directions):
                raise DesignFreezeError(
                    f"{delta_path}.directions must be unique"
                )
            if arm["dataset_kind"] != "noisy":
                raise DesignFreezeError(
                    f"{delta_path} requires dataset_kind 'noisy'"
                )
            if arm["fit_mode"] != "freed":
                raise DesignFreezeError(
                    f"{delta_path} requires fit_mode 'freed'"
                )
            if arm["rung"] != "top":
                raise DesignFreezeError(
                    f"{delta_path} requires rung 'top'"
                )
            if "noise_replicate" in arm:
                raise DesignFreezeError(
                    f"{delta_path} cannot carry noise_replicate"
                )
            pair = (amplitude, arm["subhalo_in_truth"])
            if pair in delta_pairs:
                raise DesignFreezeError(
                    "nonlinear_validation fit_psf_delta arms duplicate the "
                    f"(amplitude, subhalo_in_truth) pair {pair!r}: "
                    f"{delta_pairs[pair]!r} and {name!r}"
                )
            delta_pairs[pair] = name
        if "noise_replicate" in arm:
            replicate = _require_positive_int(
                arm["noise_replicate"],
                f"nonlinear_validation.arms.{name}.noise_replicate",
            )
            if arm["dataset_kind"] != "noisy":
                raise DesignFreezeError(
                    f"nonlinear_validation.arms.{name}.noise_replicate "
                    "requires dataset_kind 'noisy'"
                )
            if arm["subhalo_in_truth"] is not False:
                raise DesignFreezeError(
                    f"nonlinear_validation.arms.{name}.noise_replicate "
                    "requires subhalo_in_truth false"
                )
            if replicate not in null_replicate_indices:
                raise DesignFreezeError(
                    f"nonlinear_validation.arms.{name}.noise_replicate "
                    f"{replicate} is not in seeds.streams.null_noise."
                    "replicate_indices"
                )
            replicate_by_arm[name] = replicate
        indices.append(int(arm["arm_index"]))
    if len(set(indices)) != len(indices):
        raise DesignFreezeError(
            "nonlinear_validation arm indices must be unique, got "
            f"{sorted(indices)}"
        )

    seeds = _require_mapping(block["seeds"], "nonlinear_validation.seeds")
    campaign_seeds = _require_mapping(freeze["seeds"], "seeds")
    if int(seeds.get("entropy", -1)) != int(campaign_seeds["entropy"]):
        raise DesignFreezeError(
            "nonlinear_validation.seeds.entropy must equal the frozen "
            "campaign seed entropy"
        )
    spawn_key = seeds.get("spawn_key")
    if not isinstance(spawn_key, list) or spawn_key[:1] != [5]:
        raise DesignFreezeError(
            "nonlinear_validation.seeds.spawn_key must be a list starting "
            "with 5, beyond the frozen campaign spawn keys 0-4"
        )
    if int(freeze["freeze"]["version"]) >= 5:
        direction_spawn_key = _required(
            seeds,
            "psf_knowledge_direction_spawn_key",
            "nonlinear_validation.seeds",
        )
        if direction_spawn_key != [7, "direction_index", "system_index"]:
            raise DesignFreezeError(
                "nonlinear_validation.seeds."
                "psf_knowledge_direction_spawn_key must be "
                "[7, 'direction_index', 'system_index']"
            )

    if set(replicate_by_arm.values()) != set(null_replicate_indices):
        raise DesignFreezeError(
            "nonlinear_validation noise_replicate values over declared arms "
            f"must equal seeds.streams.null_noise.replicate_indices: "
            f"declared {sorted(replicate_by_arm.values())}, expected "
            f"{sorted(null_replicate_indices)}"
        )

    member_sets = _require_mapping(
        block["member_sets"], "nonlinear_validation.member_sets"
    )
    if not member_sets:
        raise DesignFreezeError("nonlinear_validation.member_sets is empty")
    for name, member_set in member_sets.items():
        path = f"nonlinear_validation.member_sets.{name}"
        member_set = _require_mapping(member_set, path)
        for key in ("rule", "source", "tier", "n_systems"):
            _required(member_set, key, path)
        _require_positive_int(
            member_set["n_systems"], f"{path}.n_systems"
        )

    campaigns = _require_mapping(
        block["campaigns"], "nonlinear_validation.campaigns"
    )
    if not campaigns:
        raise DesignFreezeError("nonlinear_validation.campaigns is empty")
    replicate_indices = []
    for name, campaign in campaigns.items():
        path = f"nonlinear_validation.campaigns.{name}"
        campaign = _require_mapping(campaign, path)
        member_set_name = _required(campaign, "member_set", path)
        if (
            not isinstance(member_set_name, str)
            or member_set_name not in member_sets
        ):
            raise DesignFreezeError(
                f"{path}.member_set {member_set_name!r} is not a declared "
                "member set"
            )
        campaign_arms = _required(campaign, "arms", path)
        if not isinstance(campaign_arms, list) or not campaign_arms:
            raise DesignFreezeError(f"{path}.arms must be a non-empty list")
        if any(not isinstance(arm_name, str) for arm_name in campaign_arms):
            raise DesignFreezeError(f"{path}.arms must contain arm names")
        if len(set(campaign_arms)) != len(campaign_arms):
            raise DesignFreezeError(f"{path}.arms must contain unique arms")
        for arm_name in campaign_arms:
            if arm_name not in arms:
                raise DesignFreezeError(
                    f"{path}.arms names undeclared arm {arm_name!r}"
                )
        campaign_delta_flags = [
            "fit_psf_delta" in arms[arm_name] for arm_name in campaign_arms
        ]
        has_delta_arms = any(campaign_delta_flags)
        if has_delta_arms and not all(campaign_delta_flags):
            raise DesignFreezeError(
                f"{path}.arms must not mix fit_psf_delta and non-delta arms"
            )
        if has_delta_arms:
            for source_key in ("reference_source", "null_source"):
                if source_key not in campaign:
                    raise DesignFreezeError(
                        f"{path} with fit_psf_delta arms requires {source_key}"
                    )
                _validate_nonlinear_source(
                    campaign[source_key], f"{path}.{source_key}"
                )
            reference_source = _require_mapping(
                campaign["reference_source"], f"{path}.reference_source"
            )
            reference_arms = _required(
                reference_source,
                "arms",
                f"{path}.reference_source",
            )
            if not isinstance(reference_arms, list) or not reference_arms:
                raise DesignFreezeError(
                    f"{path}.reference_source.arms must be a non-empty list"
                )
            for reference_arm in reference_arms:
                if (
                    not isinstance(reference_arm, str)
                    or reference_arm not in arms
                ):
                    raise DesignFreezeError(
                        f"{path}.reference_source.arms names undeclared arm "
                        f"{reference_arm!r}"
                    )
        campaign_replicates = {
            replicate_by_arm[arm_name]
            for arm_name in campaign_arms
            if arm_name in replicate_by_arm
        }
        if campaign_replicates and campaign_replicates != set(
            null_replicate_indices
        ):
            raise DesignFreezeError(
                f"{path}.arms noise_replicate values must equal "
                "seeds.streams.null_noise.replicate_indices: declared "
                f"{sorted(campaign_replicates)}, expected "
                f"{sorted(null_replicate_indices)}"
            )
        positions_source = _required(campaign, "positions_source", path)
        if positions_source != "self":
            if not isinstance(positions_source, str) or not positions_source:
                raise DesignFreezeError(
                    f"{path}.positions_source must be 'self' or a non-empty "
                    "campaign name"
                )
            source_uuid = _required(
                campaign, "positions_source_campaign_uuid", path
            )
            if not isinstance(source_uuid, str) or not source_uuid:
                raise DesignFreezeError(
                    f"{path}.positions_source_campaign_uuid must be a "
                    "non-empty string"
                )
        for source_key in ("replicate_zero_source", "pooled_source"):
            if source_key in campaign:
                _validate_nonlinear_source(
                    campaign[source_key], f"{path}.{source_key}"
                )
        smoke_rule = _require_mapping(
            _required(campaign, "smoke_rule", path), f"{path}.smoke_rule"
        )
        smoke_arms = _required(smoke_rule, "arms", f"{path}.smoke_rule")
        if not isinstance(smoke_arms, list) or not smoke_arms:
            raise DesignFreezeError(
                f"{path}.smoke_rule.arms must be a non-empty list"
            )
        for arm_name in smoke_arms:
            if not isinstance(arm_name, str):
                raise DesignFreezeError(
                    f"{path}.smoke_rule.arms must contain arm names"
                )
            if arm_name not in arms:
                raise DesignFreezeError(
                    f"{path}.smoke_rule.arms names undeclared arm "
                    f"{arm_name!r}"
                )
            if arm_name not in campaign_arms:
                raise DesignFreezeError(
                    f"{path}.smoke_rule.arms names arm {arm_name!r} "
                    "outside the campaign arm list"
                )
        member_rule = _required(smoke_rule, "member", f"{path}.smoke_rule")
        if member_rule not in (
            "smallest_image_per_template",
            "smallest_image_non_censored_per_template",
            "smallest_image_golden",
        ):
            raise DesignFreezeError(
                f"{path}.smoke_rule.member has unknown rule "
                f"{member_rule!r}"
            )
        if has_delta_arms:
            smoke_directions = _required(
                smoke_rule,
                "directions",
                f"{path}.smoke_rule",
            )
            if not isinstance(smoke_directions, list) or not smoke_directions:
                raise DesignFreezeError(
                    f"{path}.smoke_rule.directions must be a non-empty list"
                )
            declared_directions = {
                direction
                for arm_name in campaign_arms
                for direction in arms[arm_name]["fit_psf_delta"]["directions"]
            }
            if any(
                isinstance(direction, bool)
                or not isinstance(direction, int)
                or direction not in declared_directions
                for direction in smoke_directions
            ):
                raise DesignFreezeError(
                    f"{path}.smoke_rule.directions must be a subset of the "
                    "campaign fit_psf_delta directions"
                )
    for name, declaration in arms.items():
        if "noise_replicate" in declaration:
            replicate_indices.append(
                int(declaration["noise_replicate"])
            )
    if len(set(replicate_indices)) != len(replicate_indices):
        raise DesignFreezeError(
            "nonlinear_validation noise_replicate values must be unique"
        )

    criteria = block["success_criteria"]
    if not isinstance(criteria, list) or not criteria:
        raise DesignFreezeError(
            "nonlinear_validation.success_criteria must be a non-empty list"
        )


def _validate_nonlinear_source(source: Any, path: str) -> None:
    """Validate one declared external nonlinear-campaign source binding.

    Parameters
    ----------
    source : `object`
        Source declaration to validate.
    path : `str`
        Configuration path used in validation messages.

    Raises
    ------
    DesignFreezeError
        Raised when a required source field is absent or malformed.
    """
    source = _require_mapping(source, path)
    for key in ("campaign", "campaign_uuid", "harvest"):
        value = _required(source, key, path)
        if not isinstance(value, str) or not value:
            raise DesignFreezeError(f"{path}.{key} must be a non-empty string")
    _require_sha256(
        _required(source, "harvest_sha256", path), f"{path}.harvest_sha256"
    )
    _require_sha256(
        _required(source, "review_sha256", path), f"{path}.review_sha256"
    )


def load_design_freeze(
    path=None, skip_bound_artifact_verification: bool = False
) -> dict:
    """Load, validate and verify the design freeze.

    Loading is the only supported way to obtain a freeze mapping, so the
    load also hashes every artifact the freeze binds. A freeze whose
    observing reference or template assets no longer carry the frozen
    digests cannot be loaded at all.

    Parameters
    ----------
    path : path-like, optional
        Freeze artifact to read. Defaults to
        `DEFAULT_DESIGN_FREEZE_PATH`.
    skip_bound_artifact_verification : `bool`, optional
        Skip the `verify_bound_artifacts` pass. This exists for the
        hash-only contexts that read the design without ever rendering
        with the assets, and it is deliberately verbose because using it
        on a production path removes the guarantee the loader exists to
        give.

    Returns
    -------
    freeze : `dict`
        Validated freeze document whose bound artifacts have been
        verified unless the caller explicitly opted out.

    Raises
    ------
    DesignFreezeError
        Raised when the file is missing, is not a YAML mapping, fails
        `validate_design_freeze`, or binds an artifact that is missing
        or no longer hashes to the frozen digest.
    """
    resolved = Path(path or DEFAULT_DESIGN_FREEZE_PATH).expanduser().resolve()
    if not resolved.is_file():
        raise DesignFreezeError(f"Design freeze {resolved} does not exist")
    with resolved.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict):
        raise DesignFreezeError(f"Design freeze {resolved} must contain a mapping")
    freeze = validate_design_freeze(document)
    if not skip_bound_artifact_verification:
        verify_bound_artifacts(freeze)
    return freeze


def design_freeze_digest(path=None) -> str:
    """Return the SHA-256 digest of the freeze artifact.

    Parameters
    ----------
    path : path-like, optional
        Freeze artifact to digest. Defaults to
        `DEFAULT_DESIGN_FREEZE_PATH`.

    Returns
    -------
    digest : `str`
        Full hexadecimal digest, recorded in every campaign built from it.
    """
    return file_sha256(Path(path or DEFAULT_DESIGN_FREEZE_PATH))


def template_levels(freeze: dict) -> tuple[dict, ...]:
    """Return the template bank in declared level order.

    Parameters
    ----------
    freeze : `dict`
        Validated freeze document.

    Returns
    -------
    levels : `tuple` [`dict`]
        One mapping per template, in the order the design declares.
    """
    return tuple(freeze["templates"]["levels"])


def verify_bound_artifacts(freeze: dict, root: Optional[Path] = None) -> dict:
    """Verify every file the freeze binds by digest.

    The observing reference and the five template assets are committed
    artifacts. They are always required: a clone of this repository at
    the freeze's revision has them, so their absence is a broken
    checkout rather than a legitimate state.

    Exactly two bound references are genuinely optional, and only these
    two. The selection pre-registration document and the parent design
    source live in the untracked ``scratch`` tree, which is not
    distributed with the repository, so a clean clone cannot have them.
    Both are embedded in, or restated by, this freeze precisely so the
    freeze stands alone without them: the pre-registration's definitions
    are restated block by block under ``selection`` and the whole parent
    design travels under ``parent_design``. They are verified whenever
    they are on disk and named in the report's ``absent`` list when they
    are not, so their absence is recorded rather than assumed.

    The optional ``selection.pre_registration.committed_path`` names a
    committed copy of the pre-registration document carrying the same
    frozen digest. When the freeze declares it, that copy is a required
    artifact like any other committed one, so the signed selection rule
    is bound into a clean clone rather than reachable only through an
    untracked ``scratch`` tree that may never have been distributed.

    Parameters
    ----------
    freeze : `dict`
        Validated freeze document.
    root : `pathlib.Path`, optional
        Repository root the relative paths resolve against. Defaults to
        `repo_root`.

    Returns
    -------
    report : `dict`
        ``verified`` maps logical name to the confirmed digest and
        ``absent`` lists the untracked references that were not on disk.

    Raises
    ------
    DesignFreezeError
        Raised when a committed artifact is missing, or when any bound
        file that is present carries a different digest. Nothing is
        repaired and no mismatch is tolerated.
    """
    base = Path(root or repo_root())
    committed = {
        "observing_reference": (
            freeze["observing"]["reference"]["path"],
            freeze["observing"]["reference"]["sha256"],
        ),
    }
    for level in template_levels(freeze):
        committed[f"template_{level['id']}"] = (level["asset_path"], level["sha256"])
    pre_registration = freeze["selection"]["pre_registration"]
    if "committed_path" in pre_registration:
        committed["selection_pre_registration_committed"] = (
            pre_registration["committed_path"],
            pre_registration["sha256"],
        )
    golden_anchor = freeze["observing"].get("golden_anchor")
    if golden_anchor is not None:
        committed["golden_magnitude_anchor"] = (
            golden_anchor["path"],
            golden_anchor["sha256"],
        )
    untracked = {
        "selection_pre_registration": (
            pre_registration["path"],
            pre_registration["sha256"],
        ),
        "parent_design_source": (
            freeze["parent_design_source"]["path"],
            freeze["parent_design_source"]["sha256"],
        ),
    }

    verified: dict = {}
    absent: list = []
    problems: list = []
    for bound, required in ((committed, True), (untracked, False)):
        for name, (relative, expected) in bound.items():
            path = base/relative
            if not path.is_file():
                if required:
                    problems.append(f"{name}: {path} does not exist")
                else:
                    absent.append(name)
                continue
            digest = file_sha256(path)
            if digest != expected:
                problems.append(
                    f"{name}: {path} digest {digest} does not match the frozen "
                    f"value {expected}"
                )
                continue
            verified[name] = digest
    if problems:
        raise DesignFreezeError(
            "Design freeze bound artifacts failed verification: "
            + "; ".join(problems)
        )
    return {"verified": verified, "absent": sorted(absent)}

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

import hashlib
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
)
"""Keys the version-3 nonlinear-validation block must carry (`tuple`)."""

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
    _validate_extraction_settings(freeze)
    _validate_module_constants(freeze)
    _validate_nonlinear_validation(freeze)
    return freeze


def _validate_nonlinear_validation(freeze: dict) -> None:
    """Validate the version-3 nonlinear-validation protocol block.

    Parameters
    ----------
    freeze : `dict`
        Freeze document whose ``nonlinear_validation`` block is checked.

    Raises
    ------
    DesignFreezeError
        Raised for missing protocol keys, missing fit settings, a
        malformed arm declaration, duplicate arm indices, or a seed
        declaration that does not extend the frozen campaign streams.
    """
    block = _require_mapping(
        freeze["nonlinear_validation"], "nonlinear_validation"
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

    arms = _require_mapping(block["arms"], "nonlinear_validation.arms")
    if not arms:
        raise DesignFreezeError("nonlinear_validation.arms is empty")
    indices = []
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

    criteria = block["success_criteria"]
    if not isinstance(criteria, list) or not criteria:
        raise DesignFreezeError(
            "nonlinear_validation.success_criteria must be a non-empty list"
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

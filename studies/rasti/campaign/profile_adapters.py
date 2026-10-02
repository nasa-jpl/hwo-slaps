"""Adapters preserving the RASTI release declaration and Fisher protocol.

These adapters intentionally preserve the submitted study's fixed 999-pixel
Fisher convention. They are not part of the installed forecasting engine.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from hwoslaps.modeling.nonlinear.profile_calibration import _instance_value
from hwoslaps.modeling.nonlinear.profile_settings import (
    FreshProfileSettings, PROCEDURE_VERSION,
)


def profile_settings_from_release_protocol(release: Mapping[str, Any]) -> FreshProfileSettings:
    """Build settings only from the explicit v7 release declaration."""
    protocol = release.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("release declaration has no protocol")
    optimizer = protocol.get("optimizer")
    acceptance = protocol.get("acceptance")
    if not isinstance(optimizer, Mapping) or not isinstance(acceptance, Mapping):
        raise ValueError("release declaration lacks optimizer/acceptance settings")
    search = optimizer.get("current_search_optimization", {})
    if not isinstance(search, Mapping):
        raise ValueError(
            "release optimizer current_search_optimization must be a mapping"
        )
    repeat = optimizer.get("tighter_repeat", {})
    if not isinstance(repeat, Mapping):
        raise ValueError("release optimizer tighter_repeat must be a mapping")
    repeat_aliases = {
        "repeat_maxiter": ("repeat_maxiter", "maxiter"),
        "repeat_ftol": ("repeat_ftol", "ftol"),
        "repeat_gtol": ("repeat_gtol", "gtol"),
    }
    resolved = dict(search)
    for target, aliases in repeat_aliases.items():
        if target not in resolved:
            for alias in aliases:
                if alias in repeat:
                    resolved[target] = repeat[alias]
                    break
    required = (
        "original_start_count", "start_separation_normalized_l2",
        "start_separation_posterior_sigma", "maxiter",
        "ftol", "gtol", "maxls", "repeat_maxiter", "repeat_ftol",
        "repeat_gtol", "scalar_residual_tolerance",
    )
    missing = [key for key in required if key not in resolved]
    if missing:
        raise ValueError(
            "release declaration must explicitly bind profile settings: "
            + ", ".join(missing)
        )
    acceptance_required = (
        "support_log_likelihood_tolerance",
        "tighter_repeat_tolerance",
        "distinct_original_starts",
    )
    missing_acceptance = [key for key in acceptance_required if key not in acceptance]
    if missing_acceptance:
        raise ValueError(
            "release declaration must explicitly bind acceptance settings: "
            + ", ".join(missing_acceptance)
        )
    values = {
        key: resolved[key]
        for key in required
    }
    values.update(
        support_log_likelihood_tolerance=acceptance.get(
            "support_log_likelihood_tolerance"
        ),
        repeat_log_likelihood_tolerance=acceptance.get(
            "tighter_repeat_tolerance"
        ),
        minimum_distinct_original_start_support=acceptance.get(
            "distinct_original_starts"
        ),
        version=PROCEDURE_VERSION,
    )
    return FreshProfileSettings(**values)


def evaluate_established_fisher_q(
    *,
    config: Mapping[str, Any],
    position_yx_arcsec: Sequence[float],
    log10_m200: float,
    kernel_shape_native: Sequence[int] = (999, 999),
) -> dict[str, Any]:
    """Evaluate the established production 999-pixel Fisher q at one point.

    This is a narrow adapter around the tested ``run_ladder`` evaluator.  It
    uses the full square Fisher geometry and the existing matched-PSF
    detector; it introduces no new derivative, PSF, or threshold prescription.
    The function is intentionally opt-in because bracket materialization does
    not need to run a Fisher calculation merely to create an H1 anchor.
    """
    if list(kernel_shape_native) != [999, 999]:
        raise ValueError("production Fisher bracket evaluation requires the 999x999 kernel")
    if len(position_yx_arcsec) != 2 or not np.all(np.isfinite(position_yx_arcsec)):
        raise ValueError("Fisher-q position must contain two finite coordinates")
    log10_m200 = float(log10_m200)
    if not np.isfinite(log10_m200):
        raise ValueError("Fisher-q mass must be finite")
    from studies.rasti.scripts import run_ladder
    from hwoslaps.config.validation import validate_or_raise
    from hwoslaps.psf.generator import generate_psf_system

    source_config = deepcopy(dict(config))
    ladder = source_config.get("ladder")
    if not isinstance(ladder, dict) or not isinstance(ladder.get("aperture"), dict):
        raise ValueError("Fisher-q adapter requires the established ladder aperture declaration")
    rung_config = run_ladder._rung_config(
        source_config,
        ladder,
        ladder["aperture"],
    )
    if list(rung_config["psf"]["kernel"]["shape_native"]) != [999, 999]:
        raise RuntimeError("established Fisher-q adapter did not construct the 999x999 kernel")
    validate_or_raise(rung_config)
    psf_data = generate_psf_system(rung_config["psf"], full_config=rung_config)
    detector = run_ladder._build_detector(rung_config, psf_data)
    run_ladder._point_detector_at_rung(detector, log10_m200)
    results = detector._evaluate_grid_positions(
        [tuple(float(value) for value in position_yx_arcsec)]
    )
    if len(results) != 1:
        raise RuntimeError("Fisher-q adapter returned an unexpected number of positions")
    result = results[0]
    q_value = float(result.q_asimov_local)
    if not np.isfinite(q_value):
        raise ValueError("established Fisher-q evaluator returned a non-finite value")
    return {
        "q_f_production_at_position": q_value,
        "log10_m200": log10_m200,
        "position_yx_arcsec": [float(value) for value in position_yx_arcsec],
        "kernel_shape_native": [999, 999],
        "evaluator": "run_ladder._rung_config/_build_detector/_evaluate_grid_positions",
        "full_square_geometry": True,
        "new_prescription": False,
    }


def materialize_bracket_case(
    *,
    full_config: Mapping[str, Any],
    positions: Mapping[str, Any],
    trial: Any,
    output_dir: Path,
    case_id: str,
    bracket_rung: str,
    target_log10_m200: float,
    target_mass_msun: float,
    position_yx_arcsec: Sequence[float],
) -> dict[str, Any]:
    """Create a bracket rung and an H1 truth anchor without sampling.

    The caller supplies the already generated physical truth ``trial``.  The
    anchor is derived from the same fixed-point model builder used by the
    validator and is checked again by :class:`ZeroResidualAnchorRunner` using
    the actual corrected dataset.  This helper intentionally does not render,
    fit, or copy a prior result.
    """
    import copy
    import yaml
    from hwoslaps.modeling.nonlinear.autolens_model_builder import (
        autofit_model_from_spec,
        fixed_point_model_spec_from_trial,
        subhalo_model_spec_from_trial,
    )

    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"bracket materialization output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(position_yx_arcsec) != 2 or not np.all(np.isfinite(position_yx_arcsec)):
        raise ValueError("bracket position must contain two finite coordinates")
    target_log10_m200 = float(target_log10_m200)
    target_mass_msun = float(target_mass_msun)
    if not np.isfinite(target_log10_m200) or not np.isfinite(target_mass_msun) or target_mass_msun <= 0:
        raise ValueError("bracket target mass must be finite and positive")
    if not np.isclose(target_mass_msun, 10.0 ** target_log10_m200, rtol=1.0e-12, atol=0.0):
        raise ValueError("bracket target mass and log mass disagree")
    source_positions = copy.deepcopy(dict(positions))
    rungs = source_positions.get("rungs")
    if not isinstance(rungs, dict) or "top" not in rungs:
        raise ValueError("bracket materialization requires the existing top rung")
    top = rungs["top"]
    top_position = top.get("position_yx_arcsec")
    if list(top_position or []) != [float(value) for value in position_yx_arcsec]:
        raise ValueError("bracket position differs from the frozen upper-rung position")
    rungs[bracket_rung] = {
        "logm": target_log10_m200,
        "mass_msun": target_mass_msun,
        "position_yx_arcsec": [float(value) for value in position_yx_arcsec],
        "source": "stage3_runtime_bracket_materializer",
    }
    positions_path = output_dir / "positions.json"
    positions_path.write_text(
        json.dumps(source_positions, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    config_path = output_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(dict(full_config), sort_keys=False), encoding="utf-8")

    fixed_model = autofit_model_from_spec(
        fixed_point_model_spec_from_trial(dict(full_config), trial)
    )
    target_model = autofit_model_from_spec(
        subhalo_model_spec_from_trial(
            dict(full_config), trial, fit_mode="fixed_template"
        )
    )
    truth_instance = fixed_model.instance_from_prior_medians()
    names = [".".join(path) for path in target_model.unique_prior_paths]
    lower = np.asarray(
        [prior.lower_limit for prior in target_model.priors_ordered_by_id],
        dtype=float,
    )
    upper = np.asarray(
        [prior.upper_limit for prior in target_model.priors_ordered_by_id],
        dtype=float,
    )
    vector = np.asarray(
        [_instance_value(truth_instance, path) for path in target_model.unique_prior_paths],
        dtype=float,
    )
    if vector.shape != lower.shape or np.any(vector < lower) or np.any(vector > upper):
        raise ValueError("generated bracket H1 anchor is outside the runtime model support")
    anchor = {
        "schema_version": 1,
        "generator": "fresh_profile.materialize_bracket_case",
        "case_id": case_id,
        "parameter_names": names,
        "vector": vector.tolist(),
        "target_log10_m200": target_log10_m200,
        "target_mass_msun": target_mass_msun,
        "position_yx_arcsec": [float(value) for value in position_yx_arcsec],
        "fit_mode": "fixed_template",
        "verification": "ZeroResidualAnchorRunner must evaluate actual corrected objective",
        "sampler_executed": False,
        "evidence_claim": False,
    }
    anchor_path = output_dir / "h1_anchor.json"
    anchor_path.write_text(
        json.dumps(anchor, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    return {
        "status": "MATERIALIZED_NOT_FIT",
        "case_id": case_id,
        "config": str(config_path),
        "config_sha256": digest(config_path),
        "positions": str(positions_path),
        "positions_sha256": digest(positions_path),
        "h1_anchor": str(anchor_path),
        "h1_anchor_sha256": digest(anchor_path),
        "bracket_rung": bracket_rung,
        "target_log10_m200": target_log10_m200,
        "target_mass_msun": target_mass_msun,
        "position_yx_arcsec": [float(value) for value in position_yx_arcsec],
        "sampler_executed": False,
        "evidence_claim": False,
    }


def materialize_bracket_case_from_files(
    *,
    config_path: Path,
    positions_path: Path,
    output_dir: Path,
    case_id: str,
    bracket_rung: str,
    target_log10_m200: float,
    target_mass_msun: float,
    position_yx_arcsec: Sequence[float],
) -> dict[str, Any]:
    """Materialize a bracket from a catalog-bound config and position."""
    import math
    import yaml
    from hwoslaps.modeling.nonlinear.trial import subhalo_truth_config, trial_from_fisher_map_position
    from hwoslaps.lensing.generator import generate_lensing_system

    config_path = Path(config_path).expanduser().resolve()
    positions_path = Path(positions_path).expanduser().resolve()
    if not config_path.is_file() or not positions_path.is_file():
        raise FileNotFoundError("bracket config or positions input is missing")
    full_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    positions = json.loads(positions_path.read_text(encoding="utf-8"))
    if not isinstance(full_config, dict) or not isinstance(positions, dict):
        raise ValueError("bracket inputs must contain a config and positions mapping")
    target_mass_msun = float(target_mass_msun)
    position_yx_arcsec = tuple(float(value) for value in position_yx_arcsec)
    # The staged ladder config carries no subhalo (run_ladder injects one per
    # rung); the reference must be rendered from the declared target exactly
    # as the validation arm renders it at fit time.
    injected_config = subhalo_truth_config(
        full_config, target_mass_msun, position_yx_arcsec
    )
    lensing_reference = generate_lensing_system(
        injected_config["lensing"],
        full_config=injected_config,
    )
    reference_mass = getattr(lensing_reference, "subhalo_mass", None)
    reference_position = getattr(lensing_reference, "subhalo_position", None)
    if (
        reference_mass is None
        or reference_position is None
        or not math.isclose(
            float(reference_mass), target_mass_msun, rel_tol=1.0e-12, abs_tol=0.0
        )
        or tuple(float(value) for value in reference_position) != position_yx_arcsec
    ):
        raise RuntimeError(
            "generated bracket reference does not carry the declared target "
            f"subhalo: rendered mass {reference_mass!r} at {reference_position!r}, "
            f"declared {target_mass_msun} at {position_yx_arcsec}"
        )
    trial = trial_from_fisher_map_position(
        injected_config,
        lensing_reference,
        target_mass_msun,
        position_yx_arcsec,
        fisher_q=None,
        case_id=case_id,
    )
    if trial.metadata.get("profile_scales_source") != "reference":
        raise RuntimeError(
            "bracket trial did not take its profile scales from the rendered reference"
        )
    return materialize_bracket_case(
        full_config=full_config,
        positions=positions,
        trial=trial,
        output_dir=output_dir,
        case_id=case_id,
        bracket_rung=bracket_rung,
        target_log10_m200=target_log10_m200,
        target_mass_msun=target_mass_msun,
        position_yx_arcsec=position_yx_arcsec,
    )

"""
Approved position production and immutable dependency resolution for v7.

Importing, validating, resolving or materializing specs does not render
science. Only ``produce_position`` invokes the established Fisher-map
extractor, after checking the immutable approval and the selected physical CUDA
device.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from .release_catalog import (
    ReleaseCatalogError,
    _read_json,
    _restamp_config,
    _runner_spec_template,
    _verify_declared_file,
    input_source,
    sha256_file,
    validate_case_approval,
)


def _path(ref):
    """
    Use a checked local mirror, otherwise its declared execution location.
    """
    return input_source(ref, "dependency input")


def position_input_identity(task):
    """
    Bind the full existing ladder artifact, without recomputing its geometry.
    """
    config_path = _path(task["input_records"]["config"])
    artifact_path = _path(task["input_records"]["v6_ladder_artifact"])
    config = yaml.safe_load(config_path.read_text())
    with np.load(artifact_path, allow_pickle=False) as artifact:
        for key in (
            "system_id",
            "source_asset_sha256",
            "config_hash",
            "campaign_uuid",
            "aperture_sha256",
            "m_best_bracket_logm",
            "psf_kernel_shape_native",
        ):
            if key not in artifact:
                raise ReleaseCatalogError(f"ladder artifact missing {key}")
        if str(artifact["system_id"]) != config["run_name"]:
            raise ReleaseCatalogError("ladder/config run_name mismatch")
        if config["stage0"]["system_id"] != task["system_id"]:
            raise ReleaseCatalogError("task/config system_id mismatch")
        if str(artifact["source_asset_sha256"]) != config["stage0"]["source_asset_sha256"]:
            raise ReleaseCatalogError("ladder/config source identity mismatch")
        if list(artifact["psf_kernel_shape_native"]) != [999, 999]:
            raise ReleaseCatalogError("position producer requires the production k999 ladder")
        rungs = np.asarray(artifact["m_best_bracket_logm"], dtype=float)
        if rungs.shape != (2,) or not np.all(np.isfinite(rungs)):
            raise ReleaseCatalogError("four-arm production requires finite below/top rungs")
        identity = {
            key: str(artifact[key])
            for key in (
                "system_id",
                "source_asset_sha256",
                "config_hash",
                "campaign_uuid",
                "aperture_sha256",
            )
        }
        identity["rungs"] = {"below": float(rungs[0]), "top": float(rungs[1])}
    return config_path, artifact_path, identity


def validate_position_payload(payload, identity):
    """
    Check map-output identity, rung, geometry and the production q canary.
    """
    mappings = {
        "system_id": "system_id",
        "source_asset_sha256": "source_asset_sha256",
        "ladder_config_hash": "config_hash",
        "ladder_campaign_uuid": "campaign_uuid",
        "aperture_sha256": "aperture_sha256",
    }
    for output, source in mappings.items():
        if payload.get(output) != identity[source]:
            raise ReleaseCatalogError(f"position identity mismatch: {output}")
    if payload.get("fit_kernel_shape_native") != [51, 51]:
        raise ReleaseCatalogError("position artifact is not fit-kernel matched")
    half_widths = np.asarray(payload.get("support_half_widths_arcsec"), dtype=float)
    centre = np.asarray(payload.get("aperture_centre_arcsec"), dtype=float)
    radius = float(payload.get("aperture_radius_arcsec", 0))
    if (
        half_widths.shape != (2,)
        or centre.shape != (2,)
        or not np.all(np.isfinite(half_widths))
        or np.any(half_widths <= 0)
        or not np.all(np.isfinite(centre))
        or not np.isfinite(radius)
        or radius <= 0
    ):
        raise ReleaseCatalogError("invalid position geometry")
    for name, logm in identity["rungs"].items():
        rung = payload.get("rungs", {}).get(name, {})
        if abs(float(rung.get("logm", float("nan"))) - logm) > 1e-9 or not np.isfinite(
            float(rung.get("logm", float("nan")))
        ):
            raise ReleaseCatalogError("position rung mass mismatch")
        point = np.asarray(rung.get("position_yx_arcsec"), dtype=float)
        if (
            point.shape != (2,)
            or not np.all(np.isfinite(point))
            or np.any(np.abs(point) > half_widths + 1e-12)
            or np.sum((point - centre) ** 2) > radius**2 + 1e-12
        ):
            raise ReleaseCatalogError("position lies outside declared support/aperture")
        delta = float(rung.get("q_max_relative_difference", float("nan")))
        if not np.isfinite(delta) or abs(delta) > 1e-6:
            raise ReleaseCatalogError("production aperture q reproduction failed")
        if not np.isfinite(float(rung.get("q_f_matched", float("nan")))):
            raise ReleaseCatalogError("nonfinite matched Fisher comparator")


def _write_new(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _verify_runtime_config(config_path):
    """Use the production provenance checks before any dependency renderer."""
    repo = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo / "scripts"))
    from run_stage0_observation import _verify_code_revision, _verify_source_asset

    config = yaml.safe_load(Path(config_path).read_text())
    _verify_code_revision(config)
    _verify_source_asset(config)


def produce_position(catalog_path, task_id, output_root, revision, approval_path, release_path):
    """
    Run exactly one approved extractor; preserve output and immutable receipt.
    """
    catalog_path, release_path, approval_path = map(Path, (catalog_path, release_path, approval_path))
    catalog = _read_json(catalog_path)
    tasks = [task for task in catalog["position_tasks"] if task["task_id"] == task_id]
    if len(tasks) != 1 or tasks[0]["status"] != "PENDING_POSITION_EXTRACTION":
        raise ReleaseCatalogError("task must name exactly one pending position producer")
    task = tasks[0]
    approval = _read_json(approval_path)
    cases = {case["case_id"]: case for case in catalog["cases"]}
    for case_id in task["downstream_case_ids"]:
        validate_case_approval(cases[case_id], approval, sha256_file(catalog_path))
        _verify_declared_file(release_path, cases[case_id]["release_freeze_sha256"], "release freeze")
    source_config, artifact, identity = position_input_identity(task)
    output = Path(output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise ReleaseCatalogError("position output must be a new empty namespace")
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.yaml"
    restamp = _restamp_config(source_config, config_path, revision)
    import jax

    jax.config.update("jax_enable_x64", True)
    from hwoslaps.modeling.nonlinear.experimental_device import require_cuda_execution

    device = require_cuda_execution()
    _verify_runtime_config(config_path)
    # The existing extractor resolves the allowed tier set via the v5 loader.
    # This process-local adapter adds only the exact approved v6 tier after
    # checking v7 authority; it does not alter any fit or map prescription.
    from . import design_freeze

    release = design_freeze.load_release_freeze(release_path)
    original_loader = design_freeze.load_design_freeze

    def scoped_loader(*args, **kwargs):
        freeze = copy.deepcopy(original_loader(*args, **kwargs))
        freeze["nonlinear_validation"]["member_sets"]["v7_approved_position"] = {"tier": ["full_pool"]}
        return freeze

    repo = Path(__file__).resolve().parents[3]
    extractor = repo / "scripts" / "extract_injection_positions.py"
    sys.path.insert(0, str(repo / "scripts"))
    module_spec = importlib.util.spec_from_file_location("v7_position_extractor", extractor)
    module = importlib.util.module_from_spec(module_spec)
    try:
        design_freeze.load_design_freeze = scoped_loader
        module_spec.loader.exec_module(module)
        module.main([str(config_path), str(artifact), str(output)])
    finally:
        design_freeze.load_design_freeze = original_loader
    position = output / "injection_position.json"
    payload = _read_json(position)
    validate_position_payload(payload, identity)
    receipt = {
        "status": "COMPLETE",
        "task_id": task_id,
        "catalog_sha256": sha256_file(catalog_path),
        "release_freeze_sha256": sha256_file(release_path),
        "approval_sha256": sha256_file(approval_path),
        "position": str(position),
        "position_sha256": sha256_file(position),
        "ladder_artifact_sha256": sha256_file(artifact),
        "source_config_sha256": sha256_file(source_config),
        "extractor_sha256": sha256_file(extractor),
        "config_restamp": restamp,
        "device": device,
        "protocol_version": release["schema_version"],
    }
    _write_new(output / "dependency_receipt.json", receipt)
    return receipt


def resolve_position(catalog, task_id, receipt_path, catalog_sha256):
    """
    Return four runnable case declarations from a verified completed producer.

    The parent catalog remains immutable; this resolution is an append-only
    dependency product that carries its parent digest and generated hashes.
    """
    task = next((t for t in catalog["position_tasks"] if t["task_id"] == task_id), None)
    if task is None:
        raise ReleaseCatalogError("unknown position task")
    receipt = _read_json(Path(receipt_path))
    if (
        receipt.get("status") != "COMPLETE"
        or receipt.get("task_id") != task_id
        or receipt.get("catalog_sha256") != catalog_sha256
    ):
        raise ReleaseCatalogError("position completion receipt does not bind this task/catalog")
    _, artifact, identity = position_input_identity(task)
    if receipt.get("ladder_artifact_sha256") != sha256_file(artifact):
        raise ReleaseCatalogError("position receipt names a different ladder")
    position = Path(receipt["position"])
    _verify_declared_file(position, receipt["position_sha256"], "completed position")
    payload = _read_json(position)
    validate_position_payload(payload, identity)
    resolved = []
    for original in catalog["cases"]:
        if original["case_id"] not in task["downstream_case_ids"]:
            continue
        case = copy.deepcopy(original)
        case["input_records"]["positions"] = {
            "path": str(position),
            "execution_path": str(position),
            "sha256": sha256_file(position),
        }
        case["frozen_mass_position"] = payload["rungs"][case["rung"]]
        case["status"], case["dispatchable"] = "READY_FRESH_SEARCH", True
        case["dependency_receipt"] = str(Path(receipt_path).resolve())
        case["dependency_receipt_sha256"] = sha256_file(Path(receipt_path))
        case["parent_catalog_sha256"] = catalog_sha256
        template = case["runner_spec_template"]
        case["runner_spec_template"] = _runner_spec_template(
            case,
            template["release_freeze_path"],
            case["release_freeze_sha256"],
            template["procedure_source"],
            case["procedure_source_sha256"],
            template["approval_receipt"],
        )
        resolved.append(case)
    if len(resolved) != 4:
        raise ReleaseCatalogError("position dependency must resolve exactly four fresh arms")
    return resolved


def produce_bracket(catalog_path, case_id, output_root, revision, approval_path, release_path):
    """
    Generate one approved physical truth anchor; no sampler or refit is run.
    """
    catalog_path, release_path, approval_path = map(Path, (catalog_path, release_path, approval_path))
    catalog = _read_json(catalog_path)
    found = [case for case in catalog["cases"] if case["case_id"] == case_id]
    if len(found) != 1 or found[0]["scope"] != "selected12_brackets":
        raise ReleaseCatalogError("case_id must identify exactly one declared bracket")
    case = found[0]
    validate_case_approval(case, _read_json(approval_path), sha256_file(catalog_path))
    _verify_declared_file(release_path, case["release_freeze_sha256"], "release freeze")
    source_config, positions = (
        _path(case["input_records"]["config"]),
        _path(case["input_records"]["positions"]),
    )
    for role in ("source_asset",):
        _path(case["input_records"][role])
    output = Path(output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise ReleaseCatalogError("bracket output must be a new empty namespace")
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "inputs" / "config.yaml"
    restamp = _restamp_config(source_config, config_path, revision)
    import jax

    jax.config.update("jax_enable_x64", True)
    from hwoslaps.modeling.nonlinear.experimental_device import require_cuda_execution

    device = require_cuda_execution()
    _verify_runtime_config(config_path)
    from hwoslaps.modeling.nonlinear.fresh_profile import (
        materialize_bracket_case_from_files,
    )

    target = case["frozen_mass_position"]
    generated = materialize_bracket_case_from_files(
        config_path=config_path,
        positions_path=positions,
        output_dir=output / "generated",
        case_id=case_id,
        bracket_rung=case_id.rsplit(":", 1)[-1],
        target_log10_m200=target["target_log10_m200"],
        target_mass_msun=target["target_mass_msun"],
        position_yx_arcsec=target["position_yx_arcsec"],
    )
    receipt = {
        "status": "COMPLETE",
        "case_id": case_id,
        "catalog_sha256": sha256_file(catalog_path),
        "release_freeze_sha256": sha256_file(release_path),
        "approval_sha256": sha256_file(approval_path),
        "source_config_sha256": sha256_file(source_config),
        "source_positions_sha256": sha256_file(positions),
        "generated": generated,
        "config_restamp": restamp,
        "device": device,
        "h1_zero_residual_gate": "PENDING_CURRENT_OBJECTIVE_EVALUATION_BEFORE_H0_FIT",
        "sampler_executed": False,
    }
    _write_new(output / "dependency_receipt.json", receipt)
    return receipt


def resolve_bracket(catalog, case_id, receipt_path, catalog_sha256):
    """
    Bind generated anchor bytes; the current-objective zero-residual gate
    remains required.
    """
    found = [
        case
        for case in catalog["cases"]
        if case["case_id"] == case_id and case["scope"] == "selected12_brackets"
    ]
    if len(found) != 1:
        raise ReleaseCatalogError("unknown bracket case")
    case = copy.deepcopy(found[0])
    receipt = _read_json(Path(receipt_path))
    if (
        receipt.get("status") != "COMPLETE"
        or receipt.get("case_id") != case_id
        or receipt.get("catalog_sha256") != catalog_sha256
    ):
        raise ReleaseCatalogError("bracket completion receipt does not bind this case/catalog")
    for role in ("config", "positions"):
        if receipt.get(f"source_{role}_sha256") != case["input_records"][role]["sha256"]:
            raise ReleaseCatalogError("bracket source identity mismatch")
    generated = receipt["generated"]
    for role in ("config", "positions", "h1_anchor"):
        _verify_declared_file(Path(generated[role]), generated[f"{role}_sha256"], role)
    target = case["frozen_mass_position"]
    for key in ("target_log10_m200", "target_mass_msun", "position_yx_arcsec"):
        if generated.get(key) != target[key]:
            raise ReleaseCatalogError("bracket target identity mismatch")
    anchor = _read_json(Path(generated["h1_anchor"]))
    if (
        anchor.get("case_id") != case_id
        or anchor.get("evidence_claim") is not False
        or anchor.get("sampler_executed") is not False
    ):
        raise ReleaseCatalogError("bracket anchor identity/policy mismatch")
    for role in ("config", "positions"):
        case["input_records"][role] = {
            "path": generated[role],
            "execution_path": generated[role],
            "sha256": generated[f"{role}_sha256"],
        }
    case["status"], case["dispatchable"] = "READY_FRESH_SEARCH", True
    case["parent_catalog_sha256"] = catalog_sha256
    case["dependency_receipt"] = str(Path(receipt_path).resolve())
    case["dependency_receipt_sha256"] = sha256_file(Path(receipt_path))
    template = case["runner_spec_template"]
    spec = _runner_spec_template(
        case,
        template["release_freeze_path"],
        case["release_freeze_sha256"],
        template["procedure_source"],
        case["procedure_source_sha256"],
        template["approval_receipt"],
    )
    spec.pop("pending", None)
    spec.update(
        h1_anchor=generated["h1_anchor"],
        h1_anchor_sha256=generated["h1_anchor_sha256"],
        bracket_rung=generated["bracket_rung"],
    )
    spec["hashes"][generated["h1_anchor"]] = generated["h1_anchor_sha256"]
    case["runner_spec_template"] = spec
    return case

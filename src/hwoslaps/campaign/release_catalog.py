"""Build and route the additive v7 fresh-search case catalog.

The catalog is a declaration and identity product.  It consumes the already
harvested C inventory and v6 ladder manifest, performs no rendering or fitting,
and refuses to construct a runner invocation for a case whose required
position or bracket-generation input is pending.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .design_freeze import load_release_freeze

STANDARD_ARMS = (
    "asimov_injected",
    "asimov_below",
    "noisy_injected",
    "noisy_control",
)
ARM_INDEX = {
    "asimov_injected": 0,
    "noisy_injected": 1,
    "noisy_control": 2,
    "asimov_below": 3,
    "asimov_fixed_bridge": 4,
    "asimov_injected_r1": 5,
    "asimov_injected_r2": 6,
    **{f"noisy_control_r{i}": 6 + i for i in range(1, 10)},
    "noisy_control_d2": 16,
    "noisy_control_d5": 17,
    "noisy_control_d10": 18,
    "noisy_control_d20": 19,
    "noisy_injected_d2": 20,
    "noisy_injected_d5": 21,
    "noisy_injected_d10": 22,
    "noisy_injected_d20": 23,
}


class ReleaseCatalogError(ValueError):
    """Raised when a v7 catalog input or route is inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ReleaseCatalogError(f"Expected JSON object at {path}")
    return value


def _verify_declared_file(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise ReleaseCatalogError(f"Missing {label}: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise ReleaseCatalogError(f"{label} sha256 {observed} does not match declared {expected}")


def input_source(record: dict[str, Any], label: str) -> Path:
    """Return the checked local mirror of a declared input, else its execution location.

    Catalog input records name the file where the catalog was generated
    (``path``) and where it runs (``execution_path``); a consumer on either
    host takes the first location that exists and proves its declared hash.
    """
    for key in ("path", "execution_path"):
        value = record.get(key)
        if value and Path(value).expanduser().is_file():
            path = Path(value).expanduser().resolve()
            _verify_declared_file(path, record["sha256"], f"{label} {key}")
            return path
    raise ReleaseCatalogError(
        f"{label} is absent at both declared locations: "
        f"{record.get('path')!r}, {record.get('execution_path')!r}"
    )


def _verify_inventory_input_records(jobs: list[dict[str, Any]]) -> dict[str, Any]:
    """Verify every concrete C case config/position/case/asset binding once."""
    roles = ("config", "positions", "aggregate_artifact", "source_asset")
    digest_cache: dict[str, str] = {}
    verified_records = 0
    role_counts = {role: 0 for role in roles}
    for job in jobs:
        records = job.get("input_records")
        if not isinstance(records, dict):
            raise ReleaseCatalogError(f"C inventory row {job.get('row_id')} lacks input_records")
        for role in roles:
            record = records.get(role)
            if not isinstance(record, dict):
                raise ReleaseCatalogError(
                    f"C inventory row {job.get('row_id')} lacks concrete {role} binding"
                )
            path_value = record.get("path")
            expected = record.get("sha256")
            if not isinstance(path_value, str) or not path_value:
                raise ReleaseCatalogError(f"C inventory {role} path is empty")
            if not isinstance(expected, str) or len(expected) != 64:
                raise ReleaseCatalogError(f"C inventory {role} hash is missing")
            path = Path(path_value)
            key = str(path.resolve())
            actual = digest_cache.get(key)
            if actual is None:
                _verify_declared_file(path, expected, f"C inventory {role}")
                actual = expected
                digest_cache[key] = actual
            elif actual != expected:
                raise ReleaseCatalogError(f"C inventory repeats {role} path with conflicting hashes: {path}")
            verified_records += 1
            role_counts[role] += 1
    return {
        "records_verified": verified_records,
        "unique_files_verified": len(digest_cache),
        "role_counts": role_counts,
    }


def _bare_system_id(value: str) -> str:
    matches = re.findall(r"sys\d{4}", str(value))
    if not matches:
        raise ReleaseCatalogError(f"No sysNNNN identifier in {value!r}")
    return matches[-1]


def _system_index(system_id: str) -> int:
    return int(_bare_system_id(system_id)[3:])


def _seed(entropy: int, system_id: str, arm_index: int) -> int:
    sequence = np.random.SeedSequence(
        entropy=int(entropy),
        spawn_key=(5, _system_index(system_id), int(arm_index)),
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _noise_seed(entropy: int, system_id: str, replicate: int) -> int:
    index = _system_index(system_id)
    sequence = np.random.SeedSequence(
        entropy=int(entropy),
        spawn_key=(1, index) if int(replicate) == 0 else (6, int(replicate), index),
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _direction_seed(entropy: int, system_id: str, direction: int) -> int:
    sequence = np.random.SeedSequence(
        entropy=int(entropy),
        spawn_key=(7, int(direction), _system_index(system_id)),
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _ref(value: Any, *, role: str) -> dict[str, Any] | None:
    """Normalize a path record while retaining local and execution paths."""
    if value is None:
        return None
    if isinstance(value, str):
        return {"path": value, "execution_path": value, "sha256": None, "role": role}
    if not isinstance(value, dict):
        raise ReleaseCatalogError(f"Malformed {role} input record")
    path = value.get("path")
    execution_path = value.get("execution_path") or path
    if path is None and execution_path is None:
        raise ReleaseCatalogError(f"{role} has no path")
    result = {
        "path": path,
        "execution_path": execution_path,
        "sha256": value.get("sha256"),
        "size_bytes": value.get("size_bytes"),
        "exists": value.get("exists"),
        "role": role,
    }
    return result


def _archive_case(job: dict, entropy: int) -> dict[str, Any]:
    row = job.get("archived_row", {})
    system_id = _bare_system_id(job.get("system_id", row.get("system_id", "")))
    arm = str(job["arm"])
    if arm not in ARM_INDEX:
        raise ReleaseCatalogError(f"Unmapped archived arm {arm!r}")
    records = job.get("input_records", {})
    config = _ref(records.get("config", job.get("config")), role="config")
    positions = _ref(records.get("positions", job.get("positions")), role="positions")
    aggregate = _ref(
        records.get("aggregate_artifact", job.get("aggregate_artifact")),
        role="archived_case_artifact",
    )
    source = _ref(records.get("source_asset", job.get("source_asset")), role="source_asset")
    arm_index = ARM_INDEX[arm]
    replicate = job.get("noise_replicate")
    direction = job.get("direction")
    case_id = f"archived:{job['row_id']}"
    derived_sampler_seed = _seed(entropy, system_id, arm_index)
    stored_sampler_seed = row.get("sampler_seed")
    noise_seed = _noise_seed(entropy, system_id, int(replicate or 0))
    if row.get("noise_seed") is not None and int(row["noise_seed"]) != noise_seed:
        raise ReleaseCatalogError(f"archived noise seed mismatch for {job['row_id']}")
    if stored_sampler_seed is not None and int(stored_sampler_seed) != derived_sampler_seed:
        raise ReleaseCatalogError(
            f"archived sampler seed mismatch for {job['row_id']}: "
            f"derived {derived_sampler_seed}, stored {stored_sampler_seed}"
        )
    input_signature = {
        "case_id": case_id,
        "system_id": system_id,
        "arm": arm,
        "campaign": job.get("campaign"),
        "row_index": job.get("row_index"),
        "row_id": job.get("row_id"),
        "direction": direction,
        "noise_replicate": replicate,
    }
    case = {
        "case_id": case_id,
        "scope": "archived_cases",
        "status": "READY_FRESH_SEARCH",
        "dispatchable": True,
        "system_id": system_id,
        "run_name": job.get("system_id"),
        "template": job.get("template", row.get("template")),
        "campaign": job.get("campaign"),
        "campaign_expected_rows": job.get("campaign_expected_rows"),
        "row_index": job.get("row_index"),
        "row_id": job.get("row_id"),
        "arm": arm,
        "arm_index": arm_index,
        "direction": direction,
        "noise_replicate": replicate,
        "dataset_kind": job.get("dataset_kind", row.get("dataset_kind")),
        "fit_mode": job.get("fit_mode", row.get("fit_mode")),
        "rung": job.get("rung_name", row.get("rung_name")),
        "censored": bool(job.get("censored", row.get("censored", False))),
        "frozen_mass_position": {
            "mass_log10_m200": row.get("injection_logm"),
            "position_yx_arcsec": row.get("position_yx_arcsec"),
        },
        "sampler_seed": derived_sampler_seed,
        "sampler_seed_spawn_key": [5, _system_index(system_id), arm_index],
        "noise_seed": noise_seed,
        "direction_seed": (
            _direction_seed(entropy, system_id, int(direction)) if direction is not None else None
        ),
        "input_records": {
            "config": config,
            "positions": positions,
            "archived_case_artifact": aggregate,
            "source_asset": source,
        },
        "archived_values": {
            key: row.get(key)
            for key in (
                "q_f_production",
                "q_f_matched",
                "q_fit",
                "delta_log_likelihood",
                "delta_log_evidence",
                "sampler_seed",
                "quality_flags",
            )
            if key in row
        },
        "old_state_policy": "identity_only_not_loaded",
        "fresh_search_namespace": f"v7/archived/{job['row_id'].replace(':', '_')}",
        "case_identity_payload": input_signature,
        "case_identity_signature": _sha256_json(input_signature),
        "runner": {
            "entrypoint": "scripts/run_nonlinear_validation.py",
            "mode": "standard_arm",
            "argv_template": ["{config}", "{positions}", arm, "{output_dir}"],
            "fresh_search": True,
            "reuse_archived_fit_state": False,
        },
    }
    return case


def _new_top50_cases(
    job: dict,
    entropy: int,
    verified_positions: dict[str, dict[str, Any]],
    template_by_hash: dict[str, str],
    v6_artifact: dict[str, Any],
) -> list[dict[str, Any]]:
    system_id = _bare_system_id(job["system_id"])
    position = verified_positions.get(system_id)
    source_hash = job.get("source_asset_sha256")
    config_ref = _ref(job.get("config"), role="v6_config")
    ladder_ref = _ref(job.get("artifact"), role="v6_ladder_artifact")
    if config_ref is None or ladder_ref is None:
        raise ReleaseCatalogError(f"v6 job {system_id} lacks config or ladder artifact")
    config_ref["sha256"] = job.get("config_sha256")
    config_ref["path"] = v6_artifact["config_path"]
    config_ref["execution_path"] = v6_artifact["config_execution_path"]
    ladder_ref["sha256"] = v6_artifact["sha256"]
    ladder_ref["path"] = v6_artifact["path"]
    ladder_ref["execution_path"] = v6_artifact["execution_path"]
    if not config_ref["sha256"] or not ladder_ref["sha256"]:
        raise ReleaseCatalogError(f"v6 job {system_id} lacks config/artifact hashes")
    cases = []
    for arm in STANDARD_ARMS:
        arm_index = ARM_INDEX[arm]
        is_ready = position is not None
        case_id = f"new_top50:{system_id}:{arm}"
        input_signature = {
            "case_id": case_id,
            "system_id": system_id,
            "arm": arm,
            "source": "v6_ladder_manifest",
            "config_sha256": job.get("config_sha256"),
            "ladder_artifact_sha256": v6_artifact["sha256"],
            "position_sha256": position.get("sha256") if position else None,
        }
        case = {
            "case_id": case_id,
            "scope": "new_top50_standard_cases",
            "status": "READY_FRESH_SEARCH" if is_ready else "PENDING_POSITION_EXTRACTION",
            "dispatchable": bool(is_ready),
            "system_id": system_id,
            "run_name": f"ladder_full_pool_{system_id}",
            "template": template_by_hash.get(source_hash),
            "campaign": "v6_fisher_production_20260915",
            "arm": arm,
            "arm_index": arm_index,
            "dataset_kind": "asimov" if arm.startswith("asimov") else "noisy",
            "fit_mode": "freed",
            "rung": "top" if arm in {"asimov_injected", "noisy_injected", "noisy_control"} else "below",
            "source_asset_sha256": source_hash,
            "sampler_seed": _seed(entropy, system_id, arm_index),
            "sampler_seed_spawn_key": [5, _system_index(system_id), arm_index],
            "noise_seed_policy": "preserve_v6_config_global_seed" if arm.startswith("noisy") else None,
            "input_records": {
                "config": config_ref,
                "v6_ladder_artifact": ladder_ref,
                "source_asset": {
                    "path": None,
                    "execution_path": None,
                    "sha256": source_hash,
                    "role": "source_asset",
                    "path_policy": "resolve_from_v6_config_at_execution",
                },
                "positions": position,
            },
            "frozen_mass_position": (
                position.get("rungs", {}).get("top" if arm != "asimov_below" else "below")
                if position
                else {
                    "status": "PENDING_EXTRACTION",
                    "source_ladder_artifact_sha256": v6_artifact["sha256"],
                }
            ),
            "old_state_policy": "none_available_and_never_loaded",
            "fresh_search_namespace": f"v7/new_top50/{system_id}/{arm}",
            "case_identity_payload": input_signature,
            "case_identity_signature": _sha256_json(input_signature),
            "runner": {
                "entrypoint": "scripts/run_nonlinear_validation.py",
                "mode": "standard_arm",
                "argv_template": ["{config}", "{positions}", arm, "{output_dir}"],
                "fresh_search": True,
                "reuse_archived_fit_state": False,
                "requires_position_extraction": not is_ready,
            },
        }
        cases.append(case)
    return cases


def _bracket_cases(
    selected_rows: dict[str, dict[str, Any]],
    entropy: int,
) -> list[dict[str, Any]]:
    cases = []
    for system_id, selected in sorted(selected_rows.items()):
        position_payload = selected["position_payload"]
        top = position_payload["rungs"]["top"]
        for offset in (0.1, 0.2, 0.3):
            target_logm = float(top["logm"]) + offset
            case_id = f"selected12_bracket:{system_id}:plus_{offset:.1f}dex"
            signature = {
                "case_id": case_id,
                "system_id": system_id,
                "arm": "h0_bracket",
                "scope": "selected12_brackets",
                "offset_dex": offset,
                "config_sha256": selected["config"]["sha256"],
                "positions_sha256": selected["positions"]["sha256"],
            }
            cases.append(
                {
                    "case_id": case_id,
                    "scope": "selected12_brackets",
                    "status": "PENDING_BRACKET_GENERATION",
                    "dispatchable": False,
                    "system_id": system_id,
                    "run_name": selected["run_name"],
                    "template": selected["template"],
                    "campaign": selected["campaign"],
                    "arm": "h0_bracket",
                    "dataset_kind": "asimov",
                    "fit_mode": "h0_only_after_h1_anchor",
                    "rung": "above_upper_rung",
                    "sampler_seed": _seed(entropy, system_id, 24 + round(offset * 10)),
                    "sampler_seed_spawn_key": [5, _system_index(system_id), 24 + round(offset * 10)],
                    "input_records": {
                        "config": selected["config"],
                        "positions": selected["positions"],
                        "base_case_artifact": selected["case_artifact"],
                        "source_asset": selected["source_asset"],
                    },
                    "frozen_mass_position": {
                        "base_upper_rung_log10_m200": float(top["logm"]),
                        "offset_dex": offset,
                        "target_log10_m200": target_logm,
                        "target_mass_msun": float(10.0**target_logm),
                        "position_yx_arcsec": list(top["position_yx_arcsec"]),
                        "position_source": selected["positions"]["execution_path"],
                    },
                    "mass_mapping_policy": "recompute_physical_subhalo_parameters_at_generation",
                    "truth_anchor": "H1_fixed_truth_zero_residual_required_before_H0_profile",
                    "noise_policy": "same_declared_Asimov_observation_convention; no noise seed",
                    "old_state_policy": "none_loaded",
                    "fresh_search_namespace": f"v7/selected12_brackets/{system_id}/plus_{offset:.1f}dex",
                    "case_identity_payload": signature,
                    "case_identity_signature": _sha256_json(signature),
                    "runner": {
                        "entrypoint": "scripts/run_nonlinear_validation.py",
                        "mode": "h0_bracket_adapter_owned_by_optimizer_stage3",
                        "argv_template": [
                            "{generated_config}",
                            "{positions}",
                            "{bracket_arm}",
                            "{output_dir}",
                        ],
                        "fresh_search": True,
                        "reuse_archived_fit_state": False,
                        "requires_bracket_generation": True,
                        "h1_nautilus_forbidden": True,
                        "evidence_claim_forbidden": True,
                    },
                }
            )
    return cases


def _position_tasks(
    missing_systems: list[str],
    v6_by_system: dict[str, dict[str, Any]],
    verified_positions: dict[str, dict[str, Any]],
    v6_artifacts: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Create the producer-to-fit dependency records for all 28 additions."""
    tasks = []
    for system_id in missing_systems:
        job = v6_by_system[system_id]
        verified = verified_positions.get(system_id)
        config = _ref(job.get("config"), role="v6_config")
        ladder = _ref(job.get("artifact"), role="v6_ladder_artifact")
        if config is None or ladder is None:
            raise ReleaseCatalogError(f"v6 position task {system_id} lacks inputs")
        config["sha256"] = job.get("config_sha256")
        config["path"] = v6_artifacts[system_id]["config_path"]
        config["execution_path"] = v6_artifacts[system_id]["config_execution_path"]
        ladder["sha256"] = v6_artifacts[system_id]["sha256"]
        ladder["path"] = v6_artifacts[system_id]["path"]
        ladder["execution_path"] = v6_artifacts[system_id]["execution_path"]
        downstream = [f"new_top50:{system_id}:{arm}" for arm in STANDARD_ARMS]
        if verified:
            tasks.append(
                {
                    "task_id": f"position:{system_id}",
                    "system_id": system_id,
                    "status": "VERIFIED_REUSED",
                    "dispatchable": False,
                    "completion_verified": True,
                    "input_records": {"config": config, "v6_ladder_artifact": ladder},
                    "output": verified,
                    "downstream_case_ids": downstream,
                    "producer": {
                        "entrypoint": "scripts/extract_injection_positions.py",
                        "argv_template": ["{config}", "{v6_ladder_artifact}", "{position_output_dir}"],
                        "no_fit": True,
                    },
                }
            )
            continue
        tasks.append(
            {
                "task_id": f"position:{system_id}",
                "system_id": system_id,
                "status": "PENDING_POSITION_EXTRACTION",
                "dispatchable": False,
                "completion_verified": False,
                "input_records": {"config": config, "v6_ladder_artifact": ladder},
                "output": {
                    "status": "PENDING",
                    "path": f"{{position_output_root}}/{system_id}/injection_position.json",
                    "sha256": None,
                },
                "downstream_case_ids": downstream,
                "producer": {
                    "entrypoint": "scripts/prepare_nonlinear_dependencies.py",
                    "argv_template": [
                        "produce-position",
                        "--catalog",
                        "{catalog}",
                        "--task-id",
                        f"position:{system_id}",
                        "--output",
                        "{new_empty_position_output}",
                        "--revision",
                        "{revision_json}",
                        "--approval",
                        "{approval_receipt}",
                        "--release-freeze",
                        "{release_freeze}",
                    ],
                    "resolver_entrypoint": "scripts/prepare_nonlinear_dependencies.py resolve-position",
                    "no_fit": True,
                    "requires_v7_identity_check": True,
                },
            }
        )
    return tasks


def _bracket_tasks(bracket_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create explicit materialization dependencies for the 36 H0 brackets."""
    tasks = []
    for case in bracket_cases:
        target = case["frozen_mass_position"]
        tasks.append(
            {
                "task_id": f"bracket_generation:{case['case_id']}",
                "case_id": case["case_id"],
                "system_id": case["system_id"],
                "status": "PENDING_BRACKET_GENERATION",
                "dispatchable": False,
                "input_records": case["input_records"],
                "target": target,
                "output": {
                    "generated_config": "{bracket_output_root}/config.yaml",
                    "h1_anchor": "{bracket_output_root}/h1_anchor.json",
                    "generated_case_sha256": None,
                    "h1_anchor_sha256": None,
                },
                "producer": {
                    "mode": "approved_bracket_materialization",
                    "entrypoint": "scripts/prepare_nonlinear_dependencies.py",
                    "argv_template": [
                        "produce-bracket",
                        "--catalog",
                        "{catalog}",
                        "--case-id",
                        case["case_id"],
                        "--output",
                        "{new_empty_bracket_output}",
                        "--revision",
                        "{revision_json}",
                        "--approval",
                        "{approval_receipt}",
                        "--release-freeze",
                        "{release_freeze}",
                    ],
                    "resolver_entrypoint": "scripts/prepare_nonlinear_dependencies.py resolve-bracket",
                    "action": "generate_physical_anchor_then_verify_zero_residual_in_current_fit",
                    "h1_nautilus_forbidden": True,
                    "evidence_claim_forbidden": True,
                    "same_upper_position_required": True,
                    "future_hashes_forbidden": True,
                },
                "downstream": case["case_id"],
            }
        )
    return tasks


def _runner_spec_template(
    case: dict[str, Any],
    release_execution_path: str,
    release_digest: str,
    procedure_source_execution_path: str,
    procedure_source_sha256: str,
    approval_receipt_path: str,
) -> dict[str, Any]:
    """Build the optimizer-owned per-case spec consumed by its Stage 3 CLI."""
    inputs = case.get("input_records", {})
    identity_payload = case.get("case_identity_payload")
    if not isinstance(identity_payload, dict):
        raise ReleaseCatalogError(f"case {case['case_id']} lacks its canonical identity payload")
    compute_tangent = case.get("compute_tangent_comparator", case.get("scope") == "selected12_brackets")
    if not isinstance(compute_tangent, bool):
        raise ReleaseCatalogError("compute_tangent_comparator must be an explicit boolean")
    identity_payload = json.loads(json.dumps(identity_payload))
    identity_payload["compute_tangent_comparator"] = compute_tangent
    compute_bracket_fisher = case.get("compute_bracket_fisher_q", case.get("scope") == "selected12_brackets")
    if not isinstance(compute_bracket_fisher, bool) or compute_bracket_fisher != (
        case.get("scope") == "selected12_brackets"
    ):
        raise ReleaseCatalogError("compute_bracket_fisher_q must be true exactly for brackets")
    identity_payload["compute_bracket_fisher_q"] = compute_bracket_fisher
    config = inputs.get("config")
    positions = inputs.get("positions")
    if not isinstance(config, dict) or not isinstance(positions, dict):
        return {
            "case_id": case["case_id"],
            "system_id": case["system_id"],
            "direction": case.get("direction"),
            "case_identity": case["case_id"],
            "case_identity_payload": identity_payload,
            "case_identity_signature": _sha256_json(identity_payload),
            "compute_tangent_comparator": compute_tangent,
            "compute_bracket_fisher_q": compute_bracket_fisher,
            "case_kind": "bracket" if case["scope"] == "selected12_brackets" else "standard",
            "scope": case["scope"],
            "release_freeze_path": release_execution_path,
            "release_freeze_sha256": release_digest,
            "objective_version": "consistent_sampling_v2",
            "procedure_version": "fresh_nonlinear_v7_lbfgsb_v1",
            "procedure_source": procedure_source_execution_path,
            "source_assets": [],
            "approval_receipt": approval_receipt_path,
            "approval_receipt_sha256": None,
            "catalog_sha256": None,
            "execution_policy_version": "stage3_v7",
            "pending": True,
        }
    config_path = config.get("execution_path") or config.get("path")
    positions_path = positions.get("execution_path") or positions.get("path")
    source_asset = inputs.get("source_asset")
    source_asset_path = source_asset.get("execution_path") if isinstance(source_asset, dict) else None
    source_asset_sha256 = source_asset.get("sha256") if isinstance(source_asset, dict) else None
    hashes = {
        release_execution_path: release_digest,
        procedure_source_execution_path: procedure_source_sha256,
    }
    if config_path and config.get("sha256"):
        hashes[str(config_path)] = str(config["sha256"])
    if positions_path and positions.get("sha256"):
        hashes[str(positions_path)] = str(positions["sha256"])
    if source_asset_path and source_asset_sha256:
        hashes[str(source_asset_path)] = str(source_asset_sha256)
    template = {
        "case_id": case["case_id"],
        "system_id": case["system_id"],
        "direction": case.get("direction"),
        "case_identity": case["case_id"],
        "case_identity_payload": identity_payload,
        "case_identity_signature": _sha256_json(identity_payload),
        "compute_tangent_comparator": compute_tangent,
        "compute_bracket_fisher_q": compute_bracket_fisher,
        "case_kind": "bracket" if case["scope"] == "selected12_brackets" else "standard",
        "scope": case["scope"],
        "config": config_path,
        "positions": positions_path,
        "arm": case["arm"],
        "release_freeze_path": release_execution_path,
        "release_freeze_sha256": release_digest,
        "objective_version": "consistent_sampling_v2",
        "procedure_version": "fresh_nonlinear_v7_lbfgsb_v1",
        "procedure_source": procedure_source_execution_path,
        "source_assets": [source_asset_path] if source_asset_path else [],
        "approval_receipt": approval_receipt_path,
        "approval_receipt_sha256": None,
        "catalog_sha256": None,
        "execution_policy_version": "stage3_v7",
        "output": "{output_dir}",
        "hashes": hashes,
    }
    if case["scope"] == "selected12_brackets":
        template.update(
            {
                "pending": True,
                "h1_anchor": "{h1_anchor}",
                "h1_anchor_sha256": None,
                "bracket_rung": case["case_id"].rsplit(":", 1)[-1],
                "bracket_arm_index": case["sampler_seed_spawn_key"][-1],
            }
        )
    return template


def validate_case_approval(case, approval, catalog_sha256):
    """Require exact case membership in a hash-bound launch authorization."""
    if not isinstance(approval, dict) or approval.get("status") != "APPROVED":
        raise ReleaseCatalogError("an APPROVED immutable approval receipt is required")
    if approval.get("release_freeze_sha256") != case.get("release_freeze_sha256"):
        raise ReleaseCatalogError("approval receipt names a different release freeze")
    if (
        not isinstance(catalog_sha256, str)
        or len(catalog_sha256) != 64
        or approval.get("catalog_sha256") != catalog_sha256
    ):
        raise ReleaseCatalogError("approval receipt names a different catalog digest")
    case_ids = approval.get("case_ids")
    if (
        not isinstance(case_ids, list)
        or len(case_ids) != len(set(case_ids))
        or case.get("case_id") not in case_ids
    ):
        raise ReleaseCatalogError("approval receipt does not authorize this exact case_id")
    scope = approval.get("authorized_scope")
    allowed = {case.get("scope"), "all_standard_and_brackets"}
    if case.get("scope") != "selected12_brackets":
        allowed.add("all_standard_cases")
    if scope not in allowed:
        raise ReleaseCatalogError("approval receipt does not authorize this catalog scope")
    limit = approval.get("authorized_gpu_limit")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 8:
        raise ReleaseCatalogError("approval receipt must authorize between one and eight GPUs")


def runner_invocation(
    case: dict,
    output_dir: str | Path,
    *,
    approval_receipt: dict[str, Any] | None = None,
    catalog_sha256: str | None = None,
) -> dict[str, Any]:
    """Return a concrete runner route after the immutable approval gate."""
    if not case.get("dispatchable"):
        raise ReleaseCatalogError(
            f"Case {case.get('case_id')} is {case.get('status')} and cannot be dispatched"
        )
    if not isinstance(approval_receipt, dict):
        raise ReleaseCatalogError("an immutable v7 approval receipt is required before dispatch")
    if approval_receipt.get("status") != "APPROVED":
        raise ReleaseCatalogError("v7 approval receipt status is not APPROVED")
    if approval_receipt.get("release_freeze_sha256") != case.get("release_freeze_sha256"):
        raise ReleaseCatalogError("approval receipt does not name this release freeze digest")
    if (
        not isinstance(catalog_sha256, str)
        or len(catalog_sha256) != 64
        or approval_receipt.get("catalog_sha256") != catalog_sha256
    ):
        raise ReleaseCatalogError("approval receipt does not name this catalog digest")
    approval_path = approval_receipt.get("path")
    approval_sha256 = approval_receipt.get("sha256")
    if (
        not isinstance(approval_path, str)
        or not approval_path
        or not isinstance(approval_sha256, str)
        or len(approval_sha256) != 64
    ):
        raise ReleaseCatalogError("approval receipt must carry its immutable path and SHA-256")
    validate_case_approval(case, approval_receipt, catalog_sha256)
    route = case.get("runner", {})
    if route.get("mode") != "standard_case_spec":
        raise ReleaseCatalogError(f"Case {case.get('case_id')} needs route adapter {route.get('mode')!r}")
    spec_template = case.get("runner_spec_template")
    if not isinstance(spec_template, dict):
        raise ReleaseCatalogError(f"Case {case['case_id']} has no production spec template")
    spec = json.loads(json.dumps(spec_template))
    spec["output"] = str(Path(output_dir).expanduser().resolve())
    spec["approval_receipt"] = approval_path
    spec["approval_receipt_sha256"] = approval_sha256
    spec["catalog_sha256"] = catalog_sha256
    spec["hashes"][approval_path] = approval_sha256
    environment = {
        "HWOSLAPS_RELEASE_CASE_ID": case["case_id"],
        "HWOSLAPS_RELEASE_NAMESPACE": case["fresh_search_namespace"],
        "HWOSLAPS_REUSE_ARCHIVED_FIT_STATE": "0",
    }
    aggregate = case["input_records"].get("archived_case_artifact")
    if aggregate and aggregate.get("sha256"):
        environment["HWOSLAPS_ARCHIVED_CASE_IDENTITY_SHA256"] = str(aggregate["sha256"])
    return {
        "entrypoint": route["entrypoint"],
        "argv": [route["entrypoint"], "{case_spec_path}"],
        "spec": spec,
        "environment": environment,
    }


def _scientific_config_view(config: dict[str, Any]) -> dict[str, Any]:
    """Return a config view whose only allowed restamp field is removed."""
    view = deepcopy(config)
    try:
        del view["stage0"]["code_revision"]
    except KeyError as exc:
        raise ReleaseCatalogError("case config has no stage0.code_revision restamp field") from exc
    return view


def _restamp_config(
    source_path: Path,
    destination_path: Path,
    revision: dict[str, Any],
    *,
    fit_kernel: bool = False,
) -> dict[str, Any]:
    """Write a restamped config after proving no scientific field changed."""
    if not isinstance(revision, dict) or set(revision) != {"git_hash", "git_dirty", "sha256"}:
        raise ReleaseCatalogError("revision must contain exactly git_hash, git_dirty and sha256")
    if revision["git_dirty"] is not False:
        raise ReleaseCatalogError("refusing to materialize from a dirty revision")
    if not isinstance(revision["git_hash"], str) or not revision["git_hash"]:
        raise ReleaseCatalogError("revision.git_hash must be nonempty")
    if not isinstance(revision["sha256"], str) or len(revision["sha256"]) != 64:
        raise ReleaseCatalogError("revision.sha256 must be a full SHA-256 digest")
    with source_path.open(encoding="utf-8") as stream:
        original = yaml.safe_load(stream)
    if not isinstance(original, dict):
        raise ReleaseCatalogError(f"config is not a mapping: {source_path}")
    updated = deepcopy(original)
    updated["stage0"]["code_revision"] = revision
    if _scientific_config_view(original) != _scientific_config_view(updated):
        raise ReleaseCatalogError(f"restamp changed scientific config fields: {source_path}")
    if fit_kernel:
        original_shape = updated.get("psf", {}).get("kernel", {}).get("shape_native")
        if original_shape not in ([999, 999], [51, 51]):
            raise ReleaseCatalogError("new-system config has an undeclared PSF kernel")
        updated["psf"]["kernel"]["shape_native"] = [51, 51]
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_text(
        yaml.safe_dump(updated, sort_keys=False),
        encoding="utf-8",
    )
    return {
        "original_path": str(source_path.resolve()),
        "original_sha256": sha256_file(source_path),
        "materialized_path": str(destination_path.resolve()),
        "materialized_sha256": sha256_file(destination_path),
        "allowed_changed_path": "stage0.code_revision",
        "declared_kernel_conversion": "999_to_51_fit_only" if fit_kernel else None,
        "revision": revision,
    }


def materialize_case_spec(
    case: dict[str, Any],
    output_root: Path,
    revision: dict[str, Any],
    *,
    approval_receipt: dict[str, Any] | None = None,
    approval_receipt_path: str | None = None,
    catalog_sha256: str | None = None,
    execution_output_root: Path | None = None,
) -> dict[str, Any]:
    """Materialize one ready case and return its non-launching spec receipt."""
    if case.get("status") != "READY_FRESH_SEARCH" or not case.get("dispatchable"):
        raise ReleaseCatalogError(f"case {case.get('case_id')} is not ready for materialization")
    inputs = case.get("input_records", {})
    config = inputs.get("config")
    positions = inputs.get("positions")
    if not isinstance(config, dict) or not isinstance(positions, dict):
        raise ReleaseCatalogError(f"case {case['case_id']} has incomplete inputs")
    config_source = input_source(config, f"case {case['case_id']} config")
    positions_source = input_source(positions, f"case {case['case_id']} positions")
    case_dir = Path(output_root).expanduser().resolve() / case["case_id"].replace(":", "__")
    if case_dir.exists() and any(case_dir.iterdir()):
        raise ReleaseCatalogError(f"materialization output is not empty: {case_dir}")
    config_destination = case_dir / "inputs" / "config.yaml"
    receipt = _restamp_config(
        config_source,
        config_destination,
        revision,
        fit_kernel=case.get("scope") == "new_top50_standard_cases",
    )
    template = case.get("runner_spec_template")
    if not isinstance(template, dict) or template.get("pending") is True:
        raise ReleaseCatalogError(f"case {case['case_id']} has no runnable spec template")
    spec = json.loads(json.dumps(template))
    execution_dir = (Path(execution_output_root) / case_dir.name) if execution_output_root else case_dir
    config_execution = str(execution_dir / "inputs" / "config.yaml")
    spec["config"] = config_execution
    spec["scope"] = case["scope"]
    spec["catalog_sha256"] = catalog_sha256
    identity = json.loads(json.dumps(spec.get("case_identity_payload", {})))
    spec["catalog_case_identity_signature"] = case.get("case_identity_signature")
    identity["source_config_sha256"] = config["sha256"]
    identity["config_sha256"] = receipt["materialized_sha256"]
    identity["positions_sha256"] = positions["sha256"]
    spec["case_identity_payload"] = identity
    spec["case_identity_signature"] = _sha256_json(identity)
    positions_execution = positions.get("execution_path") or str(positions_source)
    spec["positions"] = positions_execution
    spec["output"] = str(execution_dir / "attempt")
    spec["hashes"] = {
        config_execution: receipt["materialized_sha256"],
        positions_execution: positions["sha256"],
        str(spec["release_freeze_path"]): spec["release_freeze_sha256"],
    }
    if case.get("dependency_receipt"):
        spec["dependency_receipt"] = case["dependency_receipt"]
        spec["hashes"][case["dependency_receipt"]] = case["dependency_receipt_sha256"]
    if spec.get("h1_anchor") and spec.get("h1_anchor_sha256"):
        spec["hashes"][spec["h1_anchor"]] = spec["h1_anchor_sha256"]
    procedure_source = spec.get("procedure_source")
    for source_asset in spec.get("source_assets", []):
        source_ref = inputs.get("source_asset", {})
        source_hash = source_ref.get("sha256") if isinstance(source_ref, dict) else None
        if source_hash:
            spec["hashes"][source_asset] = source_hash
    procedure_source_hash = case.get("procedure_source_sha256")
    if procedure_source and procedure_source_hash:
        spec["hashes"][procedure_source] = procedure_source_hash
    materialization_status = "MATERIALIZED_PENDING_APPROVAL"
    if approval_receipt is not None:
        validate_case_approval(case, approval_receipt, catalog_sha256)
        if not isinstance(approval_receipt, dict):
            raise ReleaseCatalogError("approval_receipt must be a mapping")
        if approval_receipt.get("status") != "APPROVED":
            raise ReleaseCatalogError("approval_receipt is not APPROVED")
        if approval_receipt_path is None:
            approval_receipt_path = approval_receipt.get("path")
        approval_sha256 = approval_receipt.get("sha256")
        if (
            not isinstance(approval_receipt_path, str)
            or not approval_receipt_path
            or not isinstance(approval_sha256, str)
            or len(approval_sha256) != 64
            or not isinstance(catalog_sha256, str)
            or len(catalog_sha256) != 64
            or approval_receipt.get("catalog_sha256") != catalog_sha256
            or approval_receipt.get("release_freeze_sha256") != spec["release_freeze_sha256"]
        ):
            raise ReleaseCatalogError("approval receipt does not bind the exact release/catalog digests")
        spec["approval_receipt"] = approval_receipt_path
        spec["approval_receipt_sha256"] = approval_sha256
        spec["catalog_sha256"] = catalog_sha256
        spec["hashes"][approval_receipt_path] = approval_sha256
        materialization_status = "MATERIALIZED_APPROVED_NOT_LAUNCHED"
    spec_path = case_dir / "case_spec.json"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(
        json.dumps(spec, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    materialization = {
        "schema_version": 1,
        "status": materialization_status,
        "case_id": case["case_id"],
        "scope": case["scope"],
        "case_spec": str(spec_path),
        "case_spec_sha256": sha256_file(spec_path),
        "config_restamp": receipt,
        "positions": {
            "path": str(positions_source.resolve()),
            "execution_path": positions.get("execution_path"),
            "sha256": positions["sha256"],
        },
        "runner": {
            "entrypoint": "scripts/run_nonlinear_production.py",
            "argv": ["scripts/run_nonlinear_production.py", str(spec_path)],
            "fresh_search": True,
            "reuse_archived_fit_state": False,
            "approved": materialization_status == "MATERIALIZED_APPROVED_NOT_LAUNCHED",
        },
    }
    (case_dir / "materialization.json").write_text(
        json.dumps(materialization, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return materialization


def materialize_ready_cases(
    catalog: dict[str, Any],
    output_root: Path,
    revision: dict[str, Any],
    *,
    approval_receipt: dict[str, Any] | None = None,
    approval_receipt_path: str | None = None,
    catalog_sha256: str | None = None,
    execution_output_root: Path | None = None,
) -> dict[str, Any]:
    """Materialize ready cases while leaving pending cases untouched."""
    results = []
    for case in catalog.get("cases", []):
        if case.get("status") != "READY_FRESH_SEARCH":
            continue
        results.append(
            materialize_case_spec(
                case,
                output_root,
                revision,
                approval_receipt=approval_receipt,
                approval_receipt_path=approval_receipt_path,
                catalog_sha256=catalog_sha256,
                execution_output_root=execution_output_root,
            )
        )
    manifest = {
        "schema_version": 1,
        "status": (
            "MATERIALIZED_APPROVED_NOT_LAUNCHED"
            if approval_receipt is not None
            else "MATERIALIZED_PENDING_APPROVAL"
        ),
        "release_freeze": catalog.get("release_freeze"),
        "catalog_case_count": len(catalog.get("cases", [])),
        "materialized_case_count": len(results),
        "pending_case_count": sum(
            case.get("status") != "READY_FRESH_SEARCH" for case in catalog.get("cases", [])
        ),
        "materializations": results,
        "no_rendering_or_fitting": True,
        "launch_required_approval": True,
    }
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "MATERIALIZATION_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def _validate_source_bindings(release: dict, paths: dict[str, Path]) -> dict[str, Any]:
    sources = release["sources"]
    observed = {}
    for name, path in paths.items():
        expected = sources[name]["sha256"]
        _verify_declared_file(path, expected, name)
        observed[name] = {"path": str(path.resolve()), "sha256": expected}
    return observed


def _load_verified_positions(
    release: dict,
    local_position_root: Path | None,
) -> dict[str, dict[str, Any]]:
    declared = release.get("verified_position_inputs", {})
    systems = declared.get("systems", {})
    if len(systems) != 3:
        raise ReleaseCatalogError("v7 must declare exactly three reusable position inputs")
    result = {}
    for system_id, item in systems.items():
        remote = Path(item["path"])
        local = None
        if local_position_root is not None:
            try:
                local = local_position_root / remote.relative_to(
                    Path("/data/home/gvassilakis/nonlinear_release_20260917/stage12_20260917")
                )
            except ValueError as exc:
                raise ReleaseCatalogError(f"Verified position path outside stage root: {remote}") from exc
            _verify_declared_file(local, item["sha256"], f"verified position {system_id}")
            payload = _read_json(local)
            path = str(local.resolve())
        else:
            payload = None
            path = str(remote)
        result[system_id] = {
            "path": path,
            "execution_path": str(remote),
            "sha256": item["sha256"],
            "config_sha256": item["config_sha256"],
            "ladder_artifact_sha256": item["ladder_artifact_sha256"],
            "extraction_receipt": item["extraction_receipt"],
            "status": "VERIFIED_POSITION_REUSED",
            "rungs": payload.get("rungs") if payload else None,
            "position_payload": payload,
        }
    return result


def _validate_v6_ladder_artifact(
    job: dict[str, Any],
    output_root: Path,
    campaign_uuid: str,
) -> dict[str, Any]:
    """Hash and identity-check one actual v6 ladder_result.npz file."""
    system_id = _bare_system_id(job.get("system_id", ""))
    expected_run_name = f"ladder_full_pool_{system_id}"
    execution_path = str(job["artifact"])
    artifact_path = output_root / Path(execution_path).parent.name / Path(execution_path).name
    if not artifact_path.is_file():
        raise ReleaseCatalogError(f"Actual v6 ladder artifact is missing for {system_id}: {artifact_path}")
    artifact_sha256 = sha256_file(artifact_path)
    config_execution_path = str(job["config"])
    config_path = output_root.parent / "configs" / Path(config_execution_path).name
    _verify_declared_file(config_path, str(job["config_sha256"]), f"v6 config {system_id}")
    with np.load(artifact_path, allow_pickle=False) as record:

        def scalar(name: str) -> Any:
            if name not in record.files:
                raise ReleaseCatalogError(f"v6 ladder artifact {artifact_path} lacks {name}")
            value = record[name]
            return value.item() if getattr(value, "shape", ()) == () else value

        if str(scalar("system_id")) != expected_run_name:
            raise ReleaseCatalogError(
                f"v6 ladder artifact {artifact_path} system_id does not match {expected_run_name}"
            )
        if str(scalar("config_hash")) != str(job.get("config_hash")):
            raise ReleaseCatalogError(
                f"v6 ladder artifact {artifact_path} config hash does not match manifest"
            )
        if str(scalar("campaign_uuid")) != campaign_uuid:
            raise ReleaseCatalogError(
                f"v6 ladder artifact {artifact_path} campaign UUID does not match manifest"
            )
        if str(scalar("source_asset_sha256")) != str(job.get("source_asset_sha256")):
            raise ReleaseCatalogError(
                f"v6 ladder artifact {artifact_path} source asset hash does not match manifest"
            )
        psf_kernel_shape_native = [
            int(value) for value in np.asarray(scalar("psf_kernel_shape_native"), dtype=int)
        ]
    return {
        "config_path": str(config_path.resolve()),
        "config_execution_path": config_execution_path,
        "config_sha256": str(job["config_sha256"]),
        "path": str(artifact_path.resolve()),
        "execution_path": execution_path,
        "sha256": artifact_sha256,
        "manifest_stage0_artifact_sha256": job.get("stage0_artifact_sha256"),
        "manifest_config_sha256": job.get("config_sha256"),
        "run_name": expected_run_name,
        "psf_kernel_shape_native": psf_kernel_shape_native,
    }


def build_catalog(
    release_path: Path,
    inventory_path: Path,
    dispatch_path: Path,
    v6_manifest_path: Path,
    cohort_path: Path,
    output_dir: Path,
    *,
    local_position_root: Path | None = None,
    v6_output_root: Path | None = None,
) -> dict[str, Any]:
    """Build the complete declaration catalog without rendering or fitting."""
    release = load_release_freeze(release_path)
    source_paths = {
        "archived_case_inventory": inventory_path,
        "archived_dispatch": dispatch_path,
        "v6_ladder_manifest": v6_manifest_path,
        "cohort_reference": cohort_path,
    }
    observed_sources = _validate_source_bindings(release, source_paths)
    inventory = _read_json(inventory_path)
    dispatch = _read_json(dispatch_path)
    v6 = _read_json(v6_manifest_path)
    cohort = _read_json(cohort_path)
    jobs = inventory.get("jobs")
    dispatch_jobs = dispatch.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 1453:
        raise ReleaseCatalogError("C inventory must contain exactly 1453 jobs")
    if not isinstance(dispatch_jobs, list) or len(dispatch_jobs) != 1453:
        raise ReleaseCatalogError("C dispatch inventory must contain exactly 1453 jobs")
    if inventory.get("counts", {}).get("duplicate_case_signatures") != 0:
        raise ReleaseCatalogError("C inventory reports duplicate case signatures")
    row_ids = [job.get("row_id") for job in jobs]
    dispatch_ids = [job.get("row_id") for job in dispatch_jobs]
    if any(not row_id for row_id in row_ids) or len(set(row_ids)) != 1453:
        raise ReleaseCatalogError("C inventory row IDs are missing or duplicated")
    if set(row_ids) != set(dispatch_ids):
        raise ReleaseCatalogError("C dispatch row IDs do not match C inventory")
    archived_input_verification = _verify_inventory_input_records(jobs)
    selected12 = list(cohort.get("original_selected12", []))
    top50 = list(cohort.get("frozen_R_top50", []))
    missing = list(cohort.get("top50_missing_historical_nonlinear", []))
    if len(selected12) != 12 or len(top50) != 50 or len(missing) != 28:
        raise ReleaseCatalogError("Cohort reference has unexpected selected/top50 counts")
    if not set(selected12).issubset(set(top50)):
        raise ReleaseCatalogError("selected12 is not a subset of the frozen top50")
    historical_top50 = {
        _bare_system_id(job["system_id"])
        for job in jobs
        if job.get("campaign") in {"nonlinear_validation_v1", "nonlinear_validation100_v1"}
        and job.get("arm") in STANDARD_ARMS
        and _bare_system_id(job["system_id"]) in set(top50)
    }
    if len(historical_top50) != 22:
        raise ReleaseCatalogError(
            f"Top-50 historical standard membership has {len(historical_top50)} systems, expected 22"
        )
    if set(missing) != set(top50) - historical_top50:
        raise ReleaseCatalogError("Top-50 missing membership does not match C inventory")
    entropy = int(release["mass_position_seed_policy"]["sampler_seed"]["entropy"])
    archived_cases = [_archive_case(job, entropy) for job in jobs]
    verified_positions = _load_verified_positions(release, local_position_root)
    asset_dir = release_path.resolve().parents[2] / "configs" / "source_assets"
    template_by_hash = {
        sha256_file(path): path.stem.removesuffix("_hlr011")
        for path in sorted(asset_dir.glob("*_hlr011.npz"))
    }
    v6_jobs = v6.get("jobs")
    if not isinstance(v6_jobs, list) or len(v6_jobs) != 798:
        raise ReleaseCatalogError("v6 ladder manifest must contain exactly 798 jobs")
    v6_by_system = {}
    for job in v6_jobs:
        system_id = _bare_system_id(job.get("system_id", ""))
        if system_id in v6_by_system:
            raise ReleaseCatalogError(f"v6 manifest duplicates {system_id}")
        v6_by_system[system_id] = job
    if set(missing) - set(v6_by_system):
        raise ReleaseCatalogError("v6 manifest is missing one or more top-50 additions")
    if v6_output_root is None:
        raise ReleaseCatalogError(
            "actual v6 output root is required; future ladder artifact hashes are forbidden"
        )
    v6_output_root = Path(v6_output_root).expanduser().resolve()
    v6_artifacts = {
        system_id: _validate_v6_ladder_artifact(
            v6_by_system[system_id],
            v6_output_root,
            str(v6.get("campaign_uuid")),
        )
        for system_id in missing
    }
    for system_id, position in verified_positions.items():
        if position["ladder_artifact_sha256"] != v6_artifacts[system_id]["sha256"]:
            raise ReleaseCatalogError(
                f"verified position {system_id} is bound to a different v6 ladder artifact"
            )
        payload = position.get("position_payload")
        if payload is not None and payload.get("ladder_config_hash") != v6_by_system[system_id].get(
            "config_hash"
        ):
            raise ReleaseCatalogError(f"verified position {system_id} is bound to a different v6 config hash")
    new_cases = []
    for system_id in missing:
        new_cases.extend(
            _new_top50_cases(
                v6_by_system[system_id],
                entropy,
                verified_positions,
                template_by_hash,
                v6_artifacts[system_id],
            )
        )
    position_tasks = _position_tasks(
        missing,
        v6_by_system,
        verified_positions,
        v6_artifacts,
    )

    selected_rows = {}
    for job in jobs:
        system_id = _bare_system_id(job.get("system_id", ""))
        if system_id not in set(selected12) or job.get("arm") != "asimov_injected":
            continue
        if job.get("campaign") != "nonlinear_validation_v1":
            continue
        records = job.get("input_records", {})
        position_ref = _ref(records.get("positions", job.get("positions")), role="positions")
        if position_ref is None:
            continue
        verified_local = Path(position_ref["path"])
        if not verified_local.is_file():
            raise ReleaseCatalogError(f"Selected12 bracket position is missing: {verified_local}")
        payload = _read_json(verified_local)
        selected_rows[system_id] = {
            "config": _ref(records.get("config", job.get("config")), role="config"),
            "positions": {
                **position_ref,
                "sha256": sha256_file(verified_local),
                "execution_path": (position_ref.get("execution_path") or position_ref.get("path")),
            },
            "case_artifact": _ref(
                records.get("aggregate_artifact", job.get("aggregate_artifact")),
                role="base_case_artifact",
            ),
            "source_asset": _ref(records.get("source_asset", job.get("source_asset")), role="source_asset"),
            "position_payload": payload,
            "run_name": job.get("system_id"),
            "template": job.get("template"),
            "campaign": job.get("campaign"),
        }
    if set(selected_rows) != set(selected12):
        raise ReleaseCatalogError(
            "Selected12 bracket bindings do not cover exactly the frozen selected12 membership"
        )
    bracket_cases = _bracket_cases(selected_rows, entropy)
    bracket_tasks = _bracket_tasks(bracket_cases)
    cases = archived_cases + new_cases + bracket_cases
    release_digest = sha256_file(release_path)
    release_execution_path = release["runner_contract"]["release_freeze_execution_path"]
    procedure_source_execution_path = release["runner_contract"]["procedure_source_execution_path"]
    procedure_source_local_path = release_path.resolve().parents[2] / (
        "src/hwoslaps/modeling/nonlinear/fresh_profile.py"
    )
    _verify_declared_file(
        procedure_source_local_path,
        sha256_file(procedure_source_local_path),
        "v7 procedure source",
    )
    procedure_source_sha256 = sha256_file(procedure_source_local_path)
    approval_receipt_path = release["runner_contract"]["approval_gate"]["sidecar_path"]
    source_asset_execution_root = release["runner_contract"]["source_asset_execution_root"]
    tangent_policy = release["protocol"].get("tangent_comparator")
    if tangent_policy != {
        "enabled_views": ["selected12_standard", "selected12_brackets"],
        "other_cases": False,
    }:
        raise ReleaseCatalogError(
            "v7 tangent comparator routing policy differs from declared selected12 views"
        )
    if release["protocol"].get("bracket_fisher_q") != {
        "enabled_views": ["selected12_brackets"],
        "kernel_shape_native": [999, 999],
        "position": "unchanged_upper_rung",
        "mass": "current_shifted_mass",
        "other_cases": False,
    }:
        raise ReleaseCatalogError("v7 bracket Fisher policy must use k999 at the current shifted mass")
    for case in cases:
        case["compute_bracket_fisher_q"] = case["scope"] == "selected12_brackets"
        case["case_identity_payload"]["compute_bracket_fisher_q"] = case["compute_bracket_fisher_q"]
        case["compute_tangent_comparator"] = case["scope"] == "selected12_brackets" or (
            case["scope"] == "archived_cases"
            and case["system_id"] in selected12
            and case["arm"] in STANDARD_ARMS
        )
        case["case_identity_payload"]["compute_tangent_comparator"] = case["compute_tangent_comparator"]
        case["case_identity_signature"] = _sha256_json(case["case_identity_payload"])
        case["release_freeze_sha256"] = release_digest
        case["procedure_source_sha256"] = procedure_source_sha256
        source_asset = case["input_records"].get("source_asset")
        if isinstance(source_asset, dict) and not source_asset.get("execution_path") and case.get("template"):
            asset_name = f"{case['template']}_hlr011.npz"
            source_asset["path"] = str(asset_dir / asset_name)
            source_asset["execution_path"] = f"{source_asset_execution_root}/{asset_name}"
        elif isinstance(source_asset, dict) and case.get("template"):
            asset_name = f"{case['template']}_hlr011.npz"
            source_asset["original_execution_path"] = source_asset.get("execution_path")
            source_asset["execution_path"] = f"{source_asset_execution_root}/{asset_name}"
        case["runner"]["entrypoint"] = release["runner_contract"]["entrypoint"]
        case["runner"]["mode"] = "standard_case_spec"
        case["runner_spec_template"] = _runner_spec_template(
            case,
            release_execution_path,
            release_digest,
            procedure_source_execution_path,
            procedure_source_sha256,
            approval_receipt_path,
        )
    case_ids = [case["case_id"] for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ReleaseCatalogError("Execution catalog contains duplicate case IDs")

    standard_archived = [case for case in archived_cases if case["arm"] in STANDARD_ARMS]
    selected_view = [case["case_id"] for case in standard_archived if case["system_id"] in set(selected12)]
    if len(selected_view) != 48:
        raise ReleaseCatalogError(f"selected12 standard view has {len(selected_view)} cases")
    top50_archived = [
        case["case_id"]
        for case in standard_archived
        if case["system_id"] in set(top50)
        and case["campaign"] in {"nonlinear_validation_v1", "nonlinear_validation100_v1"}
    ]
    top50_new = [case["case_id"] for case in new_cases]
    if len(top50_archived) != 88 or len(top50_new) != 112:
        raise ReleaseCatalogError(
            f"top50 standard view components are {len(top50_archived)} archived and {len(top50_new)} new"
        )
    null_cases = [
        case["case_id"]
        for case in archived_cases
        if (case["campaign"] == "nonlinear_validation_v1" and case["arm"] == "noisy_control")
        or case["campaign"] == "nonlinear_null_v1"
    ]
    if len(null_cases) != 590:
        raise ReleaseCatalogError(f"null590 view has {len(null_cases)} cases")
    psf_cases = [
        case["case_id"] for case in archived_cases if case["campaign"] == "psf_knowledge_nonlinear_v1"
    ]
    if len(psf_cases) != 288:
        raise ReleaseCatalogError(f"psf288 view has {len(psf_cases)} cases")
    bracket_ids = [case["case_id"] for case in bracket_cases]
    views = {
        "archived_cases": [case["case_id"] for case in archived_cases],
        "selected12_standard": selected_view,
        "top50_standard": top50_archived + top50_new,
        "null590": null_cases,
        "psf288": psf_cases,
        "selected12_brackets": bracket_ids,
        "all_standard_and_brackets": [case["case_id"] for case in cases],
    }
    if any(len(ids) != len(set(ids)) for ids in views.values()):
        raise ReleaseCatalogError("A catalog view repeats a case ID")
    catalog = {
        "schema_version": 7,
        "status": "PREPARED_NOT_LAUNCHED",
        "release_freeze": {
            "path": str(release_path.resolve()),
            "sha256": sha256_file(release_path),
            "version": 7,
            "consumed_freeze": release["consumed_freeze"],
        },
        "inputs": {
            "sources": observed_sources,
            "v6_expected_jobs": 798,
            "v6_actual_artifacts": v6_artifacts,
            "c_inventory_expected_rows": 1453,
            "archived_input_verification": archived_input_verification,
            "cohort_membership": {
                "selected12": selected12,
                "top50": top50,
                "top50_missing_historical_nonlinear": missing,
            },
        },
        "procedure": release["protocol"],
        "counts": {
            "archived_cases": len(archived_cases),
            "new_top50_standard_cases": len(new_cases),
            "selected12_brackets": len(bracket_cases),
            "standard_cases": len(archived_cases) + len(new_cases),
            "execution_catalog": len(cases),
            "selected12_standard": len(selected_view),
            "top50_standard": len(views["top50_standard"]),
            "null590": len(null_cases),
            "psf288": len(psf_cases),
        },
        "views": views,
        "position_tasks": position_tasks,
        "bracket_tasks": bracket_tasks,
        "dependencies": {
            "new_top50_standard_cases": {
                case["case_id"]: f"position:{case['system_id']}" for case in new_cases
            },
            "selected12_brackets": {
                case["case_id"]: "bracket_case_generation_and_h1_anchor" for case in bracket_cases
            },
        },
        "duplicate_policy": release["duplicate_policy"],
        "verified_position_inputs": {
            system_id: {key: value for key, value in item.items() if key not in {"position_payload", "rungs"}}
            for system_id, item in verified_positions.items()
        },
        "cases": cases,
        "runner_contract": release["runner_contract"],
        "limitations": {
            "no_rendering_or_fitting": True,
            "new_position_extraction_pending_count": len(missing) - len(verified_positions),
            "new_position_extraction_verified_reuse_count": len(verified_positions),
            "h0_bracket_generation_pending_count": len(bracket_cases),
            "archived_fit_state_reused": False,
            "area_or_census_inference": False,
            "paper_or_manuscript_edits": False,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "release_catalog.json").write_text(
        json.dumps(catalog, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 7,
        "status": catalog["status"],
        "release_freeze_sha256": catalog["release_freeze"]["sha256"],
        "catalog_sha256": sha256_file(output_dir / "release_catalog.json"),
        "counts": catalog["counts"],
        "views": {name: len(ids) for name, ids in views.items()},
        "pending": catalog["limitations"],
    }
    (output_dir / "release_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return catalog


__all__ = [
    "ARM_INDEX",
    "STANDARD_ARMS",
    "ReleaseCatalogError",
    "build_catalog",
    "runner_invocation",
]

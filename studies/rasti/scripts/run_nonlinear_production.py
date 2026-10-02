#!/usr/bin/env python3
"""Execute one explicitly declared Stage 3 fresh nonlinear case.

The case specification is the only routing input.  Standard cases run fresh
H0/H1 Nautilus searches followed by the versioned local profile.  Bracket
cases run a fresh H0 search and verify the supplied numerical-zero H1 anchor;
the anchor never produces an H1 evidence claim.  This CLI performs no
fallback to archived fits or checkpoints and leaves scheduling to the stage
controller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import sys
_STUDY_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(_STUDY_REPO_ROOT), str(_STUDY_REPO_ROOT / "src")]
from typing import Any


REQUIRED_FIELDS = {
    "case_id", "system_id", "config", "positions", "arm",
    "release_freeze_path", "release_freeze_sha256", "objective_version",
    "procedure_version", "output", "case_identity_payload",
    "case_identity_signature", "procedure_source", "source_assets",
    "approval_receipt", "approval_receipt_sha256", "catalog_sha256",
    "compute_tangent_comparator", "compute_bracket_fisher_q", "scope",
}
FORBIDDEN_STATE_FIELDS = {
    "replay", "baseline", "b002", "starts", "checkpoint", "search_state",
    "old_result", "posterior", "likelihood", "evidence",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def validate_spec(spec: dict[str, Any], spec_path: Path) -> None:
    missing = sorted(REQUIRED_FIELDS - set(spec))
    if missing:
        raise ValueError("production case spec is missing: " + ", ".join(missing))
    if spec.get("execution_policy_version") != "stage3_v7":
        raise ValueError("production case requires execution_policy_version=stage3_v7")
    if spec.get("objective_version") != "consistent_sampling_v2":
        raise ValueError("production case must select consistent_sampling_v2")
    if spec.get("procedure_version") != "fresh_nonlinear_v7_lbfgsb_v2":
        raise ValueError("production case must select the reviewed fresh profile procedure")
    if not isinstance(spec.get("compute_tangent_comparator"), bool):
        raise ValueError("compute_tangent_comparator must be an explicit boolean")
    if not isinstance(spec.get("compute_bracket_fisher_q"), bool):
        raise ValueError("compute_bracket_fisher_q must be an explicit boolean")
    if spec.get("case_kind", "standard") not in {"standard", "bracket"}:
        raise ValueError("case_kind must be standard or bracket")
    if FORBIDDEN_STATE_FIELDS.intersection(spec):
        raise ValueError("production case contains archived search-state input")
    output = Path(spec["output"]).expanduser().resolve()
    wrapper_names = {
        "worker.log", "worker.stdout.log", "worker.stderr.log",
        "worker_started.json", "worker_exit.json", "launch.json", "execution.json",
    }
    if output.exists():
        unexpected = [path.name for path in output.iterdir() if path.name not in wrapper_names]
        if unexpected:
            raise ValueError(
                "production attempt namespace contains unexpected output: "
                + ",".join(sorted(unexpected))
            )
    for field in ("config", "positions", "release_freeze_path"):
        if not Path(spec[field]).expanduser().is_file():
            raise FileNotFoundError(f"production input is missing: {spec[field]}")
    approval_path = Path(spec["approval_receipt"]).expanduser()
    if not approval_path.is_file():
        raise FileNotFoundError(f"approval receipt is missing: {approval_path}")
    if spec.get("case_kind", "standard") == "bracket":
        anchor = spec.get("h1_anchor")
        if not isinstance(anchor, str) or not Path(anchor).expanduser().is_file():
            raise FileNotFoundError("bracket case requires a hash-bound h1_anchor JSON")
        if not isinstance(spec.get("bracket_rung"), str) or not spec["bracket_rung"]:
            raise ValueError("bracket case requires an explicit bracket_rung")
        if isinstance(spec.get("bracket_arm_index"), bool) or not isinstance(
            spec.get("bracket_arm_index"), int
        ):
            raise ValueError("bracket case requires an integer bracket_arm_index")
        if spec["compute_bracket_fisher_q"] is not True:
            raise ValueError("bracket case requires established 999x999 Fisher-q evaluation")
    case_output = Path(spec.get("case_output", output / "case")).expanduser().resolve()
    if not case_output.is_relative_to(output) or case_output == output:
        raise ValueError("case_output must be a child of the outer attempt output")
    if case_output.exists() and any(case_output.iterdir()):
        raise ValueError(f"case output namespace is not empty: {case_output}")
    hashes = spec.get("hashes")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("production case requires non-empty source/input hashes")
    for filename, expected in hashes.items():
        path = Path(filename).expanduser()
        if not path.is_file() or sha256_file(path).lower() != str(expected).lower():
            raise ValueError(f"production input/source hash mismatch: {filename}")
    release_digest = sha256_file(Path(spec["release_freeze_path"]).expanduser())
    if release_digest.lower() != str(spec["release_freeze_sha256"]).lower():
        raise ValueError("release_freeze_sha256 does not match the release declaration")
    for required_path in (
        str(Path(spec["config"]).expanduser()),
        str(Path(spec["positions"]).expanduser()),
        str(Path(spec["release_freeze_path"]).expanduser()),
    ):
        if required_path not in hashes:
            raise ValueError(f"production required input is not hash-bound: {required_path}")
    if str(approval_path) not in hashes:
        raise ValueError("approval receipt is not hash-bound")
    approval_digest = sha256_file(approval_path)
    if approval_digest.lower() != str(spec["approval_receipt_sha256"]).lower():
        raise ValueError("approval_receipt_sha256 does not match the receipt")
    approval = read_json(approval_path)
    if approval.get("status") != "APPROVED":
        raise ValueError("approval receipt is not APPROVED")
    if approval.get("release_freeze_sha256") != spec["release_freeze_sha256"]:
        raise ValueError("approval receipt names a different release freeze")
    if approval.get("catalog_sha256") != spec["catalog_sha256"]:
        raise ValueError("approval receipt names a different release catalog")
    if approval.get("authorized_scope") not in {
        "archived_cases", "new_top50_standard_cases", "all_standard_cases",
        "selected12_brackets", "all_standard_and_brackets",
    }:
        raise ValueError("approval receipt has no supported authorized scope")
    if spec["scope"] not in {
        "archived_cases", "new_top50_standard_cases", "selected12_brackets",
    }:
        raise ValueError("production case has no supported catalog scope")
    approval_scope = approval.get("authorized_scope")
    scope_compatible = {
        "archived_cases": {"archived_cases", "all_standard_cases", "all_standard_and_brackets"},
        "new_top50_standard_cases": {
            "new_top50_standard_cases", "all_standard_cases", "all_standard_and_brackets"
        },
        "selected12_brackets": {"selected12_brackets", "all_standard_and_brackets"},
    }
    if approval_scope not in scope_compatible[spec["scope"]]:
        raise ValueError("approval receipt scope does not authorize this case scope")
    approved_case_ids = approval.get("case_ids")
    if (
        not isinstance(approved_case_ids, list)
        or len(approved_case_ids) != len(set(approved_case_ids))
        or spec["case_id"] not in approved_case_ids
    ):
        raise ValueError("approval receipt does not authorize this case_id")
    if isinstance(approval.get("authorized_gpu_limit"), bool) or not isinstance(
        approval.get("authorized_gpu_limit"), int
    ) or approval["authorized_gpu_limit"] < 1:
        raise ValueError("approval receipt has no positive GPU authorization")
    procedure_source = spec.get("procedure_source")
    if not isinstance(procedure_source, str) or procedure_source not in hashes:
        raise ValueError("production procedure_source must be hash-bound")
    source_assets = spec.get("source_assets")
    if not isinstance(source_assets, list) or not source_assets:
        raise ValueError("production source_assets must be a non-empty list")
    for source_asset in source_assets:
        if not isinstance(source_asset, str) or source_asset not in hashes:
            raise ValueError(f"production source asset is not hash-bound: {source_asset}")
    if spec.get("case_kind", "standard") == "bracket":
        anchor_path = str(Path(spec["h1_anchor"]).expanduser())
        if anchor_path not in hashes:
            raise ValueError("bracket h1_anchor is not hash-bound")
    expected_case = spec.get("expected_case_sha256")
    if expected_case is not None and not isinstance(expected_case, str):
        raise ValueError("expected_case_sha256 must be a string when supplied")
    identity = spec.get("case_identity")
    if identity != spec.get("case_id"):
        raise ValueError("case_id does not match the explicit case_identity")
    identity_payload = spec.get("case_identity_payload")
    identity_signature = spec.get("case_identity_signature")
    if not isinstance(identity_payload, dict) or not isinstance(identity_signature, str):
        raise ValueError(
            "production case requires case_identity_payload and its canonical signature"
        )
    canonical = json.dumps(
        identity_payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    observed_signature = hashlib.sha256(canonical).hexdigest()
    if observed_signature != identity_signature:
        raise ValueError("case_identity_signature does not match its payload")
    for key in (
        "case_id",
        "system_id",
        "arm",
        "compute_tangent_comparator",
        "compute_bracket_fisher_q",
    ):
        if identity_payload.get(key) != spec.get(key):
            raise ValueError(f"case identity payload disagrees for {key}")
    catalog_signature = spec.get("catalog_case_identity_signature")
    if catalog_signature is not None and not isinstance(catalog_signature, str):
        raise ValueError("catalog_case_identity_signature must be a string")
    env_case_id = os.environ.get("HWOSLAPS_RELEASE_CASE_ID")
    if env_case_id is not None and env_case_id != spec["case_id"]:
        raise ValueError("HWOSLAPS_RELEASE_CASE_ID does not match the case spec")
    expected_case = spec.get("expected_case_sha256")
    env_case_sha = os.environ.get("HWOSLAPS_EXPECTED_CASE_SHA256")
    if expected_case is not None and env_case_sha != expected_case:
        raise ValueError("HWOSLAPS_EXPECTED_CASE_SHA256 does not match the case spec")
    if os.environ.get("HWOSLAPS_REUSE_ARCHIVED_FIT_STATE") not in (None, "0"):
        raise ValueError("archived fit-state reuse is forbidden for Stage 3")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = Path(path).with_name(Path(path).name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _process_start() -> float | None:
    try:
        import psutil

        return float(psutil.Process(os.getpid()).create_time())
    except Exception:
        return None


def _receipt_artifacts(output: Path) -> dict[str, str]:
    live_logs = {"worker.log", "worker.stdout.log", "worker.stderr.log"}
    return {
        str(path): sha256_file(path)
        for path in output.rglob("*")
        if path.is_file() and path.name not in {"worker_exit.json", *live_logs}
    }


def _canonical_payload(case_output: Path, spec: dict[str, Any]) -> dict[str, Any]:
    """Read the canonical nonlinear payload the route wrote for this case."""
    direction = spec.get("direction")
    suffix = "" if direction is None else f"_dir{direction}"
    payload_path = case_output / f"nonlinear_validation_{spec['arm']}{suffix}.json"
    payload = read_json(payload_path)
    if payload.get("artifact_completeness_status") not in {"complete", "incomplete"}:
        raise ValueError(
            "canonical payload has no artifact_completeness_status; "
            f"the v7 route did not run: {payload_path}"
        )
    return payload


def _incomplete_roles(retention_contract: Any) -> list[str]:
    roles = (retention_contract or {}).get("roles", {})
    incomplete = sorted(
        role for role, record in roles.items() if not record.get("complete")
    )
    return incomplete or ["unknown"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    args = parser.parse_args(argv)
    spec_path = args.spec.expanduser().resolve()
    spec = read_json(spec_path)
    validate_spec(spec, spec_path)
    output = Path(spec["output"]).expanduser().resolve()
    case_output = Path(spec.get("case_output", output / "case")).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "worker_started.json").exists():
        _write_json(
            output / "worker_started.json",
            {
                "pid": os.getpid(),
                "process_start": _process_start(),
                "spec_path": str(spec_path),
                "task_kind": "STAGE3_PRODUCTION",
            },
        )
    started = time.monotonic()
    status = "FAILED"
    error = None
    try:
        import run_nonlinear_validation

        route_args = [
            str(spec["config"]),
            str(spec["positions"]),
            str(spec["arm"]),
            str(case_output),
            "--release-freeze-path",
            str(spec["release_freeze_path"]),
            "--objective-version",
            str(spec["objective_version"]),
            "--procedure",
            "fresh_profile_v1",
        ]
        if spec.get("direction") is not None:
            if isinstance(spec["direction"], bool) or not isinstance(spec["direction"], int):
                raise ValueError("direction must be an integer when supplied")
            route_args.extend(["--direction", str(spec["direction"])])
        if spec["compute_tangent_comparator"]:
            route_args.append("--compute-tangent-comparator")
        if spec["compute_bracket_fisher_q"]:
            route_args.append("--compute-bracket-fisher-q")
        if spec.get("case_kind", "standard") == "bracket":
            route_args.extend(
                [
                    "--h1-anchor",
                    str(spec["h1_anchor"]),
                    "--bracket-rung",
                    str(spec["bracket_rung"]),
                    "--bracket-arm-index",
                    str(spec["bracket_arm_index"]),
                ]
            )
        run_nonlinear_validation.main(route_args)
        payload = _canonical_payload(case_output, spec)
        artifact_completeness = payload.get("artifact_completeness_status")
        retention_contract = payload.get("retention_contract")
        if artifact_completeness == "complete":
            status = "COMPLETE"
        else:
            status = "INCOMPLETE_ARTIFACTS"
            error = (
                "required sampler state was not retained for: "
                + ", ".join(_incomplete_roles(retention_contract))
            )
        _write_json(
            output / "production_run.json",
            {
                "status": status,
                "artifact_completeness_status": artifact_completeness,
                "retention_contract": retention_contract,
                "case_id": spec["case_id"],
                "system_id": spec["system_id"],
                "arm": spec["arm"],
                "direction": spec.get("direction"),
                "compute_tangent_comparator": spec["compute_tangent_comparator"],
                "compute_bracket_fisher_q": spec["compute_bracket_fisher_q"],
                "spec_sha256": sha256_file(spec_path),
                "catalog_sha256": spec["catalog_sha256"],
                "case_identity_signature": spec["case_identity_signature"],
                "catalog_case_identity_signature": spec.get(
                    "catalog_case_identity_signature"
                ),
                "scope": spec["scope"],
                "config_sha256": sha256_file(Path(spec["config"])),
                "positions_sha256": sha256_file(Path(spec["positions"])),
                "case_kind": spec.get("case_kind", "standard"),
                "objective_version": spec["objective_version"],
                "procedure_version": spec["procedure_version"],
                "release_freeze_path": str(Path(spec["release_freeze_path"]).resolve()),
                "release_freeze_sha256": sha256_file(Path(spec["release_freeze_path"])),
                "case_output": str(case_output),
                "fresh_searches": spec.get("case_kind", "standard") == "standard",
                "h1_anchor_sampler_executed": False
                if spec.get("case_kind", "standard") == "bracket"
                else None,
                "h1_anchor_evidence_claim": False
                if spec.get("case_kind", "standard") == "bracket"
                else None,
                "archived_state_imported": False,
            },
        )
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        _write_json(
            output / "production_failure.json",
            {
                "status": "FAILED",
                "case_id": spec.get("case_id"),
                "case_output": str(case_output),
                "error": error,
                "archived_state_imported": False,
            },
        )
        raise
    finally:
        _write_json(
            output / "worker_exit.json",
            {
                "status": status,
                "error": error,
                "elapsed_s": time.monotonic() - started,
                "case_id": spec.get("case_id"),
                "system_id": spec.get("system_id"),
                "arm": spec.get("arm"),
                "spec_sha256": sha256_file(spec_path),
                "catalog_sha256": spec.get("catalog_sha256"),
                "case_identity_signature": spec.get("case_identity_signature"),
                "catalog_case_identity_signature": spec.get(
                    "catalog_case_identity_signature"
                ),
                "scope": spec.get("scope"),
                "config_sha256": (
                    sha256_file(Path(spec["config"]))
                    if spec.get("config") and Path(spec["config"]).is_file()
                    else None
                ),
                "positions_sha256": (
                    sha256_file(Path(spec["positions"]))
                    if spec.get("positions") and Path(spec["positions"]).is_file()
                    else None
                ),
                "objective_version": spec.get("objective_version"),
                "procedure_version": spec.get("procedure_version"),
                "release_freeze_path": spec.get("release_freeze_path"),
                "release_freeze_sha256": spec.get("release_freeze_sha256"),
                "artifacts": _receipt_artifacts(output),
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

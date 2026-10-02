"""Read-only, standard-library-only harvest of the canonical v7 catalog.

Only explicitly supplied attempt specifications are considered. Historical
results and directory discovery are deliberately not inputs. A completed
worker must have an intact receipt and exact release/case/spec bindings.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from hwoslaps.campaign.system_ids import SystemIdError, bare_system_id


class ProductionHarvestError(ValueError):
    """The supplied catalog or attempt inventory is ambiguous."""


REQUIRED_VIEW_COUNTS = {
    "selected12_standard": 48,
    "top50_standard": 200,
    "null590": 590,
    "psf288": 288,
    "selected12_brackets": 36,
}
IDENTITY_KEYS = (
    "case_id",
    "catalog_sha256",
    "case_identity_signature",
    "objective_version",
    "procedure_version",
    "release_freeze_sha256",
    "spec_sha256",
    "config_sha256",
    "positions_sha256",
)
CSV_FIELDS = (
    "case_id",
    "system_id",
    "campaign",
    "arm",
    "direction",
    "case_kind",
    "status",
    "attempt_count",
    "completed_attempt_count",
    "selected_attempt",
    "q_signed",
    "q_clipped",
    "q_f_production_at_position",
    "marginal_q_flag",
    "profile_decision",
    "delta_log_evidence",
    "h1_evidence_claim",
    "integrity_errors",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ProductionHarvestError(f"JSON object required: {path}")
    return value


def _config_run_name(config_path: Path) -> str:
    """Return the run name the route stamps into its payload as ``system_id``."""
    with Path(config_path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, Mapping) or not isinstance(config.get("run_name"), str):
        raise ProductionHarvestError(f"case config declares no run_name: {config_path}")
    return config["run_name"]


def _same(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ProductionHarvestError(f"{label} mismatch: {actual!r} != {expected!r}")


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ProductionHarvestError(f"{label} is not a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ProductionHarvestError(f"{label} is not finite")
    return result


class Paths:
    """Explicit prefix relocation of an immutable remote evidence mirror."""

    def __init__(self, mappings: Mapping[str, str] | None = None):
        self.mappings = sorted(
            ((Path(a), Path(b)) for a, b in (mappings or {}).items()),
            key=lambda pair: len(pair[0].parts),
            reverse=True,
        )

    def __call__(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        for source, destination in self.mappings:
            if path.is_relative_to(source):
                return destination / path.relative_to(source)
        return path


def _bracket_source_chain(case, spec, catalog_digest, paths):
    """Verify original inputs -> generated bracket -> restamped fit inputs."""
    filename = spec.get("dependency_receipt")
    digest = spec.get("hashes", {}).get(filename)
    if not filename or not digest:
        raise ProductionHarvestError("bracket requires a hash-bound dependency receipt")
    _same(sha256_file(paths(filename)), digest, "bracket dependency receipt hash")
    receipt = _read(paths(filename))
    for key, expected in (
        ("status", "COMPLETE"),
        ("case_id", case["case_id"]),
        ("catalog_sha256", catalog_digest),
    ):
        _same(receipt.get(key), expected, f"bracket dependency {key}")
    for role in ("config", "positions"):
        _same(
            receipt.get(f"source_{role}_sha256"),
            case["input_records"][role]["sha256"],
            f"bracket original {role}",
        )
    generated = receipt["generated"]
    for role in ("config", "positions", "h1_anchor"):
        _same(
            sha256_file(paths(generated[role])),
            generated[f"{role}_sha256"],
            f"generated bracket {role} hash",
        )
    for key in ("target_log10_m200", "target_mass_msun", "position_yx_arcsec"):
        _same(
            generated.get(key),
            case["frozen_mass_position"][key],
            f"bracket target {key}",
        )
    _same(
        generated["positions_sha256"],
        spec["hashes"][spec["positions"]],
        "bracket generated position binding",
    )
    _same(generated["h1_anchor"], spec.get("h1_anchor"), "bracket anchor path")
    _same(
        generated["h1_anchor_sha256"],
        spec["hashes"].get(spec.get("h1_anchor")),
        "bracket anchor binding",
    )
    anchor = _read(paths(generated["h1_anchor"]))
    _same(anchor.get("case_id"), case["case_id"], "bracket anchor case")
    _same(anchor.get("evidence_claim"), False, "bracket anchor evidence policy")
    _same(anchor.get("sampler_executed"), False, "bracket anchor sampler policy")
    _same(generated["bracket_rung"], spec.get("bracket_rung"), "bracket rung")
    return generated["config_sha256"]


def _verify_retained_sampler_state(
    payload: dict,
    kind: str,
    case_output_original: Path,
    receipt_digests: Mapping[Path, str],
    paths: Paths,
) -> None:
    """Fail closed unless every required sampler-state file is receipt-bound.

    A standard pair needs the raw state of both fresh searches; a bracket
    needs the H0 state and an H1 anchor that claims none. Each required file
    must sit under its own search result path inside the case output, carry
    nonzero size, and match the worker receipt's digest byte for byte.
    """
    contract = payload.get("retention_contract")
    if not isinstance(contract, dict) or not isinstance(contract.get("roles"), dict):
        raise ProductionHarvestError("completed payload has no retention contract")
    _same(payload.get("artifact_completeness_status"), "complete", "artifact completeness")
    _same(contract.get("complete"), True, "retention contract completeness")
    case_record = payload.get("case")
    if not isinstance(case_record, dict):
        raise ProductionHarvestError("completed payload has no case fit summaries")
    for role, fit_key in (("smooth", "smooth_fit"), ("subhalo", "subhalo_fit")):
        record = contract["roles"].get(role)
        fit = case_record.get(fit_key)
        if not isinstance(record, dict) or not isinstance(fit, dict):
            raise ProductionHarvestError(f"retention record or fit summary missing for {role}")
        required = kind == "standard" or role == "smooth"
        _same(record.get("sampler_state_required"), required, f"{role} sampler-state requirement")
        _same(record.get("complete"), True, f"{role} retention completeness")
        if not required:
            _same(fit.get("search_internal_retention_requested"), False, f"{role} anchor retention request")
            _same(fit.get("search_internal_retained"), False, f"{role} anchor retained state")
            _same(fit.get("search_engine"), "VerifiedZeroResidualAnchor", f"{role} anchor engine")
            continue
        _same(fit.get("status"), "success", f"{role} fresh search status")
        _same(fit.get("search_internal_retention_requested"), True, f"{role} retention request")
        _same(fit.get("search_internal_retained"), True, f"{role} retained state")
        _same(record.get("retained"), True, f"{role} contract retained state")
        inventory = fit.get("search_internal_payload")
        if not isinstance(inventory, dict):
            raise ProductionHarvestError(f"{role} fit has no sampler-state inventory")
        _same(inventory.get("missing_required"), [], f"{role} missing sampler-state files")
        if inventory.get("bound_to_result_path") is False:
            raise ProductionHarvestError(f"{role} sampler state belongs to a different result path")
        required_files = inventory.get("required_files")
        if not isinstance(required_files, list) or not required_files:
            raise ProductionHarvestError(f"{role} declares no required sampler-state files")
        result_path = fit.get("result_path")
        if not isinstance(result_path, str) or not result_path:
            raise ProductionHarvestError(f"{role} fresh search has no result path")
        result_original = Path(result_path)
        if not result_original.is_relative_to(case_output_original):
            raise ProductionHarvestError(f"{role} search result escapes the case output")
        files = inventory.get("files")
        if not isinstance(files, dict):
            raise ProductionHarvestError(f"{role} sampler-state inventory has no files")
        for name in required_files:
            entry = files.get(name)
            if not isinstance(entry, dict):
                raise ProductionHarvestError(f"{role} required sampler-state file missing: {name}")
            size = entry.get("bytes")
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                raise ProductionHarvestError(f"{role} sampler-state file is empty: {name}")
            if isinstance(entry.get("path"), str):
                original = Path(entry["path"])
                if not original.is_relative_to(result_original):
                    raise ProductionHarvestError(f"{role} sampler-state file escapes its search: {name}")
                actual = paths(original)
                digest = receipt_digests.get(actual.resolve())
                if digest is None:
                    raise ProductionHarvestError(f"{role} sampler-state file is not receipt-bound: {name}")
                _same(entry.get("sha256"), digest, f"{role} sampler-state hash {name}")
                _same(actual.stat().st_size, size, f"{role} sampler-state size {name}")
            elif isinstance(entry.get("container"), str) and isinstance(entry.get("member"), str):
                container_original = Path(entry["container"])
                if not container_original.is_relative_to(case_output_original):
                    raise ProductionHarvestError(f"{role} sampler-state archive escapes the case output")
                container = paths(container_original)
                if receipt_digests.get(container.resolve()) is None:
                    raise ProductionHarvestError(f"{role} sampler-state archive is not receipt-bound")
                with zipfile.ZipFile(container) as archive:
                    data = archive.read(entry["member"])
                digest = hashlib.sha256(data).hexdigest()
                _same(digest, entry.get("sha256"), f"{role} sampler-state hash {name}")
                _same(len(data), size, f"{role} sampler-state size {name}")
            else:
                raise ProductionHarvestError(
                    f"{role} sampler-state file has no disk or archive location: {name}"
                )


def _verify_complete(
    case: dict,
    spec: dict,
    spec_path: Path,
    receipt: dict,
    catalog: dict,
    catalog_digest: str,
    paths: Paths,
) -> dict:
    output_original = Path(spec["output"]).expanduser()
    output = paths(output_original)
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ProductionHarvestError("completed receipt has no artifact hashes")
    verified = set()
    receipt_digests: dict[Path, str] = {}
    for filename, digest in artifacts.items():
        original = Path(filename)
        if not original.is_absolute() or not original.is_relative_to(output_original):
            raise ProductionHarvestError(f"receipt artifact escapes attempt: {filename}")
        path = paths(original)
        if not path.is_file():
            raise ProductionHarvestError(f"receipt artifact is missing: {filename}")
        _same(sha256_file(path), digest, f"artifact SHA256 {filename}")
        verified.add(path.resolve())
        receipt_digests[path.resolve()] = str(digest)
    run_path = output / "production_run.json"
    if run_path.resolve() not in verified:
        raise ProductionHarvestError("production_run.json is not receipt-bound")
    run = _read(run_path)
    _same(run.get("status"), "COMPLETE", "production run status")
    expected = {
        "case_id": case["case_id"],
        "catalog_sha256": catalog_digest,
        "case_identity_signature": spec["case_identity_signature"],
        "objective_version": "consistent_sampling_v2",
        "procedure_version": "fresh_nonlinear_v7_lbfgsb_v2",
        "release_freeze_sha256": catalog["release_freeze"]["sha256"],
        "spec_sha256": sha256_file(spec_path),
    }
    for field in ("config", "positions"):
        filename = str(spec[field])
        digest = spec["hashes"].get(filename)
        if not digest:
            raise ProductionHarvestError(f"{field} is not hash-bound in spec")
        _same(sha256_file(paths(filename)), digest, f"{field} input hash")
        expected[f"{field}_sha256"] = digest
    for key in IDENTITY_KEYS:
        _same(run.get(key), expected[key], f"production run {key}")
        _same(receipt.get(key), expected[key], f"worker receipt {key}")
    _same(run.get("archived_state_imported"), False, "archived state policy")
    original_identity = case["case_identity_payload"]
    materialized_identity = spec["case_identity_payload"]
    if spec.get("catalog_case_identity_signature") is not None:
        for record, label in ((spec, "spec"), (run, "run"), (receipt, "receipt")):
            _same(
                record.get("catalog_case_identity_signature"),
                case["case_identity_signature"],
                f"{label} original catalog identity",
            )
        allowed_updates = {"source_config_sha256", "config_sha256", "positions_sha256"}
        _same(
            {k: v for k, v in materialized_identity.items() if k not in allowed_updates},
            {k: v for k, v in original_identity.items() if k not in allowed_updates},
            "materialization changed scientific case identity",
        )
        for field in ("config", "positions"):
            _same(
                materialized_identity.get(f"{field}_sha256"),
                expected[f"{field}_sha256"],
                f"materialized identity {field} hash",
            )
        source_config = case.get("input_records", {}).get("config", {}).get("sha256")
        if source_config is None:
            raise ProductionHarvestError("catalog lacks source config identity")
        if case.get("scope") == "selected12_brackets":
            source_config = _bracket_source_chain(case, spec, catalog_digest, paths)
        _same(
            materialized_identity.get("source_config_sha256"),
            source_config,
            "source config identity",
        )
    else:
        _same(materialized_identity, original_identity, "catalog/spec identity payload")
    signature = hashlib.sha256(
        json.dumps(
            materialized_identity,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    _same(signature, spec["case_identity_signature"], "materialized identity signature")
    for key in ("case_id", "system_id", "arm", "direction"):
        _same(run.get(key), case.get(key), f"production run {key}")
        _same(spec.get(key), case.get(key), f"spec {key}")
    for key in (
        "catalog_sha256",
        "case_identity_signature",
        "objective_version",
        "procedure_version",
        "release_freeze_sha256",
    ):
        _same(spec.get(key), expected[key], f"spec {key}")
    child_original = Path(spec.get("case_output", output_original / "case"))
    if child_original == output_original or not child_original.is_relative_to(output_original):
        raise ProductionHarvestError("case output escapes outer attempt")
    _same(run.get("case_output"), str(child_original), "case output")
    direction = spec.get("direction")
    suffix = "" if direction is None else f"_dir{direction}"
    payload_path = paths(child_original) / f"nonlinear_validation_{spec['arm']}{suffix}.json"
    if payload_path.resolve() not in verified:
        raise ProductionHarvestError("canonical nonlinear payload is not receipt-bound")
    payload = _read(payload_path)
    for key in (
        "arm",
        "objective_version",
        "procedure_version",
        "release_freeze_sha256",
    ):
        _same(payload.get(key), spec.get(key), f"nonlinear payload {key}")
    # The route identifies a system by its configuration's run name, the
    # catalog by the bare sysNNNN identifier that run name carries.
    run_name = _config_run_name(paths(Path(spec["config"])))
    _same(payload.get("system_id"), run_name, "nonlinear payload run name")
    try:
        run_system = bare_system_id(run_name)
    except SystemIdError as error:
        raise ProductionHarvestError(str(error)) from error
    _same(run_system, spec.get("system_id"), "config run name system")
    _same(
        payload.get("positions_artifact_sha256"),
        expected["positions_sha256"],
        "payload positions hash",
    )
    _same(payload.get("sampler_seed"), case.get("sampler_seed"), "sampler seed")
    psf = payload.get("fit_psf_delta")
    _same(None if psf is None else psf.get("direction"), direction, "PSF direction")
    roles = payload.get("profile_role_statuses", {})
    kind = "bracket" if case.get("scope") == "selected12_brackets" else "standard"
    _verify_retained_sampler_state(payload, kind, child_original, receipt_digests, paths)
    expected_fisher_flag = kind == "bracket"
    for record, label in ((case, "catalog"), (spec, "spec"), (run, "production run")):
        _same(
            record.get("compute_bracket_fisher_q"),
            expected_fisher_flag,
            f"{label} bracket Fisher computation flag",
        )
    _same(
        spec["case_identity_payload"].get("compute_bracket_fisher_q"),
        expected_fisher_flag,
        "identity bracket Fisher computation flag",
    )
    bracket_fisher = payload.get("bracket_fisher_q")
    fisher_q = None
    if kind == "bracket":
        if not isinstance(bracket_fisher, dict):
            raise ProductionHarvestError("bracket is missing its fresh production Fisher comparator")
        fisher_q = _finite(bracket_fisher.get("q_f_production_at_position"), "bracket Fisher q")
        _same(
            bracket_fisher.get("kernel_shape_native"),
            [999, 999],
            "bracket Fisher kernel",
        )
        target = case["frozen_mass_position"]
        _same(
            bracket_fisher.get("log10_m200"),
            target["target_log10_m200"],
            "bracket Fisher mass",
        )
        _same(
            bracket_fisher.get("position_yx_arcsec"),
            target["position_yx_arcsec"],
            "bracket Fisher position",
        )
        _same(bracket_fisher.get("full_square_geometry"), True, "bracket Fisher geometry")
        _same(bracket_fisher.get("new_prescription"), False, "bracket Fisher prescription")
    elif bracket_fisher is not None:
        raise ProductionHarvestError("standard case unexpectedly carries bracket Fisher comparator")
    _same(spec.get("case_kind", "standard"), kind, "spec case kind")
    _same(run.get("case_kind"), kind, "run case kind")
    allowed = {"accepted_repeatable_profile"}
    accepted = roles.get("smooth") in allowed and roles.get("subhalo") in (
        {"verified_zero_residual_anchor"} if kind == "bracket" else allowed
    )
    numerical = "accepted" if accepted else "unresolved"
    _same(payload.get("numerical_status"), numerical, "numerical status")
    delta = payload.get("delta_log_likelihood")
    if delta is None and accepted:
        raise ProductionHarvestError("accepted result has no finite likelihood ratio")
    q = None if delta is None else 2.0 * _finite(delta, "delta log likelihood")
    clipped = None if q is None else max(0.0, q)
    if payload.get("q_fit") is not None and (
        clipped is None
        or not math.isclose(_finite(payload["q_fit"], "q_fit"), clipped, rel_tol=1e-12, abs_tol=1e-7)
    ):
        raise ProductionHarvestError("q_fit differs from clipped profile likelihood ratio")
    marginal = None if q is None else abs(q - 10.0) < 1.0
    decision = q >= 10.0 if accepted else None
    _same(payload.get("marginal_q_flag"), marginal, "marginal flag")
    _same(payload.get("profile_decision"), decision, "profile classification")
    evidence = payload.get("delta_log_evidence")
    if evidence is not None:
        evidence = _finite(evidence, "delta log evidence")
    if kind == "bracket":
        anchor = payload.get("h1_anchor", {})
        _same(anchor.get("sampler_executed"), False, "bracket H1 sampler")
        _same(anchor.get("evidence_claim"), False, "bracket H1 evidence claim")
        _same(evidence, None, "bracket delta log evidence")
        _same(run.get("h1_anchor_evidence_claim"), False, "run bracket evidence")
    return {
        "status": numerical,
        "q_signed": q,
        "q_clipped": clipped,
        "marginal_q_flag": marginal,
        "profile_decision": decision,
        "delta_log_evidence": evidence,
        "h1_evidence_claim": kind != "bracket",
        "bracket_fisher_q": bracket_fisher,
        "q_f_production_at_position": fisher_q,
        "profile_role_statuses": roles,
        "artifact_completeness_status": payload.get("artifact_completeness_status"),
        "quality_flags": payload.get("quality_flags", []),
        "likelihood_matched_tangent": payload.get("likelihood_matched_tangent"),
        "payload_path": str(payload_path),
        "payload_sha256": sha256_file(payload_path),
        "verified_artifact_count": len(verified),
    }


def harvest_production(
    catalog_path: str | Path,
    spec_paths: Sequence[str | Path],
    *,
    path_mappings: Mapping[str, str] | None = None,
    enforce_production_counts: bool = True,
) -> dict:
    """Return all expected rows, never optimize selection across attempts.

    Multiple COMPLETE receipts for one case are a fatal ambiguity even if one
    later fails integrity checks. Integrity failures become failed rows without
    scientific values; missing and explicitly failed attempts remain visible.
    An attempt whose fits finished but whose required sampler state was not
    retained is reported as ``incomplete``, never as a production result.
    """
    catalog_path = Path(catalog_path)
    catalog = _read(catalog_path)
    digest = sha256_file(catalog_path)
    paths = Paths(path_mappings)
    cases = catalog["cases"]
    by_id = {case["case_id"]: case for case in cases}
    if len(by_id) != len(cases):
        raise ProductionHarvestError("duplicate case IDs in catalog")
    views = catalog["views"]
    for name, ids in views.items():
        if len(ids) != len(set(ids)) or set(ids) - set(by_id):
            raise ProductionHarvestError(f"invalid or duplicate membership in {name}")
    if enforce_production_counts:
        for name, expected in REQUIRED_VIEW_COUNTS.items():
            _same(len(views.get(name, [])), expected, f"{name} denominator")
        _same(len(cases), 1601, "canonical case count")
    attempts: dict[str, list] = {case_id: [] for case_id in by_id}
    seen_outputs = set()
    for filename in spec_paths:
        spec_path = paths(filename)
        spec = _read(spec_path)
        case_id = spec.get("case_id")
        if case_id not in by_id:
            raise ProductionHarvestError(f"attempt case not in catalog: {case_id}")
        output = paths(spec["output"])
        if output.resolve() in seen_outputs:
            raise ProductionHarvestError(f"attempt output repeated: {output}")
        seen_outputs.add(output.resolve())
        exit_path = output / "worker_exit.json"
        receipt = None
        error = None
        if exit_path.exists():
            try:
                receipt = _read(exit_path)
            except (ValueError, OSError) as exc:
                error = str(exc)
        attempts[case_id].append((spec_path, spec, receipt, error))
    rows = []
    for case in cases:
        assigned = attempts[case["case_id"]]
        complete = [a for a in assigned if a[2] and a[2].get("status") == "COMPLETE"]
        incomplete = [
            a for a in assigned if a[2] and a[2].get("status") == "INCOMPLETE_ARTIFACTS"
        ]
        if len(complete) > 1:
            raise ProductionHarvestError(f"duplicate COMPLETE attempts: {case['case_id']}")
        row = {key: case.get(key) for key in ("case_id", "system_id", "campaign", "arm", "direction")}
        row.update(
            {
                "case_kind": "bracket" if case.get("scope") == "selected12_brackets" else "standard",
                "status": "missing",
                "attempt_count": len(assigned),
                "completed_attempt_count": len(complete),
                "selected_attempt": None,
                "q_signed": None,
                "q_clipped": None,
                "marginal_q_flag": None,
                "profile_decision": None,
                "delta_log_evidence": None,
                "h1_evidence_claim": False,
                "bracket_fisher_q": None,
                "q_f_production_at_position": None,
                "integrity_errors": [],
                "attempts": [
                    {
                        "spec": str(a[0]),
                        "status": (a[2] or {}).get("status", "NO_RECEIPT"),
                        "receipt_error": a[3],
                    }
                    for a in assigned
                ],
            }
        )
        if complete:
            spec_path, spec, receipt, _ = complete[0]
            row["selected_attempt"] = str(spec_path)
            try:
                row.update(_verify_complete(case, spec, spec_path, receipt, catalog, digest, paths))
            except (KeyError, TypeError, ValueError, OSError, zipfile.BadZipFile) as exc:
                row["status"] = "failed"
                row["integrity_errors"].append(str(exc))
        elif incomplete:
            row["status"] = "incomplete"
        elif any(a[2] is not None or a[3] is not None for a in assigned):
            row["status"] = "failed"
        rows.append(row)
    rows_by_id = {row["case_id"]: row for row in rows}
    counts = {}
    for name, ids in views.items():
        members = [rows_by_id[case_id] for case_id in ids]
        statuses = Counter(row["status"] for row in members)
        counts[name] = {
            "expected": len(ids),
            **{
                key: statuses[key]
                for key in ("missing", "failed", "incomplete", "unresolved", "accepted")
            },
            "marginal": sum(row["marginal_q_flag"] is True for row in members),
            "accepted_detections": sum(row["profile_decision"] is True for row in members),
            "accepted_non_detections": sum(row["profile_decision"] is False for row in members),
            "integrity_failed": sum(bool(row["integrity_errors"]) for row in members),
        }
    return {
        "schema_version": 7,
        "catalog_path": str(catalog_path),
        "catalog_sha256": digest,
        "status": "COMPLETE" if all(r["status"] == "accepted" for r in rows) else "INCOMPLETE_OR_UNRESOLVED",
        "selection_rule": "exactly one COMPLETE attempt; never select by q",
        "path_mappings": dict(path_mappings or {}),
        "views": counts,
        "rows": rows,
    }


def write_harvest(result: dict, output_dir: str | Path) -> None:
    """Write JSON, one canonical-row CSV, and a readable denominator report."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "PRODUCTION_HARVEST.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    with (output / "PRODUCTION_HARVEST.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in result["rows"]:
            writer.writerow({**row, "integrity_errors": json.dumps(row["integrity_errors"])})
    lines = [
        "# Canonical nonlinear production harvest",
        "",
        f"Status: {result['status']}",
        "",
        f"Catalog SHA-256: `{result['catalog_sha256']}`",
        "",
        "Every expected case remains in its denominator. Missing, failed, incomplete and "
        "unresolved cases have no classification. Incomplete means the fits finished but the "
        "required raw sampler state was not retained. Marginal flags are independent of "
        "numerical acceptance. The signed q is retained separately from max(0, q). Brackets "
        "carry no H1 evidence claim.",
        "",
        "| View | Expected | Accepted | Unresolved | Incomplete | Failed | Missing | Marginal "
        "| Detections |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, c in result["views"].items():
        lines.append(
            f"| {name} | {c['expected']} | {c['accepted']} | {c['unresolved']} | {c['incomplete']} "
            f"| {c['failed']} | {c['missing']} | {c['marginal']} | {c['accepted_detections']} |"
        )
    (output / "PRODUCTION_HARVEST.md").write_text("\n".join(lines) + "\n")

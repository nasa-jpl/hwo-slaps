"""Synthetic, CPU-only integrity/denominator tests for the production harvester."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from hwoslaps.campaign.production_harvest import (
    ProductionHarvestError,
    harvest_production,
    sha256_file,
    write_harvest,
)


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True))
    return path


def _fisher_fixture():
    return {
        "q_f_production_at_position": 11.0,
        "kernel_shape_native": [999, 999],
        "log10_m200": 8.2,
        "position_yx_arcsec": [0.2, 0.1],
        "full_square_geometry": True,
        "new_prescription": False,
    }


def _retained_search(child, role, *, retain=True, size=16, route="directory"):
    """Write one fresh search's output tree with or without sampler state."""
    result_dir = child / f"{role}_search" / "identifier"
    internal = result_dir / "files" / "search_internal"
    files = {}
    if route == "directory":
        internal.mkdir(parents=True, exist_ok=True)
        (internal / ".time").write_text("0.1")
        files[".time"] = {
            "path": str(internal / ".time"),
            "bytes": 3,
            "sha256": sha256_file(internal / ".time"),
        }
        if retain:
            dill = internal / "search_internal.dill"
            dill.write_bytes(b"s" * size)
            files["search_internal.dill"] = {
                "path": str(dill),
                "bytes": size,
                "sha256": sha256_file(dill),
            }
    else:
        import zipfile

        result_dir.parent.mkdir(parents=True, exist_ok=True)
        container = Path(f"{result_dir}.zip")
        with zipfile.ZipFile(container, "w") as archive:
            archive.writestr("files/model.json", "{}")
            if retain:
                archive.writestr("files/search_internal/search_internal.dill", b"s" * size)
        if retain:
            files["search_internal.dill"] = {
                "container": str(container),
                "member": "files/search_internal/search_internal.dill",
                "bytes": size,
                "sha256": hashlib.sha256(b"s" * size).hexdigest(),
            }
    return result_dir, files


def _fit_summary(role, result_dir, files, *, retained=True, anchor=False):
    if anchor:
        return {
            "model_role": role,
            "status": "success",
            "search_engine": "VerifiedZeroResidualAnchor",
            "result_path": None,
            "search_internal_retention_requested": False,
            "search_internal_retained": False,
            "search_internal_payload": None,
        }
    return {
        "model_role": role,
        "status": "success",
        "search_engine": "Nautilus",
        "result_path": str(result_dir),
        "search_internal_retention_requested": True,
        "search_internal_retained": retained,
        "search_internal_payload": {
            "backend": "Nautilus",
            "required_files": ["search_internal.dill"],
            "output_path": str(result_dir),
            "route": "directory" if any("path" in f for f in files.values()) else "zip",
            "files": files,
            "missing_required": [] if retained else ["search_internal.dill"],
            "bound_to_result_path": True,
            "retained": retained,
            "error": None,
        },
    }


def _retention_contract(case_record, *, bracket):
    roles = {}
    for role, key in (("smooth", "smooth_fit"), ("subhalo", "subhalo_fit")):
        fit = case_record[key]
        payload = fit.get("search_internal_payload") or {}
        required = not (bracket and role == "subhalo")
        if required:
            complete = (
                fit["search_internal_retention_requested"] is True
                and fit["search_internal_retained"] is True
                and payload.get("missing_required") == []
            )
        else:
            complete = (
                fit["search_internal_retention_requested"] is False
                and fit["search_internal_retained"] is False
            )
        roles[role] = {
            "sampler_state_required": required,
            "fit_status": fit["status"],
            "retention_requested": fit["search_internal_retention_requested"],
            "retained": fit["search_internal_retained"],
            "route": payload.get("route"),
            "files": payload.get("files"),
            "missing_required": payload.get("missing_required"),
            "bound_to_result_path": payload.get("bound_to_result_path"),
            "complete": complete,
        }
    return {
        "policy": "standard pairs retain both fresh searches; brackets retain H0 only",
        "roles": roles,
        "complete": all(r["complete"] for r in roles.values()),
    }


def _receipt_artifacts(output):
    return {
        str(p): sha256_file(p)
        for p in output.rglob("*")
        if p.is_file() and p.name != "worker_exit.json"
    }


def fixture(
    tmp_path,
    *,
    q=12.0,
    accepted=True,
    bracket=False,
    direction=None,
    retain=("smooth", "subhalo"),
    route="directory",
):
    case = {
        "case_id": "case-1",
        "system_id": "sys0001",
        "arm": "asimov_injected",
        "campaign": "test",
        "direction": direction,
        "sampler_seed": 100,
        "compute_bracket_fisher_q": bracket,
        "frozen_mass_position": {
            "target_log10_m200": 8.2,
            "position_yx_arcsec": [0.2, 0.1],
        },
        "scope": "selected12_brackets" if bracket else "archived_cases",
    }
    identity = {k: case[k] for k in ("case_id", "system_id", "arm", "direction")}
    identity["compute_bracket_fisher_q"] = bracket
    case["case_identity_payload"] = identity
    case["case_identity_signature"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    catalog_path = dump(
        tmp_path / "catalog.json",
        {
            "cases": [case],
            "release_freeze": {"sha256": "freeze-hash"},
            "views": {"test": ["case-1"]},
        },
    )
    config = dump(
        tmp_path / "config.json",
        {"test": True, "run_name": f"ladder_selected_{case['system_id']}"},
    )
    positions = dump(tmp_path / "positions.json", {"position": [0, 1]})
    output = tmp_path / "attempt"
    child = output / "case"
    spec = dict(
        case,
        case_kind="bracket" if bracket else "standard",
        output=str(output),
        config=str(config),
        positions=str(positions),
        hashes={
            str(config): sha256_file(config),
            str(positions): sha256_file(positions),
        },
        catalog_sha256=sha256_file(catalog_path),
        objective_version="consistent_sampling_v2",
        procedure_version="fresh_nonlinear_v7_lbfgsb_v2",
        release_freeze_sha256="freeze-hash",
    )
    spec_path = dump(tmp_path / "spec.json", spec)
    identity_bindings = {
        key: spec[key]
        for key in (
            "case_id",
            "catalog_sha256",
            "case_identity_signature",
            "objective_version",
            "procedure_version",
            "release_freeze_sha256",
        )
    }
    identity_bindings.update(
        spec_sha256=sha256_file(spec_path),
        config_sha256=sha256_file(config),
        positions_sha256=sha256_file(positions),
    )
    run = dict(
        identity_bindings,
        status="COMPLETE",
        system_id=case["system_id"],
        arm=case["arm"],
        direction=direction,
        case_kind=spec["case_kind"],
        archived_state_imported=False,
        case_output=str(child),
        h1_anchor_evidence_claim=False if bracket else None,
        compute_bracket_fisher_q=bracket,
    )
    dump(output / "production_run.json", run)
    payload = {
        k: spec[k]
        for k in (
            "system_id",
            "arm",
            "objective_version",
            "procedure_version",
            "release_freeze_sha256",
        )
    }
    payload["system_id"] = f"ladder_selected_{case['system_id']}"
    payload.update(
        positions_artifact_sha256=sha256_file(positions),
        sampler_seed=100,
        fit_psf_delta=None if direction is None else {"direction": direction},
        delta_log_likelihood=q / 2,
        q_fit=max(0.0, q),
        delta_log_evidence=None if bracket else 2.0,
        bracket_fisher_q=_fisher_fixture() if bracket else None,
        profile_role_statuses={
            "smooth": "accepted_repeatable_profile",
            "subhalo": (
                "verified_zero_residual_anchor"
                if bracket
                else "accepted_repeatable_profile"
            )
            if accepted
            else "unresolved",
        },
        numerical_status="accepted" if accepted else "unresolved",
        marginal_q_flag=abs(q - 10.0) < 1.0,
        profile_decision=q >= 10.0 if accepted else None,
        h1_anchor={"sampler_executed": False, "evidence_claim": False}
        if bracket
        else None,
    )
    smooth_dir, smooth_files = _retained_search(
        child, "smooth", retain="smooth" in retain, route=route
    )
    case_record = {
        "smooth_fit": _fit_summary(
            "smooth", smooth_dir, smooth_files, retained="smooth" in retain
        )
    }
    if bracket:
        case_record["subhalo_fit"] = _fit_summary("subhalo", None, {}, anchor=True)
    else:
        subhalo_dir, subhalo_files = _retained_search(
            child, "subhalo", retain="subhalo" in retain, route=route
        )
        case_record["subhalo_fit"] = _fit_summary(
            "subhalo", subhalo_dir, subhalo_files, retained="subhalo" in retain
        )
    contract = _retention_contract(case_record, bracket=bracket)
    payload.update(
        case=case_record,
        retention_contract=contract,
        artifact_completeness_status="complete" if contract["complete"] else "incomplete",
    )
    suffix = "" if direction is None else f"_dir{direction}"
    payload_path = dump(
        child / f"nonlinear_validation_{case['arm']}{suffix}.json", payload
    )
    receipt = dict(
        identity_bindings,
        status="COMPLETE",
        artifacts=_receipt_artifacts(output),
    )
    dump(output / "worker_exit.json", receipt)
    return catalog_path, spec_path, output, payload_path


def harvest(catalog, specs):
    return harvest_production(catalog, specs, enforce_production_counts=False)


def refresh(output):
    path = output / "worker_exit.json"
    receipt = json.loads(path.read_text())
    receipt["artifacts"] = _receipt_artifacts(output)
    dump(path, receipt)


def _payload_case_fits(payload_path):
    data = json.loads(payload_path.read_text())
    return data, data["case"]["smooth_fit"], data["case"]["subhalo_fit"]


def test_accepted_and_report(tmp_path):
    catalog, spec, _output, _payload = fixture(tmp_path)
    result = harvest(catalog, [spec])
    assert result["status"] == "COMPLETE"
    assert result["rows"][0]["q_signed"] == 12.0
    assert result["rows"][0]["artifact_completeness_status"] == "complete"
    assert result["views"]["test"]["accepted_detections"] == 1
    assert result["views"]["test"]["incomplete"] == 0
    write_harvest(result, tmp_path / "harvest")
    assert len(list((tmp_path / "harvest").iterdir())) == 3
    assert "Incomplete" in (tmp_path / "harvest" / "PRODUCTION_HARVEST.md").read_text()


def test_reviewer_missing_retention_flags_are_never_production_complete(tmp_path):
    """GPT Pro's reproduction: requested True, retained False, COMPLETE."""
    catalog, spec, output, payload_path = fixture(tmp_path)
    data, smooth, subhalo = _payload_case_fits(payload_path)
    for fit in (smooth, subhalo):
        fit["search_internal_retained"] = False
    dump(payload_path, data)
    refresh(output)
    result = harvest(catalog, [spec])
    row = result["rows"][0]
    assert result["status"] == "INCOMPLETE_OR_UNRESOLVED"
    assert row["status"] == "failed"
    assert row["q_signed"] is None
    assert row["h1_evidence_claim"] is False
    assert any("retained state" in error for error in row["integrity_errors"])


@pytest.mark.parametrize(
    "defect",
    [
        "smooth_dill_deleted",
        "subhalo_dill_deleted",
        "zero_bytes",
        "hash_mismatch",
        "not_receipt_bound",
        "time_only_inventory",
        "escapes_result_path",
        "foreign_result_path",
        "contract_incomplete",
        "payload_status_incomplete",
        "no_contract",
    ],
)
def test_required_sampler_state_is_verified_not_declared(tmp_path, defect):
    catalog, spec, output, payload_path = fixture(tmp_path)
    data, smooth, subhalo = _payload_case_fits(payload_path)
    smooth_dill = Path(
        smooth["search_internal_payload"]["files"]["search_internal.dill"]["path"]
    )
    if defect == "smooth_dill_deleted":
        smooth_dill.unlink()
    elif defect == "subhalo_dill_deleted":
        Path(
            subhalo["search_internal_payload"]["files"]["search_internal.dill"]["path"]
        ).unlink()
    elif defect == "zero_bytes":
        smooth_dill.write_bytes(b"")
        smooth["search_internal_payload"]["files"]["search_internal.dill"].update(
            bytes=0, sha256=sha256_file(smooth_dill)
        )
    elif defect == "hash_mismatch":
        smooth_dill.write_bytes(b"different-state")
    elif defect == "time_only_inventory":
        smooth_dill.unlink()
        del smooth["search_internal_payload"]["files"]["search_internal.dill"]
    elif defect == "escapes_result_path":
        outside = tmp_path / "elsewhere.dill"
        outside.write_bytes(b"s" * 16)
        smooth["search_internal_payload"]["files"]["search_internal.dill"]["path"] = str(
            outside
        )
    elif defect == "foreign_result_path":
        smooth["search_internal_payload"]["bound_to_result_path"] = False
    elif defect == "contract_incomplete":
        data["retention_contract"]["roles"]["smooth"]["complete"] = False
    elif defect == "payload_status_incomplete":
        data["artifact_completeness_status"] = "incomplete"
    elif defect == "no_contract":
        del data["retention_contract"]
    dump(payload_path, data)
    refresh(output)
    if defect == "not_receipt_bound":
        receipt_path = output / "worker_exit.json"
        receipt = json.loads(receipt_path.read_text())
        del receipt["artifacts"][str(smooth_dill)]
        dump(receipt_path, receipt)
    row = harvest(catalog, [spec])["rows"][0]
    assert row["status"] == "failed", defect
    assert row["integrity_errors"], defect
    assert row["q_signed"] is None
    assert row["profile_decision"] is None


def test_zip_route_sampler_state_is_verified_inside_the_archive(tmp_path):
    import zipfile

    catalog, spec, output, payload_path = fixture(tmp_path, route="zip")
    assert harvest(catalog, [spec])["rows"][0]["status"] == "accepted"
    data, smooth, _ = _payload_case_fits(payload_path)
    container = Path(
        smooth["search_internal_payload"]["files"]["search_internal.dill"]["container"]
    )
    with zipfile.ZipFile(container, "w") as archive:
        archive.writestr("files/model.json", "{}")
        archive.writestr("files/search_internal/search_internal.dill", b"tampered")
    refresh(output)
    row = harvest(catalog, [spec])["rows"][0]
    assert row["status"] == "failed"
    assert any("sampler-state hash" in error for error in row["integrity_errors"])


def test_bracket_retains_h0_state_only_and_anchor_claims_none(tmp_path):
    catalog, spec, output, payload_path = fixture(tmp_path, bracket=True)
    assert harvest(catalog, [spec])["rows"][0]["status"] == "accepted"
    data, _, anchor = _payload_case_fits(payload_path)
    anchor["search_internal_retained"] = True
    dump(payload_path, data)
    refresh(output)
    assert harvest(catalog, [spec])["rows"][0]["status"] == "failed"

    catalog, spec, output, payload_path = fixture(tmp_path / "h0_missing", bracket=True)
    data, smooth, _ = _payload_case_fits(payload_path)
    Path(smooth["search_internal_payload"]["files"]["search_internal.dill"]["path"]).unlink()
    refresh(output)
    row = harvest(catalog, [spec])["rows"][0]
    assert row["status"] == "failed"
    assert row["q_f_production_at_position"] is None


def test_incomplete_artifact_attempt_is_reported_separately_without_q(tmp_path):
    catalog, spec, output, payload_path = fixture(tmp_path, retain=("subhalo",))
    data = json.loads(payload_path.read_text())
    assert data["artifact_completeness_status"] == "incomplete"
    for filename in ("production_run.json", "worker_exit.json"):
        path = output / filename
        record = json.loads(path.read_text())
        record["status"] = "INCOMPLETE_ARTIFACTS"
        dump(path, record)
    refresh(output)
    result = harvest(catalog, [spec])
    row = result["rows"][0]
    assert result["status"] == "INCOMPLETE_OR_UNRESOLVED"
    assert row["status"] == "incomplete"
    assert row["q_signed"] is None
    assert row["profile_decision"] is None
    assert row["integrity_errors"] == []
    assert result["views"]["test"]["incomplete"] == 1
    assert result["views"]["test"]["accepted"] == 0
    write_harvest(result, tmp_path / "harvest")
    report = (tmp_path / "harvest" / "PRODUCTION_HARVEST.md").read_text()
    assert "| test | 1 | 0 | 0 | 1 | 0 | 0 |" in report


@pytest.mark.parametrize(
    "q,accepted,marginal,decision",
    [(-1.0, True, False, False), (9.5, True, True, False), (10.5, False, True, None)],
)
def test_signed_clipped_and_orthogonal_status(
    tmp_path, q, accepted, marginal, decision
):
    catalog, spec, _, _ = fixture(tmp_path, q=q, accepted=accepted)
    row = harvest(catalog, [spec])["rows"][0]
    assert row["q_signed"] == q
    assert row["q_clipped"] == max(0.0, q)
    assert row["marginal_q_flag"] is marginal
    assert row["profile_decision"] is decision
    assert row["status"] == ("accepted" if accepted else "unresolved")


def test_missing_and_failed_rows_kept(tmp_path):
    catalog, spec, output, _ = fixture(tmp_path)
    assert harvest(catalog, [])["views"]["test"]["missing"] == 1
    dump(output / "worker_exit.json", {"status": "FAILED", "error": "fixture"})
    result = harvest(catalog, [spec])
    assert result["views"]["test"]["failed"] == 1
    assert result["rows"][0]["q_signed"] is None


def test_duplicate_complete_rejected_without_selection(tmp_path):
    catalog, spec, _output, _ = fixture(tmp_path)
    data = json.loads(spec.read_text())
    data["output"] = str(tmp_path / "attempt2")
    spec2 = dump(tmp_path / "spec2.json", data)
    dump(tmp_path / "attempt2" / "worker_exit.json", {"status": "COMPLETE"})
    with pytest.raises(ProductionHarvestError, match="duplicate COMPLETE"):
        harvest(catalog, [spec, spec2])


@pytest.mark.parametrize("target", ["payload", "config", "positions"])
def test_artifact_or_input_tamper_fails_closed(tmp_path, target):
    catalog, spec, _output, payload = fixture(tmp_path)
    path = payload if target == "payload" else tmp_path / f"{target}.json"
    path.write_text("{}")
    row = harvest(catalog, [spec])["rows"][0]
    assert row["status"] == "failed"
    assert row["integrity_errors"]
    assert row["q_signed"] is None


@pytest.mark.parametrize(
    "key",
    [
        "case_id",
        "catalog_sha256",
        "case_identity_signature",
        "config_sha256",
        "positions_sha256",
        "objective_version",
        "procedure_version",
        "spec_sha256",
        "release_freeze_sha256",
    ],
)
def test_identity_tamper_fails_even_with_new_receipt_hash(tmp_path, key):
    catalog, spec, output, _payload = fixture(tmp_path)
    path = output / "production_run.json"
    data = json.loads(path.read_text())
    data[key] = "different"
    dump(path, data)
    refresh(output)
    row = harvest(catalog, [spec])["rows"][0]
    assert row["status"] == "failed"
    assert row["q_signed"] is None


def test_receipt_must_bind_payload(tmp_path):
    catalog, spec, output, payload = fixture(tmp_path)
    path = output / "worker_exit.json"
    data = json.loads(path.read_text())
    del data["artifacts"][str(payload)]
    dump(path, data)
    assert harvest(catalog, [spec])["rows"][0]["status"] == "failed"


def test_bracket_forbids_evidence(tmp_path):
    catalog, spec, output, payload = fixture(tmp_path, bracket=True)
    row = harvest(catalog, [spec])["rows"][0]
    assert row["status"] == "accepted"
    assert row["h1_evidence_claim"] is False
    data = json.loads(payload.read_text())
    data["delta_log_evidence"] = 1.0
    dump(payload, data)
    refresh(output)
    assert harvest(catalog, [spec])["rows"][0]["status"] == "failed"


def test_psf_direction_is_identity_not_deduplicated_axis(tmp_path):
    catalog, spec, output, payload = fixture(tmp_path, direction=3)
    assert harvest(catalog, [spec])["rows"][0]["status"] == "accepted"
    data = json.loads(payload.read_text())
    data["fit_psf_delta"]["direction"] = 1
    dump(payload, data)
    refresh(output)
    assert harvest(catalog, [spec])["rows"][0]["status"] == "failed"


def test_default_requires_full_canonical_denominators(tmp_path):
    catalog, _, _, _ = fixture(tmp_path)
    with pytest.raises(ProductionHarvestError, match="denominator"):
        harvest_production(catalog, [])


def test_import_does_not_load_science_runtime():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from hwoslaps.campaign import production_harvest; assert not {'autolens', 'jax', 'numpy'} & sys.modules.keys()",
        ],
        check=True,
    )


def test_unresolved_without_finite_candidate_stays_unresolved(tmp_path):
    catalog, spec, output, payload = fixture(tmp_path, accepted=False)
    data = json.loads(payload.read_text())
    data.update(delta_log_likelihood=None, q_fit=None, marginal_q_flag=None)
    dump(payload, data)
    refresh(output)
    row = harvest(catalog, [spec])["rows"][0]
    assert row["status"] == "unresolved"
    assert row["q_signed"] is None
    assert row["profile_decision"] is None


def test_explicit_mirror_mapping_preserves_remote_identity(tmp_path):
    import shutil

    original = tmp_path / "remote"
    catalog, spec, _, _ = fixture(original)
    mirror = tmp_path / "mirror"
    shutil.copytree(original, mirror)
    shutil.rmtree(original)
    result = harvest_production(
        mirror / catalog.name,
        [str(spec)],
        path_mappings={str(original): str(mirror)},
        enforce_production_counts=False,
    )
    assert result["rows"][0]["status"] == "accepted"


def test_completed_after_failed_attempt_keeps_failure_history(tmp_path):
    catalog, spec, _, _ = fixture(tmp_path)
    data = json.loads(spec.read_text())
    data["output"] = str(tmp_path / "failed-attempt")
    failed_spec = dump(tmp_path / "failed-spec.json", data)
    dump(tmp_path / "failed-attempt" / "worker_exit.json", {"status": "FAILED"})
    row = harvest(catalog, [failed_spec, spec])["rows"][0]
    assert row["status"] == "accepted"
    assert row["attempt_count"] == 2
    assert row["completed_attempt_count"] == 1
    assert row["attempts"][0]["status"] == "FAILED"


@pytest.mark.parametrize("change_science", [False, True])
def test_config_restamp_preserves_catalog_identity(tmp_path, change_science):
    catalog, spec_path, output, _ = fixture(tmp_path)
    cat = json.loads(catalog.read_text())
    cat["cases"][0]["input_records"] = {"config": {"sha256": "source-config"}}
    dump(catalog, cat)
    spec = json.loads(spec_path.read_text())
    original_signature = spec["case_identity_signature"]
    spec["catalog_sha256"] = sha256_file(catalog)
    spec["catalog_case_identity_signature"] = original_signature
    spec["case_identity_payload"].update(
        source_config_sha256="source-config",
        config_sha256=spec["hashes"][spec["config"]],
        positions_sha256=spec["hashes"][spec["positions"]],
    )
    if change_science:
        spec["case_identity_payload"]["direction"] = 7
    spec["case_identity_signature"] = hashlib.sha256(
        json.dumps(
            spec["case_identity_payload"], sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    dump(spec_path, spec)
    for filename in ("production_run.json", "worker_exit.json"):
        path = output / filename
        data = json.loads(path.read_text())
        data.update(
            catalog_sha256=sha256_file(catalog),
            spec_sha256=sha256_file(spec_path),
            case_identity_signature=spec["case_identity_signature"],
            catalog_case_identity_signature=original_signature,
        )
        dump(path, data)
    refresh(output)
    row = harvest(catalog, [spec_path])["rows"][0]
    assert row["status"] == ("failed" if change_science else "accepted")


def _actual_materialization_fixture(tmp_path, bracket=False):
    """Use actual catalog routing/materialization, only fake input file bytes."""
    import importlib.util

    from hwoslaps.campaign.release_catalog import _runner_spec_template

    catalog_path, _old_spec_path, _, _ = fixture(tmp_path, bracket=bracket)
    catalog = json.loads(catalog_path.read_text())
    case = catalog["cases"][0]
    case.update(
        status="READY_FRESH_SEARCH",
        dispatchable=True,
        sampler_seed_spawn_key=[5, 1, 25],
        fresh_search_namespace="test-only",
    )
    source = tmp_path / "source.yaml"
    source.write_text(
        f"run_name: ladder_selected_{case['system_id']}\n"
        "stage0:\n  code_revision:\n    git_hash: old\n    git_dirty: false\n    sha256: old\nscience_fixture: unchanged\n"
    )
    positions = tmp_path / "positions.json"
    asset = dump(tmp_path / "asset.json", {"source": "synthetic"})
    procedure = dump(tmp_path / "procedure.json", {"source": "synthetic"})
    freeze = dump(tmp_path / "freeze.json", {"release": "synthetic"})
    case["input_records"] = {
        role: {
            "path": str(path),
            "execution_path": str(path),
            "sha256": sha256_file(path),
        }
        for role, path in (
            ("config", source),
            ("positions", positions),
            ("source_asset", asset),
        )
    }
    case["compute_tangent_comparator"] = bracket
    case["compute_bracket_fisher_q"] = bracket
    case["case_identity_payload"]["compute_bracket_fisher_q"] = bracket
    case["case_identity_payload"]["compute_tangent_comparator"] = bracket
    case["case_identity_signature"] = hashlib.sha256(
        json.dumps(
            case["case_identity_payload"], sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    case["procedure_source_sha256"] = sha256_file(procedure)
    case["release_freeze_sha256"] = sha256_file(freeze)
    case["runner_spec_template"] = _runner_spec_template(
        case,
        str(freeze),
        sha256_file(freeze),
        str(procedure),
        sha256_file(procedure),
        str(tmp_path / "approval.json"),
    )
    catalog["release_freeze"]["sha256"] = sha256_file(freeze)
    if bracket:
        case.update(
            status="PENDING_BRACKET_GENERATION",
            dispatchable=False,
            frozen_mass_position={
                "target_log10_m200": 8.2,
                "target_mass_msun": 10**8.2,
                "position_yx_arcsec": [0.2, 0.1],
            },
        )
    dump(catalog_path, catalog)
    approval = {
        "status": "APPROVED",
        "catalog_sha256": sha256_file(catalog_path),
        "release_freeze_sha256": sha256_file(freeze),
        "authorized_scope": "all_standard_and_brackets",
        "authorized_gpu_limit": 8,
        "case_ids": [case["case_id"]],
    }
    approval_path = dump(tmp_path / "approval.json", approval)
    approval.update(path=str(approval_path), sha256=sha256_file(approval_path))
    location = (
        Path(__file__).resolve().parents[1] / "scripts/run_nonlinear_production.py"
    )
    module_spec = importlib.util.spec_from_file_location(
        "production_contract_cli", location
    )
    cli = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(cli)
    return catalog_path, case, approval, cli


def _emit_synthetic_completion(spec_path, case):
    """Mock completed output schema, never execute a scientific runner."""
    spec = json.loads(spec_path.read_text())
    output = Path(spec["output"])
    bindings = {
        key: spec[key]
        for key in (
            "case_id",
            "catalog_sha256",
            "case_identity_signature",
            "objective_version",
            "procedure_version",
            "release_freeze_sha256",
            "catalog_case_identity_signature",
        )
    }
    bindings.update(
        spec_sha256=sha256_file(spec_path),
        config_sha256=sha256_file(Path(spec["config"])),
        positions_sha256=sha256_file(Path(spec["positions"])),
    )
    bracket = spec["case_kind"] == "bracket"
    run = dict(
        bindings,
        status="COMPLETE",
        system_id=spec["system_id"],
        arm=spec["arm"],
        direction=spec.get("direction"),
        case_kind=spec["case_kind"],
        case_output=str(output / "case"),
        archived_state_imported=False,
        h1_anchor_evidence_claim=False if bracket else None,
        compute_bracket_fisher_q=bracket,
    )
    dump(output / "production_run.json", run)
    payload = {
        key: spec[key]
        for key in (
            "system_id",
            "arm",
            "objective_version",
            "procedure_version",
            "release_freeze_sha256",
        )
    }
    payload["system_id"] = f"ladder_selected_{case['system_id']}"
    payload.update(
        positions_artifact_sha256=bindings["positions_sha256"],
        sampler_seed=case["sampler_seed"],
        fit_psf_delta=None,
        profile_role_statuses={
            "smooth": "accepted_repeatable_profile",
            "subhalo": "verified_zero_residual_anchor"
            if bracket
            else "accepted_repeatable_profile",
        },
        numerical_status="accepted",
        delta_log_likelihood=6.0,
        q_fit=12.0,
        profile_decision=True,
        marginal_q_flag=False,
        delta_log_evidence=None if bracket else 1.0,
        bracket_fisher_q=_fisher_fixture() if bracket else None,
        h1_anchor={"sampler_executed": False, "evidence_claim": False}
        if bracket
        else None,
    )
    child = output / "case"
    smooth_dir, smooth_files = _retained_search(child, "smooth")
    case_record = {"smooth_fit": _fit_summary("smooth", smooth_dir, smooth_files)}
    if bracket:
        case_record["subhalo_fit"] = _fit_summary("subhalo", None, {}, anchor=True)
    else:
        subhalo_dir, subhalo_files = _retained_search(child, "subhalo")
        case_record["subhalo_fit"] = _fit_summary("subhalo", subhalo_dir, subhalo_files)
    payload.update(
        case=case_record,
        retention_contract=_retention_contract(case_record, bracket=bracket),
        artifact_completeness_status="complete",
    )
    dump(child / f"nonlinear_validation_{spec['arm']}.json", payload)
    dump(
        output / "worker_exit.json",
        dict(bindings, status="COMPLETE", artifacts=_receipt_artifacts(output)),
    )


def test_actual_archive_materialize_validate_and_harvest(tmp_path):
    from hwoslaps.campaign.release_catalog import materialize_case_spec

    catalog, case, approval, cli = _actual_materialization_fixture(tmp_path)
    result = materialize_case_spec(
        case,
        tmp_path / "materialized",
        {"git_hash": "new", "git_dirty": False, "sha256": "c" * 64},
        approval_receipt=approval,
        catalog_sha256=sha256_file(catalog),
    )
    spec_path = Path(result["case_spec"])
    cli.validate_spec(json.loads(spec_path.read_text()), spec_path)
    _emit_synthetic_completion(spec_path, case)
    row = harvest(catalog, [spec_path])["rows"][0]
    assert row["status"] == "accepted", row["integrity_errors"]


@pytest.mark.parametrize(
    "tamper", [None, "source_config", "target_mass", "target_position"]
)
def test_actual_bracket_resolve_materialize_validate_and_harvest(tmp_path, tamper):
    from hwoslaps.campaign.release_catalog import materialize_case_spec
    from hwoslaps.campaign.release_dependencies import resolve_bracket

    catalog, case, approval, cli = _actual_materialization_fixture(
        tmp_path, bracket=True
    )
    generated = {}
    for role in ("config", "positions", "h1_anchor"):
        path = tmp_path / f"generated_{role}.json"
        data = {
            "stage0": {
                "code_revision": {
                    "git_hash": "old",
                    "git_dirty": False,
                    "sha256": "old",
                }
            },
            "bracket_fixture": True,
        }
        if role == "config":
            data["run_name"] = f"ladder_selected_{case['system_id']}"
        if role == "h1_anchor":
            data = {
                "case_id": case["case_id"],
                "evidence_claim": False,
                "sampler_executed": False,
            }
        dump(path, data)
        generated[role] = str(path)
        generated[f"{role}_sha256"] = sha256_file(path)
    receipt = {
        "status": "COMPLETE",
        "case_id": case["case_id"],
        "catalog_sha256": sha256_file(catalog),
        "source_config_sha256": case["input_records"]["config"]["sha256"],
        "source_positions_sha256": case["input_records"]["positions"]["sha256"],
        "generated": {
            **generated,
            **case["frozen_mass_position"],
            "bracket_rung": "plus_0.1dex",
        },
    }
    receipt_path = dump(tmp_path / "dependency.json", receipt)
    resolved = resolve_bracket(
        json.loads(catalog.read_text()),
        case["case_id"],
        receipt_path,
        sha256_file(catalog),
    )
    result = materialize_case_spec(
        resolved,
        tmp_path / "materialized",
        {"git_hash": "new", "git_dirty": False, "sha256": "c" * 64},
        approval_receipt=approval,
        catalog_sha256=sha256_file(catalog),
    )
    spec_path = Path(result["case_spec"])
    if tamper is not None:
        changed = json.loads(receipt_path.read_text())
        if tamper == "source_config":
            changed["source_config_sha256"] = "incorrect-root"
        elif tamper == "target_mass":
            changed["generated"]["target_mass_msun"] *= 2.0
        else:
            changed["generated"]["position_yx_arcsec"] = [0.9, 0.9]
        dump(receipt_path, changed)
        changed_spec = json.loads(spec_path.read_text())
        changed_spec["hashes"][str(receipt_path)] = sha256_file(receipt_path)
        dump(spec_path, changed_spec)
    cli.validate_spec(json.loads(spec_path.read_text()), spec_path)
    _emit_synthetic_completion(spec_path, case)
    row = harvest(catalog, [spec_path])["rows"][0]
    assert row["status"] == ("failed" if tamper else "accepted"), row[
        "integrity_errors"
    ]
    if tamper:
        assert row["q_signed"] is None


@pytest.mark.parametrize(
    "change", ["missing", "nonfinite", "kernel", "mass", "position", "disabled"]
)
def test_bracket_fisher_comparator_required_at_exact_target(tmp_path, change):
    catalog, spec, output, payload = fixture(tmp_path, bracket=True)
    data = json.loads(payload.read_text())
    if change == "missing":
        data["bracket_fisher_q"] = None
    elif change == "nonfinite":
        data["bracket_fisher_q"]["q_f_production_at_position"] = float("nan")
    elif change == "kernel":
        data["bracket_fisher_q"]["kernel_shape_native"] = [151, 151]
    elif change == "mass":
        data["bracket_fisher_q"]["log10_m200"] = 8.3
    elif change == "position":
        data["bracket_fisher_q"]["position_yx_arcsec"] = [0.3, 0.1]
    else:
        run_path = output / "production_run.json"
        run = json.loads(run_path.read_text())
        run["compute_bracket_fisher_q"] = False
        dump(run_path, run)
    dump(payload, data)
    refresh(output)
    row = harvest(catalog, [spec])["rows"][0]
    assert row["status"] == "failed"
    assert row["q_signed"] is None
    assert row["q_f_production_at_position"] is None

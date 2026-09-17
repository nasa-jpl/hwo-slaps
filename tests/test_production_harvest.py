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


def fixture(tmp_path, *, q=12.0, accepted=True, bracket=False, direction=None):
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
    config = dump(tmp_path / "config.json", {"test": True})
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
        procedure_version="fresh_nonlinear_v7_lbfgsb_v1",
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
    suffix = "" if direction is None else f"_dir{direction}"
    payload_path = dump(
        child / f"nonlinear_validation_{case['arm']}{suffix}.json", payload
    )
    receipt = dict(
        identity_bindings,
        status="COMPLETE",
        artifacts={str(p): sha256_file(p) for p in output.rglob("*.json")},
    )
    dump(output / "worker_exit.json", receipt)
    return catalog_path, spec_path, output, payload_path


def harvest(catalog, specs):
    return harvest_production(catalog, specs, enforce_production_counts=False)


def refresh(output):
    path = output / "worker_exit.json"
    receipt = json.loads(path.read_text())
    receipt["artifacts"] = {
        str(p): sha256_file(p) for p in output.rglob("*.json") if p != path
    }
    dump(path, receipt)


def test_accepted_and_report(tmp_path):
    catalog, spec, _output, _payload = fixture(tmp_path)
    result = harvest(catalog, [spec])
    assert result["status"] == "COMPLETE"
    assert result["rows"][0]["q_signed"] == 12.0
    assert result["views"]["test"]["accepted_detections"] == 1
    write_harvest(result, tmp_path / "harvest")
    assert len(list((tmp_path / "harvest").iterdir())) == 3


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
    dump(output / "case" / f"nonlinear_validation_{spec['arm']}.json", payload)
    dump(
        output / "worker_exit.json",
        dict(
            bindings,
            status="COMPLETE",
            artifacts={str(p): sha256_file(p) for p in output.rglob("*.json")},
        ),
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

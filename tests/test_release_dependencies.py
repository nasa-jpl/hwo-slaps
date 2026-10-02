"""Synthetic dependency gates; no scientific rendering or sampler calls."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from studies.rasti.campaign.release_catalog import (
    ReleaseCatalogError,
    _restamp_config,
    sha256_file,
    validate_case_approval,
)
from studies.rasti.campaign.release_dependencies import (
    position_input_identity,
    produce_position,
    resolve_position,
    validate_position_payload,
)


def identity_fixture(tmp_path):
    config = {
        "run_name": "ladder_full_pool_sys0001",
        "stage0": {
            "system_id": "sys0001",
            "source_asset_sha256": "s" * 64,
            "code_revision": {},
        },
        "psf": {"kernel": {"shape_native": [999, 999]}},
        "geometry": "full-square-preserved",
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    artifact = tmp_path / "ladder.npz"
    np.savez(
        artifact,
        system_id=config["run_name"],
        source_asset_sha256="s" * 64,
        config_hash="h" * 64,
        campaign_uuid="campaign",
        aperture_sha256="a" * 64,
        m_best_bracket_logm=[8.0, 8.1],
        psf_kernel_shape_native=[999, 999],
    )
    task = {
        "task_id": "position:sys0001",
        "system_id": "sys0001",
        "status": "PENDING_POSITION_EXTRACTION",
        "downstream_case_ids": [
            f"new_top50:sys0001:{arm}"
            for arm in (
                "asimov_injected",
                "asimov_below",
                "noisy_injected",
                "noisy_control",
            )
        ],
        "input_records": {
            "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
            "v6_ladder_artifact": {
                "path": str(artifact),
                "sha256": sha256_file(artifact),
            },
        },
    }
    identity = position_input_identity(task)[2]
    payload = {
        "system_id": config["run_name"],
        "source_asset_sha256": "s" * 64,
        "ladder_config_hash": "h" * 64,
        "ladder_campaign_uuid": "campaign",
        "aperture_sha256": "a" * 64,
        "fit_kernel_shape_native": [51, 51],
        "support_half_widths_arcsec": [1, 1],
        "aperture_centre_arcsec": [0, 0],
        "aperture_radius_arcsec": 0.8,
        "rungs": {
            name: {
                "logm": mass,
                "position_yx_arcsec": [0.2, 0.1],
                "q_max_relative_difference": 1e-8,
                "q_f_matched": 10.0,
            }
            for name, mass in identity["rungs"].items()
        },
    }
    return task, identity, payload


def test_exact_case_approval_and_bracket_scope():
    case = {
        "case_id": "bracket:one",
        "scope": "selected12_brackets",
        "release_freeze_sha256": "f" * 64,
    }
    approval = {
        "status": "APPROVED",
        "case_ids": ["bracket:one"],
        "authorized_scope": "all_standard_and_brackets",
        "authorized_gpu_limit": 8,
        "release_freeze_sha256": "f" * 64,
        "catalog_sha256": "c" * 64,
    }
    validate_case_approval(case, approval, "c" * 64)
    for change in (
        {"case_ids": ["bracket:other"]},
        {"authorized_scope": "all_standard_cases"},
        {"authorized_gpu_limit": True},
    ):
        with pytest.raises(ReleaseCatalogError):
            validate_case_approval(case, {**approval, **change}, "c" * 64)


def test_position_geometry_and_identity_fail_closed(tmp_path):
    task, identity, payload = identity_fixture(tmp_path)
    validate_position_payload(payload, identity)
    for key, value in (
        ("position_yx_arcsec", [0.9, 0.9]),
        ("q_max_relative_difference", 1e-3),
        ("logm", float("nan")),
    ):
        broken = copy.deepcopy(payload)
        broken["rungs"]["top"][key] = value
        with pytest.raises(ReleaseCatalogError):
            validate_position_payload(broken, identity)
    Path(task["input_records"]["config"]["path"]).write_text("corrupt")
    with pytest.raises(ReleaseCatalogError, match="sha256"):
        position_input_identity(task)


def test_new_fit_kernel_changes_only_declared_fields(tmp_path):
    task, _, _ = identity_fixture(tmp_path)
    source = Path(task["input_records"]["config"]["path"])
    dest = tmp_path / "fit.yaml"
    revision = {"git_hash": "new", "git_dirty": False, "sha256": "r" * 64}
    _restamp_config(source, dest, revision, fit_kernel=True)
    original, fit = yaml.safe_load(source.read_text()), yaml.safe_load(dest.read_text())
    assert fit["psf"]["kernel"]["shape_native"] == [51, 51]
    fit["psf"]["kernel"]["shape_native"] = [999, 999]
    fit["stage0"]["code_revision"] = original["stage0"]["code_revision"]
    assert fit == original


def test_producer_requires_approval_before_importing_science(tmp_path):
    task, _, _ = identity_fixture(tmp_path)
    catalog = {
        "position_tasks": [task],
        "cases": [
            {
                "case_id": case_id,
                "scope": "new_top50_standard_cases",
                "release_freeze_sha256": "f" * 64,
            }
            for case_id in task["downstream_case_ids"]
        ],
    }
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog))
    approval = tmp_path / "approval.json"
    approval.write_text('{"status":"PENDING"}')
    with pytest.raises(ReleaseCatalogError, match="APPROVED"):
        produce_position(
            path, task["task_id"], tmp_path / "out", {}, approval, tmp_path / "freeze"
        )
    assert not (tmp_path / "out").exists()


def test_resolver_requires_complete_catalog_bound_receipt(tmp_path):
    task, _identity, payload = identity_fixture(tmp_path)
    position = tmp_path / "position.json"
    position.write_text(json.dumps(payload))
    cases = []
    for case_id in task["downstream_case_ids"]:
        arm = case_id.split(":")[-1]
        cases.append(
            {
                "case_id": case_id,
                "case_identity_payload": {"case_id": case_id},
                "case_identity_signature": "x",
                "scope": "new_top50_standard_cases",
                "rung": "below" if arm == "asimov_below" else "top",
                "arm": arm,
                "system_id": "sys0001",
                "input_records": {**task["input_records"], "positions": None},
                "release_freeze_sha256": "f" * 64,
                "procedure_source_sha256": "p" * 64,
                "runner_spec_template": {
                    "release_freeze_path": "/freeze",
                    "procedure_source": "/source",
                    "approval_receipt": "/approval",
                },
            }
        )
    catalog = {"position_tasks": [task], "cases": cases}
    receipt = {
        "status": "COMPLETE",
        "task_id": task["task_id"],
        "catalog_sha256": "c" * 64,
        "ladder_artifact_sha256": task["input_records"]["v6_ladder_artifact"]["sha256"],
        "position": str(position),
        "position_sha256": sha256_file(position),
    }
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt))
    resolved = resolve_position(catalog, task["task_id"], path, "c" * 64)
    assert len(resolved) == 4 and all(case["dispatchable"] for case in resolved)
    assert all(not case["runner_spec_template"].get("pending") for case in resolved)
    assert catalog["cases"][0]["input_records"]["positions"] is None
    receipt["catalog_sha256"] = "wrong"
    path.write_text(json.dumps(receipt))
    with pytest.raises(ReleaseCatalogError, match="receipt"):
        resolve_position(catalog, task["task_id"], path, "c" * 64)


def test_bracket_resolver_rejects_shifted_target_and_emits_anchor_gate(tmp_path):
    from studies.rasti.campaign.release_dependencies import resolve_bracket

    case_id = "selected12_bracket:sys0001:plus_0.1dex"
    target = {
        "target_log10_m200": 8.2,
        "target_mass_msun": 10**8.2,
        "position_yx_arcsec": [0.2, 0.1],
    }
    files = {}
    for role in ("config", "positions", "h1_anchor"):
        path = tmp_path / f"{role}.json"
        payload = (
            {"case_id": case_id, "evidence_claim": False, "sampler_executed": False}
            if role == "h1_anchor"
            else {}
        )
        path.write_text(json.dumps(payload))
        files[role] = str(path)
        files[f"{role}_sha256"] = sha256_file(path)
    case = {
        "case_id": case_id,
        "system_id": "sys0001",
        "scope": "selected12_brackets",
        "arm": "h0_bracket",
        "sampler_seed_spawn_key": [5, 1, 25],
        "case_identity_payload": {"case_id": case_id},
        "case_identity_signature": "i",
        "input_records": {
            "config": {"sha256": "c" * 64},
            "positions": {"sha256": "p" * 64},
        },
        "frozen_mass_position": target,
        "release_freeze_sha256": "f" * 64,
        "procedure_source_sha256": "s" * 64,
        "runner_spec_template": {
            "release_freeze_path": "/freeze",
            "procedure_source": "/source",
            "approval_receipt": "/approval",
        },
    }
    receipt = {
        "status": "COMPLETE",
        "case_id": case_id,
        "catalog_sha256": "d" * 64,
        "source_config_sha256": "c" * 64,
        "source_positions_sha256": "p" * 64,
        "generated": {**files, **target, "bracket_rung": "plus_0.1dex"},
    }
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt))
    resolved = resolve_bracket({"cases": [case]}, case_id, path, "d" * 64)
    spec = resolved["runner_spec_template"]
    assert resolved["dispatchable"] and not spec.get("pending")
    assert spec["hashes"][spec["h1_anchor"]] == files["h1_anchor_sha256"]
    receipt["generated"]["target_log10_m200"] = 8.3
    path.write_text(json.dumps(receipt))
    with pytest.raises(ReleaseCatalogError, match="target"):
        resolve_bracket({"cases": [case]}, case_id, path, "d" * 64)


def test_actual_freeze_defines_declared_sampler_and_brackets():
    from studies.rasti.campaign.design_freeze import load_release_freeze

    freeze = load_release_freeze(
        Path(__file__).resolve().parents[1] / "configs/design/design_freeze_v7.yaml"
    )
    assert freeze["protocol"]["sampler"]["n_eff"] == 500
    assert freeze["protocol"]["sampler"]["f_live"] == 0.01


def test_real_freeze_tangent_policy_and_pending_template_identity(tmp_path):
    from studies.rasti.campaign.design_freeze import load_release_freeze
    from studies.rasti.campaign.release_catalog import _runner_spec_template, _sha256_json

    freeze = load_release_freeze(
        Path(__file__).resolve().parents[1] / "configs/design/design_freeze_v7.yaml"
    )
    assert freeze["protocol"]["tangent_comparator"] == {
        "enabled_views": ["selected12_standard", "selected12_brackets"],
        "other_cases": False,
    }
    case = {
        "case_id": "pending",
        "system_id": "sys0001",
        "arm": "h0_bracket",
        "scope": "selected12_brackets",
        "input_records": {},
        "case_identity_payload": {"case_id": "pending"},
        "case_identity_signature": "old",
        "compute_tangent_comparator": True,
    }
    spec = _runner_spec_template(
        case, "/freeze", "f" * 64, "/procedure", "p" * 64, "/approval"
    )
    assert spec["pending"] and spec["compute_tangent_comparator"] is True
    assert spec["compute_bracket_fisher_q"] is True
    assert spec["case_identity_payload"]["compute_bracket_fisher_q"] is True
    assert freeze["protocol"]["bracket_fisher_q"]["kernel_shape_native"] == [999, 999]
    assert spec["case_identity_payload"]["compute_tangent_comparator"] is True
    assert spec["case_identity_signature"] == _sha256_json(
        spec["case_identity_payload"]
    )
    case["compute_tangent_comparator"] = "true"
    with pytest.raises(ReleaseCatalogError, match="boolean"):
        _runner_spec_template(
            case, "/freeze", "f" * 64, "/procedure", "p" * 64, "/approval"
        )

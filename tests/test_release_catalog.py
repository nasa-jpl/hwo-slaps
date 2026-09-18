"""Import-safe tests for the additive v7 release declaration and routing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hwoslaps.campaign.design_freeze import load_release_freeze
from hwoslaps.campaign.release_catalog import (
    ReleaseCatalogError,
    materialize_case_spec,
    runner_invocation,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
RELEASE_PATH = ROOT / "configs/design/design_freeze_v7.yaml"


def test_v7_loader_consumes_v5_by_exact_hash():
    release = load_release_freeze(RELEASE_PATH)
    assert release["schema_version"] == 7
    assert release["freeze"]["version"] == 7
    assert release["consumed_freeze"]["version"] == 5
    assert release["protocol"]["fresh_searches"] is True
    assert release["protocol"]["reuse_archived_fit_state"] is False
    assert release["protocol"]["sampler"]["n_eff"] == 500
    assert release["runner_contract"]["approval_gate"]["required"] is True


def _ready_case():
    return {
        "case_id": "archived:test:0000",
        "scope": "archived_cases",
        "dispatchable": True,
        "status": "READY_FRESH_SEARCH",
        "release_freeze_sha256": "f" * 64,
        "system_id": "sys0000",
        "arm": "asimov_injected",
        "fresh_search_namespace": "v7/test",
        "runner": {"mode": "standard_case_spec", "entrypoint": "scripts/run_nonlinear_production.py"},
        "input_records": {
            "config": {"execution_path": "/remote/config.yaml"},
            "positions": {"execution_path": "/remote/positions.json"},
            "archived_case_artifact": {"sha256": "a" * 64},
        },
        "runner_spec_template": {
            "case_id": "archived:test:0000",
            "case_identity": "archived:test:0000",
            "case_kind": "standard",
            "config": "/remote/config.yaml",
            "positions": "/remote/positions.json",
            "arm": "asimov_injected",
            "release_freeze_path": "/remote/design_freeze_v7.yaml",
            "release_freeze_sha256": "f" * 64,
            "objective_version": "consistent_sampling_v2",
            "procedure_version": "fresh_nonlinear_v7_lbfgsb_v1",
            "execution_policy_version": "stage3_v7",
            "output": "{output_dir}",
            "hashes": {
                "/remote/design_freeze_v7.yaml": "f" * 64,
                "/remote/config.yaml": "b" * 64,
                "/remote/positions.json": "d" * 64,
            },
        },
    }


def test_runner_route_requires_immutable_approval_receipt():
    with pytest.raises(ReleaseCatalogError, match="approval receipt"):
        runner_invocation(_ready_case(), "/tmp/output")
    receipt = {
        "status": "APPROVED",
        "release_freeze_sha256": "f" * 64,
        "catalog_sha256": "c" * 64,
        "authorized_scope": "archived_cases",
        "case_ids": ["archived:test:0000"],
        "authorized_gpu_limit": 8,
        "path": "/remote/approval.json",
        "sha256": "e" * 64,
    }
    route = runner_invocation(
        _ready_case(),
        "/tmp/output",
        approval_receipt=receipt,
        catalog_sha256="c" * 64,
    )
    assert route["argv"] == ["scripts/run_nonlinear_production.py", "{case_spec_path}"]
    assert route["spec"]["output"] == str(Path("/tmp/output").resolve())
    assert route["spec"]["approval_receipt"] == "/remote/approval.json"
    assert route["spec"]["approval_receipt_sha256"] == "e" * 64
    assert route["environment"]["HWOSLAPS_REUSE_ARCHIVED_FIT_STATE"] == "0"


def test_pending_position_case_cannot_route_even_with_approval():
    case = _ready_case()
    case.update(
        {
            "case_id": "new_top50:sys0016:asimov_injected",
            "dispatchable": False,
            "status": "PENDING_POSITION_EXTRACTION",
        }
    )
    receipt = {
        "status": "APPROVED",
        "release_freeze_sha256": "f" * 64,
        "catalog_sha256": "c" * 64,
        "authorized_scope": "all_standard_cases",
        "authorized_gpu_limit": 8,
    }
    with pytest.raises(ReleaseCatalogError, match="PENDING_POSITION_EXTRACTION"):
        runner_invocation(case, "/tmp/output", approval_receipt=receipt, catalog_sha256="c" * 64)


def test_materialization_changes_only_code_revision_and_writes_hash_receipt(tmp_path):
    config_path = tmp_path / "source.yaml"
    config_path.write_text(
        "run_name: ladder_selected_sys0001\n"
        "stage0:\n"
        "  source_asset_sha256: 'a'\n"
        "  code_revision:\n"
        "    git_hash: old\n"
        "    git_dirty: false\n"
        "    sha256: 'b'\n"
        "lensing:\n"
        "  source_galaxy:\n"
        "    light:\n"
        "      asset_path: configs/source_assets/test.npz\n"
    )
    positions_path = tmp_path / "positions.json"
    positions_path.write_text('{"system_id": "ladder_selected_sys0001"}\n')
    case = {
        "case_id": "archived:test:0001",
        "scope": "archived_cases",
        "status": "READY_FRESH_SEARCH",
        "dispatchable": True,
        "release_freeze_sha256": "f" * 64,
        "fresh_search_namespace": "v7/test/0001",
        "arm": "asimov_injected",
        "input_records": {
            "config": {
                "path": str(config_path),
                "execution_path": "/remote/source.yaml",
                "sha256": sha256_file(config_path),
            },
            "positions": {
                "path": str(positions_path),
                "execution_path": "/remote/positions.json",
                "sha256": sha256_file(positions_path),
            },
        },
        "runner_spec_template": {
            "case_id": "archived:test:0001",
            "case_identity": "archived:test:0001",
            "case_kind": "standard",
            "config": "/remote/source.yaml",
            "positions": "/remote/positions.json",
            "arm": "asimov_injected",
            "release_freeze_path": "/remote/release.yaml",
            "release_freeze_sha256": "f" * 64,
            "objective_version": "consistent_sampling_v2",
            "procedure_version": "fresh_nonlinear_v7_lbfgsb_v1",
            "execution_policy_version": "stage3_v7",
            "output": "{output_dir}",
            "hashes": {},
        },
    }
    revision = {"git_hash": "new", "git_dirty": False, "sha256": "c" * 64}
    receipt = materialize_case_spec(case, tmp_path / "materialized", revision)
    assert receipt["status"] == "MATERIALIZED_PENDING_APPROVAL"
    assert receipt["config_restamp"]["allowed_changed_path"] == "stage0.code_revision"
    spec = json.loads(Path(receipt["case_spec"]).read_text())
    assert spec["positions"] == "/remote/positions.json"
    assert spec["hashes"]["/remote/positions.json"] == sha256_file(positions_path)
    assert spec["output"].endswith("/attempt")


def test_materialize_case_spec_uses_the_execution_path_when_the_mirror_is_absent(tmp_path):
    """A case resolved on the execution host has no local mirror of its inputs."""
    config_path = tmp_path / "remote_config.yaml"
    config_path.write_text(
        "run_name: ladder_selected_sys0002\n"
        "stage0:\n"
        "  source_asset_sha256: 'a'\n"
        "  code_revision:\n"
        "    git_hash: old\n"
        "    git_dirty: false\n"
        "    sha256: 'b'\n"
        "psf:\n"
        "  kernel:\n"
        "    shape_native: [999, 999]\n"
    )
    positions_path = tmp_path / "remote_positions.json"
    positions_path.write_text('{"system_id": "ladder_selected_sys0002"}\n')
    case = {
        "case_id": "new_top50:test:0002",
        "scope": "new_top50_standard_cases",
        "status": "READY_FRESH_SEARCH",
        "dispatchable": True,
        "release_freeze_sha256": "f" * 64,
        "fresh_search_namespace": "v7/test/0002",
        "arm": "asimov_injected",
        "input_records": {
            "config": {
                "path": "/Users/nobody/absent/config.yaml",
                "execution_path": str(config_path),
                "sha256": sha256_file(config_path),
            },
            "positions": {
                "path": str(tmp_path / "absent_positions.json"),
                "execution_path": str(positions_path),
                "sha256": sha256_file(positions_path),
            },
        },
        "runner_spec_template": {
            "case_id": "new_top50:test:0002",
            "case_identity": "new_top50:test:0002",
            "case_kind": "standard",
            "config": str(config_path),
            "positions": str(positions_path),
            "arm": "asimov_injected",
            "release_freeze_path": "/remote/release.yaml",
            "release_freeze_sha256": "f" * 64,
            "objective_version": "consistent_sampling_v2",
            "procedure_version": "fresh_nonlinear_v7_lbfgsb_v1",
            "execution_policy_version": "stage3_v7",
            "output": "{output_dir}",
            "hashes": {},
        },
    }
    revision = {"git_hash": "new", "git_dirty": False, "sha256": "c" * 64}
    receipt = materialize_case_spec(case, tmp_path / "materialized", revision)
    assert receipt["config_restamp"]["original_path"] == str(config_path.resolve())
    assert receipt["config_restamp"]["declared_kernel_conversion"] == "999_to_51_fit_only"
    spec = json.loads(Path(receipt["case_spec"]).read_text())
    assert spec["positions"] == str(positions_path)
    assert spec["hashes"][str(positions_path)] == sha256_file(positions_path)

    case["input_records"]["config"]["execution_path"] = str(tmp_path / "also_absent.yaml")
    with pytest.raises(ReleaseCatalogError, match="absent at both declared locations"):
        materialize_case_spec(case, tmp_path / "materialized_again", revision)


def test_restamp_preserves_yaml_integer_psf_keys(tmp_path):
    import yaml
    from hwoslaps.campaign.release_catalog import _restamp_config

    original = {
        "stage0": {"code_revision": {"git_hash": "old"}},
        "psf": {"aberrations": {"global_zernikes": {4: 0.03, 10: -0.02}}},
    }
    source, target = tmp_path / "source.yaml", tmp_path / "target.yaml"
    source.write_text(yaml.safe_dump(original))
    revision = {"git_hash": "new", "git_dirty": False, "sha256": "c" * 64}
    _restamp_config(source, target, revision)
    output = yaml.safe_load(target.read_text())
    assert output["psf"] == original["psf"]
    assert all(isinstance(key, int) for key in output["psf"]["aberrations"]["global_zernikes"])


def test_psf_direction_survives_case_to_worker_routing():
    from hwoslaps.campaign.release_catalog import _runner_spec_template

    case = _ready_case()
    case.update(arm="noisy_control_d2", direction=1)
    case["case_identity_payload"] = {
        "case_id": case["case_id"], "system_id": case["system_id"],
        "arm": case["arm"], "direction": 1,
    }
    spec = _runner_spec_template(case, "/release", "f" * 64, "/procedure", "a" * 64, "/approval")
    assert spec["direction"] == spec["case_identity_payload"]["direction"] == 1


def test_primary_psf_noise_keeps_the_archived_seed():
    from hwoslaps.campaign.release_catalog import _noise_seed

    assert _noise_seed(20260823, "sys0043", 0) == 1703553965
    assert _noise_seed(20260823, "sys0043", 1) != 1703553965

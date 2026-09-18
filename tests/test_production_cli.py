"""CPU-only contract tests for the Stage 3 per-case production CLI."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def load_cli():
    path = Path(__file__).parents[1] / "scripts/run_nonlinear_production.py"
    spec = importlib.util.spec_from_file_location("stage3_production_cli", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_validation():
    path = Path(__file__).parents[1] / "scripts/run_nonlinear_validation.py"
    spec = importlib.util.spec_from_file_location("stage3_validation_route", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def valid_spec(tmp_path, *, case_kind="standard"):
    cli = load_cli()
    output = tmp_path / "attempt"
    output.mkdir()
    (output / "worker.log").write_text("controller wrapper\n")
    paths = {}
    for name in (
        "config.yaml", "positions.json", "release.yaml", "fresh_profile.py",
        "source.npz", "approval.json", "catalog.json",
    ):
        path = tmp_path / name
        path.write_text(name)
        paths[name] = path
    identity_payload = {
        "case_id": "archived:case-0001",
        "system_id": "sys0001",
        "arm": "asimov_injected",
        "config_sha256": hashlib.sha256(paths["config.yaml"].read_bytes()).hexdigest(),
        "positions_sha256": hashlib.sha256(paths["positions.json"].read_bytes()).hexdigest(),
        "compute_tangent_comparator": case_kind == "standard",
        "compute_bracket_fisher_q": case_kind == "bracket",
    }
    signature = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    approval = {
        "status": "APPROVED",
        "case_ids": ["archived:case-0001"],
        "release_freeze_sha256": hashlib.sha256(paths["release.yaml"].read_bytes()).hexdigest(),
        "catalog_sha256": hashlib.sha256(paths["catalog.json"].read_bytes()).hexdigest(),
        "authorized_scope": (
            "selected12_brackets" if case_kind == "bracket" else "all_standard_cases"
        ),
        "authorized_gpu_limit": 8,
    }
    paths["approval.json"].write_text(json.dumps(approval))
    spec = {
        "case_id": "archived:case-0001",
        "case_identity": "archived:case-0001",
        "case_identity_payload": identity_payload,
        "case_identity_signature": signature,
        "system_id": "sys0001",
        "arm": "asimov_injected",
        "config": str(paths["config.yaml"]),
        "positions": str(paths["positions.json"]),
        "release_freeze_path": str(paths["release.yaml"]),
        "release_freeze_sha256": hashlib.sha256(paths["release.yaml"].read_bytes()).hexdigest(),
        "approval_receipt": str(paths["approval.json"]),
        "approval_receipt_sha256": hashlib.sha256(paths["approval.json"].read_bytes()).hexdigest(),
        "catalog_sha256": hashlib.sha256(paths["catalog.json"].read_bytes()).hexdigest(),
        "objective_version": "consistent_sampling_v2",
        "procedure_version": "fresh_nonlinear_v7_lbfgsb_v1",
        "case_kind": case_kind,
        "scope": "selected12_brackets" if case_kind == "bracket" else "archived_cases",
        "compute_tangent_comparator": case_kind == "standard",
        "compute_bracket_fisher_q": case_kind == "bracket",
        "procedure_source": str(paths["fresh_profile.py"]),
        "source_assets": [str(paths["source.npz"])],
        "execution_policy_version": "stage3_v7",
        "output": str(output),
        "hashes": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths.values()
            if path.name != "approval.json"
        },
    }
    spec["hashes"][str(paths["approval.json"])] = hashlib.sha256(
        paths["approval.json"].read_bytes()
    ).hexdigest()
    if case_kind == "bracket":
        anchor = tmp_path / "anchor.json"
        anchor.write_text("{}")
        spec.update(
            bracket_rung="above_upper_rung",
            bracket_arm_index=24,
            h1_anchor=str(anchor),
        )
        spec["hashes"][str(anchor)] = hashlib.sha256(anchor.read_bytes()).hexdigest()
    return cli, spec


def test_cli_accepts_controller_wrappers_but_requires_identity_and_hashes(tmp_path):
    cli, spec = valid_spec(tmp_path)
    cli.validate_spec(spec, tmp_path / "spec.json")
    bad = dict(spec)
    bad["case_identity_signature"] = "0" * 64
    with pytest.raises(ValueError, match="signature"):
        cli.validate_spec(bad, tmp_path / "spec.json")


def test_cli_requires_bracket_anchor_contract(tmp_path):
    cli, spec = valid_spec(tmp_path, case_kind="bracket")
    cli.validate_spec(spec, tmp_path / "spec.json")
    bad = dict(spec)
    bad.pop("h1_anchor")
    with pytest.raises(FileNotFoundError, match="h1_anchor"):
        cli.validate_spec(bad, tmp_path / "spec.json")


def test_worker_receipt_excludes_live_controller_logs(tmp_path):
    cli = load_cli()
    output = tmp_path / "attempt"
    output.mkdir()
    (output / "science.json").write_text("{}")
    (output / "worker_exit.json").write_text("old")
    for name in ("worker.log", "worker.stdout.log", "worker.stderr.log"):
        (output / name).write_text("controller output")
    receipt = cli._receipt_artifacts(output)
    assert str(output / "science.json") in receipt
    assert all(str(output / name) not in receipt for name in (
        "worker.log", "worker.stdout.log", "worker.stderr.log"
    ))
    assert str(output / "worker_exit.json") not in receipt


@pytest.mark.parametrize("completeness", ["complete", "incomplete"])
def test_worker_marks_missing_sampler_state_as_not_complete(tmp_path, monkeypatch, completeness):
    """The receipt says COMPLETE only when all sampler state was retained."""
    import sys
    from types import ModuleType

    cli, spec = valid_spec(tmp_path)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    route_calls = []
    fake_route = ModuleType("run_nonlinear_validation")

    def fake_main(route_args):
        route_calls.append(list(route_args))
        case_output = Path(route_args[3])
        case_output.mkdir(parents=True, exist_ok=True)
        contract = {
            "roles": {
                "smooth": {"sampler_state_required": True, "complete": True},
                "subhalo": {
                    "sampler_state_required": True,
                    "complete": completeness == "complete",
                },
            },
            "complete": completeness == "complete",
        }
        (case_output / "nonlinear_validation_asimov_injected.json").write_text(
            json.dumps(
                {
                    "numerical_status": "accepted",
                    "delta_log_likelihood": 6.0,
                    "artifact_completeness_status": completeness,
                    "retention_contract": contract,
                }
            )
        )

    fake_route.main = fake_main
    monkeypatch.setitem(sys.modules, "run_nonlinear_validation", fake_route)
    assert cli.main([str(spec_path)]) == 0
    assert route_calls and route_calls[0][2] == "asimov_injected"
    output = Path(spec["output"])
    run = json.loads((output / "production_run.json").read_text())
    receipt = json.loads((output / "worker_exit.json").read_text())
    expected = "COMPLETE" if completeness == "complete" else "INCOMPLETE_ARTIFACTS"
    assert run["status"] == expected
    assert receipt["status"] == expected
    assert run["artifact_completeness_status"] == completeness
    assert run["retention_contract"]["complete"] is (completeness == "complete")
    if completeness == "complete":
        assert receipt["error"] is None
    else:
        assert "subhalo" in receipt["error"]
        assert "smooth" not in receipt["error"]
    assert not (output / "production_failure.json").exists()
    assert str(output / "production_run.json") in receipt["artifacts"]


def test_worker_refuses_a_payload_without_completeness_status(tmp_path, monkeypatch):
    """A legacy payload without the v7 completeness field cannot complete."""
    import sys
    from types import ModuleType

    cli, spec = valid_spec(tmp_path)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    fake_route = ModuleType("run_nonlinear_validation")

    def fake_main(route_args):
        case_output = Path(route_args[3])
        case_output.mkdir(parents=True, exist_ok=True)
        (case_output / "nonlinear_validation_asimov_injected.json").write_text(
            json.dumps({"numerical_status": "accepted"})
        )

    fake_route.main = fake_main
    monkeypatch.setitem(sys.modules, "run_nonlinear_validation", fake_route)
    with pytest.raises(ValueError, match="artifact_completeness_status"):
        cli.main([str(spec_path)])
    output = Path(spec["output"])
    assert json.loads((output / "worker_exit.json").read_text())["status"] == "FAILED"
    assert not (output / "production_run.json").exists()


def test_cli_rejects_unapproved_sidecar(tmp_path):
    cli, spec = valid_spec(tmp_path)
    approval_path = Path(spec["approval_receipt"])
    approval = json.loads(approval_path.read_text())
    approval["status"] = "PENDING"
    approval_path.write_text(json.dumps(approval))
    spec["hashes"][str(approval_path)] = hashlib.sha256(
        approval_path.read_bytes()
    ).hexdigest()
    spec["approval_receipt_sha256"] = spec["hashes"][str(approval_path)]
    with pytest.raises(ValueError, match="APPROVED"):
        cli.validate_spec(spec, tmp_path / "spec.json")


def test_v7_route_pins_sampler_contract_without_changing_legacy_defaults():
    route = load_validation()
    protocol = {
        "fit": {
            "n_live_smooth": 100,
            "n_live_subhalo_search": 200,
            "n_live_subhalo_fixed": 100,
        }
    }
    release = {
        "protocol": {
            "sampler": {
                "n_eff": 500,
                "n_live_smooth": 100,
                "n_live_subhalo_search": 200,
                "n_live_subhalo_fixed": 100,
                "retain_sampler_internals": True,
                "n_shell": 1,
                "f_live": 0.01,
                "discard_exploration": False,
            }
        }
    }
    settings = route._v7_sampler_settings(protocol, release)
    assert settings == {
        "n_eff": 500,
        "n_shell": 1,
        "f_live": 0.01,
        "discard_exploration": False,
        "retain_search_internal": True,
        "sampler_contract": {
            "n_eff": 500,
            "n_shell": 1,
            "f_live": 0.01,
            "discard_exploration": False,
            "n_live_by_fit_mode": {
                "smooth": 100,
                "freed": 200,
                "fixed_template": 100,
            },
        },
    }
    assert route.V7_OBJECTIVE_VERSION == "consistent_sampling_v2"
    assert route.V7_PROFILE_PROCEDURE == "fresh_profile_v1"

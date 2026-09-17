"""CPU-only execution planning and approval tests; supervision is always mocked."""

import json
from pathlib import Path

import pytest

from hwoslaps.campaign.execution_prepare import (
    activate,
    digest,
    prepare,
    read,
    resource_job,
    scientific_config_sha256,
    validate_prepared,
)


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def setup_case(tmp_path, cards=4):
    work = tmp_path / "work"
    worker = work / "scripts/run_nonlinear_production.py"
    worker.parent.mkdir(parents=True)
    worker.write_text(
        'def validate_spec(spec, path):\n    assert spec["prepared_only"] is False\n'
    )
    config = tmp_path / "config.json"
    dump(
        config,
        {
            "lensing": {"grid": {"shape": [790, 790]}},
            "stage0": {"code_revision": {"git_hash": "old"}},
        },
    )
    position = tmp_path / "positions.json"
    dump(position, {"x": 0})
    freeze = tmp_path / "freeze.json"
    dump(freeze, {"version": 7})
    spec = {
        "case_id": "archived:test:0000",
        "scope": "archived_cases",
        "system_id": "sys0987",
        "config": str(config),
        "positions": str(position),
        "release_freeze_path": str(freeze),
        "release_freeze_sha256": digest(freeze),
        "catalog_sha256": "a" * 64,
        "hashes": {str(path): digest(path) for path in (config, position, freeze)},
    }
    spec_path = tmp_path / "spec.json"
    dump(spec_path, spec)
    output = tmp_path / f"prepared{cards}"
    kwargs = {
        "task_root": tmp_path / "future_task",
        "source_worktree": work,
        "worktree": work,
        "python": Path("/test/pinned/python"),
        "gpus": list(range(cards)),
        "wallclock_seconds": 3600,
        "worker_wall_seconds": 10000,
        "timeout_seconds": 300,
    }
    result = prepare([spec_path], output, **kwargs)
    return output / "manifest.json", spec_path, kwargs, result


def approval_fixture(manifest_path):
    manifest = read(manifest_path)
    spec = read(manifest_path.parent / manifest["jobs"][0]["prepared_spec"])
    approval = {
        "status": "APPROVED",
        "authorized_scope": "all_standard_and_brackets",
        "prepared_manifest_sha256": digest(manifest_path),
        "prepared_spec_sha256": {
            job["case_id"]: job["prepared_spec_sha256"] for job in manifest["jobs"]
        },
        "case_ids": [job["case_id"] for job in manifest["jobs"]],
        "authorized_gpu_limit": len(manifest["gpus"]),
        "gpus": manifest["gpus"],
        "task_root": manifest["task_root"],
        "wallclock_seconds": manifest["proposed_wallclock_seconds"],
        "worker_wall_seconds": manifest["cap_seconds"],
        "catalog_sha256": spec["catalog_sha256"],
        "release_freeze_sha256": spec["release_freeze_sha256"],
    }
    path = manifest_path.parent / "synthetic_test_approval.json"
    dump(path, approval)
    return path


@pytest.mark.parametrize("cards", [4, 8])
def test_prepare_and_dry_run_never_activate(tmp_path, cards):
    manifest_path, _, kwargs, result = setup_case(tmp_path, cards)
    assert result["status"] == "PREPARED_DRY_RUN_VALID"
    assert result["worker_ceiling"] == cards * 3
    assert result["deadline_created"] is False
    assert not kwargs["task_root"].exists()
    manifest = read(manifest_path)
    assert manifest["deadline"] is None
    assert manifest["approval"] is None
    assert manifest["jobs"][0]["exclusive_gpu"] is True
    assert manifest["jobs"][0]["peak_mib"] == 140000
    assert validate_prepared(manifest_path) == result


def test_prepared_spec_tamper_rejected(tmp_path):
    manifest_path, _, _, _ = setup_case(tmp_path)
    spec_path = manifest_path.parent / read(manifest_path)["jobs"][0]["prepared_spec"]
    spec_path.write_text(spec_path.read_text() + " ")
    with pytest.raises(ValueError, match="binding differs"):
        validate_prepared(manifest_path)


@pytest.mark.parametrize(
    "key,value",
    [
        ("prepared_manifest_sha256", "f" * 64),
        ("prepared_spec_sha256", {}),
        ("case_ids", ["wrong"]),
        ("authorized_gpu_limit", 8),
        ("gpus", [4, 5, 6, 7]),
        ("task_root", "/elsewhere"),
        ("wallclock_seconds", 100000),
        ("worker_wall_seconds", 100000),
        ("catalog_sha256", "f" * 64),
        ("release_freeze_sha256", "f" * 64),
        ("authorized_scope", "new_top50_standard_cases"),
        ("status", "PREPARED"),
    ],
)
def test_approval_requires_exact_batch_and_resources(tmp_path, key, value):
    manifest_path, _, kwargs, _ = setup_case(tmp_path)
    approval_path = approval_fixture(manifest_path)
    approval = read(approval_path)
    approval[key] = value
    dump(approval_path, approval)
    with pytest.raises(ValueError):
        activate(
            manifest_path,
            approval_path,
            supervisor=lambda _: pytest.fail("must not supervise"),
        )
    assert not kwargs["task_root"].exists()


def test_approved_activation_and_restart_preserve_deadline(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "hwoslaps.campaign.execution_prepare.clock_epoch", lambda: "synthetic-test-boot"
    )
    monkeypatch.setattr(
        "hwoslaps.modeling.nonlinear.profile_execution.clock_epoch",
        lambda: "synthetic-test-boot",
    )
    manifest_path, _, kwargs, _ = setup_case(tmp_path)
    approval_path = approval_fixture(manifest_path)
    calls = []
    activate(manifest_path, approval_path, supervisor=calls.append)
    root = kwargs["task_root"]
    deadline_hash = digest(root / "state/deadline.json")
    active = read(calls[0])
    spec = read(active["jobs"][0]["spec"])
    assert spec["prepared_only"] is False
    assert spec["approval_receipt_sha256"] == digest(approval_path)
    assert spec["case_output"] == str(Path(spec["output"]) / "case")
    activate(manifest_path, approval_path, supervisor=calls.append)
    assert len(calls) == 2
    assert digest(root / "state/deadline.json") == deadline_hash
    deadline = read(root / "state/deadline.json")
    deadline["hard_stop_monotonic"] += 100
    dump(root / "state/deadline.json", deadline)
    with pytest.raises(ValueError, match="deadline changed"):
        activate(manifest_path, approval_path, supervisor=calls.append)
    assert len(calls) == 2


def test_partial_activation_does_not_mint_new_time(tmp_path):
    manifest_path, _, kwargs, _ = setup_case(tmp_path)
    approval_path = approval_fixture(manifest_path)
    (kwargs["task_root"] / "activated").mkdir(parents=True)
    with pytest.raises(ValueError, match="partial or preexisting"):
        activate(
            manifest_path,
            approval_path,
            supervisor=lambda _: pytest.fail("must not supervise"),
        )
    assert not (kwargs["task_root"] / "state/deadline.json").exists()


def test_source_change_blocks_activation_before_state(tmp_path):
    manifest_path, _, kwargs, _ = setup_case(tmp_path)
    approval_path = approval_fixture(manifest_path)
    (kwargs["worktree"] / "scripts/run_nonlinear_production.py").write_text(
        "raise Exception('wrong source')"
    )
    with pytest.raises(ValueError, match="source hash mismatch"):
        activate(
            manifest_path,
            approval_path,
            supervisor=lambda _: pytest.fail("must not supervise"),
        )
    assert not kwargs["task_root"].exists()


def test_input_change_blocks_activation_before_state(tmp_path):
    manifest_path, spec_path, kwargs, _ = setup_case(tmp_path)
    approval_path = approval_fixture(manifest_path)
    Path(read(spec_path)["positions"]).write_text("{}")
    with pytest.raises(ValueError, match="source hash mismatch"):
        activate(
            manifest_path,
            approval_path,
            supervisor=lambda _: pytest.fail("must not supervise"),
        )
    assert not kwargs["task_root"].exists()


def test_memory_receipt_requires_actual_config_and_measurements(tmp_path):
    _, spec_path, _, _ = setup_case(tmp_path)
    spec = read(spec_path)
    evidence = tmp_path / "measurement.json"
    dump(evidence, {"physical_card_memory": 150512})
    receipt_path = tmp_path / "memory.json"
    receipt = {
        "status": "MEASURED",
        "case_id": spec["case_id"],
        "config_sha256": spec["hashes"][spec["config"]],
        "registry_id": "stage3_b200_790_v1",
        "image_shape": [790, 790],
        "kernel_shape": [51, 51],
        "batch_size": 32,
        "precision": "float64",
        "measurement_artifacts": {str(evidence): digest(evidence)},
    }
    dump(receipt_path, receipt)
    spec["memory_profile_receipt"] = str(receipt_path)
    spec["hashes"].update(
        {str(receipt_path): digest(receipt_path), str(evidence): digest(evidence)}
    )
    assert resource_job(spec)["peak_mib"] == 51200
    receipt["kernel_shape"] = [999, 999]
    dump(receipt_path, receipt)
    spec["hashes"][str(receipt_path)] = digest(receipt_path)
    with pytest.raises(ValueError, match="kernel_shape"):
        resource_job(spec)


def test_memory_reuses_identical_physics_with_only_revision_restamped(tmp_path):
    _, spec_path, _, _ = setup_case(tmp_path)
    spec = read(spec_path)
    measured = tmp_path / "old_config.json"
    measured.write_text(Path(spec["config"]).read_text())
    config = read(spec["config"])
    config["stage0"]["code_revision"]["git_hash"] = "fresh"
    dump(Path(spec["config"]), config)
    spec["hashes"][spec["config"]] = digest(spec["config"])
    evidence = tmp_path / "measurement.json"
    dump(evidence, {"physical_card_memory": 150512})
    receipt = {
        "status": "MEASURED",
        "case_id": spec["case_id"],
        "config_sha256": digest(measured),
        "measured_config_path": str(measured),
        "config_scientific_sha256": scientific_config_sha256(measured),
        "registry_id": "stage3_b200_790_v1",
        "image_shape": [790, 790],
        "kernel_shape": [51, 51],
        "batch_size": 32,
        "precision": "float64",
        "measurement_artifacts": {str(evidence): digest(evidence)},
    }
    receipt_path = tmp_path / "memory.json"
    dump(receipt_path, receipt)
    spec["memory_profile_receipt"] = str(receipt_path)
    spec["hashes"].update(
        {
            str(receipt_path): digest(receipt_path),
            str(evidence): digest(evidence),
            str(measured): digest(measured),
        }
    )
    assert resource_job(spec)["peak_mib"] == 51200
    config["lensing"]["grid"]["shape"] = [900, 900]
    dump(Path(spec["config"]), config)
    with pytest.raises(ValueError, match="beyond code_revision"):
        resource_job(spec)


def test_synthetic_examples_reject_even_exact_approval(tmp_path):
    manifest_path, _, _, _ = setup_case(tmp_path)
    manifest = read(manifest_path)
    manifest["synthetic_example"] = True
    dump(manifest_path, manifest)
    approval_path = approval_fixture(manifest_path)
    with pytest.raises(ValueError, match="synthetic examples"):
        activate(
            manifest_path,
            approval_path,
            supervisor=lambda _: pytest.fail("must not supervise"),
        )


@pytest.mark.parametrize("case_kind", ["standard", "bracket"])
def test_preparation_activates_through_real_worker_validator_only(
    tmp_path, monkeypatch, case_kind
):
    import runpy

    fixture = runpy.run_path(str(Path(__file__).with_name("test_production_cli.py")))[
        "valid_spec"
    ]
    data_root = tmp_path / "real_contract"
    data_root.mkdir()
    _, spec = fixture(data_root, case_kind=case_kind)
    spec_path = data_root / "source_spec.json"
    dump(spec_path, spec)
    work = tmp_path / "code"
    (work / "scripts").mkdir(parents=True)
    worker = Path(__file__).parents[1] / "scripts/run_nonlinear_production.py"
    (work / "scripts/run_nonlinear_production.py").write_bytes(worker.read_bytes())
    output = tmp_path / "review"
    prepare(
        [spec_path],
        output,
        task_root=tmp_path / "execution",
        source_worktree=work,
        worktree=work,
        python=Path("/test/pinned/python"),
        gpus=[0, 1, 2, 3],
        wallclock_seconds=3600,
        worker_wall_seconds=10000,
        timeout_seconds=300,
    )
    manifest_path = output / "manifest.json"
    approval_path = approval_fixture(manifest_path)
    monkeypatch.setattr(
        "hwoslaps.campaign.execution_prepare.clock_epoch", lambda: "synthetic-test-boot"
    )
    monkeypatch.setattr(
        "hwoslaps.modeling.nonlinear.profile_execution.clock_epoch",
        lambda: "synthetic-test-boot",
    )
    calls = []
    activate(manifest_path, approval_path, supervisor=calls.append)
    assert len(calls) == 1
    active_spec = read(read(calls[0])["jobs"][0]["spec"])
    assert active_spec["case_kind"] == case_kind
    assert active_spec["compute_bracket_fisher_q"] is (case_kind == "bracket")
    assert active_spec["case_identity_payload"]["compute_bracket_fisher_q"] is (
        case_kind == "bracket"
    )
    assert active_spec["approval_receipt_sha256"] == digest(approval_path)

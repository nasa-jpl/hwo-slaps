"""Tests for the fail-closed S1-lite campaign layer."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import hwoslaps.campaign.s1_lite as s1
from hwoslaps.provenance import config_hash


PINNED_UUID = "11111111-2222-3333-4444-555555555555"

_STUB_RUNNER_SOURCE = '''"""Stub campaign runner used only by the S1-lite tests."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "__SRC_ROOT__")

import numpy as np
import yaml

from hwoslaps.provenance import config_hash


def main():
    config_path = Path(sys.argv[1])
    control = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    job_id = config["run_name"]
    if job_id in control.get("exit_nonzero", []):
        print("stub runner injected failure for " + job_id)
        return 3
    if job_id in control.get("write_nothing", []):
        print("stub runner wrote nothing for " + job_id)
        return 0
    identity = {
        "config_hash": np.asarray(config_hash(config)),
        "campaign_uuid": np.asarray(os.environ["HWOSLAPS_CAMPAIGN_UUID"]),
    }
    for member in control.get("omit_identity", {}).get(job_id, []):
        del identity[member]
    output_dir = Path(config["plotting"]["output_dir"]) / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = output_dir / "result.npz"
    payload = {"metrics": {"num_detectable": len(job_id)}}
    np.savez(
        artifact,
        detectable_area_arcsec2=np.asarray(0.25 * len(job_id), dtype=float),
        kernel_rows=np.asarray(config["psf"]["kernel"]["shape_native"][0]),
        payload_json=np.asarray(json.dumps(payload)),
        **identity,
    )
    if job_id in control.get("second_artifact", []):
        np.savez(output_dir / "metrics.npz", **identity)
    if job_id in control.get("extra_npz", []):
        extra_dir = output_dir / "extra"
        extra_dir.mkdir(parents=True, exist_ok=True)
        np.savez(extra_dir / "undeclared.npz", **identity)
    if job_id in control.get("symlink_artifact", []):
        target = output_dir.parent / (job_id + "_target.npz")
        artifact.rename(target)
        artifact.symlink_to(target)
    if job_id in control.get("truncate", []):
        data = artifact.read_bytes()
        artifact.write_bytes(data[: len(data) // 2])
    print("stub runner wrote " + str(artifact))
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def _default_jobs():
    """Build the three-job set used by the executor tests."""
    return [
        {
            "job_id": "job_a",
            "scene": "scene_a",
            "overrides": {"psf": {"kernel": {"shape_native": [101, 101]}}},
        },
        {
            "job_id": "job_b",
            "scene": "scene_a",
            "overrides": {
                "psf": {"kernel": {"shape_native": [101, 101]}},
                "lensing": {"subhalo": {"mass": 3.0e7}},
            },
        },
        {
            "job_id": "job_c",
            "scene": "scene_b",
            "overrides": {"psf": {"kernel": {"shape_native": [201, 201]}}},
        },
    ]


def _scene_config(label):
    """Build a minimal scene configuration with merge-visible structure."""
    return {
        "run_name": label,
        "global_seed": 11,
        "plotting": {"enabled": True, "output_dir": "outputs"},
        "tags": ["scene", label],
        "lensing": {
            "cosmology": "Planck15",
            "source_galaxy": {
                "redshift": 0.6,
                "light": {
                    "type": "Exponential",
                    "intensity": 2.0,
                    "effective_radius": 0.11,
                },
            },
            "subhalo": {"enabled": True, "model": "NFW", "mass": 1.0e7},
        },
        "psf": {"kernel": {"shape_native": [51, 51]}},
        "observation": {
            "exposure_time": 900.0,
            "throughput": 1.0,
            "detector": {"read_noise": 0.2, "sky_background": 1.0},
        },
    }


def _write_yaml(path, payload):
    """Write one YAML mapping and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, sort_keys=False)
    return path


def _control_path(tmp_path):
    """Return the stub runner control file shared by every job."""
    return tmp_path / "stub_control.json"


def _stub_command(tmp_path, control):
    """Write the stub runner and return its runner_command template."""
    stub_path = tmp_path / "stub_runner.py"
    stub_path.write_text(
        _STUB_RUNNER_SOURCE.replace("__SRC_ROOT__", str(SRC_ROOT)),
        encoding="utf-8",
    )
    control_path = _control_path(tmp_path)
    control_path.write_text(json.dumps(control), encoding="utf-8")
    return [sys.executable, str(stub_path), "{config}", str(control_path)]


def _campaign_manifest(
    tmp_path,
    root,
    control=None,
    jobs=None,
    manifest_name="manifest.yaml",
    **changes,
):
    """Write scene configs and one campaign manifest, returning its path."""
    scene_paths = {
        label: _write_yaml(
            tmp_path / "scenes" / f"{label}.yaml",
            _scene_config(label),
        )
        for label in ("scene_a", "scene_b")
    }
    resolved_jobs = copy.deepcopy(_default_jobs() if jobs is None else jobs)
    campaign = {
        "name": "s1_lite_test",
        "output_root": str(root),
        "runner_command": _stub_command(tmp_path, control or {}),
        "base_scene_configs": {
            label: str(path) for label, path in scene_paths.items()
        },
        "jobs": resolved_jobs,
        "expected_job_count": len(resolved_jobs),
        "expected_artifacts": ["result.npz"],
        "seed_policy": {"root_seed": 11, "per_job": "job_id_digest"},
        "campaign_uuid": PINNED_UUID,
    }
    campaign.update(changes)
    return _write_yaml(tmp_path / manifest_name, {"campaign": campaign})


def _staged_config(root, job_id):
    """Load one staged campaign configuration."""
    path = root / "configs" / f"{job_id}.yaml"
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _frozen_campaign(root):
    """Load the frozen manifest campaign block."""
    path = root / "manifest.frozen.yaml"
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)["campaign"]


def _read_json(path):
    """Load one JSON document."""
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _sha256(path):
    """Return the full SHA-256 digest of one file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_gate1_injected_worker_failure_is_fail_closed(tmp_path):
    """A nonzero worker exit leaves no DONE and blocks the harvest."""
    root = (tmp_path / "campaign").resolve()
    manifest = _campaign_manifest(
        tmp_path,
        root,
        control={"exit_nonzero": ["job_b"]},
    )
    s1.freeze_campaign(manifest)

    with pytest.raises(s1.CampaignError) as run_error:
        s1.run_campaign(root, 1, timeout_s=120.0)
    message = str(run_error.value)
    assert "job_b" in message
    assert "exit_nonzero" in message

    assert not (root / "sentinels" / "job_b.DONE").exists()
    assert not (root / "sentinels" / "CAMPAIGN_COMPLETE").exists()
    assert (root / "logs" / "job_b.log").is_file()

    with pytest.raises(s1.CampaignError) as harvest_error:
        s1.harvest_campaign(root)
    assert "CAMPAIGN_COMPLETE" in str(harvest_error.value)
    assert not (root / "harvest" / "harvest.json").exists()


def test_gate1b_corrupt_artifact_fails_validation(tmp_path):
    """A zero exit with an unloadable artifact still fails the job."""
    root = (tmp_path / "campaign").resolve()
    manifest = _campaign_manifest(
        tmp_path,
        root,
        control={"truncate": ["job_b"]},
    )
    s1.freeze_campaign(manifest)

    with pytest.raises(s1.CampaignError) as run_error:
        s1.run_campaign(root, 1, timeout_s=120.0)
    message = str(run_error.value)
    assert "job_b" in message
    assert "validation_failed" in message
    assert "does not load" in message

    assert (root / "outputs" / "job_b" / "result.npz").is_file()
    assert not (root / "sentinels" / "job_b.DONE").exists()
    assert not (root / "sentinels" / "CAMPAIGN_COMPLETE").exists()

    with pytest.raises(s1.CampaignError):
        s1.harvest_campaign(root)


def test_gate2_stale_artifact_rejection_and_allowed_rerun(tmp_path):
    """A tampered artifact blocks harvest and resume until allow_rerun."""
    root = (tmp_path / "campaign").resolve()
    manifest = _campaign_manifest(tmp_path, root)
    s1.freeze_campaign(manifest)
    s1.run_campaign(root, 2, timeout_s=120.0)

    artifact = root / "outputs" / "job_b" / "result.npz"
    tampered = bytearray(artifact.read_bytes())
    tampered[len(tampered) // 2] ^= 0xFF
    artifact.write_bytes(bytes(tampered))

    with pytest.raises(s1.CampaignError) as harvest_error:
        s1.harvest_campaign(root)
    assert str(artifact) in str(harvest_error.value)
    assert not (root / "harvest" / "harvest.json").exists()

    with pytest.raises(s1.CampaignError) as resume_error:
        s1.run_campaign(root, 1, timeout_s=120.0)
    resume_message = str(resume_error.value)
    assert str(artifact) in resume_message
    assert "allow_rerun" in resume_message

    summary = s1.run_campaign(root, 1, timeout_s=120.0, allow_rerun=True)
    assert summary["executed_job_ids"] == ["job_b"]
    assert sorted(summary["skipped_job_ids"]) == ["job_a", "job_c"]
    assert (root / "sentinels" / "CAMPAIGN_COMPLETE").is_file()

    harvest_path = s1.harvest_campaign(root)
    assert harvest_path.is_file()


def test_gate3_artifact_without_identity_fails_validation(tmp_path):
    """An artifact carrying neither identity member is refused."""
    root = (tmp_path / "campaign").resolve()
    manifest = _campaign_manifest(
        tmp_path,
        root,
        control={"omit_identity": {"job_a": ["config_hash", "campaign_uuid"]}},
        jobs=_default_jobs()[:1],
    )
    s1.freeze_campaign(manifest)

    with pytest.raises(s1.CampaignError) as run_error:
        s1.run_campaign(root, 1, timeout_s=120.0)
    message = str(run_error.value)
    assert "job_a" in message
    assert "validation_failed" in message
    assert "config_hash" in message

    assert (root / "outputs" / "job_a" / "result.npz").is_file()
    assert not (root / "sentinels" / "job_a.DONE").exists()
    assert not (root / "sentinels" / "CAMPAIGN_COMPLETE").exists()

    with pytest.raises(s1.CampaignError) as harvest_error:
        s1.harvest_campaign(root)
    assert "CAMPAIGN_COMPLETE" in str(harvest_error.value)


@pytest.mark.parametrize("member", ["config_hash", "campaign_uuid"])
def test_gate3_artifact_missing_one_identity_member(tmp_path, member):
    """Dropping either identity member alone still fails the job."""
    root = (tmp_path / "campaign").resolve()
    manifest = _campaign_manifest(
        tmp_path,
        root,
        control={"omit_identity": {"job_a": [member]}},
        jobs=_default_jobs()[:1],
    )
    s1.freeze_campaign(manifest)

    with pytest.raises(s1.CampaignError) as run_error:
        s1.run_campaign(root, 1, timeout_s=120.0)
    message = str(run_error.value)
    assert "validation_failed" in message
    assert f"required '{member}'" in message

    assert not (root / "sentinels" / "job_a.DONE").exists()
    assert not (root / "sentinels" / "CAMPAIGN_COMPLETE").exists()


def test_gate3_successful_no_op_cannot_certify_a_prior_artifact(tmp_path):
    """A zero-exit run that writes nothing cannot inherit old outputs."""
    root = (tmp_path / "campaign").resolve()
    manifest = _campaign_manifest(tmp_path, root, jobs=_default_jobs()[:1])
    s1.freeze_campaign(manifest)
    s1.run_campaign(root, 1, timeout_s=120.0)

    artifact = root / "outputs" / "job_a" / "result.npz"
    assert artifact.is_file()
    (root / "sentinels" / "job_a.DONE").unlink()
    (root / "sentinels" / "CAMPAIGN_COMPLETE").unlink()
    _control_path(tmp_path).write_text(
        json.dumps({"write_nothing": ["job_a"]}),
        encoding="utf-8",
    )

    with pytest.raises(s1.CampaignError) as run_error:
        s1.run_campaign(root, 1, timeout_s=120.0)
    message = str(run_error.value)
    assert "job_a" in message
    assert "validation_failed" in message
    assert "produced no output directory" in message

    assert not artifact.exists()
    assert not (root / "outputs" / "job_a").exists()
    assert not (root / "sentinels" / "job_a.DONE").exists()
    assert not (root / "sentinels" / "CAMPAIGN_COMPLETE").exists()

    with pytest.raises(s1.CampaignError) as harvest_error:
        s1.harvest_campaign(root)
    assert "CAMPAIGN_COMPLETE" in str(harvest_error.value)


def test_gate3_undeclared_artifact_fails_job_validation(tmp_path):
    """An .npz the job never declared fails that job outright."""
    root = (tmp_path / "campaign").resolve()
    manifest = _campaign_manifest(
        tmp_path,
        root,
        control={"extra_npz": ["job_a"]},
        jobs=_default_jobs()[:1],
    )
    s1.freeze_campaign(manifest)

    with pytest.raises(s1.CampaignError) as run_error:
        s1.run_campaign(root, 1, timeout_s=120.0)
    message = str(run_error.value)
    assert "validation_failed" in message
    assert "undeclared" in message
    assert "extra/undeclared.npz" in message

    assert not (root / "sentinels" / "job_a.DONE").exists()
    assert not (root / "sentinels" / "CAMPAIGN_COMPLETE").exists()


def test_gate3_undeclared_artifact_added_later_blocks_harvest(tmp_path):
    """An .npz dropped in after validation fails reconciliation."""
    root = (tmp_path / "campaign").resolve()
    manifest = _campaign_manifest(tmp_path, root, jobs=_default_jobs()[:1])
    s1.freeze_campaign(manifest)
    s1.run_campaign(root, 1, timeout_s=120.0)

    stray = root / "outputs" / "job_a" / "stray.npz"
    stray.write_bytes((root / "outputs" / "job_a" / "result.npz").read_bytes())

    with pytest.raises(s1.CampaignError) as harvest_error:
        s1.harvest_campaign(root)
    harvest_message = str(harvest_error.value)
    assert "undeclared" in harvest_message
    assert "stray.npz" in harvest_message
    assert not (root / "harvest" / "harvest.json").exists()

    with pytest.raises(s1.CampaignError) as resume_error:
        s1.run_campaign(root, 1, timeout_s=120.0)
    assert "allow_rerun" in str(resume_error.value)


def test_gate3_symlinked_artifact_fails_validation(tmp_path):
    """A declared artifact that is a symlink is refused outright."""
    root = (tmp_path / "campaign").resolve()
    manifest = _campaign_manifest(
        tmp_path,
        root,
        control={"symlink_artifact": ["job_a"]},
        jobs=_default_jobs()[:1],
    )
    s1.freeze_campaign(manifest)

    with pytest.raises(s1.CampaignError) as run_error:
        s1.run_campaign(root, 1, timeout_s=120.0)
    message = str(run_error.value)
    assert "validation_failed" in message
    assert "symlink" in message

    assert not (root / "sentinels" / "job_a.DONE").exists()
    assert not (root / "sentinels" / "CAMPAIGN_COMPLETE").exists()


def test_gate3_symlinked_artifact_blocks_harvest(tmp_path):
    """Swapping a certified artifact for a symlink fails reconciliation."""
    root = (tmp_path / "campaign").resolve()
    s1.freeze_campaign(_campaign_manifest(tmp_path, root, jobs=_default_jobs()[:1]))
    s1.run_campaign(root, 1, timeout_s=120.0)

    artifact = root / "outputs" / "job_a" / "result.npz"
    target = root / "preserved_result.npz"
    artifact.rename(target)
    artifact.symlink_to(target)

    with pytest.raises(s1.CampaignError) as harvest_error:
        s1.harvest_campaign(root)
    assert "symlink" in str(harvest_error.value)
    assert not (root / "harvest" / "harvest.json").exists()

    with pytest.raises(s1.CampaignError) as resume_error:
        s1.run_campaign(root, 1, timeout_s=120.0)
    assert "allow_rerun" in str(resume_error.value)


def test_gate3_symlinked_output_dir_blocks_harvest(tmp_path):
    """A job output directory replaced by a symlink fails reconciliation."""
    root = (tmp_path / "campaign").resolve()
    s1.freeze_campaign(_campaign_manifest(tmp_path, root, jobs=_default_jobs()[:1]))
    s1.run_campaign(root, 1, timeout_s=120.0)

    output_dir = root / "outputs" / "job_a"
    relocated = root / "job_a_relocated"
    output_dir.rename(relocated)
    output_dir.symlink_to(relocated, target_is_directory=True)

    with pytest.raises(s1.CampaignError) as harvest_error:
        s1.harvest_campaign(root)
    assert "symlink" in str(harvest_error.value)


def test_gate3_partial_artifact_set_fails_the_job(tmp_path):
    """A zero-exit job that writes part of its declared set fails."""
    root = (tmp_path / "campaign").resolve()
    manifest = _campaign_manifest(
        tmp_path,
        root,
        jobs=_default_jobs()[:1],
        expected_artifacts=["metrics.npz", "result.npz"],
    )
    s1.freeze_campaign(manifest)

    with pytest.raises(s1.CampaignError) as run_error:
        s1.run_campaign(root, 1, timeout_s=120.0)
    message = str(run_error.value)
    assert "validation_failed" in message
    assert "metrics.npz" in message

    assert (root / "outputs" / "job_a" / "result.npz").is_file()
    assert not (root / "sentinels" / "job_a.DONE").exists()
    assert not (root / "sentinels" / "CAMPAIGN_COMPLETE").exists()


def test_gate3_declared_artifact_set_is_validated_and_recorded(tmp_path):
    """Every declared artifact is identity-checked and recorded."""
    root = (tmp_path / "campaign").resolve()
    manifest = _campaign_manifest(
        tmp_path,
        root,
        control={"second_artifact": ["job_a"]},
        jobs=_default_jobs()[:1],
        expected_artifacts=["metrics.npz", "result.npz"],
    )
    s1.freeze_campaign(manifest)
    s1.run_campaign(root, 1, timeout_s=120.0)

    sentinel = _read_json(root / "sentinels" / "job_a.DONE")
    assert [entry["path"] for entry in sentinel["artifacts"]] == [
        "outputs/job_a/metrics.npz",
        "outputs/job_a/result.npz",
    ]
    for entry in sentinel["artifacts"]:
        assert entry["sha256"] == _sha256(root / entry["path"])
    assert (root / "sentinels" / "CAMPAIGN_COMPLETE").is_file()
    assert s1.harvest_campaign(root).is_file()


def test_gate3_run_refuses_a_campaign_without_declared_artifacts(tmp_path):
    """A campaign may stage without a manifest but never run one."""
    root = (tmp_path / "campaign").resolve()
    manifest = _campaign_manifest(
        tmp_path,
        root,
        jobs=_default_jobs()[:1],
        expected_artifacts=None,
    )
    s1.freeze_campaign(manifest)

    with pytest.raises(s1.CampaignError) as run_error:
        s1.run_campaign(root, 1, timeout_s=120.0)
    message = str(run_error.value)
    assert "job_a" in message
    assert "expected_artifacts" in message
    assert not (root / "outputs" / "job_a").exists()


def test_happy_path_sentinels_completion_and_harvest(tmp_path):
    """Three clean jobs produce sentinels, completion, and a harvest."""
    root = (tmp_path / "campaign").resolve()
    manifest = _campaign_manifest(
        tmp_path,
        root,
        collect={
            "scalars": [
                "detectable_area_arcsec2",
                "payload_json.metrics.num_detectable",
            ],
            "artifact": "result.npz",
        },
    )
    s1.freeze_campaign(manifest)
    summary = s1.run_campaign(root, 3, timeout_s=120.0)

    assert summary["campaign_uuid"] == PINNED_UUID
    assert summary["executed_job_ids"] == ["job_a", "job_b", "job_c"]

    frozen = _frozen_campaign(root)
    for job in frozen["jobs"]:
        sentinel = _read_json(root / "sentinels" / f"{job['job_id']}.DONE")
        assert set(sentinel) == {
            "job_id",
            "campaign_uuid",
            "config_sha256",
            "artifacts",
            "wall_s",
            "validated_utc",
        }
        assert sentinel["job_id"] == job["job_id"]
        assert sentinel["campaign_uuid"] == PINNED_UUID
        assert sentinel["config_sha256"] == job["staged_config_sha256"]
        assert sentinel["wall_s"] > 0.0
        assert sentinel["validated_utc"].endswith("Z")
        artifact = root / "outputs" / job["job_id"] / "result.npz"
        assert sentinel["artifacts"] == [
            {
                "path": f"outputs/{job['job_id']}/result.npz",
                "sha256": _sha256(artifact),
            }
        ]

    complete = _read_json(root / "sentinels" / "CAMPAIGN_COMPLETE")
    assert complete["campaign_uuid"] == PINNED_UUID
    assert complete["expected_job_count"] == 3
    assert complete["actual_job_count"] == 3
    assert complete["job_ids"] == ["job_a", "job_b", "job_c"]

    harvest_path = s1.harvest_campaign(root)
    harvest = _read_json(harvest_path)
    assert harvest["campaign_uuid"] == PINNED_UUID
    assert harvest["manifest_frozen_sha256"] == _sha256(
        root / "manifest.frozen.yaml"
    )
    assert "revision_provenance" in harvest
    assert harvest["reconciliation"] == {
        "expected_job_count": 3,
        "found_job_count": 3,
        "expected_job_ids": ["job_a", "job_b", "job_c"],
        "found_job_ids": ["job_a", "job_b", "job_c"],
        "missing_job_ids": [],
        "unexpected_job_ids": [],
    }
    assert [record["job_id"] for record in harvest["jobs"]] == [
        "job_a",
        "job_b",
        "job_c",
    ]
    assert harvest["jobs"][0]["scene"] == "scene_a"
    assert harvest["jobs"][2]["scene"] == "scene_b"
    assert harvest["jobs"][0]["scalars"] == pytest.approx({
        "detectable_area_arcsec2": 0.25*len("job_a"),
        "payload_json.metrics.num_detectable": float(len("job_a")),
    })

    with np.load(root / "harvest" / "harvest.npz", allow_pickle=False) as npz:
        assert list(npz["job_ids"]) == ["job_a", "job_b", "job_c"]
        assert npz["detectable_area_arcsec2"] == pytest.approx(
            [1.25, 1.25, 1.25]
        )


def test_manifest_rejects_expected_job_count_mismatch(tmp_path):
    """An expected_job_count that disagrees with jobs is refused."""
    manifest = _campaign_manifest(
        tmp_path,
        (tmp_path / "campaign").resolve(),
        expected_job_count=7,
    )
    with pytest.raises(s1.CampaignError) as error:
        s1.freeze_campaign(manifest)
    assert "campaign.expected_job_count" in str(error.value)


def test_manifest_rejects_duplicate_job_id(tmp_path):
    """A repeated job_id is refused with its manifest path."""
    jobs = _default_jobs()
    jobs[1]["job_id"] = "job_a"
    manifest = _campaign_manifest(tmp_path, (tmp_path / "campaign").resolve(), jobs=jobs)
    with pytest.raises(s1.CampaignError) as error:
        s1.freeze_campaign(manifest)
    message = str(error.value)
    assert "campaign.jobs[1].job_id" in message
    assert "job_a" in message


def test_manifest_rejects_missing_kernel_shape_override(tmp_path):
    """A job without an explicit kernel shape override is refused."""
    jobs = _default_jobs()
    jobs[0]["overrides"] = {"lensing": {"subhalo": {"mass": 2.0e7}}}
    manifest = _campaign_manifest(tmp_path, (tmp_path / "campaign").resolve(), jobs=jobs)
    with pytest.raises(s1.CampaignError) as error:
        s1.freeze_campaign(manifest)
    message = str(error.value)
    assert "campaign.jobs[0].overrides" in message
    assert "psf.kernel.shape_native" in message


def test_manifest_rejects_missing_config_placeholder(tmp_path):
    """A runner_command without the config placeholder is refused."""
    manifest = _campaign_manifest(
        tmp_path,
        (tmp_path / "campaign").resolve(),
        runner_command=[sys.executable, "runner.py", "--quiet"],
    )
    with pytest.raises(s1.CampaignError) as error:
        s1.freeze_campaign(manifest)
    message = str(error.value)
    assert "campaign.runner_command" in message
    assert "{config}" in message


def test_manifest_rejects_reserved_collect_key(tmp_path):
    """A collect key that would shadow a harvest identity member is refused."""
    manifest = _campaign_manifest(
        tmp_path,
        (tmp_path / "campaign").resolve(),
        collect={"scalars": ["job_ids"], "artifact": "result.npz"},
    )
    with pytest.raises(s1.CampaignError) as error:
        s1.freeze_campaign(manifest)
    message = str(error.value)
    assert "campaign.collect.scalars[0]" in message
    assert "reserved" in message


def test_manifest_rejects_an_escaping_artifact_path(tmp_path):
    """An expected artifact outside the job output directory is refused."""
    manifest = _campaign_manifest(
        tmp_path,
        (tmp_path / "campaign").resolve(),
        expected_artifacts=["../result.npz"],
    )
    with pytest.raises(s1.CampaignError) as error:
        s1.freeze_campaign(manifest)
    message = str(error.value)
    assert "campaign.expected_artifacts[0]" in message
    assert "relative .npz path" in message


def test_manifest_rejects_non_empty_output_root_without_resume(tmp_path):
    """A populated output root without a frozen manifest is refused."""
    root = (tmp_path / "campaign").resolve()
    root.mkdir(parents=True)
    (root / "stray.txt").write_text("prior run", encoding="utf-8")
    manifest = _campaign_manifest(tmp_path, root)
    with pytest.raises(s1.CampaignError) as error:
        s1.freeze_campaign(manifest)
    message = str(error.value)
    assert str(root) in message
    assert "manifest.frozen.yaml" in message


def test_frozen_manifest_is_immutable(tmp_path):
    """Editing the frozen manifest hard-errors on the next freeze or run."""
    root = (tmp_path / "campaign").resolve()
    manifest = _campaign_manifest(tmp_path, root)
    frozen_path = s1.freeze_campaign(manifest)

    frozen = _frozen_campaign(root)
    frozen["jobs"][0]["config_hash"] = "0"*16
    frozen_path.write_text(
        yaml.safe_dump({"campaign": frozen}, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(s1.CampaignError) as freeze_error:
        s1.freeze_campaign(manifest)
    freeze_message = str(freeze_error.value)
    assert str(frozen_path) in freeze_message
    assert "immutable" in freeze_message

    frozen["jobs"][0]["staged_config_sha256"] = "0"*64
    frozen_path.write_text(
        yaml.safe_dump({"campaign": frozen}, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(s1.CampaignError) as run_error:
        s1.run_campaign(root, 1, timeout_s=120.0)
    assert "freeze-time digest" in str(run_error.value)


@pytest.mark.parametrize("field", ["name", "seed_policy", "runner_command"])
def test_frozen_manifest_benign_field_tamper_refuses_run(tmp_path, field):
    """Any frozen-manifest edit fails the digest check, however benign."""
    root = (tmp_path / "campaign").resolve()
    frozen_path = s1.freeze_campaign(_campaign_manifest(tmp_path, root))

    frozen = _frozen_campaign(root)
    if field == "name":
        frozen["name"] = "tampered_name"
    elif field == "seed_policy":
        frozen["seed_policy"]["root_seed"] = 999
    else:
        frozen["runner_command"] = frozen["runner_command"] + ["--extra"]
    frozen_path.write_text(
        yaml.safe_dump({"campaign": frozen}, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(s1.CampaignError) as run_error:
        s1.run_campaign(root, 1, timeout_s=120.0)
    assert "freeze-time digest" in str(run_error.value)
    assert not (root / "sentinels" / "CAMPAIGN_COMPLETE").exists()


def test_frozen_manifest_tamper_after_run_refuses_harvest(tmp_path):
    """A post-run frozen-manifest edit blocks the harvest gate."""
    root = (tmp_path / "campaign").resolve()
    frozen_path = s1.freeze_campaign(_campaign_manifest(tmp_path, root))
    s1.run_campaign(root, 1, timeout_s=120.0)

    original = frozen_path.read_bytes()
    frozen = _frozen_campaign(root)
    frozen["seed_policy"]["root_seed"] = 999
    frozen_path.write_text(
        yaml.safe_dump({"campaign": frozen}, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(s1.CampaignError) as harvest_error:
        s1.harvest_campaign(root)
    assert "freeze-time digest" in str(harvest_error.value)

    frozen_path.write_bytes(original)
    assert s1.harvest_campaign(root).is_file()


def test_frozen_digest_missing_or_tampered_refuses_run(tmp_path):
    """The digest sidecar is required and must match the frozen bytes."""
    root = (tmp_path / "campaign").resolve()
    s1.freeze_campaign(_campaign_manifest(tmp_path, root))
    digest_path = root / "manifest.frozen.sha256"

    original = digest_path.read_bytes()
    digest_path.write_text("0" * 64 + "\n", encoding="ascii")
    with pytest.raises(s1.CampaignError) as tamper_error:
        s1.run_campaign(root, 1, timeout_s=120.0)
    assert "freeze-time digest" in str(tamper_error.value)

    digest_path.unlink()
    with pytest.raises(s1.CampaignError) as missing_error:
        s1.run_campaign(root, 1, timeout_s=120.0)
    assert "manifest.frozen.sha256" in str(missing_error.value)

    digest_path.write_bytes(original)
    s1.run_campaign(root, 1, timeout_s=120.0)
    assert (root / "sentinels" / "CAMPAIGN_COMPLETE").is_file()


def test_freeze_is_deterministic_across_output_roots(tmp_path):
    """Two roots stage byte-identical configs apart from the root path."""
    root_a = (tmp_path / "campaign_a").resolve()
    root_b = (tmp_path / "campaign_b").resolve()
    manifest_a = _campaign_manifest(
        tmp_path,
        root_a,
        manifest_name="manifest_a.yaml",
    )
    manifest_b = _campaign_manifest(
        tmp_path,
        root_b,
        manifest_name="manifest_b.yaml",
    )
    s1.freeze_campaign(manifest_a)
    s1.freeze_campaign(manifest_b)

    for job_id in ("job_a", "job_b", "job_c"):
        bytes_a = (root_a / "configs" / f"{job_id}.yaml").read_bytes()
        bytes_b = (root_b / "configs" / f"{job_id}.yaml").read_bytes()
        assert bytes_a != bytes_b
        assert bytes_a.replace(str(root_a).encode("utf-8"), b"<ROOT>") == (
            bytes_b.replace(str(root_b).encode("utf-8"), b"<ROOT>")
        )
        assert _staged_config(root_a, job_id)["plotting"]["output_dir"] == str(
            root_a / "outputs"
        )

    jobs_a = {job["job_id"]: job for job in _frozen_campaign(root_a)["jobs"]}
    jobs_b = {job["job_id"]: job for job in _frozen_campaign(root_b)["jobs"]}
    for job_id in ("job_a", "job_b", "job_c"):
        assert jobs_a[job_id]["overrides_digest"] == (
            jobs_b[job_id]["overrides_digest"]
        )


def test_override_merge_semantics(tmp_path):
    """Overrides replace scalars and lists while deep-merging mappings."""
    root = (tmp_path / "campaign").resolve()
    jobs = [
        {
            "job_id": "job_merge",
            "scene": "scene_a",
            "overrides": {
                "psf": {"kernel": {"shape_native": [201, 201]}},
                "lensing": {"subhalo": {"mass": 5.0e7}},
                "observation": {"detector": {"read_noise": 3.536}},
                "tags": ["override"],
            },
        }
    ]
    manifest = _campaign_manifest(tmp_path, root, jobs=jobs)
    s1.freeze_campaign(manifest)

    staged = _staged_config(root, "job_merge")
    assert staged["lensing"]["subhalo"]["mass"] == 5.0e7
    assert staged["lensing"]["subhalo"]["model"] == "NFW"
    assert staged["lensing"]["source_galaxy"]["light"]["intensity"] == 2.0
    assert staged["observation"]["detector"]["read_noise"] == 3.536
    assert staged["observation"]["detector"]["sky_background"] == 1.0
    assert staged["observation"]["exposure_time"] == 900.0
    assert staged["psf"]["kernel"]["shape_native"] == [201, 201]
    assert staged["tags"] == ["override"]
    assert staged["run_name"] == "job_merge"
    assert staged["plotting"]["output_dir"] == str(root / "outputs")

    frozen_job = _frozen_campaign(root)["jobs"][0]
    assert frozen_job["config_hash"] == config_hash(staged)
    assert frozen_job["staged_config_sha256"] == _sha256(
        root / "configs" / "job_merge.yaml"
    )


def test_observing_reference_patches_every_job(tmp_path):
    """The reference observation block and source patches reach each job."""
    root = (tmp_path / "campaign").resolve()
    reference = _write_yaml(
        tmp_path / "observing" / "reference.yaml",
        {
            "observation": {
                "exposure_time": 1800.0,
                "detector": {"read_noise": 3.536},
            },
            "source_normalization": {
                "scene_a": {
                    "lensing": {
                        "source_galaxy": {"light": {"intensity": 0.5}}
                    }
                },
                "scene_b": {
                    "lensing": {
                        "source_galaxy": {"light": {"intensity": 0.7}}
                    }
                },
            },
        },
    )
    manifest = _campaign_manifest(
        tmp_path,
        root,
        observing_reference=str(reference),
    )
    s1.freeze_campaign(manifest)

    staged_a = _staged_config(root, "job_a")
    assert staged_a["observation"]["exposure_time"] == 1800.0
    assert staged_a["observation"]["detector"]["read_noise"] == 3.536
    assert staged_a["observation"]["detector"]["sky_background"] == 1.0
    assert staged_a["lensing"]["source_galaxy"]["light"]["intensity"] == 0.5
    assert staged_a["lensing"]["source_galaxy"]["light"]["type"] == (
        "Exponential"
    )
    staged_c = _staged_config(root, "job_c")
    assert staged_c["lensing"]["source_galaxy"]["light"]["intensity"] == 0.7

    frozen = _frozen_campaign(root)
    assert frozen["observing_reference"]["path"] == str(reference.resolve())
    assert frozen["observing_reference"]["sha256"] == _sha256(reference)


def test_observing_reference_requires_every_scene_patch(tmp_path):
    """A reference missing one scene's source patch is refused."""
    root = (tmp_path / "campaign").resolve()
    reference = _write_yaml(
        tmp_path / "observing" / "partial.yaml",
        {
            "observation": {"exposure_time": 1800.0},
            "source_normalization": {
                "scene_a": {
                    "lensing": {
                        "source_galaxy": {"light": {"intensity": 0.5}}
                    }
                }
            },
        },
    )
    manifest = _campaign_manifest(
        tmp_path,
        root,
        observing_reference=str(reference),
    )
    with pytest.raises(s1.CampaignError) as error:
        s1.freeze_campaign(manifest)
    message = str(error.value)
    assert str(reference.resolve()) in message
    assert "scene_b" in message

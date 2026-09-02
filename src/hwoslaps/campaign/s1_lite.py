"""Fail-closed S1-lite campaign manifest, executor, and harvest.

The layer turns one explicit YAML manifest into an immutable frozen
campaign, executes every declared job as an isolated subprocess, and
harvests only a campaign that reconciles exactly against the frozen
expected job set.

Three entry points cover the whole lifecycle:

``freeze_campaign``
    Validates the manifest, stages one merged configuration per job, and
    writes ``manifest.frozen.yaml``. Re-freezing the same manifest into
    an already frozen campaign byte-compares instead of rewriting, so the
    frozen manifest and every staged configuration are immutable. The
    frozen manifest is bound to a freeze-time sha256 digest that both
    ``run_campaign`` and ``harvest_campaign`` verify before trusting it.
``run_campaign``
    Executes the jobs that lack a valid ``DONE`` sentinel. Every attempt
    starts from a cleared job output directory, and a job earns a
    sentinel only after the artifacts it declared are validated, so a
    zero exit status alone never marks a job complete.
``harvest_campaign``
    Refuses unless ``CAMPAIGN_COMPLETE`` exists and every sentinel still
    re-validates against the artifacts on disk.

Layout under ``output_root``::

    manifest.frozen.yaml    resolved manifest, written once
    configs/<job_id>.yaml   staged merged configuration per job
    outputs/<job_id>/...    runner outputs
    logs/<job_id>.log       combined stdout and stderr
    sentinels/<job_id>.DONE per-job validation record
    sentinels/CAMPAIGN_COMPLETE
    harvest/harvest.json    reconciled harvest record

Manifest paths are resolved against the manifest's own directory, so a
manifest and the scene configurations it names travel together. Staged
configurations carry ``run_name`` equal to the job id and
``plotting.output_dir`` equal to ``output_root/outputs``, which is the
only campaign-root dependent content in the staged bytes.

``campaign.expected_artifacts`` declares the artifact paths every job
must write under ``outputs/<job_id>``. A campaign may be frozen without
that declaration for staging inspection, but it can neither run nor
harvest until it declares them. Every job subprocess receives
``HWOSLAPS_CAMPAIGN_UUID`` in its environment, and every declared
artifact must embed both that value and the staged configuration hash.
A staged configuration that pins provenance digests, as a Stage 0
configuration pins the source revision and the template asset, also
requires the artifact to carry those digests unchanged.
Declared artifacts must be regular files: any symlink in a job output
tree, or a declared artifact resolving outside it, fails validation
and harvest reconciliation.

Every failure raises `CampaignError`. There is no warning path, no
partial harvest, and no silent fallback.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import datetime
import json
import math
import os
from pathlib import Path, PurePosixPath
import queue
import re
import shutil
import subprocess
import threading
import time
from typing import Any, Optional
import uuid as uuid_module

import yaml

from hwoslaps.provenance import revision_provenance

from ._common import (
    _FROZEN_DIGEST_NAME,
    _FROZEN_MANIFEST_NAME,
    _OUTPUTS_DIR,
    _SCHEMA_VERSION,
    CampaignError,
    _canonical_json,
    _integer_at_least,
    _reject_unknown_keys,
    _require_list,
    _require_mapping,
    _required,
    _positive_number,
    _sha256_bytes,
    load_frozen_manifest,
    load_observing_reference,
    load_yaml_mapping,
    resolve_path,
    stage_job_config,
)
from .design_freeze import file_sha256


_IDENTIFIER_PATTERN = re.compile(r"[a-z0-9_]+")
"""Pattern required of campaign names and job ids (`re.Pattern`)."""

_COLLECT_KEY_PATTERN = re.compile(r"[A-Za-z0-9_]+(\.[A-Za-z0-9_]+)*")
"""Pattern required of dotted harvest collection keys (`re.Pattern`)."""

_RESERVED_COLLECT_KEYS = {"campaign_uuid", "job_ids"}
"""Harvest identity members a collection key may not shadow (`set`)."""

_CONFIG_PLACEHOLDER = "{config}"
"""Mandatory runner-command placeholder for the staged config path."""

_CONFIGS_DIR = "configs"
_LOGS_DIR = "logs"
_SENTINELS_DIR = "sentinels"
_HARVEST_DIR = "harvest"
_STAGING_DIR = ".staging"
_COMPLETE_SENTINEL = "CAMPAIGN_COMPLETE"

_CAMPAIGN_DIRS = (
    _CONFIGS_DIR,
    _OUTPUTS_DIR,
    _LOGS_DIR,
    _SENTINELS_DIR,
    _HARVEST_DIR,
)

_DONE_SENTINEL_MEMBERS = {
    "job_id",
    "campaign_uuid",
    "config_sha256",
    "artifacts",
    "wall_s",
    "validated_utc",
}

def _check_clean_tree(require_clean_tree: bool, allow_dirty_tree: bool) -> None:
    """Refuse production launch from a dirty source tree."""
    if not require_clean_tree or allow_dirty_tree:
        return
    revision = revision_provenance()
    if revision.get("git_dirty") is True:
        paths = ", ".join(revision.get("git_dirty_paths") or [])
        raise CampaignError(
            "refusing campaign launch from dirty source tree"
            + (f": {paths}" if paths else "")
        )


@dataclass(frozen=True, kw_only=True)
class JobResult:
    """Outcome of one campaign job execution.

    Parameters
    ----------
    job_id : `str`
        Job identifier from the frozen manifest.
    status : `str`
        One of ``"completed"``, ``"output_reset_failed"``,
        ``"exit_nonzero"``, ``"timeout"``, ``"subprocess_error"``, or
        ``"validation_failed"``.
    detail : `str`, optional
        Failure description, or `None` for a completed job.
    wall_s : `float`
        Subprocess wall-clock duration in seconds.
    """

    job_id: str
    status: str
    detail: Optional[str]
    wall_s: float


# Unversioned campaign drivers outside the repository call these private names.
_load_frozen_manifest = load_frozen_manifest
_load_yaml_mapping = load_yaml_mapping
_load_observing_reference = load_observing_reference
_stage_job_config = stage_job_config
_resolve_path = resolve_path
_file_sha256 = file_sha256


def _require_string(value: Any, path: str) -> str:
    """Require a non-empty string value."""
    if not isinstance(value, str) or not value:
        raise CampaignError(f"{path} must be a non-empty string")
    return value


def _require_identifier(value: Any, path: str) -> str:
    """Require a lowercase alphanumeric-underscore identifier."""
    text = _require_string(value, path)
    if _IDENTIFIER_PATTERN.fullmatch(text) is None:
        raise CampaignError(f"{path} must match [a-z0-9_]+")
    return text


def _require_artifact_relpath(value: Any, path: str) -> str:
    """Require a relative ``.npz`` path with no parent segments."""
    text = _require_string(value, path)
    candidate = PurePosixPath(text)
    if (
        candidate.is_absolute()
        or ".." in candidate.parts
        or not text.endswith(".npz")
    ):
        raise CampaignError(
            f"{path} must be a relative .npz path inside the job output "
            "directory"
        )
    return candidate.as_posix()


def _require_boolean(value: Any, path: str) -> bool:
    """Require a boolean value."""
    if not isinstance(value, bool):
        raise CampaignError(f"{path} must be boolean")
    return value


def _utc_now() -> str:
    """Return the current UTC time as a second-resolution ISO string."""
    now = datetime.datetime.now(datetime.timezone.utc)
    return now.isoformat(timespec="seconds").replace("+00:00", "Z")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Write bytes through a same-directory temporary file and replace."""
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def validate_campaign_manifest(manifest: dict) -> dict:
    """Validate and canonicalize an S1-lite campaign manifest.

    Parameters
    ----------
    manifest : `dict`
        Mapping whose sole top-level key is ``campaign``.

    Returns
    -------
    normalized : `dict`
        Deep normalized copy with every optional block made explicit.

    Raises
    ------
    CampaignError
        Raised for a missing, unknown, mistyped, duplicated, or
        inconsistent value. Messages carry the offending manifest path.
    """
    root = _require_mapping(manifest, "manifest")
    _reject_unknown_keys(root, {"campaign"}, "manifest")
    campaign = _require_mapping(
        _required(root, "campaign", "manifest"),
        "campaign",
    )
    _reject_unknown_keys(
        campaign,
        {
            "name",
            "output_root",
            "runner_command",
            "base_scene_configs",
            "observing_reference",
            "jobs",
            "expected_job_count",
            "expected_artifacts",
            "seed_policy",
            "campaign_uuid",
            "collect",
        },
        "campaign",
    )

    name = _require_identifier(
        _required(campaign, "name", "campaign"),
        "campaign.name",
    )
    output_root = _require_string(
        _required(campaign, "output_root", "campaign"),
        "campaign.output_root",
    )
    raw_command = _require_list(
        _required(campaign, "runner_command", "campaign"),
        "campaign.runner_command",
    )
    if not raw_command:
        raise CampaignError("campaign.runner_command must not be empty")
    runner_command = [
        _require_string(part, f"campaign.runner_command[{index}]")
        for index, part in enumerate(raw_command)
    ]
    if not any(_CONFIG_PLACEHOLDER in part for part in runner_command):
        raise CampaignError(
            "campaign.runner_command must contain the "
            f"'{_CONFIG_PLACEHOLDER}' placeholder"
        )

    scenes = _require_mapping(
        _required(campaign, "base_scene_configs", "campaign"),
        "campaign.base_scene_configs",
    )
    if not scenes:
        raise CampaignError("campaign.base_scene_configs must not be empty")
    base_scene_configs = {
        label: _require_string(
            scenes[label],
            f"campaign.base_scene_configs.{label}",
        )
        for label in sorted(scenes)
    }

    observing_reference = campaign.get("observing_reference")
    if observing_reference is not None:
        observing_reference = _require_string(
            observing_reference,
            "campaign.observing_reference",
        )

    raw_jobs = _require_list(
        _required(campaign, "jobs", "campaign"),
        "campaign.jobs",
    )
    if not raw_jobs:
        raise CampaignError("campaign.jobs must not be empty")
    jobs = []
    seen_job_ids: set[str] = set()
    for index, raw_job in enumerate(raw_jobs):
        path = f"campaign.jobs[{index}]"
        job = _require_mapping(raw_job, path)
        _reject_unknown_keys(job, {"job_id", "scene", "overrides"}, path)
        job_id = _require_identifier(
            _required(job, "job_id", path),
            f"{path}.job_id",
        )
        if job_id in seen_job_ids:
            raise CampaignError(f"{path}.job_id duplicates '{job_id}'")
        seen_job_ids.add(job_id)
        scene = _require_string(_required(job, "scene", path), f"{path}.scene")
        if scene not in base_scene_configs:
            raise CampaignError(
                f"{path}.scene '{scene}' is not declared in "
                "campaign.base_scene_configs"
            )
        overrides = _require_mapping(
            _required(job, "overrides", path),
            f"{path}.overrides",
        )
        psf = overrides.get("psf")
        kernel = psf.get("kernel") if isinstance(psf, dict) else None
        shape = kernel.get("shape_native") if isinstance(kernel, dict) else None
        if shape is None:
            raise CampaignError(
                f"{path}.overrides must set psf.kernel.shape_native "
                "explicitly; the master default is never inherited"
            )
        jobs.append({
            "job_id": job_id,
            "scene": scene,
            "overrides": deepcopy(overrides),
        })

    expected_job_count = _integer_at_least(
        _required(campaign, "expected_job_count", "campaign"),
        1,
        "campaign.expected_job_count",
    )
    if expected_job_count != len(jobs):
        raise CampaignError(
            f"campaign.expected_job_count {expected_job_count} does not equal "
            f"the declared job count {len(jobs)}"
        )

    raw_artifacts = campaign.get("expected_artifacts")
    expected_artifacts: list[str] = []
    if raw_artifacts is not None:
        declared = _require_list(raw_artifacts, "campaign.expected_artifacts")
        if not declared:
            raise CampaignError("campaign.expected_artifacts must not be empty")
        expected_artifacts = sorted(
            _require_artifact_relpath(
                value,
                f"campaign.expected_artifacts[{index}]",
            )
            for index, value in enumerate(declared)
        )
        if len(set(expected_artifacts)) != len(expected_artifacts):
            raise CampaignError(
                "campaign.expected_artifacts must not repeat paths"
            )

    seed_policy = _require_mapping(
        _required(campaign, "seed_policy", "campaign"),
        "campaign.seed_policy",
    )

    campaign_uuid = campaign.get("campaign_uuid")
    if campaign_uuid is not None:
        campaign_uuid = _require_string(campaign_uuid, "campaign.campaign_uuid")
        try:
            uuid_module.UUID(campaign_uuid)
        except ValueError as exc:
            raise CampaignError(
                f"campaign.campaign_uuid is not a valid UUID: {exc}"
            ) from exc

    collect = campaign.get("collect")
    if collect is not None:
        collect_block = _require_mapping(collect, "campaign.collect")
        _reject_unknown_keys(
            collect_block,
            {"scalars", "artifact"},
            "campaign.collect",
        )
        raw_scalars = _require_list(
            _required(collect_block, "scalars", "campaign.collect"),
            "campaign.collect.scalars",
        )
        if not raw_scalars:
            raise CampaignError("campaign.collect.scalars must not be empty")
        scalars = []
        for index, raw_key in enumerate(raw_scalars):
            key_path = f"campaign.collect.scalars[{index}]"
            key = _require_string(raw_key, key_path)
            if _COLLECT_KEY_PATTERN.fullmatch(key) is None:
                raise CampaignError(
                    f"{key_path} must be a dotted [A-Za-z0-9_] key path"
                )
            if key in _RESERVED_COLLECT_KEYS:
                raise CampaignError(
                    f"{key_path} '{key}' is a reserved harvest identity "
                    "member and cannot be collected"
                )
            scalars.append(key)
        if len(set(scalars)) != len(scalars):
            raise CampaignError("campaign.collect.scalars must not repeat keys")
        artifact = collect_block.get("artifact")
        if artifact is not None:
            artifact = _require_string(artifact, "campaign.collect.artifact")
            if expected_artifacts and artifact not in expected_artifacts:
                raise CampaignError(
                    f"campaign.collect.artifact '{artifact}' is not declared "
                    "in campaign.expected_artifacts"
                )
        collect = {"scalars": scalars, "artifact": artifact}

    return {
        "campaign": {
            "name": name,
            "output_root": output_root,
            "runner_command": runner_command,
            "base_scene_configs": base_scene_configs,
            "observing_reference": observing_reference,
            "seed_policy": deepcopy(seed_policy),
            "campaign_uuid": campaign_uuid,
            "collect": collect,
            "expected_job_count": expected_job_count,
            "expected_artifacts": expected_artifacts,
            "jobs": jobs,
        }
    }


def _build_frozen_manifest(
    campaign: dict,
    manifest_path: Path,
    output_root: Path,
    campaign_uuid: str,
) -> tuple[dict, list[tuple[str, bytes]]]:
    """Resolve inputs, stage every job, and build the frozen manifest."""
    from hwoslaps.provenance import config_hash

    manifest_dir = manifest_path.parent
    scene_paths = {
        label: _resolve_path(value, manifest_dir)
        for label, value in campaign["base_scene_configs"].items()
    }
    scene_configs = {
        label: _load_yaml_mapping(
            path,
            f"Campaign base scene config '{label}'",
        )
        for label, path in scene_paths.items()
    }
    scene_records = {
        label: {"path": str(path), "sha256": file_sha256(path)}
        for label, path in scene_paths.items()
    }

    observation = None
    source_patches: dict = {}
    reference_record = None
    if campaign["observing_reference"] is not None:
        reference_path = _resolve_path(
            campaign["observing_reference"],
            manifest_dir,
        )
        observation, source_patches = _load_observing_reference(
            reference_path,
            sorted(scene_paths),
        )
        reference_record = {
            "path": str(reference_path),
            "sha256": file_sha256(reference_path),
        }

    frozen_jobs = []
    staged_payloads = []
    for job in campaign["jobs"]:
        merged, payload = _stage_job_config(
            scene_configs[job["scene"]],
            observation,
            source_patches.get(job["scene"]),
            job["overrides"],
            job["job_id"],
            output_root,
        )
        staged_payloads.append((job["job_id"], payload))
        frozen_jobs.append({
            "job_id": job["job_id"],
            "scene": job["scene"],
            "overrides": deepcopy(job["overrides"]),
            "overrides_digest": _sha256_bytes(
                _canonical_json(job["overrides"]).encode("utf-8")
            ),
            "staged_config": f"{_CONFIGS_DIR}/{job['job_id']}.yaml",
            "staged_config_sha256": _sha256_bytes(payload),
            "config_hash": config_hash(merged),
            "expected_artifacts": list(campaign["expected_artifacts"]),
        })

    frozen = {
        "campaign": {
            "schema_version": _SCHEMA_VERSION,
            "name": campaign["name"],
            "campaign_uuid": campaign_uuid,
            "output_root": str(output_root),
            "manifest_path": str(manifest_path),
            "runner_command": list(campaign["runner_command"]),
            "base_scene_configs": scene_records,
            "observing_reference": reference_record,
            "seed_policy": deepcopy(campaign["seed_policy"]),
            "collect": deepcopy(campaign["collect"]),
            "expected_job_count": campaign["expected_job_count"],
            "jobs": frozen_jobs,
        }
    }
    return frozen, staged_payloads


def _verify_frozen_state(
    output_root: Path,
    frozen_payload: bytes,
    staged_payloads: list[tuple[str, bytes]],
) -> None:
    """Byte-compare an existing frozen campaign against a re-freeze."""
    frozen_path = output_root/_FROZEN_MANIFEST_NAME
    if frozen_path.read_bytes() != frozen_payload:
        raise CampaignError(
            f"Frozen manifest {frozen_path} does not match a re-freeze of its "
            "source manifest; the frozen manifest is immutable"
        )
    digest_path = output_root/_FROZEN_DIGEST_NAME
    if (
        not digest_path.is_file()
        or digest_path.read_text(encoding="ascii").strip()
        != _sha256_bytes(frozen_payload)
    ):
        raise CampaignError(
            f"Frozen manifest digest {digest_path} is missing or does not "
            "match the frozen manifest; the frozen manifest is immutable"
        )
    for job_id, payload in staged_payloads:
        staged_path = output_root/_CONFIGS_DIR/f"{job_id}.yaml"
        if not staged_path.is_file():
            raise CampaignError(f"Staged config {staged_path} is missing")
        if staged_path.read_bytes() != payload:
            raise CampaignError(
                f"Staged config {staged_path} does not match a re-freeze of "
                "its source manifest; staged configs are immutable"
            )


def _write_frozen_state(
    output_root: Path,
    frozen_payload: bytes,
    staged_payloads: list[tuple[str, bytes]],
) -> None:
    """Stage every config to a temporary directory, then publish it."""
    staging_root = output_root/_STAGING_DIR
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True)
    try:
        for job_id, payload in staged_payloads:
            (staging_root/f"{job_id}.yaml").write_bytes(payload)
        for name in _CAMPAIGN_DIRS:
            (output_root/name).mkdir(parents=True, exist_ok=True)
        for job_id, _ in staged_payloads:
            os.replace(
                staging_root/f"{job_id}.yaml",
                output_root/_CONFIGS_DIR/f"{job_id}.yaml",
            )
        _atomic_write_bytes(output_root/_FROZEN_MANIFEST_NAME, frozen_payload)
        _atomic_write_bytes(
            output_root/_FROZEN_DIGEST_NAME,
            (_sha256_bytes(frozen_payload) + "\n").encode("ascii"),
        )
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def freeze_campaign(manifest_path) -> Path:
    """Validate a manifest, stage every job config, and freeze it.

    Parameters
    ----------
    manifest_path : path-like
        Campaign manifest YAML path. Relative paths inside the manifest
        resolve against this file's directory.

    Returns
    -------
    frozen_path : `pathlib.Path`
        Path of ``manifest.frozen.yaml`` under the campaign output root.

    Raises
    ------
    CampaignError
        Raised for any manifest validation failure, for a non-empty
        output root without a frozen manifest to resume from, and for any
        byte difference against an already frozen campaign.

    Notes
    -----
    Freezing is idempotent. When the campaign is already frozen the
    rebuilt manifest and staged configurations are byte-compared instead
    of rewritten, and an auto-generated campaign UUID is taken from the
    existing frozen manifest so the comparison stays meaningful.
    """
    manifest_path = Path(manifest_path).expanduser().resolve()
    normalized = validate_campaign_manifest(
        _load_yaml_mapping(manifest_path, "Campaign manifest")
    )
    campaign = normalized["campaign"]
    output_root = _resolve_path(campaign["output_root"], manifest_path.parent)
    frozen_path = output_root/_FROZEN_MANIFEST_NAME
    resuming = frozen_path.is_file()
    if output_root.exists():
        if not output_root.is_dir():
            raise CampaignError(
                f"Campaign output_root {output_root} is not a directory"
            )
        if any(output_root.iterdir()) and not resuming:
            raise CampaignError(
                f"Campaign output_root {output_root} is not empty and holds no "
                f"{_FROZEN_MANIFEST_NAME} to resume from"
            )

    campaign_uuid = campaign["campaign_uuid"]
    if resuming and campaign_uuid is None:
        campaign_uuid = _load_frozen_manifest(output_root)["campaign_uuid"]
    if campaign_uuid is None:
        campaign_uuid = str(uuid_module.uuid4())

    frozen, staged_payloads = _build_frozen_manifest(
        campaign,
        manifest_path,
        output_root,
        campaign_uuid,
    )
    frozen_payload = yaml.safe_dump(frozen, sort_keys=True).encode("utf-8")
    if resuming:
        _verify_frozen_state(output_root, frozen_payload, staged_payloads)
        return frozen_path
    output_root.mkdir(parents=True, exist_ok=True)
    _write_frozen_state(output_root, frozen_payload, staged_payloads)
    return frozen_path


def _verify_staged_configs(output_root: Path, campaign: dict) -> None:
    """Check that staged configs match the frozen manifest exactly."""
    configs_dir = output_root/_CONFIGS_DIR
    if not configs_dir.is_dir():
        raise CampaignError(
            f"Campaign {output_root} has no staged config directory "
            f"{configs_dir}"
        )
    expected = {job["job_id"] for job in campaign["jobs"]}
    found = {path.stem for path in configs_dir.glob("*.yaml")}
    if found != expected:
        raise CampaignError(
            f"Staged configs in {configs_dir} do not match the frozen job "
            f"set: unexpected {sorted(found - expected)}, missing "
            f"{sorted(expected - found)}"
        )
    for job in campaign["jobs"]:
        staged_path = configs_dir/f"{job['job_id']}.yaml"
        digest = file_sha256(staged_path)
        if digest != job["staged_config_sha256"]:
            raise CampaignError(
                f"Staged config {staged_path} sha256 {digest} does not match "
                f"the frozen manifest record {job['staged_config_sha256']}"
            )


def _read_sentinel(path: Path) -> dict:
    """Read one sentinel file as a JSON object."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        raise CampaignError(f"Sentinel {path} is not readable JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise CampaignError(f"Sentinel {path} must contain a JSON object")
    return payload


def _declared_artifact_paths(job: dict) -> list[str]:
    """Return one job's declared artifacts relative to the campaign root."""
    return [
        (Path(_OUTPUTS_DIR)/job["job_id"]/relative).as_posix()
        for relative in job["expected_artifacts"]
    ]


def _undeclared_artifacts(output_dir: Path, job: dict) -> list[str]:
    """Return the ``.npz`` files a job wrote without declaring them."""
    declared = set(job["expected_artifacts"])
    return sorted(
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*.npz")
        if path.is_file()
        and path.relative_to(output_dir).as_posix() not in declared
    )


def _artifact_tree_problem(output_dir: Path, job: dict) -> Optional[str]:
    """Return a symlink or containment violation in one job's output tree."""
    if output_dir.is_symlink():
        return f"job output directory {output_dir} is a symlink"
    for path in sorted(output_dir.rglob("*")):
        if path.is_symlink():
            return f"job output tree holds a symlink: {path}"
    resolved_root = output_dir.resolve()
    for relative in job["expected_artifacts"]:
        resolved = (output_dir/relative).resolve()
        if not resolved.is_relative_to(resolved_root):
            return (
                f"declared artifact {output_dir/relative} resolves outside "
                "the job output directory"
            )
    return None


def _verify_declared_artifacts(campaign: dict) -> None:
    """Require every frozen job to declare at least one artifact."""
    undeclared = sorted(
        job["job_id"]
        for job in campaign["jobs"]
        if not job["expected_artifacts"]
    )
    if undeclared:
        raise CampaignError(
            f"Frozen jobs {', '.join(undeclared)} declare no expected "
            "artifacts; campaign.expected_artifacts must declare every "
            "artifact a job writes before the campaign can run or harvest"
        )


def _sentinel_problem(
    output_root: Path,
    campaign: dict,
    job: dict,
    done_path: Path,
) -> Optional[str]:
    """Return the first reason a DONE sentinel is stale, or `None`."""
    try:
        sentinel = _read_sentinel(done_path)
    except CampaignError as exc:
        return str(exc)
    missing = sorted(_DONE_SENTINEL_MEMBERS - set(sentinel))
    if missing:
        return (
            f"sentinel {done_path} is missing members: " + ", ".join(missing)
        )
    for name, expected in (
        ("job_id", job["job_id"]),
        ("campaign_uuid", campaign["campaign_uuid"]),
        ("config_sha256", job["staged_config_sha256"]),
    ):
        if sentinel.get(name) != expected:
            return (
                f"sentinel {done_path} {name} {sentinel.get(name)!r} does not "
                f"match the frozen manifest value {expected!r}"
            )
    artifacts = sentinel.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return f"sentinel {done_path} records no artifacts"
    for entry in artifacts:
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            return f"sentinel {done_path} has a malformed artifact entry"
    declared = _declared_artifact_paths(job)
    recorded = [entry["path"] for entry in artifacts]
    if recorded != declared:
        return (
            f"sentinel {done_path} records artifacts {recorded} instead of "
            f"the declared set {declared}"
        )
    output_dir = output_root/_OUTPUTS_DIR/job["job_id"]
    tree_problem = _artifact_tree_problem(output_dir, job)
    if tree_problem:
        return f"sentinel {done_path}: {tree_problem}"
    undeclared = _undeclared_artifacts(output_dir, job)
    if undeclared:
        return (
            f"job output {output_dir} holds undeclared .npz artifacts: "
            + ", ".join(undeclared)
        )
    for entry in artifacts:
        artifact_path = output_root/entry["path"]
        if not artifact_path.is_file():
            return f"artifact {artifact_path} recorded in {done_path} is missing"
        digest = file_sha256(artifact_path)
        if digest != entry["sha256"]:
            return (
                f"artifact {artifact_path} sha256 {digest} does not match "
                f"{entry['sha256']} recorded in {done_path}"
            )
    return None


def _scalar_text(value: Any, path: Path, member: str) -> str:
    """Return a single stored value as text."""
    import numpy as np

    array = np.asarray(value)
    if array.size != 1:
        raise CampaignError(
            f"Artifact {path} member '{member}' must hold a single value"
        )
    return str(array.reshape(-1)[0])


def _declared_provenance_digests(staged_config: dict) -> dict:
    """Return the provenance digests one staged configuration declares.

    A Stage 0 configuration pins the source revision the campaign was
    generated at and the template asset the design selected, and the
    Stage 0 runner refuses to render unless this checkout and the asset
    on disk carry exactly those digests. That check lives inside the
    runner, so a job invoked through a wrapper that never reaches the
    runner would produce an artifact nobody held to the pinned values.
    Re-checking the digests the runner stamped into the artifact closes
    that path. It protects against an honest mistake, a stale resume or
    a hand-edited command, not against a determined tamperer: anyone who
    can write the artifact can write the members too.

    Parameters
    ----------
    staged_config : `dict`
        Staged merged configuration of one job.

    Returns
    -------
    declared : `dict`
        Artifact member name to the digest the configuration pins.
        Empty for a configuration carrying no ``stage0`` block, which
        is every non-Stage-0 campaign.
    """
    stage0 = staged_config.get("stage0")
    if not isinstance(stage0, dict):
        return {}
    code_revision = stage0.get("code_revision")
    declared = {}
    for member, value in (
        (
            "code_revision_sha256",
            code_revision.get("sha256") if isinstance(code_revision, dict) else None,
        ),
        ("source_asset_sha256", stage0.get("source_asset_sha256")),
    ):
        if value is not None:
            declared[member] = str(value)
    return declared


def _validate_job_artifacts(
    output_root: Path,
    campaign: dict,
    job: dict,
) -> list[dict]:
    """Validate one job's declared artifacts and return sentinel records.

    Beyond the campaign identity members, every provenance digest the
    staged configuration pins must be embedded in the artifact and must
    match; see `_declared_provenance_digests` for what that binding is
    and is not worth.
    """
    import numpy as np

    from hwoslaps.provenance import config_hash

    job_id = job["job_id"]
    output_dir = output_root/_OUTPUTS_DIR/job_id
    if not output_dir.is_dir():
        raise CampaignError(
            f"Job '{job_id}' produced no output directory {output_dir}"
        )
    tree_problem = _artifact_tree_problem(output_dir, job)
    if tree_problem:
        raise CampaignError(f"Job '{job_id}' {tree_problem}")
    missing = [
        relative
        for relative in job["expected_artifacts"]
        if not (output_dir/relative).is_file()
    ]
    if missing:
        raise CampaignError(
            f"Job '{job_id}' did not produce the declared artifacts under "
            f"{output_dir}: " + ", ".join(missing)
        )
    undeclared = _undeclared_artifacts(output_dir, job)
    if undeclared:
        raise CampaignError(
            f"Job '{job_id}' produced undeclared .npz artifacts under "
            f"{output_dir}: " + ", ".join(undeclared)
        )

    staged_path = output_root/_CONFIGS_DIR/f"{job_id}.yaml"
    digest = file_sha256(staged_path)
    if digest != job["staged_config_sha256"]:
        raise CampaignError(
            f"Staged config {staged_path} sha256 {digest} does not match the "
            f"frozen manifest record {job['staged_config_sha256']}"
        )
    staged_config = _load_yaml_mapping(staged_path, "Staged campaign config")
    expected_members = (
        ("config_hash", config_hash(staged_config)),
        ("campaign_uuid", campaign["campaign_uuid"]),
    ) + tuple(_declared_provenance_digests(staged_config).items())

    records = []
    for relative in job["expected_artifacts"]:
        path = output_dir/relative
        embedded = {}
        try:
            with np.load(path, allow_pickle=False) as stored:
                for member, _ in expected_members:
                    if member in stored.files:
                        embedded[member] = _scalar_text(
                            stored[member],
                            path,
                            member,
                        )
        except CampaignError:
            raise
        except Exception as exc:
            raise CampaignError(
                f"Job '{job_id}' artifact {path} does not load: {exc}"
            ) from exc
        for member, expected in expected_members:
            if member not in embedded:
                raise CampaignError(
                    f"Job '{job_id}' artifact {path} does not embed the "
                    f"required '{member}' member, expected {expected}"
                )
            if embedded[member] != expected:
                raise CampaignError(
                    f"Job '{job_id}' artifact {path} {member} "
                    f"{embedded[member]} does not match the campaign value "
                    f"{expected}"
                )
        records.append({
            "path": path.relative_to(output_root).as_posix(),
            "sha256": file_sha256(path),
        })
    return records


def _write_done_sentinel(
    output_root: Path,
    campaign: dict,
    job: dict,
    artifacts: list[dict],
    wall_s: float,
) -> None:
    """Write one per-job DONE sentinel atomically."""
    payload = {
        "job_id": job["job_id"],
        "campaign_uuid": campaign["campaign_uuid"],
        "config_sha256": job["staged_config_sha256"],
        "artifacts": artifacts,
        "wall_s": wall_s,
        "validated_utc": _utc_now(),
    }
    _atomic_write_bytes(
        output_root/_SENTINELS_DIR/f"{job['job_id']}.DONE",
        (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8"),
    )


def _repo_root() -> Path:
    """Return the repository root used as the runner working directory."""
    return Path(__file__).resolve().parents[3]


def _execute_job(
    output_root: Path,
    campaign: dict,
    job: dict,
    gpu_id: Optional[int],
    timeout_s: float,
) -> JobResult:
    """Clear stale outputs, run one job, and validate before its sentinel."""
    job_id = job["job_id"]
    staged_path = output_root/_CONFIGS_DIR/f"{job_id}.yaml"
    command = [
        part.replace(_CONFIG_PLACEHOLDER, str(staged_path))
        for part in campaign["runner_command"]
    ]
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    env["HWOSLAPS_CAMPAIGN_UUID"] = campaign["campaign_uuid"]
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    log_path = output_root/_LOGS_DIR/f"{job_id}.log"
    output_dir = output_root/_OUTPUTS_DIR/job_id
    if output_dir.exists():
        try:
            shutil.rmtree(output_dir)
        except OSError as exc:
            return JobResult(
                job_id=job_id,
                status="output_reset_failed",
                detail=f"cannot clear {output_dir} before the run: {exc}",
                wall_s=0.0,
            )
    started = time.monotonic()
    try:
        with log_path.open("wb") as log:
            completed = subprocess.run(
                command,
                cwd=str(_repo_root()),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout_s,
                check=False,
            )
    except subprocess.TimeoutExpired:
        return JobResult(
            job_id=job_id,
            status="timeout",
            detail=f"exceeded timeout_s={timeout_s}; see {log_path}",
            wall_s=time.monotonic() - started,
        )
    except Exception as exc:
        return JobResult(
            job_id=job_id,
            status="subprocess_error",
            detail=f"subprocess failed: {exc}",
            wall_s=time.monotonic() - started,
        )
    wall_s = time.monotonic() - started
    if completed.returncode != 0:
        return JobResult(
            job_id=job_id,
            status="exit_nonzero",
            detail=f"exited {completed.returncode}; see {log_path}",
            wall_s=wall_s,
        )
    try:
        artifacts = _validate_job_artifacts(output_root, campaign, job)
    except CampaignError as exc:
        return JobResult(
            job_id=job_id,
            status="validation_failed",
            detail=str(exc),
            wall_s=wall_s,
        )
    _write_done_sentinel(output_root, campaign, job, artifacts, wall_s)
    return JobResult(job_id=job_id, status="completed", detail=None, wall_s=wall_s)


def _worker(
    job_queue: queue.Queue,
    gpu_id: Optional[int],
    output_root: Path,
    campaign: dict,
    timeout_s: float,
    results: dict,
    lock: threading.Lock,
    abort: threading.Event,
) -> None:
    """Drain the job queue until it empties or a job fails."""
    while not abort.is_set():
        try:
            job = job_queue.get_nowait()
        except queue.Empty:
            return
        result = _execute_job(output_root, campaign, job, gpu_id, timeout_s)
        with lock:
            results[result.job_id] = result
        if result.status != "completed":
            abort.set()


def _resume_plan(
    output_root: Path,
    campaign: dict,
    allow_rerun: bool,
) -> tuple[list[dict], list[str]]:
    """Split frozen jobs into pending and already validated jobs."""
    pending = []
    skipped = []
    for job in campaign["jobs"]:
        done_path = output_root/_SENTINELS_DIR/f"{job['job_id']}.DONE"
        if not done_path.is_file():
            pending.append(job)
            continue
        problem = _sentinel_problem(output_root, campaign, job, done_path)
        if problem is None:
            skipped.append(job["job_id"])
            continue
        if not allow_rerun:
            raise CampaignError(
                f"Campaign {output_root} job '{job['job_id']}' is stale: "
                f"{problem}; pass allow_rerun=True to discard the stale "
                "outputs and re-run the job"
            )
        done_path.unlink()
        shutil.rmtree(output_root/_OUTPUTS_DIR/job["job_id"], ignore_errors=True)
        pending.append(job)
    return pending, skipped


def run_campaign(
    output_root,
    max_workers,
    *,
    gpu_ids=None,
    timeout_s,
    allow_rerun=False,
    require_clean_tree=True,
    allow_dirty_tree=False,
) -> dict:
    """Execute every frozen job that lacks a valid DONE sentinel.

    Parameters
    ----------
    output_root : path-like
        Frozen campaign directory.
    max_workers : `int`
        Number of concurrent job subprocesses.
    gpu_ids : sequence of `int`, optional
        One GPU id per worker, exported as ``CUDA_VISIBLE_DEVICES``. When
        omitted the workers inherit the caller's device visibility.
    timeout_s : `float`
        Per-job subprocess timeout in seconds.
    allow_rerun : `bool`, optional
        Whether a job whose sentinel no longer matches its artifacts may
        have those outputs discarded and be re-run. Stale artifacts are
        rejected by default.
    require_clean_tree : `bool`, optional
        Require a clean source tree before launching production jobs.
    allow_dirty_tree : `bool`, optional
        Explicit override for ``require_clean_tree``.

    Returns
    -------
    summary : `dict`
        Campaign UUID, executed and skipped job ids, and the completion
        sentinel path.

    Raises
    ------
    CampaignError
        Raised when the campaign is not frozen, when staged configs no
        longer match the frozen manifest, when a frozen job declares no
        expected artifacts, when a stale sentinel is found without
        ``allow_rerun``, and when any job fails to run or to validate.
        ``CAMPAIGN_COMPLETE`` is never written on failure.
    """
    output_root = Path(output_root).expanduser().resolve()
    campaign = _load_frozen_manifest(output_root)
    max_workers = _integer_at_least(max_workers, 1, "max_workers")
    timeout_s = _positive_number(timeout_s, "timeout_s")
    allow_rerun = _require_boolean(allow_rerun, "allow_rerun")
    require_clean_tree = _require_boolean(
        require_clean_tree, "require_clean_tree"
    )
    allow_dirty_tree = _require_boolean(allow_dirty_tree, "allow_dirty_tree")
    _check_clean_tree(require_clean_tree, allow_dirty_tree)
    if gpu_ids is None:
        worker_gpus: list[Optional[int]] = [None]*max_workers
    else:
        raw_gpu_ids = _require_list(gpu_ids, "gpu_ids")
        if len(raw_gpu_ids) != max_workers:
            raise CampaignError(
                f"gpu_ids must provide exactly one id per worker; got "
                f"{len(raw_gpu_ids)} ids for max_workers={max_workers}"
            )
        worker_gpus = [
            _integer_at_least(value, 0, f"gpu_ids[{index}]")
            for index, value in enumerate(raw_gpu_ids)
        ]

    _verify_staged_configs(output_root, campaign)
    _verify_declared_artifacts(campaign)
    for name in _CAMPAIGN_DIRS:
        (output_root/name).mkdir(parents=True, exist_ok=True)
    pending, skipped = _resume_plan(output_root, campaign, allow_rerun)
    complete_path = output_root/_SENTINELS_DIR/_COMPLETE_SENTINEL
    if pending and complete_path.is_file():
        complete_path.unlink()

    results: dict = {}
    if pending:
        job_queue: queue.Queue = queue.Queue()
        for job in pending:
            job_queue.put(job)
        lock = threading.Lock()
        abort = threading.Event()
        threads = [
            threading.Thread(
                target=_worker,
                args=(
                    job_queue,
                    worker_gpus[index],
                    output_root,
                    campaign,
                    timeout_s,
                    results,
                    lock,
                    abort,
                ),
            )
            for index in range(min(max_workers, len(pending)))
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    failures = [
        result
        for result in results.values()
        if result.status != "completed"
    ]
    not_started = sorted(
        job["job_id"] for job in pending if job["job_id"] not in results
    )
    if failures:
        details = "; ".join(
            f"{result.job_id} [{result.status}] {result.detail}"
            for result in sorted(failures, key=lambda item: item.job_id)
        )
        message = (
            f"Campaign {output_root} failed for {len(failures)} job(s): "
            f"{details}"
        )
        if not_started:
            message += f"; jobs not started: {', '.join(not_started)}"
        raise CampaignError(message)
    if not_started:
        raise CampaignError(
            f"Campaign {output_root} left jobs unexecuted: "
            f"{', '.join(not_started)}"
        )

    for job in campaign["jobs"]:
        done_path = output_root/_SENTINELS_DIR/f"{job['job_id']}.DONE"
        if not done_path.is_file():
            raise CampaignError(
                f"Campaign {output_root} job '{job['job_id']}' has no "
                f"sentinel {done_path}"
            )
        problem = _sentinel_problem(output_root, campaign, job, done_path)
        if problem is not None:
            raise CampaignError(
                f"Campaign {output_root} cannot complete: {problem}"
            )
    executed = sorted(results)
    _atomic_write_bytes(
        complete_path,
        (
            json.dumps(
                {
                    "campaign_uuid": campaign["campaign_uuid"],
                    "name": campaign["name"],
                    "expected_job_count": campaign["expected_job_count"],
                    "actual_job_count": len(campaign["jobs"]),
                    "job_ids": [job["job_id"] for job in campaign["jobs"]],
                    "completed_utc": _utc_now(),
                },
                sort_keys=True,
                indent=2,
            )
            + "\n"
        ).encode("utf-8"),
    )
    return {
        "campaign_uuid": campaign["campaign_uuid"],
        "executed_job_ids": executed,
        "skipped_job_ids": skipped,
        "complete_sentinel": str(complete_path),
    }


def _numeric_scalar(value: Any, description: str) -> float:
    """Require one finite numeric scalar from a collected value."""
    import numpy as np

    array = np.asarray(value)
    if array.size != 1:
        raise CampaignError(f"{description} must be a single value")
    if not (
        np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.bool_)
    ):
        raise CampaignError(f"{description} must be numeric")
    result = float(array.reshape(-1)[0])
    if not math.isfinite(result):
        raise CampaignError(f"{description} must be finite")
    return result


def _extract_scalar(stored: Any, key: str, path: Path) -> float:
    """Extract one dotted collection key from a loaded artifact."""
    segments = key.split(".")
    head = segments[0]
    if head not in stored.files:
        raise CampaignError(
            f"Artifact {path} has no member '{head}' for collect key '{key}'"
        )
    value = stored[head]
    if len(segments) > 1:
        text = _scalar_text(value, path, head)
        try:
            decoded = json.loads(text)
        except ValueError as exc:
            raise CampaignError(
                f"Artifact {path} member '{head}' is not JSON for collect "
                f"key '{key}': {exc}"
            ) from exc
        for segment in segments[1:]:
            if not isinstance(decoded, dict) or segment not in decoded:
                raise CampaignError(
                    f"Artifact {path} collect key '{key}' has no element "
                    f"'{segment}'"
                )
            decoded = decoded[segment]
        value = decoded
    return _numeric_scalar(value, f"artifact {path} collect key '{key}'")


def _collect_job_scalars(
    output_root: Path,
    collect: dict,
    job_id: str,
    artifacts: list[dict],
) -> dict:
    """Collect the configured scalars from one job's artifact."""
    import numpy as np

    if collect["artifact"] is not None:
        relative = (
            Path(_OUTPUTS_DIR)/job_id/collect["artifact"]
        ).as_posix()
        matches = [item for item in artifacts if item["path"] == relative]
        if not matches:
            raise CampaignError(
                f"Job '{job_id}' has no validated artifact "
                f"{output_root/relative} for campaign.collect.artifact"
            )
        selected = matches[0]
    elif len(artifacts) == 1:
        selected = artifacts[0]
    else:
        raise CampaignError(
            f"Job '{job_id}' produced {len(artifacts)} artifacts; "
            "campaign.collect.artifact is required to select one"
        )
    path = output_root/selected["path"]
    try:
        with np.load(path, allow_pickle=False) as stored:
            return {
                key: _extract_scalar(stored, key, path)
                for key in collect["scalars"]
            }
    except CampaignError:
        raise
    except Exception as exc:
        raise CampaignError(
            f"Job '{job_id}' artifact {path} does not load: {exc}"
        ) from exc


def _write_harvest_npz(
    path: Path,
    campaign_uuid: str,
    job_ids: list[str],
    scalars: list[dict],
    keys: list[str],
) -> None:
    """Write the numeric harvest aggregation atomically.

    The identity members ``campaign_uuid`` and ``job_ids`` share the
    member namespace with the collected keys, so manifest validation
    reserves those two names.
    """
    import numpy as np

    arrays = {
        "campaign_uuid": np.asarray(campaign_uuid),
        "job_ids": np.asarray(job_ids),
    }
    for key in keys:
        arrays[key] = np.asarray(
            [record[key] for record in scalars],
            dtype=float,
        )
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def harvest_campaign(output_root) -> Path:
    """Harvest a completed campaign into ``harvest/harvest.json``.

    Parameters
    ----------
    output_root : path-like
        Frozen campaign directory.

    Returns
    -------
    harvest_path : `pathlib.Path`
        Path of the written ``harvest/harvest.json``.

    Raises
    ------
    CampaignError
        Raised when ``CAMPAIGN_COMPLETE`` is absent, when a frozen job
        declares no expected artifacts, when any sentinel fails to
        re-validate against the artifacts on disk, when a sentinel exists
        for a job outside the frozen manifest, or when a configured
        collection key is missing. Nothing partial is written.
    """
    output_root = Path(output_root).expanduser().resolve()
    campaign = _load_frozen_manifest(output_root)
    complete_path = output_root/_SENTINELS_DIR/_COMPLETE_SENTINEL
    if not complete_path.is_file():
        raise CampaignError(
            f"Campaign {output_root} has no {complete_path}; harvest requires "
            "a completed campaign"
        )
    complete = _read_sentinel(complete_path)
    if complete.get("campaign_uuid") != campaign["campaign_uuid"]:
        raise CampaignError(
            f"Sentinel {complete_path} campaign_uuid "
            f"{complete.get('campaign_uuid')!r} does not match the frozen "
            f"manifest value {campaign['campaign_uuid']!r}"
        )
    _verify_staged_configs(output_root, campaign)
    _verify_declared_artifacts(campaign)

    expected_job_ids = [job["job_id"] for job in campaign["jobs"]]
    sentinels_dir = output_root/_SENTINELS_DIR
    found_job_ids = sorted(
        path.name[: -len(".DONE")] for path in sentinels_dir.glob("*.DONE")
    )
    unexpected = sorted(set(found_job_ids) - set(expected_job_ids))
    missing = sorted(set(expected_job_ids) - set(found_job_ids))
    if unexpected or missing:
        raise CampaignError(
            f"Campaign {output_root} sentinels do not reconcile with the "
            f"frozen job set: unexpected {unexpected}, missing {missing}"
        )

    collect = campaign["collect"]
    records = []
    for job in campaign["jobs"]:
        done_path = sentinels_dir/f"{job['job_id']}.DONE"
        problem = _sentinel_problem(output_root, campaign, job, done_path)
        if problem is not None:
            raise CampaignError(
                f"Campaign {output_root} refuses to harvest: {problem}"
            )
        sentinel = _read_sentinel(done_path)
        record = {
            "job_id": job["job_id"],
            "scene": job["scene"],
            "overrides_digest": job["overrides_digest"],
            "config_sha256": job["staged_config_sha256"],
            "config_hash": job["config_hash"],
            "artifacts": sentinel["artifacts"],
            "wall_s": sentinel["wall_s"],
            "validated_utc": sentinel["validated_utc"],
        }
        if collect is not None:
            record["scalars"] = _collect_job_scalars(
                output_root,
                collect,
                job["job_id"],
                sentinel["artifacts"],
            )
        records.append(record)

    from hwoslaps.provenance import revision_provenance

    harvest = {
        "schema_version": _SCHEMA_VERSION,
        "campaign_uuid": campaign["campaign_uuid"],
        "name": campaign["name"],
        "manifest_frozen_sha256": file_sha256(
            output_root/_FROZEN_MANIFEST_NAME
        ),
        "runner_command": list(campaign["runner_command"]),
        "seed_policy": deepcopy(campaign["seed_policy"]),
        "collect": deepcopy(collect),
        "revision_provenance": revision_provenance(),
        "reconciliation": {
            "expected_job_count": campaign["expected_job_count"],
            "found_job_count": len(found_job_ids),
            "expected_job_ids": expected_job_ids,
            "found_job_ids": found_job_ids,
            "missing_job_ids": missing,
            "unexpected_job_ids": unexpected,
        },
        "jobs": records,
        "harvested_utc": _utc_now(),
    }
    harvest_dir = output_root/_HARVEST_DIR
    harvest_dir.mkdir(parents=True, exist_ok=True)
    if collect is not None:
        _write_harvest_npz(
            harvest_dir/"harvest.npz",
            campaign["campaign_uuid"],
            expected_job_ids,
            [record["scalars"] for record in records],
            collect["scalars"],
        )
    harvest_path = harvest_dir/"harvest.json"
    _atomic_write_bytes(
        harvest_path,
        (json.dumps(harvest, sort_keys=True, indent=2) + "\n").encode("utf-8"),
    )
    return harvest_path

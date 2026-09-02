"""Shared helpers for campaign construction and validation."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Optional

import yaml

from hwoslaps.provenance import revision_digest, revision_provenance

from .design_freeze import DEFAULT_DESIGN_FREEZE_PATH


_SCHEMA_VERSION = 1

_OUTPUTS_DIR = "outputs"
_FROZEN_MANIFEST_NAME = "manifest.frozen.yaml"
_FROZEN_DIGEST_NAME = "manifest.frozen.sha256"

_FROZEN_JOB_MEMBERS = {
    "job_id",
    "scene",
    "overrides",
    "overrides_digest",
    "staged_config",
    "staged_config_sha256",
    "config_hash",
    "expected_artifacts",
}

_FROZEN_CAMPAIGN_MEMBERS = {
    "schema_version",
    "name",
    "campaign_uuid",
    "output_root",
    "manifest_path",
    "runner_command",
    "base_scene_configs",
    "observing_reference",
    "seed_policy",
    "collect",
    "expected_job_count",
    "jobs",
}


class CampaignError(ValueError):
    """Raised for any campaign validation, execution, or harvest failure."""


def _reject_unknown_keys(mapping: dict, supported: set[str], path: str) -> None:
    """Reject unsupported mapping keys with a path-qualified message."""
    unsupported = sorted(set(mapping) - supported)
    if unsupported:
        raise CampaignError(
            f"{path} contains unsupported keys: " + ", ".join(unsupported)
        )


def _require_mapping(value: Any, path: str) -> dict:
    """Require a dictionary value."""
    if not isinstance(value, dict):
        raise CampaignError(f"{path} must be a mapping")
    return value


def _require_list(value: Any, path: str) -> list:
    """Require a list or tuple and return a list copy."""
    if not isinstance(value, (list, tuple)):
        raise CampaignError(f"{path} must be a list")
    return list(value)


def _required(mapping: dict, key: str, path: str) -> Any:
    """Return a required mapping value."""
    if key not in mapping:
        raise CampaignError(f"Missing required key '{key}' in {path}")
    return mapping[key]


def _positive_number(value: Any, path: str) -> float:
    """Require a positive finite non-boolean scalar number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CampaignError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise CampaignError(f"{path} must be finite")
    if result <= 0.0:
        raise CampaignError(f"{path} must be positive")
    return result


def _integer_at_least(value: Any, minimum: int, path: str) -> int:
    """Require a non-boolean integer at or above a lower bound."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise CampaignError(f"{path} must be an integer")
    if value < minimum:
        raise CampaignError(f"{path} must be at least {minimum}")
    return int(value)


def _canonical_json(value: Any) -> str:
    """Return the canonical JSON representation used by all digests."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(payload: bytes) -> str:
    """Return the full SHA-256 digest of a byte payload."""
    return hashlib.sha256(payload).hexdigest()


def _deep_merge(base: dict, patch: dict) -> dict:
    """Deep-merge ``patch`` onto ``base``.

    Nested mappings merge key by key. Scalars and lists in ``patch``
    replace the corresponding ``base`` value outright.
    """
    merged = deepcopy(base)
    for key, value in patch.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def resolve_path(value: str, base: Path) -> Path:
    """Resolve a manifest path against the manifest directory."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base/path
    return path.resolve()


def load_yaml_mapping(path: Path, description: str) -> dict:
    """Load one YAML file that must hold a top-level mapping."""
    if not path.is_file():
        raise CampaignError(f"{description} {path} does not exist")
    try:
        with path.open("r", encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream)
    except Exception as exc:
        raise CampaignError(
            f"{description} {path} is not readable YAML: {exc}"
        ) from exc
    if not isinstance(loaded, dict):
        raise CampaignError(f"{description} {path} must contain a mapping")
    return loaded


def load_observing_reference(
    path: Path,
    scene_labels: list[str],
) -> tuple[dict, dict]:
    """Load the observation block and per-scene source patches."""
    reference = load_yaml_mapping(path, "Campaign observing reference")
    observation = reference.get("observation")
    if not isinstance(observation, dict):
        raise CampaignError(
            f"Campaign observing reference {path} must contain an "
            "'observation' mapping"
        )
    normalization = reference.get("source_normalization")
    if not isinstance(normalization, dict):
        raise CampaignError(
            f"Campaign observing reference {path} must contain a "
            "'source_normalization' mapping"
        )
    patches = {}
    for label in scene_labels:
        patch = normalization.get(label)
        if not isinstance(patch, dict):
            raise CampaignError(
                f"Campaign observing reference {path} source_normalization "
                f"is missing a mapping for scene '{label}'"
            )
        patches[label] = deepcopy(patch)
    return deepcopy(observation), patches


def stage_job_config(
    scene_config: dict,
    observation: Optional[dict],
    source_patch: Optional[dict],
    overrides: dict,
    job_id: str,
    output_root: Path,
) -> tuple[dict, bytes]:
    """Merge one job configuration and render its staged bytes."""
    merged = deepcopy(scene_config)
    if observation is not None:
        merged = _deep_merge(merged, {"observation": observation})
        merged = _deep_merge(merged, source_patch)
    merged = _deep_merge(merged, overrides)
    merged["run_name"] = job_id
    plotting = merged.get("plotting")
    merged["plotting"] = deepcopy(plotting) if isinstance(plotting, dict) else {}
    merged["plotting"]["output_dir"] = str(output_root/_OUTPUTS_DIR)
    payload = yaml.safe_dump(merged, sort_keys=True).encode("utf-8")
    return merged, payload


def load_frozen_manifest(output_root: Path) -> dict:
    """Load and structurally validate ``manifest.frozen.yaml``."""
    frozen_path = output_root/_FROZEN_MANIFEST_NAME
    if not frozen_path.is_file():
        raise CampaignError(
            f"Campaign {output_root} has no {_FROZEN_MANIFEST_NAME}; "
            "freeze the manifest first"
        )
    digest_path = output_root/_FROZEN_DIGEST_NAME
    if not digest_path.is_file():
        raise CampaignError(
            f"Campaign {output_root} has no {_FROZEN_DIGEST_NAME}; the frozen "
            "manifest cannot be verified"
        )
    if (
        digest_path.read_text(encoding="ascii").strip()
        != _sha256_bytes(frozen_path.read_bytes())
    ):
        raise CampaignError(
            f"Frozen manifest {frozen_path} does not match its freeze-time "
            f"digest {digest_path}; the frozen manifest is immutable"
        )
    root = load_yaml_mapping(frozen_path, "Frozen campaign manifest")
    _reject_unknown_keys(root, {"campaign"}, str(frozen_path))
    campaign = _require_mapping(
        _required(root, "campaign", str(frozen_path)),
        f"{frozen_path} campaign",
    )
    _reject_unknown_keys(
        campaign,
        _FROZEN_CAMPAIGN_MEMBERS,
        f"{frozen_path} campaign",
    )
    missing = sorted(_FROZEN_CAMPAIGN_MEMBERS - set(campaign))
    if missing:
        raise CampaignError(
            f"{frozen_path} campaign is missing members: " + ", ".join(missing)
        )
    if campaign["schema_version"] != _SCHEMA_VERSION:
        raise CampaignError(
            f"{frozen_path} schema_version {campaign['schema_version']} is not "
            f"the supported version {_SCHEMA_VERSION}"
        )
    jobs = _require_list(campaign["jobs"], f"{frozen_path} campaign.jobs")
    for index, job in enumerate(jobs):
        path = f"{frozen_path} campaign.jobs[{index}]"
        job_block = _require_mapping(job, path)
        if set(job_block) != _FROZEN_JOB_MEMBERS:
            raise CampaignError(f"{path} members are invalid")
    job_ids = [job["job_id"] for job in jobs]
    if len(set(job_ids)) != len(job_ids):
        raise CampaignError(f"{frozen_path} campaign.jobs has duplicate job ids")
    if campaign["expected_job_count"] != len(jobs):
        raise CampaignError(
            f"{frozen_path} expected_job_count "
            f"{campaign['expected_job_count']} does not equal the frozen job "
            f"count {len(jobs)}"
        )
    return campaign


def _freeze_artifact_path(freeze_path) -> Path:
    """Return the resolved freeze artifact one build consumes."""
    return Path(
        freeze_path if freeze_path is not None else DEFAULT_DESIGN_FREEZE_PATH
    ).expanduser().resolve()


def _code_revision_record() -> dict:
    """Return the source revision this campaign is being generated at."""
    revision = revision_provenance()
    return {
        "git_hash": revision["git_hash"],
        "git_dirty": revision["git_dirty"],
        "sha256": revision_digest(revision),
    }


def _manifest_bytes(manifest: dict) -> bytes:
    """Render the manifest to its canonical bytes."""
    return yaml.safe_dump(manifest, sort_keys=True).encode("utf-8")

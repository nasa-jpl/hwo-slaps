#!/usr/bin/env python
"""Generate the frozen Fisher PSF knowledge-error campaign."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import uuid

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT/"src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT/"src"))
if str(REPO_ROOT/"scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT/"scripts"))

from run_nonlinear_validation import (  # noqa: E402
    PSF_KNOWLEDGE_DIRECTION_SPAWN_KEY,
    derive_direction_seed,
    system_index,
)
from run_psf_knowledge_map import (  # noqa: E402
    production_rung_reference,
    select_knowledge_rungs,
)

LADDER_ARTIFACT_NAME = "ladder_result.npz"


def _load_yaml(path: Path) -> dict:
    """Load one YAML mapping from a campaign or staged configuration."""
    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict):
        raise ValueError(f"{path} must contain a mapping")
    return document


def _load_npz(path: Path) -> dict:
    """Load one NPZ artifact into independent arrays."""
    with np.load(path, allow_pickle=False) as stored:
        return {
            name: np.array(stored[name], copy=True)
            for name in stored.files
        }


def _scalar(record: dict, name: str):
    """Return one scalar value from a loaded NPZ record."""
    if name not in record:
        raise ValueError(f"Ladder artifact is missing {name}")
    value = np.asarray(record[name])
    if value.ndim != 0:
        raise ValueError(f"Ladder artifact member {name} is not scalar")
    return value.item()


def _file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one ladder artifact or manifest."""
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _restamp_config(
    staged: dict,
    destination: Path,
    revision: dict,
    revision_sha256: str,
    config_hash,
) -> tuple[str, dict]:
    """Write one restamped configuration and return its new hash."""
    original_revision = copy.deepcopy(staged["stage0"]["code_revision"])
    staged["stage0"]["code_revision"] = {
        "git_hash": revision["git_hash"],
        "git_dirty": revision["git_dirty"],
        "sha256": revision_sha256,
    }
    destination.write_text(
        yaml.safe_dump(staged, sort_keys=False),
        encoding="utf-8",
    )
    return config_hash(staged), original_revision


def _job_from_ladder(
    job: dict,
    selected_run: Path,
    freeze: dict,
    config_out: Path,
    revision: dict,
    revision_sha256: str,
    config_hash,
) -> dict:
    """Validate and restamp one selected production ladder member."""
    job_id = str(job.get("job_id", ""))
    if not job_id:
        raise ValueError("Selected ladder manifest contains an empty job_id")
    overrides = job.get("overrides")
    if not isinstance(overrides, dict):
        raise ValueError(f"{job_id} is missing overrides")
    ladder_override = overrides.get("ladder")
    stage0_override = overrides.get("stage0")
    if (
        not isinstance(ladder_override, dict)
        or not isinstance(stage0_override, dict)
    ):
        raise ValueError(
            f"{job_id} is missing overrides.ladder or overrides.stage0"
        )
    if ladder_override.get("tier") != "selected":
        raise ValueError(f"{job_id} does not declare ladder tier 'selected'")
    config_path = selected_run/"configs"/f"{job_id}.yaml"
    artifact_path = selected_run/"outputs"/job_id/LADDER_ARTIFACT_NAME
    if not config_path.is_file():
        raise ValueError(f"Missing selected ladder configuration {config_path}")
    if not artifact_path.is_file():
        raise ValueError(f"Missing selected ladder artifact {artifact_path}")
    staged = _load_yaml(config_path)
    if staged.get("run_name") != job_id:
        raise ValueError(
            f"Configuration {config_path} run_name {staged.get('run_name')!r} "
            f"does not match job_id {job_id!r}"
        )
    artifact = _load_npz(artifact_path)
    if str(_scalar(artifact, "system_id")) != job_id:
        raise ValueError(
            f"Ladder artifact {artifact_path} system_id does not match "
            f"{job_id!r}"
        )
    if str(_scalar(artifact, "tier")) != "selected":
        raise ValueError(f"Ladder artifact {artifact_path} is not selected")
    expected_uuid = str(
        freeze["psf_knowledge_error"]["member_set"]["source_campaign_uuid"]
    )
    if str(_scalar(artifact, "campaign_uuid")) != expected_uuid:
        raise ValueError(
            f"Ladder artifact {artifact_path} campaign uuid does not match "
            f"{expected_uuid}"
        )
    if str(_scalar(artifact, "psf_state")) != "science35":
        raise ValueError(f"Ladder artifact {artifact_path} is not science35")
    shape = [
        int(value)
        for value in np.asarray(
            artifact["psf_kernel_shape_native"], dtype=int
        )
    ]
    if shape != [999, 999]:
        raise ValueError(
            f"Ladder artifact {artifact_path} kernel shape {shape!r} is not "
            "[999, 999]"
        )
    if str(_scalar(artifact, "stop_reason")) != "m50_reached":
        raise ValueError(
            f"Ladder artifact {artifact_path} stop_reason is not "
            "'m50_reached'"
        )
    expected_golden = bool(ladder_override.get("golden"))
    if bool(_scalar(artifact, "golden")) != expected_golden:
        raise ValueError(
            f"Ladder artifact {artifact_path} golden flag does not match "
            f"the selected manifest job {expected_golden!r}"
        )
    source_template = str(stage0_override.get("source_template", ""))
    if staged["stage0"].get("source_template") != source_template:
        raise ValueError(
            f"{job_id} source template does not match its selected manifest"
        )
    selected_rungs = select_knowledge_rungs(artifact)
    rung_records = []
    for rung in selected_rungs:
        reference = production_rung_reference(artifact, rung["logm"])
        rung_records.append({
            "logm": float(rung["logm"]),
            "classes": list(rung["classes"]),
            "production_cells": int(reference["production_cells"]),
            "production_q_max": float(reference["production_q_max"]),
            "production_detectable_area_arcsec2": float(
                reference["production_detectable_area_arcsec2"]
            ),
        })
    grid_shape = list(staged["lensing"]["grid"]["shape"])
    if len(grid_shape) != 2 or grid_shape[0] != grid_shape[1]:
        raise ValueError(f"{job_id} lensing.grid.shape must be square")
    global_seed = staged.get("global_seed")
    if isinstance(global_seed, bool) or not isinstance(global_seed, int):
        raise ValueError(f"{job_id} staged global_seed must be an int")
    original_hash = config_hash(staged)
    ladder_config_hash = str(_scalar(artifact, "config_hash"))
    if original_hash != ladder_config_hash:
        raise ValueError(
            f"{job_id} staged configuration hash {original_hash} does not "
            f"match ladder artifact hash {ladder_config_hash}"
        )
    restamped_path = config_out/f"{job_id}.yaml"
    restamped_hash, original_revision = _restamp_config(
        staged,
        restamped_path,
        revision,
        revision_sha256,
        config_hash,
    )
    index = system_index(job_id)
    entropy = int(freeze["seeds"]["entropy"])
    direction_seeds = {
        str(direction): derive_direction_seed(entropy, direction, index)
        for direction in freeze["psf_knowledge_error"][
            "residual_model"
        ]["direction_indices"]
    }
    return {
        "run_name": job_id,
        "system_id": str(_scalar(artifact, "system_id")),
        "template": source_template,
        "golden": expected_golden,
        "tier": "selected",
        "config": str(config_path),
        "restamped_config": str(restamped_path),
        "original_code_revision": original_revision,
        "original_config_hash": original_hash,
        "restamped_config_hash": restamped_hash,
        "staged_global_seed": int(global_seed),
        "image_side_px": int(grid_shape[0]),
        "ladder_artifact": str(artifact_path),
        "ladder_artifact_sha256": _file_sha256(artifact_path),
        "ladder_campaign_uuid": str(_scalar(artifact, "campaign_uuid")),
        "ladder_config_hash": ladder_config_hash,
        "psf_kernel_sha256": str(_scalar(artifact, "psf_kernel_sha256")),
        "output_dir": str(Path(config_out).parent/"outputs"/job_id),
        "rungs": rung_records,
        "seeds": direction_seeds,
    }


def _map_queue_lines(
    jobs: list[dict],
    deltas: list[float],
    direction_indices: list[int],
) -> list[str]:
    """Build the ordered Fisher map queue."""
    lines = []
    ordered_jobs = sorted(
        jobs,
        key=lambda job: (-job["image_side_px"], job["run_name"]),
    )
    for job in ordered_jobs:
        for delta in deltas:
            directions = [0] if delta == 0.0 else direction_indices
            for direction in directions:
                lines.append(
                    f"{job['restamped_config']} {job['ladder_artifact']} "
                    f"{delta:g} {direction} {job['output_dir']}"
                )
    return lines


def _smoke_queue_lines(
    jobs: list[dict],
    smoke_rule: dict,
) -> tuple[list[str], list[str]]:
    """Build the two-member Fisher smoke queue and its run-name list."""
    by_size = sorted(
        jobs,
        key=lambda job: (-job["image_side_px"], job["run_name"]),
    )
    selected = {
        "smallest_image": min(
            jobs,
            key=lambda job: (job["image_side_px"], job["run_name"]),
        ),
        "largest_image": by_size[0],
    }
    lines = []
    run_names = []
    member_names = sorted(
        smoke_rule["members"],
        key=lambda name: (
            -selected[name]["image_side_px"],
            selected[name]["run_name"],
        ),
    )
    for member_name in member_names:
        job = selected[member_name]
        if job["run_name"] not in run_names:
            run_names.append(job["run_name"])
        for delta in smoke_rule["deltas"]:
            direction = (
                0 if float(delta) == 0.0 else int(smoke_rule["direction"])
            )
            lines.append(
                f"{job['restamped_config']} {job['ladder_artifact']} "
                f"{float(delta):g} {direction} {job['output_dir']}"
            )
    return lines, run_names


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign",
        required=True,
        help="Declared PSF knowledge Fisher campaign name",
    )
    parser.add_argument(
        "--selected-run",
        required=True,
        help="ladder_selected_v1/run directory",
    )
    parser.add_argument("campaign_dir", help="Campaign directory to create")
    return parser


def main(argv=None) -> None:
    """Generate the PSF knowledge Fisher campaign and its queues."""
    args = _build_parser().parse_args(argv)
    from hwoslaps.campaign.design_freeze import load_design_freeze
    from hwoslaps.provenance import (
        config_hash,
        revision_digest,
        revision_provenance,
    )

    freeze = load_design_freeze()
    knowledge = freeze["psf_knowledge_error"]
    campaigns = knowledge["campaigns"]
    if args.campaign not in campaigns:
        raise ValueError(
            f"Campaign {args.campaign!r} is not declared under "
            "psf_knowledge_error"
        )
    if args.campaign != "psf_knowledge_fisher_v1":
        raise ValueError(
            "PSF knowledge Fisher generator requires campaign "
            f"'psf_knowledge_fisher_v1', got {args.campaign!r}"
        )
    campaign = campaigns[args.campaign]
    selected_run = Path(args.selected_run)
    source_manifest_path = selected_run.parent/"manifest.yaml"
    if not source_manifest_path.is_file():
        raise ValueError(f"Missing selected ladder manifest {source_manifest_path}")
    source_manifest = _load_yaml(source_manifest_path)
    source_campaign = source_manifest.get("campaign")
    jobs = (
        source_campaign.get("jobs")
        if isinstance(source_campaign, dict)
        else None
    )
    if not isinstance(jobs, list) or len(jobs) != 12:
        raise ValueError(
            f"{source_manifest_path} must carry exactly 12 selected jobs"
        )
    job_ids = [str(job.get("job_id", "")) for job in jobs if isinstance(job, dict)]
    if len(job_ids) != len(set(job_ids)):
        raise ValueError(
            f"{source_manifest_path} selected job_id values are not unique"
        )
    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError(f"{source_manifest_path} contains a malformed job")
        overrides = job.get("overrides")
        ladder = (
            overrides.get("ladder")
            if isinstance(overrides, dict)
            else None
        )
        if not isinstance(ladder, dict) or ladder.get("tier") != "selected":
            raise ValueError(
                "Every selected ladder manifest job must declare "
                "overrides.ladder.tier 'selected'"
            )

    revision = revision_provenance()
    if revision["git_dirty"]:
        raise ValueError(
            "Refusing to generate a campaign from a dirty tree: "
            f"{revision['git_dirty_paths']}"
        )
    revision_sha256 = revision_digest(revision)
    campaign_dir = Path(args.campaign_dir)
    manifest_path = campaign_dir/"manifest.json"
    if manifest_path.exists():
        raise ValueError(
            f"{campaign_dir} already holds a manifest; refusing to regenerate"
        )
    config_out = campaign_dir/"configs"
    config_out.mkdir(parents=True, exist_ok=True)
    manifest_jobs = [
        _job_from_ladder(
            job,
            selected_run,
            freeze,
            config_out,
            revision,
            revision_sha256,
            config_hash,
        )
        for job in jobs
    ]
    residual = knowledge["residual_model"]
    deltas = [float(value) for value in residual["amplitude_rms_nm_rungs"]]
    directions = [int(value) for value in residual["direction_indices"]]
    maps_lines = _map_queue_lines(manifest_jobs, deltas, directions)
    smoke_lines, smoke_run_names = _smoke_queue_lines(
        manifest_jobs,
        campaign["smoke_rule"],
    )
    if len(maps_lines) != 588:
        raise ValueError(
            f"PSF knowledge map queue has {len(maps_lines)} lines, expected 588"
        )
    if len(smoke_lines) != 4:
        raise ValueError(
            f"PSF knowledge smoke queue has {len(smoke_lines)} lines, "
            "expected 4"
        )
    campaign_dir.mkdir(parents=True, exist_ok=True)
    (campaign_dir/"maps_queue.txt").write_text(
        "\n".join(maps_lines) + "\n",
        encoding="utf-8",
    )
    (campaign_dir/"smokes_queue.txt").write_text(
        "\n".join(smoke_lines) + "\n",
        encoding="utf-8",
    )
    campaign_echo = copy.deepcopy(campaign)
    manifest = {
        "schema_version": 1,
        "name": args.campaign,
        "campaign_uuid": str(uuid.uuid4()),
        "design_freeze": {
            "path": "configs/design/design_freeze_v1.yaml",
            "version": freeze["freeze"]["version"],
            "protocol_block": "psf_knowledge_error",
        },
        "campaign": campaign_echo,
        "block": copy.deepcopy(knowledge),
        "code_revision": {
            "git_hash": revision["git_hash"],
            "git_dirty": revision["git_dirty"],
            "sha256": revision_sha256,
        },
        "seed_declaration": {
            "stream": "psf_knowledge_direction",
            "entropy": int(freeze["seeds"]["entropy"]),
            "spawn_key": [
                PSF_KNOWLEDGE_DIRECTION_SPAWN_KEY,
                "direction_index",
                "system_index",
            ],
        },
        "ladder_source": {
            "run": str(selected_run),
            "campaign_uuid": str(
                knowledge["member_set"]["source_campaign_uuid"]
            ),
            "manifest_sha256": _file_sha256(source_manifest_path),
        },
        "n_systems": len(manifest_jobs),
        "n_jobs": len(maps_lines),
        "n_maps": sum(len(job["rungs"]) for job in manifest_jobs)*49,
        "smoke_run_names": smoke_run_names,
        "jobs": manifest_jobs,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Campaign staged: {campaign_dir}\n"
        f"  {len(manifest_jobs)} systems, {len(maps_lines)} map jobs, "
        f"{manifest['n_maps']} maps, {len(smoke_lines)} smokes, code "
        f"revision {revision_sha256[:16]} (git {revision['git_hash'][:7]}), "
        f"campaign uuid {manifest['campaign_uuid']}"
    )


if __name__ == "__main__":
    main()

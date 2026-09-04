#!/usr/bin/env python
"""Generate a declared nonlinear-validation extension campaign.

Reads the ladder campaigns named by the v4 freeze, applies the selected
member-set rule and stages one restamped configuration copy per unique
system. Staged ladder configurations pin
``stage0.code_revision`` and every runner fails closed against the
executing tree, so the copies are restamped to THIS tree's revision and
the original revision travels in the manifest, per the freeze's
``code_revision_policy``.

Arm eligibility follows the freeze's arm table and the campaign's
declared arm subset. ``all`` arms run on every system, ``non_censored``
arms skip right-censored members, and ``golden`` arms run only on
golden-flagged members.

The campaign directory receives:

- ``configs/<run_name>.yaml``: restamped staged configuration copies.
- ``manifest.json``: identity, the declared protocol echo, and the full
  job table with every eligible arm's derived sampler seed.
- ``positions_queue.txt``: one extraction job per line, largest first,
  when the campaign declares ``positions_source: self``.
- ``smokes_queue.txt``: the campaign's declared smoke rule.
- ``fits_queue.txt``: every eligible fit arm, largest first.

Queue lines are ``<config> <ladder_artifact> <output_dir>`` for
positions and ``<config> <positions_artifact> <arm> <output_dir>`` for
smokes and fits, consumed by ``nonlinear_validation_dispatch.sh``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
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
    derive_direction_seed,
    derive_noise_seed,
    derive_sampler_seed,
    system_index,
)

LADDER_ARTIFACT_NAME = "ladder_result.npz"
POSITIONS_ARTIFACT_NAME = "injection_position.json"


def sample_members(
    parent_run: Path | None = None,
    selected_run: Path | None = None,
    validation_run: Path | None = None,
    mode: str = "production59",
    expected_tier_counts: dict | None = None,
    expected_n_systems: int | None = None,
):
    """Apply one declared member-set rule to staged ladder campaigns.

    Parameters
    ----------
    parent_run : `pathlib.Path`, optional
        The ladder_parent_v1 ``run`` directory.
    selected_run : `pathlib.Path`, optional
        The ladder_selected_v1 ``run`` directory.
    validation_run : `pathlib.Path`, optional
        The ladder_validation_v1 ``run`` directory.
    mode : `str`, optional
        Declared member-set mode, either ``production59`` or
        ``validation100``.
    expected_tier_counts : `dict`, optional
        Expected parent and selected tier counts for ``production59``.
    expected_n_systems : `int`, optional
        Expected number of unique systems.

    Returns
    -------
    members : `list` [`dict`]
        One entry per unique system: the bare ``sysNNNN`` identifier,
        the owning tier, the tiers the system reports in, the staged
        configuration and artifact paths, and the censored and golden
        flags read from the tier artifacts.

    Raises
    ------
    ValueError
        Raised when a source manifest, member artifact or tier declaration
        disagrees with the selected member-set rule.
    """
    if mode not in ("production59", "validation100", "selected12"):
        raise ValueError(f"Unknown member-set mode {mode!r}")
    if expected_n_systems is None:
        expected_n_systems = {
            "production59": 59,
            "validation100": 100,
            "selected12": 12,
        }[mode]
    if mode == "selected12":
        production_members = sample_members(
            parent_run=parent_run,
            selected_run=selected_run,
            mode="production59",
            expected_tier_counts=expected_tier_counts,
            expected_n_systems=59,
        )
        selected_members = [
            member
            for member in production_members
            if "selected" in member["report_tiers"]
        ]
        if len(selected_members) != expected_n_systems:
            raise ValueError(
                f"selected12 retains {len(selected_members)} systems, "
                f"expected {expected_n_systems}"
            )
        return selected_members
    if mode == "validation100":
        if validation_run is None:
            raise ValueError("validation100 requires validation_run")
        manifest_path = Path(validation_run).parent/"manifest.yaml"
        if not manifest_path.is_file():
            raise ValueError(f"Missing ladder validation manifest {manifest_path}")
        with manifest_path.open(encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
        campaign = document.get("campaign") if isinstance(document, dict) else None
        jobs = campaign.get("jobs") if isinstance(campaign, dict) else None
        if not isinstance(jobs, list):
            raise ValueError(
                f"{manifest_path} campaign.jobs must be a list"
            )
        selected = []
        for job in jobs:
            if not isinstance(job, dict):
                raise ValueError(f"{manifest_path} contains a malformed job")
            overrides = job.get("overrides")
            ladder = overrides.get("ladder") if isinstance(overrides, dict) else None
            if isinstance(ladder, dict) and ladder.get(
                "validation_sample_member"
            ) is True:
                selected.append(job)
        if len(selected) != expected_n_systems:
            raise ValueError(
                f"{manifest_path} selects {len(selected)} validation members, "
                f"expected {expected_n_systems}"
            )
        members = {}
        for job in selected:
            run_name = str(job.get("job_id", ""))
            overrides = job["overrides"]
            stage0 = overrides.get("stage0")
            if not isinstance(stage0, dict):
                raise ValueError(f"{run_name} is missing overrides.stage0")
            system_id_value = str(stage0.get("system_id", ""))
            if not run_name or not system_id_value:
                raise ValueError(
                    f"Validation member has empty run_name or system_id: {job!r}"
                )
            config_path = Path(validation_run)/"configs"/f"{run_name}.yaml"
            artifact = (
                Path(validation_run)/"outputs"/run_name/LADDER_ARTIFACT_NAME
            )
            if not config_path.is_file():
                raise ValueError(f"Missing ladder configuration {config_path}")
            if not artifact.is_file():
                raise ValueError(f"Missing ladder artifact {artifact}")
            with config_path.open(encoding="utf-8") as stream:
                config = yaml.safe_load(stream)
            if not isinstance(config, dict):
                raise ValueError(f"Configuration {config_path} must be a mapping")
            if config.get("run_name") != run_name:
                raise ValueError(
                    f"Configuration {config_path} run_name {config.get('run_name')!r} "
                    f"does not match job_id {run_name!r}"
                )
            with np.load(artifact, allow_pickle=False) as record:
                for key in ("campaign_uuid", "config_hash"):
                    if key not in record.files:
                        raise ValueError(
                            f"Ladder artifact {artifact} is missing {key}"
                        )
                artifact_system_id = str(record["system_id"])
                if artifact_system_id != run_name:
                    raise ValueError(
                        f"Ladder artifact {artifact} system_id "
                        f"{artifact_system_id!r} does not match run_name "
                        f"{run_name!r}"
                    )
                artifact_tier = str(record["tier"])
                if artifact_tier != "validation":
                    raise ValueError(
                        f"Ladder artifact {artifact} tier {artifact_tier!r} "
                        "is not 'validation'"
                    )
                golden = bool(record["golden"])
                if golden:
                    raise ValueError(
                        f"Validation member {run_name} is unexpectedly golden"
                    )
                censored = bool(math.isnan(float(record["m_best"])))
                ladder_campaign_uuid = str(record["campaign_uuid"])
                ladder_config_hash = str(record["config_hash"])
            if system_id_value in members:
                raise ValueError(
                    f"Validation member system_id {system_id_value!r} is duplicated"
                )
            members[system_id_value] = {
                "system_id": system_id_value,
                "run_name": run_name,
                "tier": "validation",
                "report_tiers": ["validation"],
                "config": str(config_path),
                "ladder_artifact": str(artifact),
                "ladder_campaign_uuid": ladder_campaign_uuid,
                "ladder_config_hash": ladder_config_hash,
                "censored": censored,
                "golden": False,
            }
        ordered = [members[key] for key in sorted(members)]
        if len(ordered) != expected_n_systems:
            raise ValueError(
                f"Validation member set has {len(ordered)} unique systems, "
                f"expected {expected_n_systems}"
            )
        return ordered

    if parent_run is None or selected_run is None:
        raise ValueError("production59 requires parent_run and selected_run")
    tier_counts = expected_tier_counts or {"parent": 48, "selected": 12}
    members = {}
    for tier, run_dir, expected in (
        ("parent", parent_run, tier_counts["parent"]),
        ("selected", selected_run, tier_counts["selected"]),
    ):
        config_dir = run_dir/"configs"
        config_paths = sorted(config_dir.glob(f"ladder_{tier}_sys*.yaml"))
        if len(config_paths) != expected:
            raise ValueError(
                f"{config_dir} holds {len(config_paths)} member "
                f"configurations, expected {expected}"
            )
        for config_path in config_paths:
            run_name = config_path.stem
            bare_id = "sys" + str(system_index(run_name)).zfill(4)
            artifact = run_dir/"outputs"/run_name/LADDER_ARTIFACT_NAME
            if not artifact.is_file():
                raise ValueError(f"Missing ladder artifact {artifact}")
            with np.load(artifact, allow_pickle=False) as record:
                for key in ("campaign_uuid", "config_hash"):
                    if key not in record.files:
                        raise ValueError(
                            f"Ladder artifact {artifact} is missing {key}"
                        )
                golden = bool(record["golden"])
                censored = bool(math.isnan(float(record["m_best"])))
                ladder_campaign_uuid = str(record["campaign_uuid"])
                ladder_config_hash = str(record["config_hash"])
            if bare_id in members:
                members[bare_id]["report_tiers"].append(tier)
                members[bare_id]["golden"] = (
                    members[bare_id]["golden"] or golden
                )
                continue
            members[bare_id] = {
                "system_id": bare_id,
                "run_name": run_name,
                "tier": tier,
                "report_tiers": [tier],
                "config": str(config_path),
                "ladder_artifact": str(artifact),
                "ladder_campaign_uuid": ladder_campaign_uuid,
                "ladder_config_hash": ladder_config_hash,
                "censored": censored,
                "golden": golden,
            }
    ordered = [members[key] for key in sorted(members)]
    overlaps = [m for m in ordered if len(m["report_tiers"]) > 1]
    if len(ordered) != expected_n_systems or len(overlaps) != 1:
        raise ValueError(
            f"Sample rule expects {expected_n_systems} unique systems with "
            "1 overlap, got "
            f"{len(ordered)} with {len(overlaps)}"
        )
    return ordered


def eligible_arms(member: dict, arms: dict, campaign_arms=None):
    """Return the declared arms one member is eligible for.

    Parameters
    ----------
    member : `dict`
        A `sample_members` entry with its censored and golden flags.
    arms : `dict`
        The freeze protocol's arm table.
    campaign_arms : `list` [`str`], optional
        Campaign-specific subset of the declared arms.

    Returns
    -------
    names : `list` [`str`]
        Eligible arm names in declared-index order.

    Raises
    ------
    ValueError
        Raised for an unknown sample rule.
    """
    allowed = None if campaign_arms is None else set(campaign_arms)
    if allowed is not None:
        unknown = sorted(allowed.difference(arms))
        if unknown:
            raise ValueError(
                f"Campaign names undeclared arms: {unknown}"
            )
    names = []
    for name, declaration in sorted(
        arms.items(), key=lambda item: item[1]["arm_index"]
    ):
        sample = declaration["sample"]
        if sample == "all":
            eligible = True
        elif sample == "non_censored":
            eligible = not member["censored"]
        elif sample == "golden":
            eligible = member["golden"]
        else:
            raise ValueError(f"Unknown arm sample rule {sample!r}")
        if eligible and (allowed is None or name in allowed):
            names.append(name)
    return names


def image_side_px(config: dict) -> int:
    """Return a member's lensing grid side in pixels for LPT ordering."""
    return int(config["lensing"]["grid"]["shape"][0])


def smoke_jobs(jobs, member_rule):
    """Select the freeze's smoke-gate members.

    Parameters
    ----------
    jobs : `list` [`dict`]
        Job table entries carrying ``template`` and ``image_side_px``.
    member_rule : `str`
        Smoke member enum declared by the campaign.

    Returns
    -------
    smokes : `list` [`dict`]
        The selected smallest-image member of each source template, in
        template order.
    """
    if member_rule not in (
        "smallest_image_per_template",
        "smallest_image_non_censored_per_template",
        "smallest_image_golden",
    ):
        raise ValueError(f"Unknown smoke member rule {member_rule!r}")
    if member_rule == "smallest_image_golden":
        candidates = [job for job in jobs if job.get("golden", False)]
        if not candidates:
            raise ValueError(
                "Smoke member rule 'smallest_image_golden' has no golden member"
            )
        return [
            min(candidates, key=lambda job: (job["image_side_px"], job["run_name"]))
        ]
    templates = {job["template"] for job in jobs}
    candidates = jobs
    if member_rule == "smallest_image_non_censored_per_template":
        candidates = [job for job in jobs if not job.get("censored", False)]
    by_template = {}
    for job in candidates:
        current = by_template.get(job["template"])
        if current is None or (
            (job["image_side_px"], job["run_name"])
            < (current["image_side_px"], current["run_name"])
        ):
            by_template[job["template"]] = job
    missing = sorted(templates.difference(by_template))
    if missing:
        raise ValueError(
            f"Smoke member rule {member_rule!r} has no member for templates {missing}"
        )
    return [by_template[key] for key in sorted(by_template)]


def _validate_source_files(
    source_dir: Path,
    declaration: dict,
    label: str,
) -> dict:
    """Validate a declared source campaign and its harvested review files.

    Parameters
    ----------
    source_dir : `pathlib.Path`
        Campaign directory containing ``manifest.json`` and ``harvest``.
    declaration : `dict`
        Frozen source declaration with its campaign UUID and digests.
    label : `str`
        Source label used in failure messages.

    Returns
    -------
    echo : `dict`
        The validated source declaration with resolved harvest and review
        paths and their observed digests.

    Raises
    ------
    ValueError
        Raised when a source file, UUID, digest or CLEAN review is absent
        or disagrees with the declaration.
    """
    from hwoslaps.campaign.design_freeze import file_sha256

    source_dir = Path(source_dir)
    for key in (
        "campaign",
        "campaign_uuid",
        "harvest",
        "harvest_sha256",
        "review_sha256",
    ):
        if key not in declaration:
            raise ValueError(f"{label} is missing {key}")
    source_manifest_path = source_dir/"manifest.json"
    if not source_manifest_path.is_file():
        raise ValueError(f"Missing {label} manifest {source_manifest_path}")
    with source_manifest_path.open(encoding="utf-8") as stream:
        source_manifest = json.load(stream)
    if not isinstance(source_manifest, dict):
        raise ValueError(f"{label} manifest {source_manifest_path} must be a mapping")
    expected_uuid = declaration["campaign_uuid"]
    if source_manifest.get("campaign_uuid") != expected_uuid:
        raise ValueError(
            f"{label} manifest campaign uuid "
            f"{source_manifest.get('campaign_uuid')!r} does not match "
            f"declared {expected_uuid!r}"
        )
    harvest_path = source_dir/"harvest"/"harvest.json"
    review_path = source_dir/"harvest"/"review.json"
    if not harvest_path.is_file():
        raise ValueError(f"Missing {label} harvest {harvest_path}")
    if not review_path.is_file():
        raise ValueError(f"Missing {label} review {review_path}")
    harvest_sha256 = file_sha256(harvest_path)
    if harvest_sha256 != declaration["harvest_sha256"]:
        raise ValueError(
            f"{label} harvest sha256 {harvest_sha256} does not match "
            f"declared {declaration['harvest_sha256']}"
        )
    review_sha256 = file_sha256(review_path)
    if review_sha256 != declaration["review_sha256"]:
        raise ValueError(
            f"{label} review sha256 {review_sha256} does not match "
            f"declared {declaration['review_sha256']}"
        )
    with review_path.open(encoding="utf-8") as stream:
        review = json.load(stream)
    if not isinstance(review, dict) or review.get("integrity") != "CLEAN":
        raise ValueError(
            f"{label} review {review_path} integrity is not CLEAN"
        )
    with harvest_path.open(encoding="utf-8") as stream:
        harvest = json.load(stream)
    if not isinstance(harvest, dict):
        raise ValueError(f"{label} harvest {harvest_path} must be a mapping")
    if harvest.get("campaign_uuid") != expected_uuid:
        raise ValueError(
            f"{label} harvest campaign uuid "
            f"{harvest.get('campaign_uuid')!r} does not match declared "
            f"{expected_uuid!r}"
        )
    return {
        "campaign": declaration["campaign"],
        "campaign_uuid": expected_uuid,
        "harvest": str(harvest_path.resolve()),
        "harvest_sha256": harvest_sha256,
        "review": str(review_path.resolve()),
        "review_sha256": review_sha256,
        "review_integrity": "CLEAN",
    }


def _validate_reused_positions(
    source_dir: Path,
    members: list[dict],
    expected_uuid: str,
    kernel_shape: list[int],
    require_harvest: bool,
) -> tuple[Path | None, dict[str, str], dict[str, dict]]:
    """Validate and index position artifacts reused from a prior campaign.

    Parameters
    ----------
    source_dir : `pathlib.Path`
        Prior nonlinear campaign directory.
    members : `list` [`dict`]
        Current campaign member entries.
    expected_uuid : `str`
        Frozen campaign UUID that owns the reused positions.
    kernel_shape : `list` [`int`]
        Frozen nonlinear fit-kernel shape.
    require_harvest : `bool`
        Whether the prior campaign's harvest must exist (it is the
        replicate-zero source of a null campaign).

    Returns
    -------
    harvest_path : `pathlib.Path` or `None`
        Existing prior-campaign harvest path when required.
    positions : `dict` [`str`, `str`]
        Position artifact path keyed by current run name.
    source_jobs : `dict` [`str`, `dict`]
        The prior campaign's manifest job entries keyed by run name, so
        the caller can bind each member's staged configuration to the
        one the positions were extracted from.

    Raises
    ------
    ValueError
        Raised when the prior campaign or any reused position artifact is
        missing or carries a different identity.
    """
    manifest_path = source_dir/"manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing positions source manifest {manifest_path}")
    with manifest_path.open(encoding="utf-8") as stream:
        source_manifest = json.load(stream)
    if not isinstance(source_manifest, dict):
        raise ValueError(f"Positions source manifest {manifest_path} must be a mapping")
    if source_manifest.get("campaign_uuid") != expected_uuid:
        raise ValueError(
            f"Positions source campaign uuid "
            f"{source_manifest.get('campaign_uuid')!r} does not match "
            f"the frozen {expected_uuid!r}"
        )
    source_jobs = {}
    for job in source_manifest.get("jobs", []):
        if not isinstance(job, dict) or "run_name" not in job:
            raise ValueError(
                f"Positions source manifest {manifest_path} has a malformed job"
            )
        source_jobs[str(job["run_name"])] = job

    harvest_path = None
    if require_harvest:
        harvest_path = source_dir/"harvest"/"harvest.json"
        if not harvest_path.is_file():
            raise ValueError(f"Missing replicate-zero harvest {harvest_path}")
        with harvest_path.open(encoding="utf-8") as stream:
            harvest = json.load(stream)
        if not isinstance(harvest, dict):
            raise ValueError(
                f"Replicate-zero harvest {harvest_path} must be a mapping"
            )
        if harvest.get("campaign_uuid") != expected_uuid:
            raise ValueError(
                f"Replicate-zero harvest campaign uuid "
                f"{harvest.get('campaign_uuid')!r} does not match the frozen "
                f"{expected_uuid!r}"
            )

    positions = {}
    for member in members:
        path = (
            source_dir/"outputs"/member["run_name"]/POSITIONS_ARTIFACT_NAME
        )
        if not path.is_file():
            raise ValueError(f"Missing reused positions artifact {path}")
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict):
            raise ValueError(f"Reused positions artifact {path} must be a mapping")
        if payload.get("system_id") != member["run_name"]:
            raise ValueError(
                f"Reused positions artifact {path} belongs to "
                f"{payload.get('system_id')!r}, expected {member['run_name']!r}"
            )
        source_kernel = payload.get("fit_kernel_shape_native")
        if source_kernel != list(kernel_shape):
            raise ValueError(
                f"Reused positions artifact {path} has kernel "
                f"{source_kernel!r}, expected "
                f"{kernel_shape!r}"
            )
        if payload.get("ladder_campaign_uuid") != member[
            "ladder_campaign_uuid"
        ]:
            raise ValueError(
                f"Reused positions artifact {path} ladder campaign uuid "
                f"{payload.get('ladder_campaign_uuid')!r} does not match "
                f"member {member['ladder_campaign_uuid']!r}"
            )
        if payload.get("ladder_config_hash") != member["ladder_config_hash"]:
            raise ValueError(
                f"Reused positions artifact {path} ladder config hash "
                f"{payload.get('ladder_config_hash')!r} does not match "
                f"member {member['ladder_config_hash']!r}"
            )
        if payload.get("censored") != member["censored"]:
            raise ValueError(
                f"Reused positions artifact {path} censored flag "
                f"{payload.get('censored')!r} does not match member "
                f"{member['censored']!r}"
            )
        rungs = payload.get("rungs")
        if (
            not isinstance(rungs, dict)
            or "top" not in rungs
            or not isinstance(rungs["top"], dict)
        ):
            raise ValueError(f"Reused positions artifact {path} has no top rung")
        if (
            not member["censored"]
            and (
                "below" not in rungs
                or not isinstance(rungs["below"], dict)
            )
        ):
            raise ValueError(
                f"Reused positions artifact {path} has no below rung for a "
                "non-censored member"
            )
        if member["run_name"] not in source_jobs:
            raise ValueError(
                f"Positions source manifest has no job for {member['run_name']}"
            )
        positions[member["run_name"]] = str(path)
    return harvest_path, positions, source_jobs


def main(argv=None) -> None:
    """Stage the selected campaign directory, manifest and queues."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, help="Freeze campaign name")
    parser.add_argument(
        "campaign_dir", help="Campaign directory to create"
    )
    parser.add_argument("--parent-run", help="ladder_parent_v1 run directory")
    parser.add_argument("--selected-run", help="ladder_selected_v1 run directory")
    parser.add_argument("--validation-run", help="ladder_validation_v1 run directory")
    parser.add_argument(
        "--positions-source-dir",
        help="Prior nonlinear campaign directory supplying positions",
    )
    parser.add_argument(
        "--pooled-source-dir",
        help="Prior nonlinear campaign directory supplying pooled rows",
    )
    parser.add_argument(
        "--null-source-dir",
        help="Prior nonlinear null campaign directory supplying null rows",
    )
    args = parser.parse_args(argv)

    from hwoslaps.campaign.design_freeze import load_design_freeze
    from hwoslaps.provenance import (
        config_hash,
        revision_digest,
        revision_provenance,
    )

    freeze = load_design_freeze()
    protocol = freeze["nonlinear_validation"]
    campaigns = protocol["campaigns"]
    if args.campaign not in campaigns:
        raise ValueError(
            f"Campaign {args.campaign!r} is not declared; declared campaigns: "
            f"{sorted(campaigns)}"
        )
    campaign = campaigns[args.campaign]
    member_set_name = campaign["member_set"]
    member_set = protocol["member_sets"][member_set_name]
    if member_set_name in ("production59", "selected12"):
        if args.parent_run is None or args.selected_run is None:
            raise ValueError(
                f"{member_set_name} requires --parent-run and --selected-run"
            )
        if args.validation_run is not None:
            raise ValueError(
                "production59 forbids --validation-run"
            )
        parent_run = Path(args.parent_run)
        selected_run = Path(args.selected_run)
        validation_run = None
    elif member_set_name == "validation100":
        if args.validation_run is None:
            raise ValueError("validation100 requires --validation-run")
        if args.parent_run is not None or args.selected_run is not None:
            raise ValueError(
                "validation100 forbids --parent-run and --selected-run"
            )
        parent_run = None
        selected_run = None
        validation_run = Path(args.validation_run)
    else:
        raise ValueError(f"Unsupported declared member set {member_set_name!r}")

    positions_source = campaign["positions_source"]
    if positions_source == "self":
        if args.positions_source_dir is not None:
            raise ValueError(
                "positions_source self forbids --positions-source-dir"
            )
    elif args.positions_source_dir is None:
        raise ValueError(
            f"positions_source {positions_source!r} requires "
            "--positions-source-dir"
        )

    revision = revision_provenance()
    digest = revision_digest(revision)
    if revision["git_dirty"]:
        raise ValueError(
            "Refusing to generate a campaign from a dirty tree: "
            f"{revision['git_dirty_paths']}"
        )

    campaign_dir = Path(args.campaign_dir)
    if (campaign_dir/"manifest.json").exists():
        raise ValueError(
            f"{campaign_dir} already holds a manifest; refusing to regenerate"
        )

    expected_tier_counts = None
    if member_set_name in ("production59", "selected12"):
        expected_tier_counts = {
            "parent": freeze["strata"]["parent"]["size"],
            "selected": freeze["strata"]["selected"]["size"],
        }
    members = sample_members(
        parent_run,
        selected_run,
        validation_run=validation_run,
        mode=member_set_name,
        expected_tier_counts=expected_tier_counts,
        expected_n_systems=member_set["n_systems"],
    )

    reused_positions = {}
    source_jobs = {}
    replicate_zero_echo = None
    pooled_source_echo = None
    if positions_source != "self":
        expected_positions_uuid = str(campaign["positions_source_campaign_uuid"])
        replicate_zero = campaign.get("replicate_zero_source")
        if replicate_zero is not None:
            replicate_zero_echo = _validate_source_files(
                Path(args.positions_source_dir),
                replicate_zero,
                "replicate_zero_source",
            )
        if replicate_zero is not None and (
            str(replicate_zero["campaign_uuid"]) != expected_positions_uuid
        ):
            raise ValueError(
                "Campaign positions_source_campaign_uuid does not match its "
                "replicate_zero_source campaign_uuid"
            )
        _, reused_positions, source_jobs = (
            _validate_reused_positions(
                Path(args.positions_source_dir),
                members,
                expected_positions_uuid,
                list(protocol["fit"]["kernel_shape_native"]),
                require_harvest=replicate_zero is not None,
            )
        )
        if (campaign_dir/"positions_queue.txt").exists():
            raise ValueError(
                f"{campaign_dir}/positions_queue.txt exists for a reused "
                "positions source"
            )
    elif campaign.get("replicate_zero_source") is not None:
        raise ValueError(
            "replicate_zero_source requires a reused positions source"
        )

    reference_source_echo = None
    reference_source = campaign.get("reference_source")
    if reference_source is None:
        if args.positions_source_dir is not None and positions_source == "self":
            raise ValueError(
                "Campaign without reference_source forbids an extra "
                "positions source for reference rows"
            )
    else:
        if args.positions_source_dir is None:
            raise ValueError(
                "Campaign with reference_source requires "
                "--positions-source-dir"
            )
        if (
            positions_source != "self"
            and str(reference_source["campaign_uuid"])
            != str(campaign["positions_source_campaign_uuid"])
        ):
            raise ValueError(
                "reference_source campaign_uuid must equal the declared "
                "positions source campaign uuid"
            )
        reference_source_echo = _validate_source_files(
            Path(args.positions_source_dir),
            reference_source,
            "reference_source",
        )

    null_source_echo = None
    null_source = campaign.get("null_source")
    if null_source is None:
        if args.null_source_dir is not None:
            raise ValueError(
                "Campaign without null_source forbids --null-source-dir"
            )
    else:
        if args.null_source_dir is None:
            raise ValueError(
                "Campaign with null_source requires --null-source-dir"
            )
        null_source_echo = _validate_source_files(
            Path(args.null_source_dir), null_source, "null_source"
        )

    pooled_source = campaign.get("pooled_source")
    if pooled_source is None:
        if args.pooled_source_dir is not None:
            raise ValueError(
                "Campaign without pooled_source forbids --pooled-source-dir"
            )
    else:
        if args.pooled_source_dir is None:
            raise ValueError(
                "Campaign with pooled_source requires --pooled-source-dir"
            )
        pooled_source_echo = _validate_source_files(
            Path(args.pooled_source_dir), pooled_source, "pooled_source"
        )

    config_out = campaign_dir/"configs"
    config_out.mkdir(parents=True, exist_ok=True)
    arms = protocol["arms"]
    entropy = int(protocol["seeds"]["entropy"])
    jobs = []
    for member in members:
        with open(member["config"], encoding="utf-8") as stream:
            staged = yaml.safe_load(stream)
        if not isinstance(staged, dict):
            raise ValueError(
                f"Staged configuration {member['config']} must be a mapping"
            )
        global_seed = staged.get("global_seed")
        if isinstance(global_seed, bool) or not isinstance(global_seed, int):
            raise ValueError(
                f"{member['run_name']} staged global_seed must be an int"
            )
        original_hash = config_hash(staged)
        if source_jobs:
            source_hash = source_jobs[member["run_name"]].get(
                "original_config_hash"
            )
            if source_hash != original_hash:
                raise ValueError(
                    f"{member['run_name']}: positions were extracted from a "
                    f"staged configuration with hash {source_hash!r}, this "
                    f"member's staged configuration hashes to {original_hash!r}"
                )
        original_revision = dict(staged["stage0"]["code_revision"])
        staged["stage0"]["code_revision"] = {
            "git_hash": revision["git_hash"],
            "git_dirty": revision["git_dirty"],
            "sha256": digest,
        }
        restamped_path = config_out/f"{member['run_name']}.yaml"
        restamped_path.write_text(
            yaml.safe_dump(staged, sort_keys=False), encoding="utf-8"
        )
        index = system_index(member["run_name"])
        arm_names = eligible_arms(member, arms, campaign["arms"])
        job_arms = {}
        for name in arm_names:
            declaration = arms[name]
            job_arms[name] = {
                "arm_index": declaration["arm_index"],
                "sampler_seed": derive_sampler_seed(
                    entropy, index, declaration["arm_index"]
                ),
            }
            if "noise_replicate" in declaration:
                job_arms[name]["noise_seed"] = derive_noise_seed(
                    entropy, declaration["noise_replicate"], index
                )
            if "fit_psf_delta" in declaration:
                job_arms[name]["directions"] = {
                    str(direction): {
                        "seed": derive_direction_seed(
                            entropy, direction, index
                        ),
                    }
                    for direction in declaration["fit_psf_delta"]["directions"]
                }
        jobs.append({
            **member,
            "restamped_config": str(restamped_path),
            "original_code_revision": original_revision,
            "original_config_hash": original_hash,
            "restamped_config_hash": config_hash(staged),
            "staged_global_seed": int(global_seed),
            "template": str(staged["stage0"]["source_template"]),
            "image_side_px": image_side_px(staged),
            "output_dir": str(campaign_dir/"outputs"/member["run_name"]),
            "arms": job_arms,
        })
        reused_path = reused_positions.get(member["run_name"])
        if reused_path is not None:
            jobs[-1]["positions_artifact_sha256"] = hashlib.sha256(
                Path(reused_path).read_bytes()
            ).hexdigest()

    largest_first = sorted(
        jobs, key=lambda job: job["image_side_px"], reverse=True
    )
    positions_lines = [
        f"{job['restamped_config']} {job['ladder_artifact']} "
        f"{job['output_dir']}"
        for job in largest_first
    ]

    def fit_line(job, arm, direction=None):
        """Build one fit-queue line for a job, arm and direction.

        Parameters
        ----------
        job : `dict`
            Manifest job entry.
        arm : `str`
            Declared arm name.
        direction : `int`, optional
            Direction for a PSF knowledge-error arm.

        Returns
        -------
        line : `str`
            Queue line consumed by the nonlinear dispatcher.

        Raises
        ------
        ValueError
            Raised when a direction is missing or supplied for the wrong
            kind of arm.
        """
        position = reused_positions.get(job["run_name"])
        if position is None:
            position = f"{job['output_dir']}/{POSITIONS_ARTIFACT_NAME}"
        line = (
            f"{job['restamped_config']} {position} {arm} "
            f"{job['output_dir']}"
        )
        carries_delta = "fit_psf_delta" in arms[arm]
        if carries_delta and direction is None:
            raise ValueError(f"Delta arm {arm!r} requires a direction")
        if not carries_delta and direction is not None:
            raise ValueError(
                f"Non-delta arm {arm!r} cannot carry a direction"
            )
        if direction is not None:
            line += f" {direction}"
        return line

    smoke_rule = campaign["smoke_rule"]
    smokes = smoke_jobs(jobs, smoke_rule["member"])
    smoke_lines = []
    for job in smokes:
        for arm in smoke_rule["arms"]:
            if arm not in job["arms"]:
                raise ValueError(
                    f"Smoke arm {arm!r} is not eligible for {job['run_name']}"
                )
            declaration = arms[arm]
            if "fit_psf_delta" in declaration:
                directions = smoke_rule["directions"]
                smoke_lines.extend(
                    fit_line(job, arm, direction)
                    for direction in directions
                )
            else:
                smoke_lines.append(fit_line(job, arm))
    fits_lines = []
    for job in largest_first:
        for arm in job["arms"]:
            declaration = arms[arm]
            if "fit_psf_delta" in declaration:
                fits_lines.extend(
                    fit_line(job, arm, direction)
                    for direction in declaration["fit_psf_delta"]["directions"]
                )
            else:
                fits_lines.append(fit_line(job, arm))
    if positions_source == "self":
        (campaign_dir/"positions_queue.txt").write_text(
            "\n".join(positions_lines) + "\n", encoding="utf-8"
        )
    (campaign_dir/"smokes_queue.txt").write_text(
        "\n".join(smoke_lines) + "\n", encoding="utf-8"
    )
    (campaign_dir/"fits_queue.txt").write_text(
        "\n".join(fits_lines) + "\n", encoding="utf-8"
    )

    campaign_echo = copy.deepcopy(campaign)
    if replicate_zero_echo is not None:
        campaign_echo["replicate_zero_source"].update(replicate_zero_echo)
    if pooled_source_echo is not None:
        campaign_echo["pooled_source"].update(pooled_source_echo)
    if reference_source_echo is not None:
        campaign_echo["reference_source"].update(reference_source_echo)
    if null_source_echo is not None:
        campaign_echo["null_source"].update(null_source_echo)
    manifest = {
        "schema_version": 3,
        "name": args.campaign,
        "campaign_uuid": str(uuid.uuid4()),
        "design_freeze": {
            "path": "configs/design/design_freeze_v1.yaml",
            "version": freeze["freeze"]["version"],
            "protocol_block": "nonlinear_validation",
        },
        "campaign": campaign_echo,
        "code_revision": {
            "git_hash": revision["git_hash"],
            "git_dirty": revision["git_dirty"],
            "sha256": digest,
        },
        "seed_declaration": {
            "stream": "sampler",
            "entropy": entropy,
            "spawn_key": [5, "system_index", "arm_index"],
        },
        "noise_seed_declaration": {
            "stream": "null_noise",
            "entropy": entropy,
            "spawn_key": [6, "replicate_index", "system_index"],
            "replicate_zero": "the staged configuration's global_seed",
        },
        "arms": {
            name: dict(declaration) for name, declaration in arms.items()
        },
        "n_systems": len(jobs),
        "n_fit_pairs": len(fits_lines),
        "smoke_run_names": [job["run_name"] for job in smokes],
        "jobs": jobs,
    }
    if any("fit_psf_delta" in arms[name] for name in campaign["arms"]):
        manifest["direction_seed_declaration"] = {
            "stream": "psf_knowledge_direction",
            "entropy": entropy,
            "spawn_key": [7, "direction_index", "system_index"],
        }
    (campaign_dir/"manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Campaign staged: {campaign_dir}\n"
        f"  {len(jobs)} systems, {len(fits_lines)} fit-pair jobs "
        f"({len(smoke_lines)} smokes), code revision {digest[:16]} "
        f"(git {revision['git_hash'][:7]}), campaign uuid "
        f"{manifest['campaign_uuid']}"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Generate the DesignFreeze v3 nonlinear-validation campaign.

Reads the harvested ladder campaigns, applies the declared sample rule
(every parent member plus every selected member, the overlap member kept
once from its parent artifact), and stages one restamped configuration
copy per unique system. Staged ladder configurations pin
``stage0.code_revision`` and every runner fails closed against the
executing tree, so the copies are restamped to THIS tree's revision and
the original revision travels in the manifest, per the freeze's
``code_revision_policy``.

Arm eligibility follows the freeze's arm table: ``all`` arms run on
every system, ``non_censored`` arms skip the right-censored members,
and ``golden`` arms run only on the golden-flagged members of the
selected tier (read from the selected-tier artifacts, so the overlap
member's golden flag is resolved from its selected artifact).

The campaign directory receives:

- ``configs/<run_name>.yaml``: restamped staged configuration copies.
- ``manifest.json``: identity, the declared protocol echo, and the full
  job table with every eligible arm's derived sampler seed.
- ``positions_queue.txt``: one extraction job per line, largest first.
- ``smokes_queue.txt``: the freeze's smoke gate, the asimov_injected
  arm of the smallest-image member of each source template.
- ``fits_queue.txt``: every eligible fit arm, largest first.

Queue lines are ``<config> <ladder_artifact> <output_dir>`` for
positions and ``<config> <positions_artifact> <arm> <output_dir>`` for
smokes and fits, consumed by ``nonlinear_validation_dispatch.sh``.
"""

from __future__ import annotations

import argparse
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
    derive_sampler_seed,
    load_protocol,
    system_index,
)

LADDER_ARTIFACT_NAME = "ladder_result.npz"
POSITIONS_ARTIFACT_NAME = "injection_position.json"


def sample_members(parent_run: Path, selected_run: Path):
    """Apply the declared sample rule to the two harvested ladder tiers.

    Parameters
    ----------
    parent_run : `pathlib.Path`
        The ladder_parent_v1 ``run`` directory.
    selected_run : `pathlib.Path`
        The ladder_selected_v1 ``run`` directory.

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
        Raised when a tier is empty, a member misses its artifact, or
        the tier counts disagree with the declared 48 + 12 sample.
    """
    members = {}
    for tier, run_dir, expected in (
        ("parent", parent_run, 48),
        ("selected", selected_run, 12),
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
            record = np.load(artifact, allow_pickle=False)
            golden = bool(record["golden"])
            censored = bool(math.isnan(float(record["m_best"])))
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
                "censored": censored,
                "golden": golden,
            }
    ordered = [members[key] for key in sorted(members)]
    overlaps = [m for m in ordered if len(m["report_tiers"]) > 1]
    if len(ordered) != 59 or len(overlaps) != 1:
        raise ValueError(
            f"Sample rule expects 59 unique systems with 1 overlap, got "
            f"{len(ordered)} with {len(overlaps)}"
        )
    return ordered


def eligible_arms(member: dict, arms: dict):
    """Return the declared arms one member is eligible for.

    Parameters
    ----------
    member : `dict`
        A `sample_members` entry with its censored and golden flags.
    arms : `dict`
        The freeze protocol's arm table.

    Returns
    -------
    names : `list` [`str`]
        Eligible arm names in declared-index order.

    Raises
    ------
    ValueError
        Raised for an unknown sample rule.
    """
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
        if eligible:
            names.append(name)
    return names


def image_side_px(config: dict) -> int:
    """Return a member's lensing grid side in pixels for LPT ordering."""
    return int(config["lensing"]["grid"]["shape"][0])


def smoke_jobs(jobs):
    """Select the freeze's smoke-gate jobs.

    Parameters
    ----------
    jobs : `list` [`dict`]
        Job table entries carrying ``template`` and ``image_side_px``.

    Returns
    -------
    smokes : `list` [`dict`]
        The smallest-image member of each source template, in template
        order.
    """
    by_template = {}
    for job in jobs:
        current = by_template.get(job["template"])
        if current is None or (
            (job["image_side_px"], job["run_name"])
            < (current["image_side_px"], current["run_name"])
        ):
            by_template[job["template"]] = job
    return [by_template[key] for key in sorted(by_template)]


def main(argv=None) -> None:
    """Stage the campaign directory, manifest and queues."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent_run", help="ladder_parent_v1 run directory")
    parser.add_argument(
        "selected_run", help="ladder_selected_v1 run directory"
    )
    parser.add_argument("campaign_dir", help="Campaign directory to create")
    args = parser.parse_args(argv)

    from hwoslaps.provenance import (
        config_hash,
        revision_digest,
        revision_provenance,
    )

    protocol = load_protocol()
    arms = protocol["arms"]
    entropy = int(protocol["seeds"]["entropy"])

    revision = revision_provenance()
    digest = revision_digest(revision)
    if revision["git_dirty"]:
        raise ValueError(
            "Refusing to generate a campaign from a dirty tree: "
            f"{revision['git_dirty_paths']}"
        )

    campaign_dir = Path(args.campaign_dir)
    config_out = campaign_dir/"configs"
    if (campaign_dir/"manifest.json").exists():
        raise ValueError(
            f"{campaign_dir} already holds a manifest; refusing to regenerate"
        )
    config_out.mkdir(parents=True, exist_ok=True)

    members = sample_members(Path(args.parent_run), Path(args.selected_run))

    jobs = []
    for member in members:
        with open(member["config"], encoding="utf-8") as stream:
            staged = yaml.safe_load(stream)
        original_hash = config_hash(staged)
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
        arm_names = eligible_arms(member, arms)
        jobs.append({
            **member,
            "restamped_config": str(restamped_path),
            "original_code_revision": original_revision,
            "original_config_hash": original_hash,
            "restamped_config_hash": config_hash(staged),
            "template": str(staged["stage0"]["source_template"]),
            "image_side_px": image_side_px(staged),
            "output_dir": str(campaign_dir/"outputs"/member["run_name"]),
            "arms": {
                name: {
                    "arm_index": arms[name]["arm_index"],
                    "sampler_seed": derive_sampler_seed(
                        entropy, index, arms[name]["arm_index"]
                    ),
                }
                for name in arm_names
            },
        })

    largest_first = sorted(
        jobs, key=lambda job: job["image_side_px"], reverse=True
    )
    positions_lines = [
        f"{job['restamped_config']} {job['ladder_artifact']} "
        f"{job['output_dir']}"
        for job in largest_first
    ]

    def fit_line(job, arm):
        return (
            f"{job['restamped_config']} "
            f"{job['output_dir']}/{POSITIONS_ARTIFACT_NAME} {arm} "
            f"{job['output_dir']}"
        )

    smokes = smoke_jobs(jobs)
    smoke_lines = [fit_line(job, "asimov_injected") for job in smokes]
    fits_lines = [
        fit_line(job, arm)
        for job in largest_first
        for arm in job["arms"]
    ]
    (campaign_dir/"positions_queue.txt").write_text(
        "\n".join(positions_lines) + "\n", encoding="utf-8"
    )
    (campaign_dir/"smokes_queue.txt").write_text(
        "\n".join(smoke_lines) + "\n", encoding="utf-8"
    )
    (campaign_dir/"fits_queue.txt").write_text(
        "\n".join(fits_lines) + "\n", encoding="utf-8"
    )

    manifest = {
        "schema_version": 2,
        "name": "nonlinear_validation_v1",
        "campaign_uuid": str(uuid.uuid4()),
        "design_freeze": {
            "path": "configs/design/design_freeze_v1.yaml",
            "version": 3,
            "protocol_block": "nonlinear_validation",
        },
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
        "arms": {
            name: dict(declaration) for name, declaration in arms.items()
        },
        "n_systems": len(jobs),
        "n_fit_pairs": len(fits_lines),
        "smoke_run_names": [job["run_name"] for job in smokes],
        "jobs": jobs,
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

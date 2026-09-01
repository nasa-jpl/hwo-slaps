#!/usr/bin/env python
"""Amend a nonlinear-validation campaign after a mid-campaign code fix.

The campaign's runners fail closed on the executing tree's revision, so
a job that failed on a code defect cannot rerun until its staged
configuration is restamped to the fixed revision. Restamping the shared
per-system configuration in place would break the identity binding of
that system's already-completed arms, so this script restages the
affected jobs explicitly:

- every listed arm's jobs whose artifact is still missing get a
  restamped configuration copy under ``configs/amended/``;
- the manifest gains an ``amendments`` record carrying the reason, the
  amendment revision and each amended job's configuration hash, which
  ``harvest_nonlinear_validation.py`` verifies fail-closed;
- the matching fit-queue lines are rewritten to the amended
  configurations and the fits cursor is reset so the dispatcher
  resumes (completed jobs are skipped by their artifacts).

Jobs of the listed arms whose artifacts already exist are left alone
and stay bound to the original revision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT/"src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT/"src"))

AMENDED_DIR_NAME = "amended"


def main(argv=None) -> None:
    """Restage the missing jobs of the listed arms."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_dir", help="Campaign directory")
    parser.add_argument("reason", help="One-line amendment reason")
    parser.add_argument(
        "arms", nargs="+", help="Arm names whose missing jobs to restage"
    )
    args = parser.parse_args(argv)

    from hwoslaps.provenance import (
        config_hash,
        revision_digest,
        revision_provenance,
    )

    revision = revision_provenance()
    if revision["git_dirty"]:
        raise SystemExit(
            "Refusing to amend from a dirty tree: "
            f"{revision['git_dirty_paths']}"
        )
    digest = revision_digest(revision)
    stamp = {
        "git_hash": revision["git_hash"],
        "git_dirty": revision["git_dirty"],
        "sha256": digest,
    }

    campaign_dir = Path(args.campaign_dir)
    manifest_path = campaign_dir/"manifest.json"
    with open(manifest_path, encoding="utf-8") as stream:
        manifest = json.load(stream)
    if digest == manifest["code_revision"]["sha256"]:
        raise SystemExit(
            "The tree is at the campaign's own revision; nothing to amend"
        )

    amended_dir = campaign_dir/"configs"/AMENDED_DIR_NAME
    amended_dir.mkdir(parents=True, exist_ok=True)

    amended_jobs = {}
    amended_paths = {}
    for job in manifest["jobs"]:
        for arm in args.arms:
            if arm not in job["arms"]:
                continue
            artifact = (
                Path(job["output_dir"])/f"nonlinear_validation_{arm}.json"
            )
            if artifact.is_file():
                continue
            label = f"{job['run_name']}/{arm}"
            amended_path = amended_paths.get(job["run_name"])
            if amended_path is None:
                with open(
                    job["restamped_config"], encoding="utf-8"
                ) as stream:
                    staged = yaml.safe_load(stream)
                staged["stage0"]["code_revision"] = dict(stamp)
                amended_path = amended_dir/f"{job['run_name']}.yaml"
                amended_path.write_text(
                    yaml.safe_dump(staged, sort_keys=False),
                    encoding="utf-8",
                )
                amended_paths[job["run_name"]] = amended_path
                amended_hash = config_hash(staged)
            else:
                with open(amended_path, encoding="utf-8") as stream:
                    amended_hash = config_hash(yaml.safe_load(stream))
            amended_jobs[label] = {
                "restamped_config": str(amended_path),
                "restamped_config_hash": amended_hash,
                "original_config": job["restamped_config"],
            }

    if not amended_jobs:
        raise SystemExit(
            f"No missing artifacts found for arms {args.arms}; "
            "nothing to amend"
        )

    queue_path = campaign_dir/"fits_queue.txt"
    lines = queue_path.read_text(encoding="utf-8").splitlines()
    rewritten = 0
    for index, line in enumerate(lines):
        fields = line.split()
        if not fields:
            continue
        arm = fields[2]
        run_name = Path(fields[3]).name
        entry = amended_jobs.get(f"{run_name}/{arm}")
        if entry is None:
            continue
        if fields[0] != entry["original_config"]:
            raise SystemExit(
                f"Queue line {index + 1} configuration {fields[0]} does "
                f"not match the manifest's {entry['original_config']}"
            )
        fields[0] = entry["restamped_config"]
        lines[index] = " ".join(fields)
        rewritten += 1
    if rewritten != len(amended_jobs):
        raise SystemExit(
            f"Rewrote {rewritten} queue lines for {len(amended_jobs)} "
            "amended jobs; queue and manifest disagree"
        )
    queue_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest.setdefault("amendments", []).append({
        "reason": args.reason,
        "code_revision": stamp,
        "jobs": amended_jobs,
    })
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    cursor = campaign_dir/".fits_cursor"
    if cursor.exists():
        cursor.write_text("0\n", encoding="utf-8")

    print(
        f"Amended {len(amended_jobs)} jobs at revision {digest[:16]} "
        f"(git {revision['git_hash'][:7]}):"
    )
    for label in sorted(amended_jobs):
        print(f"  {label}")


if __name__ == "__main__":
    main()

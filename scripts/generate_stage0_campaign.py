#!/usr/bin/env python
"""Generate the Stage 0 S1-lite campaign from the design freeze.

Reads ``configs/design/design_freeze_v1.yaml``, verifies every artifact
the freeze binds by digest, samples the declared pool deterministically,
extracts each system's ``theta_E_eff`` with the frozen D-F7 algorithm,
sizes each grid from that aperture, and writes the campaign manifest
beside the design catalogue.

The freeze is authoritative. The pool size and the runner are read from
it and this driver offers no flag that could re-declare either, so a
generated campaign always carries the design its freeze digest names.

This generator never runs a job. Pass ``--validate`` to check the written
manifest against the S1-lite schema and the digests it binds, and stop.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT/"src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT/"src"))


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        help="Directory the manifest and catalogue are written to",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Campaign output root recorded in the manifest",
    )
    parser.add_argument(
        "--design-freeze",
        default=None,
        help="Design freeze artifact; defaults to the committed one",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Interpreter the runner command invokes",
    )
    parser.add_argument(
        "--campaign-name",
        default="stage0_pool",
        help="S1-lite campaign name",
    )
    parser.add_argument(
        "--campaign-uuid",
        default=None,
        help="Pinned campaign UUID; omit to let the freeze step generate one",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the written manifest and the digests it binds",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="Report progress after every N systems; 0 stays silent",
    )
    return parser


def _progress_reporter(every: int):
    """Return a progress callable, or None when reporting is off."""
    if every < 1:
        return None

    def report(done: int, total: int) -> None:
        """Print one progress line at the requested cadence."""
        if done % every == 0 or done == total:
            print(f"  resolved {done}/{total} systems", flush=True)

    return report


def main(argv=None) -> None:
    """Write the Stage 0 manifest and catalogue, then report the pool."""
    args = _build_parser().parse_args(argv)

    from hwoslaps.campaign.design_freeze import (
        design_freeze_digest,
        load_design_freeze,
        verify_bound_artifacts,
    )
    from hwoslaps.campaign.stage0 import (
        validate_stage0_manifest,
        write_stage0_campaign,
    )

    freeze = load_design_freeze(args.design_freeze)
    bound = verify_bound_artifacts(freeze)
    written = write_stage0_campaign(
        args.directory,
        freeze,
        output_root=args.output_root,
        runner_command=[
            args.python,
            str(freeze["stage0"]["runner"]),
            "{config}",
        ],
        freeze_path=args.design_freeze,
        campaign_name=args.campaign_name,
        campaign_uuid=args.campaign_uuid,
        progress=_progress_reporter(args.progress_every),
    )
    if args.validate:
        validate_stage0_manifest(written["manifest_path"])

    summary = written["summary"]
    grid = summary["grid"]
    print(f"design freeze          : {design_freeze_digest(args.design_freeze)}")
    print(f"design freeze status   : {freeze['freeze']['status']}")
    print(f"provisional items      : "
          f"{[item['id'] for item in freeze['provisional_items']]}")
    print(f"bound artifacts        : {len(bound['verified'])} verified, absent "
          f"{bound['absent']}")
    print(f"manifest               : {written['manifest_path']}")
    print(f"manifest sha256        : {written['manifest_sha256']}")
    print(f"catalogue              : {written['catalogue_path']}")
    print(f"catalogue sha256       : {written['catalogue_sha256']}")
    print(f"jobs                   : {written['n_jobs']}")
    print(f"template balance       : {summary['template_balance']}")
    print(f"theta_E floor survival : "
          f"{summary['theta_e_floor_survival_fraction']:.4f} "
          f"({summary['theta_e_floor_survivors']}/{summary['n_systems']})")
    print(f"theta_E design p50/p99 : "
          f"{summary['quantiles']['theta_e_design_arcsec']['p50']:.4f} / "
          f"{summary['quantiles']['theta_e_design_arcsec']['p99']:.4f} arcsec")
    print(f"grid side px           : {grid['min_side_px']} to "
          f"{grid['max_side_px']} (declared max "
          f"{grid['declared_max_side_px']})")
    print(f"capped systems         : {len(grid['capped_systems'])} "
          f"{grid['capped_systems'][:8]}")
    print(f"theta_E realized/design: "
          f"{summary['theta_e_eff']['min_realized_over_design']:.6f} to "
          f"{summary['theta_e_eff']['max_realized_over_design']:.6f}")


if __name__ == "__main__":
    main()

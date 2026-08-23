#!/usr/bin/env python
"""Generate one tier's adaptive Fisher ladder S1-lite campaign.

Reads ``configs/design/design_freeze_v1.yaml``, verifies every artifact
the freeze binds by digest, reconciles the harvested Stage 0 campaign
against the layer 2 selection artifact, and writes the ladder manifest
for the requested tier: the 48 stratified representative members of
layer 3, or the 12 top-by-score members of layer 4 with the golden 5
flagged inside them.

The freeze is authoritative. The tier sizes, the aperture rule, the grid
sizing rule and the mass-ladder policy are read from it and this driver
offers no flag that could re-declare any of them, so a generated
campaign always carries the design its freeze digest names. A freeze
that is not ratified is refused: the strata freeze_order clause admits
an injected-subhalo job only after the selection is frozen and hashed.

This generator never runs a job. Pass ``--validate`` to check the
written manifest against the S1-lite schema, the digests it binds, the
frozen tier sizes, the aperture and grid arithmetic, the ladder policy
and the no-random-stream declaration, and stop.
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
        help="Directory the manifest is written to",
    )
    parser.add_argument(
        "--tier",
        required=True,
        choices=["parent", "selected"],
        help="Ladder tier to emit: the parent 48 or the selected 12",
    )
    parser.add_argument(
        "--stage0-root",
        required=True,
        help="Harvested Stage 0 campaign root the members are read from",
    )
    parser.add_argument(
        "--selection-artifact",
        required=True,
        help="Layer 2 selection freeze artifact of the same campaign",
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
        default=None,
        help="S1-lite campaign name; defaults to ladder_<tier>",
    )
    parser.add_argument(
        "--campaign-uuid",
        default=None,
        help="Pinned campaign UUID; omit to let the freeze step generate one",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the written manifest and everything it binds",
    )
    return parser


def main(argv=None) -> None:
    """Write one tier's ladder manifest, then report the tier."""
    args = _build_parser().parse_args(argv)

    from hwoslaps.campaign.design_freeze import (
        design_freeze_digest,
        load_design_freeze,
        verify_bound_artifacts,
    )
    from hwoslaps.campaign.ladder import (
        LADDER_RUNNER,
        validate_ladder_manifest,
        write_ladder_campaign,
    )

    freeze = load_design_freeze(args.design_freeze)
    bound = verify_bound_artifacts(freeze)
    written = write_ladder_campaign(
        args.directory,
        freeze,
        tier=args.tier,
        stage0_root=args.stage0_root,
        selection_artifact=args.selection_artifact,
        output_root=args.output_root,
        runner_command=[args.python, str(LADDER_RUNNER), "{config}"],
        freeze_path=args.design_freeze,
        campaign_name=args.campaign_name,
        campaign_uuid=args.campaign_uuid,
    )
    if args.validate:
        validate_ladder_manifest(written["manifest_path"])

    summary = written["summary"]
    print(f"design freeze          : {design_freeze_digest(args.design_freeze)}")
    print(f"design freeze status   : {freeze['freeze']['status']}")
    print(f"bound artifacts        : {len(bound['verified'])} verified, absent "
          f"{bound['absent']}")
    print(f"tier                   : {summary['tier']}")
    print(f"manifest               : {written['manifest_path']}")
    print(f"manifest sha256        : {written['manifest_sha256']}")
    print(f"jobs                   : {written['n_jobs']}")
    print(f"golden members         : {len(summary['golden_system_ids'])} "
          f"{summary['golden_system_ids']}")
    print(f"parent overlap         : "
          f"{len(summary['parent_overlap_system_ids'])} of "
          f"{summary['n_jobs']}")
    print(f"theta_E_eff range      : "
          f"{summary['theta_e_eff_arcsec_min']:.4f} to "
          f"{summary['theta_e_eff_arcsec_max']:.4f} arcsec")
    print(f"grid side px           : {summary['grid_side_px_min']} to "
          f"{summary['grid_side_px_max']} (declared max "
          f"{summary['declared_max_side_px']})")
    print(f"perimeter capped       : "
          f"{len(summary['perimeter_capped_system_ids'])} "
          f"{summary['perimeter_capped_system_ids']}")


if __name__ == "__main__":
    main()

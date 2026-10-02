#!/usr/bin/env python3
"""Prepare the v7 nonlinear release catalog without rendering or fitting."""

from __future__ import annotations

import argparse
from pathlib import Path

import sys
_STUDY_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(_STUDY_REPO_ROOT), str(_STUDY_REPO_ROOT / "src")]


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from studies.rasti.campaign.release_catalog import build_catalog  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-freeze",
        type=Path,
        default=REPO_ROOT / "configs/design/design_freeze_v7.yaml",
    )
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--dispatch", type=Path, required=True)
    parser.add_argument("--v6-manifest", type=Path, required=True)
    parser.add_argument(
        "--v6-output-root",
        type=Path,
        required=True,
        help=(
            "Actual v6 ladder-output mirror; every selected NPZ is hashed "
            "and identity-checked"
        ),
    )
    parser.add_argument("--cohort-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--local-position-root",
        type=Path,
        help="Local mirror of the remote stage12 root for the three verified positions",
    )
    args = parser.parse_args(argv)
    catalog = build_catalog(
        release_path=args.release_freeze,
        inventory_path=args.inventory,
        dispatch_path=args.dispatch,
        v6_manifest_path=args.v6_manifest,
        cohort_path=args.cohort_reference,
        output_dir=args.output_dir,
        local_position_root=args.local_position_root,
        v6_output_root=args.v6_output_root,
    )
    print(
        "Prepared v7 catalog: "
        f"{catalog['counts']['execution_catalog']} unique cases; "
        f"{catalog['limitations']['new_position_extraction_pending_count']} new positions pending; "
        f"{catalog['limitations']['h0_bracket_generation_pending_count']} brackets pending"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

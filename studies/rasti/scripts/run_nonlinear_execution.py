#!/usr/bin/env python3
"""Activate an exactly approved nonlinear batch and supervise it on xtx."""

import argparse
import sys
from pathlib import Path

_STUDY_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(_STUDY_REPO_ROOT), str(_STUDY_REPO_ROOT / "src")]

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from studies.rasti.campaign.execution_prepare import activate  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument(
        "--launch-approved",
        action="store_true",
        required=True,
        help="Explicit launch action; requires external exact-batch approval",
    )
    args = parser.parse_args(argv)
    return activate(args.manifest, args.approval)


if __name__ == "__main__":
    raise SystemExit(main())

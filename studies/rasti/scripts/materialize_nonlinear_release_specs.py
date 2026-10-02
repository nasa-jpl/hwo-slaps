#!/usr/bin/env python3
"""Materialize ready v7 case specs without launching a scientific run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
_STUDY_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(_STUDY_REPO_ROOT), str(_STUDY_REPO_ROOT / "src")]


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from studies.rasti.campaign.release_catalog import (  # noqa: E402
    materialize_ready_cases,
    sha256_file,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--revision", type=Path, required=True)
    parser.add_argument("--approval-receipt", type=Path)
    parser.add_argument("--catalog-sha256")
    args = parser.parse_args(argv)
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    revision = json.loads(args.revision.read_text(encoding="utf-8"))
    approval = None
    approval_path = None
    if args.approval_receipt is not None:
        approval_path = str(args.approval_receipt.expanduser().resolve())
        approval = json.loads(args.approval_receipt.read_text(encoding="utf-8"))
        approval["path"] = approval_path
        approval["sha256"] = sha256_file(args.approval_receipt)
    manifest = materialize_ready_cases(
        catalog,
        args.output_root,
        revision,
        approval_receipt=approval,
        approval_receipt_path=approval_path,
        catalog_sha256=args.catalog_sha256,
    )
    print(
        "Materialized v7 specs: "
        f"{manifest['materialized_case_count']} ready cases; "
        f"{manifest['pending_case_count']} cases remain pending"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Prepare review specs or resolve approved production dependency artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_STUDY_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(_STUDY_REPO_ROOT), str(_STUDY_REPO_ROOT / "src")]

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from studies.rasti.campaign.release_catalog import materialize_ready_cases, sha256_file
from studies.rasti.campaign.release_dependencies import (
    _write_new,
    produce_bracket,
    produce_position,
    resolve_bracket,
    resolve_position,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "materialize-ready",
            "produce-position",
            "resolve-position",
            "produce-bracket",
            "resolve-bracket",
        ),
    )
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execution-output-root", type=Path)
    parser.add_argument(
        "--revision", type=Path, help="JSON with git_hash/git_dirty/sha256"
    )
    parser.add_argument("--task-id")
    parser.add_argument("--case-id")
    parser.add_argument("--completion-receipt", type=Path)
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--release-freeze", type=Path)
    args = parser.parse_args(argv)
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())
    catalog = json.loads(args.catalog.read_text())
    digest = sha256_file(args.catalog)
    if args.action == "resolve-bracket":
        if not args.case_id or not args.completion_receipt:
            parser.error("resolve-bracket requires case-id and completion-receipt")
        case = resolve_bracket(catalog, args.case_id, args.completion_receipt, digest)
        _write_new(args.output, {"parent_catalog_sha256": digest, "cases": [case]})
    elif args.action == "resolve-position":
        if not args.task_id or not args.completion_receipt:
            parser.error("resolve-position requires task-id and completion-receipt")
        cases = resolve_position(catalog, args.task_id, args.completion_receipt, digest)
        _write_new(
            args.output,
            {
                "schema_version": 1,
                "parent_catalog_sha256": digest,
                "task_id": args.task_id,
                "cases": cases,
            },
        )
    else:
        if not args.revision:
            parser.error("revision is required")
        revision = json.loads(args.revision.read_text())
        if args.action == "materialize-ready":
            # Review preparation never infers launch authority from CLI flags.
            materialize_ready_cases(
                catalog,
                args.output,
                revision,
                catalog_sha256=catalog.get("parent_catalog_sha256", digest),
                execution_output_root=args.execution_output_root,
            )
        else:
            identity = (
                args.case_id if args.action == "produce-bracket" else args.task_id
            )
            if not identity or not args.approval or not args.release_freeze:
                parser.error(
                    "produce-position requires task-id, approval and release-freeze"
                )
            producer = (
                produce_bracket
                if args.action == "produce-bracket"
                else produce_position
            )
            # Existing generators resolve source assets against the checkout.
            os.chdir(ROOT)
            producer(
                args.catalog,
                identity,
                args.output,
                revision,
                args.approval,
                args.release_freeze,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

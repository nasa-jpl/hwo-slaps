#!/usr/bin/env python3
"""Harvest explicit v7 production attempts without loading scientific runtimes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_STUDY_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(_STUDY_REPO_ROOT), str(_STUDY_REPO_ROOT / "src")]

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from studies.rasti.campaign.production_harvest import harvest_production, write_harvest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument(
        "--attempt-spec-list",
        type=Path,
        required=True,
        help="JSON list of explicit case-spec paths; use [] before launch",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--path-map", action="append", default=[], metavar="REMOTE=LOCAL"
    )
    args = parser.parse_args(argv)
    specs = json.loads(args.attempt_spec_list.read_text())
    if not isinstance(specs, list) or any(not isinstance(p, str) for p in specs):
        parser.error("attempt-spec-list must be a JSON list of path strings")
    mappings = {}
    for value in args.path_map:
        if "=" not in value:
            parser.error("path-map requires REMOTE=LOCAL")
        source, destination = value.split("=", 1)
        if not source or not destination:
            parser.error("path-map cannot contain an empty path")
        mappings[source] = destination
    result = harvest_production(args.catalog, specs, path_mappings=mappings)
    write_harvest(result, args.output_dir)
    print(json.dumps({"status": result["status"], "views": result["views"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

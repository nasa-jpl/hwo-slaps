#!/usr/bin/env python3
"""
Prepare a reviewable nonlinear execution batch without launch or approval.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hwoslaps.campaign.execution_prepare import prepare, validate_prepared  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", type=Path, help="Validate an already prepared manifest only")
    parser.add_argument(
        "--synthetic-example",
        action="store_true",
        help="Irreversibly mark a nonlaunchable example",
    )
    parser.add_argument("--spec", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--task-root", type=Path)
    parser.add_argument("--source-worktree", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--worktree", type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--gpus", type=int, nargs="+")
    parser.add_argument("--wallclock-seconds", type=int)
    parser.add_argument("--worker-wall-seconds", type=int)
    parser.add_argument("--timeout-seconds", type=int)
    parser.add_argument("--admission-buffer-seconds", type=int, default=300)
    args = parser.parse_args(argv)
    if args.validate:
        result = validate_prepared(args.validate)
    else:
        for name in (
            "spec",
            "output_dir",
            "task_root",
            "worktree",
            "python",
            "gpus",
            "wallclock_seconds",
            "worker_wall_seconds",
            "timeout_seconds",
        ):
            if not getattr(args, name):
                parser.error(f"preparation requires --{name.replace('_', '-')}")
        result = prepare(
            args.spec,
            args.output_dir,
            task_root=args.task_root,
            source_worktree=args.source_worktree,
            worktree=args.worktree,
            python=args.python,
            gpus=args.gpus,
            wallclock_seconds=args.wallclock_seconds,
            worker_wall_seconds=args.worker_wall_seconds,
            timeout_seconds=args.timeout_seconds,
            admission_buffer_seconds=args.admission_buffer_seconds,
            synthetic_example=args.synthetic_example,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

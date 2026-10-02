"""Command-line entry point and artifact capture for forecasting runs."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import sys
from typing import Any

import yaml

from .config import load_config, resolve_config_paths, run_directory, validate_or_raise


class _Tee:
    """Write to both the caller's stream and a line-buffered run log."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def run_with_artifacts(
    config: Mapping[str, Any],
    *,
    verbose: bool = True,
    command: Sequence[str] | None = None,
    base_dir: str | Path | None = None,
) -> Any:
    """Run a Python configuration with a resolved snapshot, log and provenance.

    Validation happens before creating output directories. Existing run
    directories are rejected to preserve previous results. The snapshot,
    provenance hash and pipeline all receive the same resolved configuration.
    Heavy scientific imports happen inside log capture. The input mapping is
    never modified; paths in a Python mapping use ``base_dir`` or the caller's
    working directory.
    """
    resolved = resolve_config_paths(config, base_dir=base_dir)
    validate_or_raise(resolved)
    run_dir = run_directory(resolved)
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Run artifact directory already exists: {run_dir}. "
            "Choose a new run_name or output_dir to preserve previous results."
        ) from exc

    snapshot_path = run_dir / "config_used.yaml"
    with snapshot_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(resolved, stream, sort_keys=False)

    log_path = run_dir / "run.log"
    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        with redirect_stdout(_Tee(sys.stdout, log_file)), redirect_stderr(_Tee(sys.stderr, log_file)):
            print(f"Run log: {log_path}")
            print(f"Config snapshot: {snapshot_path}")
            from .pipeline import Pipeline
            from .provenance import write_provenance

            provenance_path = run_dir / "provenance.yaml"
            write_provenance(provenance_path, config=resolved, command=command)
            print(f"Provenance: {provenance_path}")
            return Pipeline(verbose=verbose).run(resolved)


def build_parser() -> argparse.ArgumentParser:
    """Build the installed command parser without scientific imports."""
    parser = argparse.ArgumentParser(description="Run a configurable strong-lensing forecast")
    parser.add_argument(
        "--config", "-c", action="append", required=True, metavar="YAML",
        help="Configuration file; repeat to compose fragments in order",
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress pipeline progress output")
    parser.add_argument("--run-name", help="Override the artifact directory name")
    parser.add_argument(
        "--output-dir", type=Path,
        help="Override output root (relative to current directory)",
    )
    parser.add_argument(
        "--base-dir", type=Path,
        help="Resolve all file-declared relative paths here (for historical configurations)",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Compose and validate the configuration without simulation or output files",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the CLI; Python callers omit the command name from ``argv``."""
    parser = build_parser()
    args = parser.parse_args(argv)
    overrides: dict[str, Any] = {}
    if args.run_name is not None:
        overrides["run_name"] = args.run_name
    if args.output_dir is not None:
        # CLI output overrides always belong to the caller, even when a
        # historical base directory is supplied for YAML inputs.
        overrides["plotting"] = {"output_dir": str(args.output_dir.expanduser().resolve())}
    try:
        config = load_config(args.config, overrides=overrides, base_dir=args.base_dir)
        run_directory(config)
        if args.validate_only:
            print("Configuration valid")
            return 0
        command = list(sys.argv) if argv is None else ["hwoslaps", *argv]
        run_with_artifacts(config, verbose=not args.quiet, command=command)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
